from sbmfi.inference.priors import _NetFluxPrior
import contextlib, io
import numpy as np
import pandas as pd
import torch
import random
import scipy
from sbmfi.core.reaction import LabellingReaction
from sbmfi.core.polytopia import FluxCoordinateMapper
from sbmfi.inference.priors import sampling_tasks
from collections import OrderedDict
from torch.distributions import Distribution
from pta.sampling.tfs import (
    FreeEnergiesSamplingResult, sample_drg,
    TFSModel, _find_point,
    sample_fluxes_from_drg, PmoProblemPool
)
from pta.constants import R

def get_initial_points(self: TFSModel, num_points: int) -> np.ndarray:

    # NB this has to be called with processes = 1 due to memory errors
    pool = PmoProblemPool(1, *self._pmo_args)

    # Find candidate optimization direactions.
    reaction_idxs_T = list(range(len(self.T.reaction_ids)))
    reaction_idxs_F = [self.F.reaction_ids.index(id) for id in self.T.reaction_ids]
    only_forward_ids_T = [
        i for i in reaction_idxs_T if self.F.lb[reaction_idxs_F[i]] >= 0
    ]
    only_backward_ids_T = [
        i for i in reaction_idxs_T if self.F.ub[reaction_idxs_F[i]] <= 0
    ]
    reversible_ids_T = [
        i
        for i in reaction_idxs_T
        if self.F.lb[reaction_idxs_F[i]] < 0 and self.F.ub[reaction_idxs_F[i]] > 0
    ]

    reversible_dirs = [(i, -1) for i in reversible_ids_T] + [
        (i, 1) for i in reversible_ids_T
    ]
    irreversible_dirs = [(i, -1) for i in only_backward_ids_T] + [
        (i, 1) for i in only_forward_ids_T
    ]

    # Select optimization directions, giving precedence to the reversible reactions.
    if num_points >= len(reversible_dirs):
        directions = reversible_dirs
        directions_pool = irreversible_dirs
        to_sample = min(num_points - len(reversible_dirs), len(irreversible_dirs))
    else:
        directions = []
        directions_pool = reversible_dirs
        to_sample = min(num_points, len(reversible_dirs))
    optimization_directions = directions + random.sample(directions_pool, to_sample)

    # Run the optimizations in the pool.
    initial_points = pool.map(_find_point, optimization_directions)
    assert all(p is not None for p in initial_points), (
        "One or more initial points could not be found. This could be due to "
        "an overconstrained model or numerical inaccuracies."
    )
    points_array = np.hstack(initial_points)
    pool.close()

    return points_array
TFSModel.get_initial_points = get_initial_points


def append_xch_flux_samples(  # TODO make this work without things being dataframes
        self: FluxCoordinateMapper, net_fluxes: pd.DataFrame = None, net_basis_samples: pd.DataFrame = None,
        xch_fluxes: pd.DataFrame = None, pandalize=False, return_type='both'
):
    index = None

    if isinstance(net_fluxes, pd.DataFrame):
        index = net_fluxes.index
        net_fluxes = self._la.get_tensor(values=net_fluxes.loc[:, self._Fn.A.columns].values)
    if isinstance(net_basis_samples, pd.DataFrame):
        index = net_basis_samples.index
        net_basis_samples = self._la.get_tensor(values=net_basis_samples.loc[:, self.net_basis_id].values)

    if net_fluxes is None:
        n = net_basis_samples.shape[0]
        if return_type in ['fluxes', 'both']:
            net_fluxes = self._sampler.to_net_fluxes(net_basis_samples, is_rounded=self._sampler._bascoor == 'rounded')
    elif net_basis_samples is None:
        n = net_fluxes.shape[0]
        if return_type in ['theta', 'both']:
            net_basis_samples = self._sampler.to_net_basis(net_fluxes)
    else:
        n = net_fluxes.shape[0]

    if self._fcm._nx > 0:
        if xch_fluxes is None:
            xch_fluxes = self._la.sample_bounded_distribution(
                shape=(n,), lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1]
            )
        elif isinstance(xch_fluxes, pd.DataFrame):
            xch_fluxes = self._la.get_tensor(values=xch_fluxes.loc[:, self._fwd_id + '_xch'].values)
        if return_type in ['fluxes', 'both']:
            thermo_fluxes = self._la.cat([net_fluxes, xch_fluxes], dim=-1)
    else:
        thermo_fluxes = net_fluxes

    if return_type in ['fluxes', 'both']:
        fluxes = self.map_thermo_2_fluxes(thermo_fluxes)

    if pandalize and (return_type in ['fluxes', 'both']):
        fluxes = pd.DataFrame(self._la.tonp(fluxes), index=index, columns=self._F.A.columns)
    if return_type == 'fluxes':
        return fluxes

    if self._fcm._nx > 0:
        if self._logxch:
            xch_fluxes = self._logit_xch(xch_fluxes)
        theta = self._la.cat([net_basis_samples, xch_fluxes], dim=-1)
    else:
        theta = net_basis_samples

    if pandalize:
        theta = pd.DataFrame(self._la.tonp(theta), index=index, columns=self.theta_id)
    if return_type == 'theta':
        return theta
    elif return_type == 'both':
        return theta, fluxes

class ThermoPrior(_NetFluxPrior):
    def __init__(
            self,
            model: FluxCoordinateMapper,
            tfs_model: TFSModel,
            coordinates='thermo',
            cache_size: int = 20000,
            num_processes: int = 0,
    ):
        super(ThermoPrior, self).__init__(model, cache_size, num_processes=num_processes)
        self._coords = coordinates
        self._tfs_model = tfs_model
        self._fulabel_pol = None
        self._drg_mvn = self.extract_drg_mvn(tfs_model)
        self._thermo_basis_points = None
        self._orthant_volumes = {}

    @staticmethod
    def reset_rhos(model):
        reset_rhos = []
        for r in model.reactions:
            if isinstance(r, LabellingReaction) and (r.rho_max > 0.0) and (r.rho_max < r._RHO_MAX):
                # this is necessary to make sure the polytope of the support has the correct bounds
                r.rho_min = 0.0
                r.rho_max = r._RHO_MAX
                reset_rhos.append(r.id)
        if len(reset_rhos) > 0:
            print(f'The rho_bounds of reactions: {reset_rhos} has been set to ({0.0, LabellingReaction._RHO_MAX})')
        return model

    def extract_drg_mvn(self, tfs_model: TFSModel, epsilon=1e-12, as_tensor=False) -> Distribution:
        # NB extract mv parameters, this is a shit show due to pint and pickling...
        tfs_reaction_ids = pd.Index(tfs_model.T.reaction_ids)
        indices = np.array([tfs_reaction_ids.get_loc(rid) for rid in self._fcm.fwd_id])

        T = tfs_model.T.parameters.T().model

        dfg0_prime_mean = tfs_model.T.dfg0_prime_mean.model
        log_conc_mean = tfs_model.T.log_conc_mean.model
        dfg_prime_mean = dfg0_prime_mean + log_conc_mean * R.model * T
        S_constraints = tfs_model.T.S_constraints
        drg_prime_mean = (S_constraints.T @ dfg_prime_mean)[indices]

        dfg0_prime_cov_sqrt = tfs_model.T._dfg0_prime_cov_sqrt.model
        dfg0_prime_cov = dfg0_prime_cov_sqrt @ dfg0_prime_cov_sqrt.T
        dfg_prime_cov = dfg0_prime_cov + tfs_model.T.log_conc_cov.model * (R.model * T) ** 2
        drg_prime_cov = (S_constraints.T @ dfg_prime_cov @ S_constraints)[:, indices][indices, :]

        psd = False
        eye = np.eye(drg_prime_cov.shape[0])
        tot_eps = 0.0
        while not psd:
            try:
                np.linalg.cholesky(drg_prime_cov)
                psd = True
                if self._fcm._pr_settings.verbose:
                    print(f'total correction epsilon to make matrix PSD: {tot_eps}')
            except:
                # TODO this is a fancier way of fixing non-PSD, but I dont get it: https://stackoverflow.com/a/66902455
                # u, s, v = np.linalg.svd(drg_prime_cov)
                # s[s < 1e-12] += epsilon
                # drg_prime_cov = u @ (np.diag(s)) @ v.T
                tot_eps += epsilon
                drg_prime_cov += epsilon * eye

        if as_tensor:
            return torch.distributions.MultivariateNormal(
                # TODO this covariance matrix is degenerate, this is why we need to work in the other basis...
                loc=torch.as_tensor(drg_prime_mean, dtype=torch.double).squeeze(),
                covariance_matrix=torch.as_tensor(drg_prime_cov, dtype=torch.double)
            )
        else:
            return scipy.stats.multivariate_normal(mean=drg_prime_mean.squeeze(), cov=drg_prime_cov)

    def _compute_orthant_volume(self, thermo_fluxes):
        pass
        # orthants = thermo_fluxes.loc[:, fcm.fwd_id] > 0
        # orthants.index = orthants.apply(lambda row: hash(tuple(row)), raw=True, axis=1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        if self._coords == 'labelling':
            #
            raise NotImplementedError(
                'this would mean that we need to compute the volume of a thermo_constrained_label_pol, '
                'which is a biatch due to large number of dimensions')
        # this one will be very difficult to implement, since we need to convert exchange fluxes to dG and evaluate those
        thermo_fluxes = self.map_theta_2_fluxes(value, return_thermo=True)
        drg = self._fcm.compute_dgibbsr(thermo_fluxes)
        orhtant_vols = self._compute_orthant_volume(thermo_fluxes)

    def _generate_thermo_constrained_label_pol(self, abs_xch_flux_tol=0.0):
        pol = self._fcm._F.copy()

        if abs_xch_flux_tol == 0.0:
            S_df = pd.DataFrame(
                0.0,
                index=(self._fcm.fwd_id + '_rho'),
                columns=pol.A.columns,
            )
            rho_rev_idx = np.zeros(S_df.shape, dtype=bool)
            rho_rev_idx[
                [S_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho'],
                [S_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
            ] = True
            S_df[rho_rev_idx] = -1.0
            rho_constraints = pol.S.index.str.contains('_rho')
            pol.S = pd.concat([pol.S.loc[~rho_constraints], S_df], axis=0).sort_index(axis=0)
            pol.h = pd.concat([pol.h.loc[~rho_constraints], pd.Series(0.0, index=S_df.index)]).loc[pol.S.index]
        else:
            A_df = pd.DataFrame(
                0.0,
                index=(self._fcm.fwd_id + '_rho_min|lb').union((self._fcm.fwd_id + '_rho_max|ub')),
                columns=pol.A.columns,
            )
            rho_rev_idx = np.zeros(A_df.shape, dtype=bool)
            rho_rev_idx[
                [A_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_min|lb'],
                [A_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
            ] = True
            A_df[rho_rev_idx] = -1.0
            rho_rev_idx[:] = False
            rho_rev_idx[
                [A_df.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_max|ub'],
                [A_df.columns.get_loc(key) for key in self._fcm.fwd_id + '_rev']
            ] = True
            A_df[rho_rev_idx] = 1.0

            rho_constraints = pol.A.index.str.contains('_rho')
            pol.A = pd.concat([pol.A.loc[~rho_constraints], A_df], axis=0).sort_index(axis=0)
            pol.b = pd.concat([pol.b.loc[~rho_constraints], pd.Series(0.0, index=A_df.index)]).loc[pol.A.index]
        return pol

    @staticmethod
    def _sample_drg_suppress_output(
            tfs_model: TFSModel,
            n: int = 20000,
            num_chains: int = 4,
            thermo_basis_points: np.array = None
    ):
        if thermo_basis_points is None:
            thermo_basis_points = tfs_model.get_initial_points(num_chains)

        num_chains = thermo_basis_points.shape[1]
        with contextlib.redirect_stdout(io.StringIO()):
            result = sample_drg(
                tfs_model,
                initial_points=thermo_basis_points,
                num_direction_samples=n,
                num_samples=n,
                max_psrf=1.0 + 1e-12,
                num_chains=num_chains,
                max_steps = 32 * n,
                num_initial_steps = n * 2,
                # max_threads=num_chains,
            )
        new_points_idx = np.random.choice(n, num_chains)
        new_thermo_basis_points = result.basis_samples.values[new_points_idx, :].T
        return result, new_thermo_basis_points

    def _make_b_constraint_df(self, orthants):
        forward = orthants.astype(int)
        reverse = (~orthants).astype(int)
        forward.columns += '|ub'
        reverse.columns += '|lb'

        b = self._fcm._Fn.b.copy()

        ub = forward * b.loc[forward.columns]
        lb = reverse * b.loc[reverse.columns]

        # NB this dataframe contains all the bounds for the b vector
        return pd.concat([lb, ub], axis=1).sort_index(axis=1)

    def _make_Fn_tasks(self, drg_result: FreeEnergiesSamplingResult, n_flux=1):
        drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
        drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)
        drg_orthants = drg_samples < 0.0

        orthants = (drg_orthants).value_counts().reset_index().rename({0: 'counts'}, axis=1).set_index('counts')
        orthants = orthants.loc[orthants.index > 1]
        orthants.index *= n_flux  # this means we take subsamples of the net space

        b_constraint_df = self._make_b_constraint_df(orthants)
        task_generator = sampling_tasks(  # transform_type=self._fcm.transform_type, basis_coordinates=self._fcm.basis_coordinates
            self._fcm._Fn, counts=None, to_basis_fn=self._fcm.to_basis_fn,
            b_constraint_df=b_constraint_df, return_basis_samples=True,
            return_kwargs=self._num_processes == 0,
        )
        return dict(
            orthants=orthants, drg_orthants=drg_orthants, b_constraint_df=b_constraint_df,
            drg_xch_fluxes=drg_xch_fluxes, sampling_task_generator=task_generator
        )

    def _make_S_constraint_df(self, drg_result: FreeEnergiesSamplingResult, S: pd.DataFrame):
        drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
        drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)
        S_template = pd.DataFrame(0.0, index=S.index[S.index.str.contains('rho')], columns=self._fcm.fwd_id)
        drg_xch_fluxes.columns += '_rho'
        drg_xch_fluxes = drg_xch_fluxes.clip(0.0, LabellingReaction._RHO_MAX)
        reverse_dir = (drg_samples > 0.0).values
        drg_xch_fluxes.values[reverse_dir] = (1.0 / drg_xch_fluxes).values[reverse_dir]
        rho_idx = np.zeros(S_template.shape, dtype=bool)
        rho_idx[
            [S_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho'],
            [S_template.columns.get_loc(key) for key in self._fcm.fwd_id]
        ] = True

        S_constraints = OrderedDict()
        for i, name in enumerate(drg_samples.index):
            S = np.zeros(S_template.shape).T
            S[rho_idx.T] = drg_xch_fluxes.iloc[i].values
            S_constraints[name] = pd.DataFrame(S.T, index=S_template.index, columns=S_template.columns)

        return pd.concat(S_constraints.values(), keys=drg_samples.index)

    def _make_A_constraint_df(self, drg_result: FreeEnergiesSamplingResult, A: pd.DataFrame, abs_xch_flux_tol=0.05):
        drg_samples = drg_result.samples.loc[:, self._fcm.fwd_id]
        drg_xch_fluxes = self._fcm.compute_xch_fluxes(dgibbsr=drg_samples)

        A_template = pd.DataFrame(0.0, index=A.index[A.index.str.contains('rho')], columns=self._fcm.fwd_id)

        xch_fluxes_max = drg_xch_fluxes + abs_xch_flux_tol / 2
        drg_xch_fluxes -= abs_xch_flux_tol / 2

        drg_xch_fluxes.columns += '_rho_min|lb'
        xch_fluxes_max.columns += '_rho_max|ub'

        drg_xch_fluxes = drg_xch_fluxes.clip(0.0, LabellingReaction._RHO_MAX)
        xch_fluxes_max = xch_fluxes_max.clip(0.0, LabellingReaction._RHO_MAX)

        reverse_dir = (drg_samples > 0.0).values
        drg_xch_fluxes.values[reverse_dir] = (1.0 / drg_xch_fluxes).values[reverse_dir]
        xch_fluxes_max.values[reverse_dir] = (1.0 / xch_fluxes_max).values[reverse_dir]

        rho_fwd_min_idx = np.zeros(A_template.shape, dtype=bool)
        rho_fwd_min_idx[
            [A_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_min|lb'],
            [A_template.columns.get_loc(key) for key in self._fcm.fwd_id]
        ] = True
        rho_fwd_max_idx = np.zeros(A_template.shape, dtype=bool)
        rho_fwd_max_idx[
            [A_template.index.get_loc(key) for key in self._fcm.fwd_id + '_rho_max|ub'],
            [A_template.columns.get_loc(key) for key in self._fcm.fwd_id]
        ] = True

        A_constraints = OrderedDict()
        for i, name in enumerate(drg_samples.index):
            A = np.zeros(A_template.shape).T
            A[rho_fwd_min_idx.T] = drg_xch_fluxes.iloc[i].values
            A[rho_fwd_max_idx.T] = -xch_fluxes_max.iloc[i].values
            A = pd.DataFrame(A.T, index=A_template.index, columns=A_template.columns)
            # NOTE this is essential, otherwise the bounds conflict for rho > 1.0
            A[A < -1.0] = -A[A < -1.0].values
            A[A > 1.0] = -A[A > 1.0].values
            A_constraints[name] = A

        return pd.concat(A_constraints.values(), keys=drg_samples.index)

    def _make_F_tasks(self, drg_result:FreeEnergiesSamplingResult, n_flux=1, abs_xch_flux_tol=0.05):
        pol = self._generate_thermo_constrained_label_pol(abs_xch_flux_tol)
        kwargs = {}
        if abs_xch_flux_tol == 0.0:
            kwargs['S_constraint_df'] = self._make_S_constraint_df(drg_result, pol.S)
        else:
            # TODO not sure this is correct!
            kwargs['A_constraint_df'] = self._make_A_constraint_df(drg_result, pol.A, abs_xch_flux_tol)

        return sampling_tasks(
            pol, counts=n_flux, to_basis_fn=None, return_basis_samples=False,
            return_kwargs=self._num_processes == 0, **kwargs
        )

    def _fill_caches(
            self, n=20000, n_flux=1, drg_result: FreeEnergiesSamplingResult = None,
            abs_xch_flux_tol=0.00, break_i=-1, close_pool=True,
    ):
        if drg_result is None:
            drg_result, self._thermo_basis_points = self._sample_drg_suppress_output(
                self._tfs_model, n, thermo_basis_points=self._thermo_basis_points
            )
        else:
            new_points_idx = np.random.choice(n, self._num_processes)
            self._thermo_basis_points = drg_result.basis_samples.values[new_points_idx, :].T

        if self._coords == 'thermo':
            tasks_dct = self._make_Fn_tasks(drg_result, n_flux)
            orthants = tasks_dct['orthants']
            drg_orthants = tasks_dct['drg_orthants']
            drg_xch_fluxes = tasks_dct['drg_xch_fluxes']
            sampling_task_generator = tasks_dct['sampling_task_generator']

            results = self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool)
            net_fluxes = pd.concat([r['fluxes'] for r in results], ignore_index=True).loc[:, self._fcm._Fn.A.columns]
            xch_cols = self._fcm.fwd_id + '_xch'
            xch_fluxes = pd.DataFrame(np.nan, index=net_fluxes.index, columns=xch_cols)

            flux_orthants = net_fluxes.loc[:, orthants.columns] > 0.0
            drg_orthants = drg_orthants.loc[:, orthants.columns]

            for (count, orthant), res in zip(orthants.iterrows(), results):
                self._orthant_volumes[hash(tuple(orthant))] = res['log_det_E']
                which_drg = (drg_orthants == orthant).all(1)
                which_flux = (flux_orthants == orthant).all(1)
                if which_flux.sum() < which_drg.sum() * n_flux:
                    # NOTE these orthants have 0 hypervolume I guess, thus flux sampling fails
                    raise ValueError('some orthant was not sampled correctly, make sure that the Fluxspace '
                                     'of self._tfs_model and self._model are the same!')

                # print(which_flux.sum(), which_drg.sum() * n_flux, count, which_flux.sum() < which_drg.sum() * n_flux)
                xch_fluxes.loc[which_flux, xch_cols] = pd.concat(
                    [drg_xch_fluxes.loc[which_drg, :], ] * n_flux, axis=0
                ).values

            net_basis_samples = pd.concat([r['basis_samples'] for r in results], ignore_index=True).loc[:, self._fcm.net_basis_id]
            theta, fluxes = self._fcm.append_xch_flux_samples(net_fluxes, net_basis_samples, xch_fluxes)
            # self.ding = self._fcm.map_theta_2_fluxes(self.theta)

        elif self._coords == 'labelling':
            if self._fcm.include_cofactors:
                raise ValueError('mapping from fluxes to thermo wont work if cofactors are included')
            sampling_task_generator = self._make_F_tasks(drg_result, n_flux, abs_xch_flux_tol)
            results = self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool)
            fluxes = pd.concat([r['fluxes'] for r in results], ignore_index=True)
            theta = self._fcm.map_fluxes_2_theta(fluxes)

        self._flux_cache  = torch.as_tensor(fluxes.values, dtype=torch.double)
        self._theta_cache = torch.as_tensor(theta.values, dtype=torch.double)

