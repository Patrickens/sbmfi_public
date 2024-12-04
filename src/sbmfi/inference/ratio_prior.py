import psutil
import multiprocessing as mp
import math
import cvxpy as cp
import numpy as np
import pandas as pd
import warnings
import torch
from torch.distributions.constraints import Constraint, _Dependent, _Interval
from PolyRound.api import PolyRoundApi
from sbmfi.core.model import LabellingModel, RatioMixin
from sbmfi.core.reaction import LabellingReaction
from sbmfi.core.linalg import LinAlg, TorchBackend, NumpyBackend
from sbmfi.core.polytopia import LabellingPolytope, FluxCoordinateMapper, \
    PolytopeSamplingModel, project_polytope, transform_polytope_keep_transform, \
    H_representation, V_representation

from collections import OrderedDict
from typing import Iterable, Union, Optional, Dict
from torch.distributions import constraints

from sbmfi.inference.priors import _BasePrior

class _RatioSupport(Constraint):
    def __init__(
            self,
            fcm: FluxCoordinateMapper,
            ratio_repo: dict,  # TODO need to figure out condensation and such
            ratio_tol: float = 0.0,
            min_denom_sum: float = 0.0001,
            project=False,
    ):
        # TODO project polytope on ratio-reactions!
        self._ratol = ratio_tol
        self._mds = min_denom_sum

        polytope = fcm._Fn
        normalize = fcm._sampler.kernel_basis != 'rref'
        simpol = PolyRoundApi.simplify_polytope(polytope, settings=fcm._sampler._pr_settings, normalize=normalize)
        polytope = LabellingPolytope.from_Polytope(simpol, polytope)
        if project:
            ratio_reactions = []
            for ratio_id, num_den in ratio_repo.items():
                ratio_reactions.extend(list(num_den['denominator'].keys()))
            ratio_reactions = pd.Index(set(ratio_reactions))
            P = pd.DataFrame(0.0, index=ratio_reactions, columns=polytope.A.columns)
            P.loc[ratio_reactions, ratio_reactions] = np.eye(len(ratio_reactions))
            polytope = project_polytope(polytope, P, number_type='float')
            polytope._objective = {polytope.A.columns[0]: 1.0}

        if len(polytope.objective) == 0:
            # no objective is set; automatically set one or raise error
            raise ValueError('set an objective')

        n_vars = polytope.A.shape[1]  # number of fluxes
        v = cp.Variable(n_vars, name='fluxes')

        locations = [polytope.A.columns.get_loc(reaction_id) for reaction_id in polytope.objective.keys()]
        c = np.zeros(n_vars)
        c[locations] = list(polytope.objective.values())  # what to optimize for
        objective = cp.Maximize(c @ v)

        self._nrat = len(ratio_repo)

        self._numarr = np.zeros((self._nrat, polytope.A.shape[1]), dtype=np.double)
        self._denarr = np.zeros((self._nrat, polytope.A.shape[1]), dtype=np.double)
        for i, (ratio_id, vals) in enumerate(ratio_repo.items()):
            num_idxs = np.array([polytope.A.columns.get_loc(key) for key in vals['numerator'].keys()])
            self._numarr[i, num_idxs] = np.array(list(vals['numerator'].values()), dtype=np.double)

            conden = OrderedDict((key, val) for key, val in vals['denominator'].items() if key not in vals['numerator'])
            den_idxs = np.array([polytope.A.columns.get_loc(key) for key in conden.keys()])
            self._denarr[i, den_idxs] = np.array(list(conden.values()), dtype=np.double)

        self._nlhs = self._nrat
        if ratio_tol > 0.0:
            self._nlhs = self._nrat * 2

        self._lhs = cp.Parameter(shape=(self._nlhs, polytope.A.shape[1]), name='ratio_constraints')
        rhs = cp.Constant(value=np.zeros(self._nlhs))

        if ratio_tol > 0.0:
            ratio_constraints = self._lhs @ v <= rhs
        else:
            ratio_constraints = self._lhs @ v == rhs

        # this is necessary to avoid numerical issues where we get a lot of fluxes in the denominator < 1e-12
        denominator_lhs = cp.Parameter(
            shape=self._numarr.shape, value=self._numarr + self._denarr, name='denominator_constraint'
        )
        denominator_rhs = cp.Constant(value=np.ones(self._nrat) * min_denom_sum)

        constraints = [
            polytope.A.values @ v <= polytope.b.values,
            denominator_lhs   @ v >= denominator_rhs,
            ratio_constraints,
        ]
        if polytope.S is not None:
            constraints.append(
                polytope.S.values @ v == polytope.h.values
            )

        self._problem = cp.Problem(objective=objective, constraints=constraints)

        # now we construct a polytope with ratio constraints
        index = pd.Index(ratio_repo.keys())
        den_lhs_df = pd.DataFrame(-denominator_lhs.value, index=index + '_den', columns=polytope.A.columns)
        den_rhs_df = pd.Series(min_denom_sum, index=index + '_den')
        polytope.A = pd.concat([polytope.A, den_lhs_df], axis=0)
        polytope.b = pd.concat([polytope.b, den_rhs_df])

        if self._ratol > 0.0:
            index  = (index + '_min').append(index + '_max')

        lhs_df = pd.DataFrame(self._lhs.value, index=index, columns=polytope.A.columns)
        rhs_sr = pd.Series(0.0, index=index)
        if ratio_tol > 0.0:
            polytope.A = pd.concat([polytope.A, lhs_df], axis=0)
            polytope.b = pd.concat([polytope.b, rhs_sr])
        else:
            polytope.S = pd.concat([polytope.S, lhs_df], axis=0)
            polytope.h = pd.concat([polytope.h, rhs_sr])

        self._ratio_pol = polytope
        self._reaction_ids = polytope.A.columns
        self._constraint_ids = index

    def construct_polytope_constraints(self, ratio_sample: np.array) -> np.array:
        ratio_sample = np.atleast_2d(ratio_sample)  # means that it is now 2D

        vape = ratio_sample.shape
        viewlue = ratio_sample.reshape((math.prod(vape[:-1]), vape[-1]))

        lhs_proposal = np.zeros(shape=(viewlue.shape[0], *self._lhs.shape))
        lhs_proposal[:, :self._nrat, :] = ((ratio_sample[..., None] - self._ratol) - 1.0) * self._numarr[None, ...]
        lhs_proposal[:, :self._nrat, :] += (ratio_sample[..., None] - self._ratol) * self._denarr[None, ...]

        if self._ratol > 0.0:
            # upper bounds
            lhs_proposal[:, self._nrat:, :] = (1.0 - (ratio_sample[..., None]  + self._ratol)) * self._numarr[None, ...]
            lhs_proposal[:, self._nrat:, :] += -(ratio_sample[..., None]  + self._ratol) * self._denarr[None, ...]
        return lhs_proposal

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if len(value.shape) == 1:
            value = value[None, :]  # means that it is now at least 2D

        vape = value.shape
        viewlue = value.view(vape[:-1].numel(), vape[-1]).to(dtype=torch.double, device='cpu').numpy()

        nv = viewlue.shape[0]
        self._accepted = torch.zeros((nv, ), dtype=torch.bool)
        self._optima = np.zeros((nv, ), dtype=np.double)

        for i in range(viewlue.shape[0]):
            ratio_sample = viewlue[i, :]
            self._lhs.value = self.construct_polytope_constraints(ratio_sample=ratio_sample)[0, ...]
            try:
                optimum = self._problem.solve(solver=cp.GUROBI, verbose=False, max_iter=1000)
                # NOTE sometimes the polytope is not empty according to cvxpy but sampling still fails
                if optimum is not None:
                    self._accepted[i] = True
                    self._optima[i] = optimum
            except:
                pass
        return self._accepted.view(vape[:-1])


def _init_worker(input_q: mp.Queue, output_q: mp.Queue, ratsupp: _RatioSupport):
    global _IQ, _OQ, _RS
    _IQ = input_q
    _OQ = output_q
    _RS = ratsupp


def _ratio_worker():
    global _IQ, _OQ, _RS
    warnings.simplefilter("ignore")
    nacceptot, ntotal = 0, 0
    # print(f'begin {mp.current_process().name}')
    while True:
        task = _IQ.get()
        if isinstance(task, int) and (task == 0):
            _OQ.put(task)
            return
        try:
            ratio_samples = task[1]
            accepted = _RS.check(value=ratio_samples)
            naccepted = accepted.sum()
            nacceptot += naccepted
            ntotal += ratio_samples.shape[0]
            _OQ.put(ratio_samples[accepted])
            if (ntotal > 500) and (nacceptot / ntotal < 0.01):
                raise ValueError(f'Acceptance fraction is below 1%: {nacceptot / ntotal}')
        except Exception as e:
            _OQ.put(e)


def _ratio_listener(output_q: mp.Queue, n, result):
    i = 0
    while i < n:
        oput = output_q.get()
        result.append(oput)
        if isinstance(oput, Exception):
            break
        i += oput.shape[0]


class RatioPrior(_BasePrior):
    def __init__(
            self,
            model: RatioMixin,
            # TODO should always be uniform, otherwise log_probs does not hold!
            cache_size: int = 20000,
            fluxes_subsamples: int = 10,
            num_processes: int = 0,
            algorithm: Union[str, torch.distributions.Distribution] = 'hypercube',
            ratio_tol: float = 0.05,
            min_denom_sum: float = 0.0001,
            coef=0,
    ):
        self._ratio_repo = model.ratio_repo
        self._model = model
        super().__init__(model, cache_size, num_processes)
        if len(model.ratio_repo) == 0:
            raise ValueError('set ratio_repo')
        self._theta_id = model.ratios_id

        if ratio_tol < 0.0:
            raise ValueError('bruegh')
        ratio_tol /= 2.0
        if ratio_tol < RatioMixin._RATIO_ATOL:
            ratio_tol = 0.0
        self._ratol = ratio_tol

        self._mds = min_denom_sum
        self._n_flux = fluxes_subsamples
        self._support = self.support

        self._cache_fill_kwargs['n_flux'] = self._n_flux

        self._naccepted = 9
        self._ntotal = 10
        if coef > 0:
            raise ValueError
        self._coef = coef

        self._lhsides = np.zeros((cache_size, *self._support._lhs.shape), dtype=np.double)

        self._fill_caches = self._fill_caches_rejection
        if algorithm in ('ratio', 'numden'):
            if algorithm == 'ratio':
                # NOTE this one is too restrictive, meaning that there are ratios that are valid
                #   according to rejection sampling that fall outside of this polytope
                self._pol = self.construct_ratio_polytope(self._fcm, model)
            else:
                # NOTE this "works" for sampling ratios, but the distribution is not uniform at all and looks more
                #   like the distribution we get from uniform sampling
                self._pol = self.construct_numden_polytope(self._fcm, model, coef=coef)
            self._vsm = PolytopeSamplingModel(self._pol)
            self._bsp = None  # these are the basis points
            self._fill_caches = self._fill_caches_usm
        elif algorithm == 'hypercube':
            ratio_bounds_df = self.ratio_variability_analysis(self._fcm, model, min_denom_sum=min_denom_sum)
            self._ratio_dist = self.construct_uniform_ratio_sampler(ratio_bounds_df=ratio_bounds_df)
        elif isinstance(algorithm, torch.distributions.Distribution):
            self._ratio_dist = algorithm
        else:
            raise ValueError
        self._algo = algorithm
        self._cache_fill_kwargs['ratio_dist'] = algorithm

    def _get_mp_pool(self, num_processes):
        self._input_q = mp.Queue(maxsize=20)
        self._output_q = mp.Queue()
        return mp.Pool(
            processes=num_processes, initializer=_init_worker,
            initargs=[self._input_q, self._output_q, self._support]
        )

    @staticmethod
    def construct_uniform_ratio_sampler(ratio_bounds_df: pd.DataFrame):
        lo = torch.as_tensor(ratio_bounds_df['min'].values, dtype=torch.double)
        hi = torch.as_tensor(ratio_bounds_df['max'].values, dtype=torch.double)
        return torch.distributions.Uniform(low=lo, high=hi, validate_args=None)

    @staticmethod
    def ratio_variability_analysis(fcm: FluxCoordinateMapper, model: RatioMixin, min_denom_sum = 0.0, positive_numerator=True) -> pd.DataFrame:
        """
        TODO rewrite this such that the objective is a parameter and its value is reset 1 or -1 for the different directions!
            this would be equal to sbmfi.util.generate_cvxpy_LP
        figure out the min and max of every ratio when not constraining others; this helps excluding stuff
        https://en.wikipedia.org/wiki/Fractional_programming
        https://en.wikipedia.org/wiki/Linear-fractional_programming
        :return:
        """
        # net_pol = thermo_2_net_polytope(thermo_pol).H_representation(simplify=True)
        net_pol = fcm._Fn

        n = net_pol.A.shape[1]  # number of fluxes
        objective = cp.Parameter(shape=n, value=np.zeros(n))
        v = cp.Variable(n, name='fluxes')
        t = cp.Variable(1, name='t')  # auxiliary variable for linear fractional programming
        A = net_pol.A.values
        b = net_pol.b.values
        ratio_bounds = {}
        ratio_repo = model.ratio_repo  # this is necessary to get the uncondensed representation

        for ratio_id in model.ratios_id:
            objective.value[:] = 0.0
            d = np.zeros(n)
            c = np.zeros(n)

            num = ratio_repo[ratio_id]['numerator']
            den = ratio_repo[ratio_id]['denominator']
            for flux_id, coeff in den.items():
                index = net_pol.A.columns.get_loc(flux_id)
                d[index] = coeff
                if flux_id in num:
                    objective.value[index] = coeff
                    c[index] = coeff

            constraints = [
                A @ v <= b * t,
                d @ v == 1.0,
                t >= 0.0,
                d @ v >= min_denom_sum, # make sure the denominator is positive
            ]
            if positive_numerator:
                constraints.append(c @ v >= 0.0)  # means that the numerator is also definitely positive
            lfp = cp.Problem(  # linear fractional programme
                objective=cp.Minimize(objective @ v),
                constraints=constraints
            )
            lfp.solve(solver=cp.GUROBI)
            val_min = lfp.value
            if lfp.status != 'optimal':
                raise ValueError(f'ish not a valid ratio: {ratio_id}')
            if val_min < -1e-3:
                print(f'minimum ratio {ratio_id} is: {round(val_min, 3)}')
            ratio_min = max(round(val_min, 3), 0.0)
            objective.value *= -1
            lfp.solve(solver=cp.GUROBI)
            val_max = lfp.value
            if lfp.status != 'optimal':
                raise ValueError(f'ish not a valid ratio: {ratio_id}')
            ratio_max = round(val_max, 3)
            ratio_bounds[ratio_id] = (ratio_min, -ratio_max)
        return pd.DataFrame(ratio_bounds, index=['min', 'max']).T

    @staticmethod
    def construct_ratio_polytope(fcm: FluxCoordinateMapper, model:RatioMixin, tolerance=1e-10):
        raise NotImplementedError
        F_simp = PolyRoundApi.simplify_polytope(fcm._sampler, settings=self._pr_settings, normalize=normalize)
        F_simp = LabellingPolytope.from_Polytope(F_simp, polytope)

        rref_pol, T_1, x = transform_polytope_keep_transform(F_simp, fcm._sampler._pr_settings, 'rref')
        net_flux_vertices = V_representation(rref_pol, number_type='float')
        linalg = NumpyBackend()
        num = model._sum_getter('numerator', model._ratio_repo, linalg, rref_pol.A.columns)
        den = model._sum_getter('denominator', model._ratio_repo, linalg, rref_pol.A.columns)
        ratio_num = num @ net_flux_vertices.T
        ratio_den = den @ net_flux_vertices.T
        ratio_den[ratio_den <= 0.0] = tolerance
        ratios = pd.DataFrame((ratio_num.values / ratio_den.values).T, columns=model.ratios_id).drop_duplicates()
        ratios[ratios < 0.0] = 0.0
        ratios[ratios > 1.0] = 1.0
        A, b = H_representation(vertices=ratios.values)
        return LabellingPolytope(
            A=pd.DataFrame(A, columns=model.ratios_id),
            b=pd.Series(b),
        )

    @staticmethod
    def construct_numden_polytope(fcm:FluxCoordinateMapper, model: RatioMixin, coef=0):
        F_simp = PolyRoundApi.simplify_polytope(fcm._sampler, settings=self._pr_settings, normalize=normalize)
        F_simp = LabellingPolytope.from_Polytope(F_simp, polytope)
        rref_pol = transform_polytope_keep_transform(F_simp, fcm._sampler._pr_settings, 'rref')
        linalg = NumpyBackend()
        num = model._sum_getter('numerator', model._ratio_repo, linalg, rref_pol.A.columns)
        den = model._sum_getter('denominator', model._ratio_repo, linalg, rref_pol.A.columns)
        num_sum = pd.DataFrame(num, columns=rref_pol.A.columns, index=model.ratios_id + '_num')
        den_sum = pd.DataFrame(den, columns=rref_pol.A.columns, index=model.ratios_id + '_den')
        P = pd.concat([
            num_sum,
            den_sum + coef * num_sum.values,
        ])
        return project_polytope(rref_pol, P)

    @property
    def theta_id(self) -> pd.Index:
        return self._theta_id.rename('theta_id')

    @property
    def n_theta(self):
        return len(self._ratio_repo)

    @constraints.dependent_property(is_discrete=False, event_dim=1)  # NB event_dim=1 means that the right-most dimension defines an event!
    def support(self):
        return _RatioSupport(
            fcm=self._fcm,
            ratio_repo=self._ratio_repo,
            ratio_tol=self._ratol,
            min_denom_sum=self._mds
        )

    def _fill_caches_rejection(
            self,
            ratio_dist: torch.distributions.Distribution=None,
            n=1000,
            batch_size=100,  # batches of samples from ratio_dist to check at once; influences m
            n_flux=20,
            close_pool=True,
            break_i=-1
    ):
        # by making ratio_dist an argument, we can later pass intermediate posteriors!

        if ratio_dist is None:
            ratio_dist = self._ratio_dist


        result = []
        nacceptot, ntotal = 0, 0
        m = math.ceil(n * 1.1)  # this is because some will still fail when rounding the polytope!
        while nacceptot < m:
            ratio_samples = ratio_dist.sample((batch_size, ))
            accepted = self._support.check(value=ratio_samples)
            naccepted = accepted.sum()
            nacceptot += naccepted
            ntotal += ratio_samples.shape[0]
            if (ntotal > 500) and (nacceptot / ntotal < 0.01):
                raise ValueError(f'Acceptance fraction is below 1%: {nacceptot / ntotal}')
            result.append(ratio_samples[accepted])

        result = torch.cat(result)
        if n_flux == 0:
            # this is to only fill the ratio_cache, useful for plotting
            self._theta_cache = result[:n, ...]
        else:
            constraint_df = pd.concat([
                pd.DataFrame(ar, index=self._support._constraint_ids, columns=self._support._reaction_ids
                ) for ar in self._support.construct_polytope_constraints(result)[:m]
            ], keys=np.arange(m))
            if self._ratol == 0.0:
                kwargs = {'S_constraint_df': constraint_df}
            else:
                kwargs = {'A_constraint_df': constraint_df}
            sampling_task_generator = sampling_tasks(
                polytope=self._support._ratio_pol, transform_type=self._fcm._sampler.kernel_basis,
                basis_coordinates=self._fcm._sampler.basis_coordinates, linalg=self._fcm._la,
                counts=n_flux, return_kwargs=self._num_processes == 0, return_basis_samples=True,
                **kwargs
            )
            self._run_tasks(sampling_task_generator, break_i=break_i, close_pool=close_pool, format=True)
            # net_fluxes = pd.concat([r['fluxes'] for r in results], ignore_index=True).loc[:, self._fcm._Fn.A.columns]
            # pool = mp.Pool(processes=3)
            # result = pool.starmap(sample_polytope, iterable=sampling_task_generator)
            # result = pd.concat(result, ignore_index=True)
            # self._flux_cache = torch.as_tensor(result.values[:n * n_flux], dtype=torch.double)
            # self._flux_cache = self._fcm._map_thermo_2_fluxes(thermo_fluxes=self._flux_cache)
            print(self._flux_cache.shape)
            print(self._theta_cache.shape)
            self._theta_cache = self._model.compute_ratios(self._flux_cache)
            # # otherwise subsequent samples will have similar ratio values
            # scramble_indices = self._fcm._la.randperm(n * n_flux)
            # self._theta_cache = self._theta_cache[scramble_indices]
            # self._flux_cache = self._flux_cache[scramble_indices]

    def _fill_caches_usm(self, n=1000, close_pool=True, n_flux=20):
        # TODO doesnt fucking work...
        samples, self._bsp = coordinate_hit_and_run_cpp(
            self._vsm, n=n, initial_points=self._bsp, return_basis_samples=True
        )
        samples = torch.as_tensor(samples.values, dtype=torch.double)
        if self._algo == 'numden':
            n_ratios = len(self._model.ratios_id)
            num = samples[:, :n_ratios]
            den = samples[:, n_ratios:] + abs(self._coef) * num
            self._theta_cache = num / den
        else:
            self._theta_cache = samples

        if n_flux > 0:
            raise NotImplementedError

    def log_prob(self, value):
        if self._algo != 'hypercube':
            raise NotImplementedError
        if self._validate_args:
            self._validate_sample(value)

        return self._ratio_dist.log_prob(value=value).sum(-1)


if __name__ == "__main__":
    pass