from sbmfi.core.simulator import _BaseSimulator, DataSetSim
from sbmfi.inference.priors import _BasePrior, _NetFluxPrior
from sbmfi.core.observation import MDV_ObservationModel, BoundaryObservationModel
from sbmfi.core.model import LabellingModel
import math
import arviz as az
import numpy as np
import pandas as pd
import tqdm
from typing import Dict
from functools import partial
from sbmfi.core.util import profile
from sbi.inference.posteriors.base_posterior import NeuralPosterior
import time
from line_profiler import line_profiler
import xarray as xr

# from line_profiler import line_profiler
# prof2 = line_profiler.LineProfiler()
# from sbmfi.core.util import profile

# profile2 = line_profiler.LineProfiler()
class _BaseBayes(_BaseSimulator):

    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _NetFluxPrior,
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        super(_BaseBayes, self).__init__(model, substrate_df, mdv_observation_models, boundary_observation_model)

        self._prior = prior
        if prior is not None:
            if not prior._fcm.labelling_fluxes_id.equals(model.labelling_fluxes_id):
                raise ValueError('prior has different labelling fluxes than model')
            if not model._fcm.theta_id.equals(prior.theta_id):
                raise ValueError('theta of model and prior are different')

        self._sampler = self._fcm._sampler
        self._K = self._sampler.dimensionality
        self._nx = self._fcm._nx

        self._x_meas = None
        self._x_meas_id = None
        self._true_theta = None
        self._true_theta_id = None

        self._potentype = None
        self._potential_fn = None
        self._sphere_samples = None
        self._A_dist = None

    @property
    def potentype(self):
        if self._potentype is not None:
            return self._potentype[:]

    @property
    def measurements(self):
        return pd.DataFrame(self._la.tonp(self._x_meas), index=self._x_meas_id, columns=self.data_id)

    @property
    def true_theta(self):
        if self._true_theta is None:
            return
        return pd.Series(self._la.tonp(self._true_theta[0]), index=self.theta_id, name=self._true_theta_id)

    def set_measurement(self, x_meas: pd.Series, atol=1e-3):
        if isinstance(x_meas, pd.Series):
            name = 'measurement' if not x_meas.name else x_meas.name
            x_meas = x_meas.to_frame(name=name).T
        x_meas_index = None
        if isinstance(x_meas, pd.DataFrame):
            x_meas_index = x_meas.index
            x_meas = x_meas.values
        if x_meas_index is None:
            x_meas_index = pd.RangeIndex(x_meas.shape[0])
        elif isinstance(x_meas_index, pd.MultiIndex):
            raise ValueError
        self._x_meas = self._la.atleast_2d(self._la.get_tensor(values=x_meas))
        self._x_meas_id = x_meas_index
        if (self._bomsize > 0) and self._bom._check:
            if not self._la.transax((self._bom._A @ self._x_meas[:, -self._bomsize].T <= self._bom._b)).all():
                raise ValueError('boundary measurements are outside polytope')
        x_meas_df = pd.DataFrame(self._la.tonp(self._x_meas), index=x_meas_index, columns=self.data_id)
        for labelling_id, obmod in self._obmods.items():
            obmod.check_x_meas(x_meas_df.loc[:, labelling_id], atol=atol)

    def set_true_theta(self, theta: pd.Series):
        if theta is None:
            return
        if isinstance(theta, pd.DataFrame):
            if theta.shape[0] > 1:
                raise ValueError
            theta = theta.iloc[0]
        self._true_theta = self._la.atleast_2d(self._la.get_tensor(values=theta.loc[self.theta_id].values))
        self._true_theta_id = theta.name

    def simulate_true_data(self, n_obs=0, pandalize=True):
        if self._true_theta is None:
            raise ValueError('set true_theta')
        tt = self._la.tile(self._true_theta.T, (self._la._batch_size, )).T
        true_data = self.simulate(tt, n_obs, pandalize=pandalize)
        if not pandalize:
            return true_data[[0]]
        true_data = true_data.iloc[[0]]
        true_data.index = pd.RangeIndex(true_data.shape[0])
        return true_data

    def log_lik(
            self,
            theta,
            return_data=False,
            sum=True,  # for debugging its useful to have log_lik split out per observation model
    ):

        if not self._is_exact:
            raise ValueError(
                'some observation models do not have a .log_prob, meaning that exact inference is impossible'
            )
        if self._x_meas is None:
            raise ValueError('set measurement')

        mu_o = self.simulate(theta, n_obs=0)

        n_f = self._model._fluxes.shape[0]
        n_meas = self._x_meas.shape[0]
        n_bom = 1 if self._bomsize > 0 else 0

        log_lik = self._la.get_tensor(shape=(n_f, n_meas, len(self._obmods) + n_bom))

        # FUCKING AROOND
        # difff = self._la.get_tensor(shape=(n_f, n_meas, len(self._obmods) + n_bom))
        truedist = ((theta - self._true_theta) ** 2).mean(1)

        if self._bomsize > 0:
            bo_meas = self._x_meas[:, -self._bomsize:]
            mu_bo = mu_o[:, 0, -self._bomsize:]
            log_lik[..., -1] = self._bom.log_lik(bo_meas=bo_meas, mu_bo=mu_bo)

            # FUCKING AROOND
            # mu_bo = self._la.atleast_2d(mu_bo)  # shape = batch x n_bo
            # bo_meas = self._la.atleast_2d(bo_meas)  # shape = n_obs x n_bo
            # diff_bo = mu_bo[:, None, :] - bo_meas[:, None, :]  # shape = batch x n_obs x n_bo
            # difff[..., -1] = (diff_bo ** 2).sum(-1)

        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            j, k = self._obsize[labelling_id]
            x_meas_o = self._x_meas[..., j:k]
            mu_o_i = mu_o[:, 0, j:k]
            ll = obmod.log_lik(x_meas_o, mu_o_i)
            log_lik[..., i] = ll

            # FUCKING AROOND
            # x_meas = self._la.atleast_2d(x_meas_o)  # shape = n_meas x n_mdv
            # mu_oo = self._la.atleast_2d(mu_o_i)  # shape = batch x n_d
            # diff = mu_oo[:, None, :] - x_meas[:, None, :]  # shape = n_obs x batch x n_d
            # difff[..., i] = (diff ** 2).sum(-1)

        if sum:
            log_lik = self._la.sum(log_lik, axis=(1, 2), keepdims=False)

        # print(truedist)
        # print(self._la.sum(log_lik, axis=(1, 2), keepdims=False))
        # print()

        if return_data:
            return log_lik, mu_o
        return log_lik

    def log_prob(
            self,
            theta,
            return_data=False,
            evaluate_prior=False,
            **kwargs,
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = theta.shape
        if len(vape) > 2:
            theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1]))

        n_f = theta.shape[0]
        k = len(self._obmods) + (1 if self._bom is None else 2)  # the 2 is for a column of prior and boundary probabilities
        n_meas = self._x_meas.shape[0]
        log_prob = self._la.get_tensor(shape=(n_f, n_meas, k))

        if evaluate_prior:
            # NB not necessary for uniform prior
            # NB this also checks support! the hr is guaranteed to sample within the support
            # NB since priors are currently torch objects, this will not work with numpy backend
            #   which has proven the faster option for the hr-sampler
            log_prob[..., -1] = self._prior.log_prob(theta)

        log_lik = self.log_lik(theta, return_data, False)
        if return_data:
            log_lik, mu_o = log_lik

        log_prob[..., :-1] = log_lik
        log_prob = self._la.view(self._la.sum(log_prob, axis=(1, 2), keepdims=False), shape=vape[:-1])
        if return_data:
            return log_prob, self._la.view(mu_o, shape=(*vape[:-1], len(self._did)))
        return log_prob

    def compute_distance(
            self,
            theta,
            epsilon,
            n_obs=5,
            metric='rmse',
            return_data=False,
            **kwargs
    ):
        if self._x_meas is None:
            raise ValueError('set an observation first')
        # NB we do not evaluate the log_prob of the measured boundary fluxes, since it is a constant for _x_meas

        vape = theta.shape
        if len(vape) > 2:
            theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1]))
        data = self.__call__(theta, n_obs=n_obs, **kwargs)

        time = None
        if isinstance(data, tuple):
            data, time = data

        data = self._la.unsqueeze(data, 0)  # artificially add a chains dimension!
        if metric == 'rmse':
            fobmod = next(iter(self._obmods.values()))
            distances = fobmod.rmse(data, self._x_meas).squeeze(0)
        else:
            # TODO think of other distance metrics
            raise ValueError

        n_obshape = max(1, n_obs)
        distances = self._la.view(distances, shape=vape[:-1])
        if epsilon > -float('inf'):
            distances[distances > epsilon] = float('nan')  # this indicates we reject samples with a large distance!
        data = self._la.view(data, shape=(*vape[:-1], n_obshape, len(self._did)))
        if return_data:
            distances = distances, data
        if time is not None:
            return distances, time
        return distances

    def evaluate_neural_density(
            self,
            theta,
            potential_fn,
            evaluate_prior=False,
            return_data=False,
    ):
        vape = theta.shape
        theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1]))
        if return_data:
            raise NotImplementedError(
                'its more efficient to sample all paramters and then use a DataSim to simulate all data'
            )

        raise NotImplementedError

    def _set_potential(
            self,
            potentype,
            potential_fn=None,
            **kwargs
    ):
        if potentype == 'exact':
            if not self._is_exact:
                self.log_lik(None)  # this raises the error
            fun = self.log_prob
            kwargs = dict(
                return_data=kwargs.get('return_data', True),
                evaluate_prior=kwargs.get('evaluate_prior', True),
            )
        elif potentype == 'approx':
            fun = self.compute_distance
            kwargs = dict(
                n_obs=kwargs.get('n_obs', 5),
                metric=kwargs.get('metric', 'rmse'),
                return_data=kwargs.get('return_data', True),
                evaluate_prior=kwargs.get('evaluate_prior', True),
            )
        elif potentype == 'density':
            self.evaluate_neural_density
            kwargs = dict(
                track_gradients=False,
                # when we use sequential neural likelihood, we need to evaluate the prior, in SNPE not
                # evaluate_prior=kwargs.get('evaluate_prior', False),
            )
            fun = potential_fn.__call__
        else:
            raise ValueError
        self._potentype = potentype
        self.potential = partial(fun, **kwargs)

    def set_density(self, density: NeuralPosterior):
        raise NotImplementedError

    def mvn_kernel_variance(
            self,
            theta,
            weights=None,
            samples_per_dim: int = 100,
            kernel_std_scale: float = 1.0,
            prev_cov=True,
    ):
        vape = theta.shape
        # shape into matrix with variables along rows and samples in columns
        if theta.ndim > 2:
            theta = self._la.view(theta, shape=(math.prod(vape[:-1]), vape[-1]))

        if prev_cov:
            # Calculate weighted covariance of particles.
            # For variant C, Beaumont et al. 2009, the kernel variance comes from the
            # previous population.
            population_cov = self._la.cov(theta.T, aweights=weights)  # rowvar=False,
            # Make sure variance is nonsingular.
            # I'd rather have this crash out if the singular, means that the parameters are not independent
            #    or constrained to a single value
            # diagonal = self._la.diag(population_cov)
            # diagonal += 0.01
            self._la.cholesky(kernel_std_scale * population_cov)
            return kernel_std_scale * population_cov
        else:
            # Toni et al. and Sisson et al. it comes from the parameter ranges.
            indices = self._la.multinomial(samples_per_dim * theta.shape[1], p=weights)
            samples = theta[indices]
            particle_ranges = self._la.max(samples, 0) - self._la.min(samples, 0)
            return kernel_std_scale * self._la.diag(particle_ranges)

    def quantile_indices(self, distances, quantiles=0.8):
        dist_cumsum = self._la.cumsum(distances, 0)
        dist_cdf = dist_cumsum / dist_cumsum[-1]
        bigger_than_quantile = self._la.get_tensor(values=(dist_cdf >= quantiles), dtype=np.uint8)
        # select the first 1 in the bigger_than_quantile matrix above along the 0 axis
        return self._la.argmax(bigger_than_quantile, 0)

    def _format_dims_coords(self, n_obs=0):
        data_dims = ['data_id']
        coords = {
            'theta_id': self.theta_id.tolist(),
            'measurement_id': self._x_meas_id.tolist(),
            'data_id': [f'{i[0]}: {i[1]}' for i in self.data_id.tolist()],
        }
        if n_obs > 0:
            data_dims = ['obs_idx', 'data_id']
            coords['obs_idx'] = np.arange(n_obs)
        dims = {
            'theta': ['theta_id'],
            'observed_data': ['measurement_id', 'data_id'],
            'data': data_dims,
        }
        return dims, coords

    def simulate_data(
            self,
            inference_data: az.InferenceData = None,
            n=20000,
            theta=None,
            include_predictive=True,
            num_processes=0,
            n_obs=0,
            show_progress=True,
    ):
        model = self._model

        from_prior = theta is None
        if from_prior:
            theta = self._prior.sample(sample_shape=(n,))
            if model._la.backend != 'torch':
                # TODO inconsistency between model and prior LinAlg, where prior has torch backend and model has numpy backend
                theta = self._prior._fcm._la.tonp(theta)

        if (inference_data is None) or not from_prior:
            result = dict(theta=theta[None, :, :])
        else:
            prior_dataset = az.convert_to_dataset(
                {'theta': theta[None, :, :]},
                dims={'theta': ['theta_id']},
                coords={'theta_id': model._fcm.theta_id.tolist()},
            )
            inference_data.add_groups(
                group_dict={'prior': prior_dataset},
            )

        if include_predictive:
            if not hasattr(self, '_dss'):
                dsim = DataSetSim(
                    model=model,
                    substrate_df=self._substrate_df,
                    mdv_observation_models=self._obmods,
                    boundary_observation_model=self._bom,
                    num_processes=num_processes,
                )
            else:
                dsim = self._dss
            data = dsim.simulate_set(theta, n_obs=n_obs, show_progress=show_progress)['data']
            dims = {'data': ['data_id']}
            coords = {'data_id': [f'{i[0]}: {i[1]}' for i in self.data_id.tolist()]}
            if n_obs == 0:
                data = model._la.transax(data, 0, 1)
            else:
                dims['data'] = ['obs_idx', 'data_id']
                data = data[None, :, :, :]

            if (inference_data is None) or not from_prior:
                result['data'] = data
            else:
                prior_dataset = az.convert_to_dataset(
                    {'data': data},
                    dims=dims,
                    coords=coords,
                )
                inference_data.add_groups(
                    group_dict={'prior_predictive': prior_dataset},
                )

        if (inference_data is None) or not from_prior:
            if inference_data is not None:
                print(
                    'returning result instead of adding to inference data, since it is not clear where theta originates')
            return result

    def perturb_particles(
            self,
            theta,
            i,
            batch_shape,
            n_cdf=5,
            chord_proposal='unif',
            chord_std=2.0,
            xch_proposal='gauss',
            xch_std=0.4,
            return_what=1,
    ):
        # TODO implement random coordinate instead of random direction
        # given x, the next point in the chain is x+alpha*r
        #   it also satisfies A(x+alpha*r)<=b which implies A*alpha*r<=b-Ax
        #   so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
        #   b - A @ x is always >= 0, clamping for numerical tolerances

        pre_sample_batch = batch_shape[0]
        ii = i % pre_sample_batch
        if ii == 0:
            # TODO: https://link.springer.com/article/10.1007/BF02591694
            #  implement coordinate hit-and-run (might be faster??)
            # uniform samples from unit ball in batch_shape dims
            self._sphere_samples = self._la.sample_hypersphere(shape=(*batch_shape, self._K))
            # batch compute distances to all planes
            self._A_dist = self._la.tensormul_T(self._sampler._G, self._sphere_samples)

        sphere_sample = self._sphere_samples[[ii]]
        A_dist = self._A_dist[ii]

        pol_dist = self._sampler._h.T - self._la.tensormul_T(self._sampler._G, theta[..., :self._K])
        pol_dist[pol_dist < 0.0] = 0.0
        allpha = pol_dist / A_dist
        alpha_min, alpha_max = self._la.min_pos_max_neg(allpha, return_what=0)

        if chord_std.ndim > 1:
            # this means that we passed a covariance matrix and we need to compute std along the line
            chord_std = self._la.sqrt(self._la.sum(((sphere_sample @ chord_std) * sphere_sample), -1))

        chord_alphas = self._la.sample_bounded_distribution(
            shape=(n_cdf,), lo=alpha_min, hi=alpha_max, which=chord_proposal,
            std=chord_std, return_log_prob=return_what>0
        )
        if return_what > 0:
            chord_alphas, log_probs = chord_alphas
        perturbed_particles = theta[..., :self._K] + chord_alphas[..., None] * sphere_sample
        if self._nx > 0:
            # in case there are exchange fluxes, construct them here
            current_xch = theta[..., -self._nx:]
            xch_fluxes = self._la.sample_bounded_distribution(
                shape=perturbed_particles.shape[:-1],
                lo=self._fcm._rho_bounds[:, 0], hi=self._fcm._rho_bounds[:, 1],
                mu=current_xch, which=xch_proposal, std=xch_std, return_log_prob=return_what>0,
            )
            if return_what > 0:
                xch_fluxes, xch_log_probs = xch_fluxes
                log_probs += self._la.sum(xch_log_probs, -1, keepdims=False)
            perturbed_particles = self._la.cat([perturbed_particles, xch_fluxes], dim=-1)

        if return_what == 0:
            return perturbed_particles
        elif return_what == 1:
            return perturbed_particles, log_probs
        return perturbed_particles, dict(  # this is just for debuggin stuff in compute_proposal_prob()
            chord_alphas=chord_alphas, alpha_min=alpha_min, alpha_max=alpha_max, directions=sphere_sample,
            log_probs=log_probs
        )

    def compute_proposal_prob(
            self,
            old_particles,
            new_particles,
            chord_proposal='unif',
            chord_std=1.0,
            xch_proposal='unif',
            xch_std=0.4,
            old_is_new=True,
    ):
        old_pol = old_particles[..., :self._K]
        new_pol = new_particles[..., :self._K]
        old_pol = self._la.unsqueeze_like(old_pol, new_pol)

        diff = self._la.unsqueeze(old_pol, 1) - self._la.unsqueeze(new_pol, 0)
        directions = diff
        if old_is_new:
            # along a chord, directions are the same, per chain we only need 1 direction computation
            directions = diff[0, 1]

        directions = directions / self._la.norm(directions, 2, -1, True)

        A_dist = self._la.tensormul_T(self._sampler._G, directions)  # this one has wrong sign on first row
        particle_pol_dist = self._sampler._h.T - self._la.tensormul_T(self._sampler._G, old_pol)

        allpha = -particle_pol_dist / A_dist
        alpha_min, alpha_max = self._la.min_pos_max_neg(allpha, return_what=0)
        alpha = diff[..., 0] / directions[..., 0]  # alpha is the same for all dimensions, so we only need to select 1

        mu = self._la.zeros(alpha.shape)

        if chord_std.ndim > 1:
            # this means that we passed a covariance matrix and we need to compute std along the line
            chord_std = self._la.sqrt(self._la.sum(((directions @ chord_std) * directions), -1))

        # print(alpha[..., None] * directions + old_pol)
        # print(new_pol)  # check whether we recover newpol from oldpol

        log_probs = self._la.bounded_distribution_log_prob(
            x=alpha, lo=alpha_min, hi=alpha_max, mu=mu, std=chord_std, which=chord_proposal, old_is_new=old_is_new,
            unsqueeze=False,
        )
        if self._nx > 0:
            old_xch = old_particles[..., -self._nx:]
            new_xch = new_particles[..., -self._nx:]
            old_xch = self._la.unsqueeze_like(old_xch, new_xch)

            xch_log_probs = self._la.bounded_distribution_log_prob(
                x=new_xch, mu=old_xch,
                lo=self._fcm._rho_bounds[:, 0], hi=self._fcm._rho_bounds[:, 1],
                std=xch_std, which=xch_proposal, old_is_new=old_is_new
            )
            log_probs += self._la.sum(xch_log_probs, -1, keepdims=False)
        return log_probs

    def map_chains_2_theta(self, chains):
        # if chains are not rounded and not log-ratio, we need to map accordingly
        if (self._sampler.basis_coordinates == 'rounded') and not self._fcm.logit_xch_fluxes:
            return chains
        theta = self._fcm._sampler._map_rounded_2_basis(rounded=chains[..., :self._K])
        if self._nx > 0:
            xch_fluxes = chains[..., -self._nx:]
            if self._fcm.logit_xch_fluxes:
                xch_fluxes = self._fcm._logit_xch(xch_fluxes)
            theta = self._la.cat([theta, xch_fluxes], dim=-1)
        return theta


class MCMC(_BaseBayes):
    SYMMETRIC_PROPOSALS = ['gauss', 'unif']
    def accept_reject(self, i, post_probs, prop_probs=None, pre_sample_batch=5000, peskunize=True):
        # based on: https://www.math.ntnu.no/preprint/statistics/2004/S4-2004.pdf
        # other notation: https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/258340/348197_FULLTEXT01.pdf?sequence=2&isAllowed=y

        if prop_probs is not None:
            prop_probs = prop_probs.sum(1)  # sum transition probs over all proposals
            if post_probs.shape != prop_probs.shape:
                raise ValueError(f'post shape: {post_probs.shape}, proposal shape: {prop_probs.shape}')
            n_cdf_1, n_chains = post_probs.shape
            P_l = post_probs + prop_probs
        else:
            P_l = post_probs

        if (n_cdf_1 == 2) and peskunize:
            # this is the single proposal Metropolis-Hastings acceptance ratio!
            ii = i % pre_sample_batch
            if ii == 0:
                if i == 0:
                    self._accept_index = self._la.zeros((n_chains, ), dtype=np.int64)
                self._rnd = self._la.log(self._la.randu((pre_sample_batch,)))
            self._accept_index[:] = 0
            rnd = self._rnd[ii]
            log_mh_ratio = P_l[1] - P_l[0]
            self._accept_index[rnd <= log_mh_ratio] = 1
            return self._accept_index

        P_l -= self._la.max(P_l, 0)
        P_l = self._la.exp(P_l)
        P_l = P_l / P_l.sum(0)
        if not peskunize:
            # this corresponds to Barkers acceptance probability, which is sub-optimal, thus longer ESS
            return self._la.multinomial(n=1, p=P_l.T).T[0]

        if i == 0:
            self._didx = self._la.arange(n_cdf_1)
            self._where = self._la.zeros(n_cdf_1, dtype=np.float64)
            self._ut = self._la.zeros(n_cdf_1)

            self._non_diag = self._la.ones((n_cdf_1, n_cdf_1)) - self._la.eye(n_cdf_1)

        P_kl = self._la.tile(P_l, (n_cdf_1, 1, 1))
        for chain_i in range(n_chains):
            P_j = P_kl[..., chain_i]
            for j in range(n_cdf_1):
                diags = P_j[self._didx, self._didx]
                diag_idxs = self._la.where(diags > 0.0)[0]

                if diag_idxs.shape[0] < 2:
                    print('mah mann')
                    break

                self._ut[:] = float('inf')

                wheres = []
                wheres2 = []
                for k in diag_idxs:
                    self._where[:] = 1
                    self._where[k] = 0
                    numerator = 1 - self._where @ P_j[k]
                    wheres.append(self._la.vecopy(self._where))

                    self._where[:] = 0
                    self._where[diag_idxs] = 1
                    self._where[k] = 0
                    denominor = self._where @ P_j[k]
                    wheres2.append(self._la.vecopy(self._where))

                    self._ut[k] = numerator / denominor
                ut = self._la.min(self._ut)

                print(self._la.stack(wheres))
                print(self._la.stack(wheres2))

                for k in diag_idxs:
                    for l in diag_idxs:
                        if k == l:
                            continue
                        P_j[k, l] *= ut

                P_j[diag_idxs, diag_idxs] = (1 - self._non_diag @ P_j)[diag_idxs]
                print(P_j)



                # P_j[]






                # non_diag = self._la.vecopy(self._non_diag)


                # self._where[:] = 0
                # num_select = self._didx != j
                # self._where[num_select] = 1
                # numerator = 1 - self._where @ P_j[j, :]
                #
                # self._where[:] = 0
                # self._where[diag_idxs] = 1
                # self._where[j] = 0
                # denominor = self._where @ P_j[j, :]
                #
                # self._ut[j] = numerator / denominor

            # ut = self._la.min(self._ut)
            # P_j_1 = self._la.vecopy(P_j)
            # for diag_k in diag_idxs:
            #     for diag_l in diag_idxs:
            #         if diag_l != diag_k:
            #             P_j_1[diag_k, diag_l] = ut * P_j[diag_k, diag_l]
            #     self._where[:] = 1
            #     self._where[diag_k] = 0
            #     P_j_1[diag_k, diag_k] = 1 - self._where @ P_j_1[diag_k]
            #
            # print(P_j_1)

        raise NotImplementedError('we should finish this after handing in the thesis')

    # @profile(profiler=profile2)
    def run(
            self,
            initial_points=None,
            n: int = 50,
            n_burn=50,
            thinning_factor=3,
            n_chains: int = 4,
            potentype='exact',
            n_cdf=6,
            chord_proposal='gauss',
            chord_std=1.0,
            xch_proposal='gauss',
            xch_std=0.4,
            return_data=True,
            evaluate_prior=False,
            potential_kwargs={},
            return_az=True,
            peskunize=True,
            pre_sample_batch=5000,
            debug=True,
    ) -> az.InferenceData:
        # TODO: this publication talks about this algo, but has a different acceptance procedure:
        #  doi:10.1080/01621459.2000.10473908
        #  doi:10.1007/BF02591694  Rinooy Kan article

        chord_std = self._la.get_tensor(values=np.array([chord_std]))
        xch_std = self._la.get_tensor(values=np.array([xch_std]))

        if self._fcm._sampler.basis_coordinates == 'transformed':
            raise NotImplementedError('transform the chains to transformed')

        batch_size = n_chains * n_cdf
        if (self._la._batch_size != batch_size) or not self._model._is_built:
            # this way the batch processing is corrected
            self._la._batch_size = batch_size
            self._model.build_simulator(**self._fcm.fcm_kwargs)

        n_rounded = self._fcm._sampler._F_round.A.shape[1] + self._nx
        chains = self._la.get_tensor(shape=(n, n_chains, n_rounded))  # TODO this should be the number of dimensions in rounded coord system!
        post_probs = self._la.get_tensor(shape=(n, n_chains))
        accept_rate = self._la.get_tensor(shape=(n_chains,), dtype=np.int64)

        if return_data:
            sim_data = self._la.get_tensor(shape=(n, n_chains, len(self.data_id)))

        if initial_points is None:
            y = self._sampler.get_initial_points(num_points=n_chains)
            if self._prior._xch_prior is not None:
                xch_basis_points = self._prior._xch_prior.sample((n_chains, ))
                y = self._la.cat([y, xch_basis_points], dim=-1)
        else:
            y = initial_points

        y = self._la.tile(y, (n_cdf, 1))  # remember that the new batch size is n_chains x n_cdf

        self._set_potential(potentype, **dict(return_data=return_data, evaluate_prior=evaluate_prior, **potential_kwargs))
        if (self._potentype == 'approx'):
            # https://www.biorxiv.org/content/10.1101/106450v1.full.pdf
            # TODO for approx, the MH acceptance is just the ratio between prior probabilities and proposal (which is symmetric, so falls out)
            if n_chains > 1:
                raise ValueError(
                    'currently not possible to simulate multiple chains at once '
                    'due to skipping distance under epsilon simulations'
                )
            raise NotImplementedError('this is complicated, since we need to weight samples by the prior somehow')

        chord_ys = self._la.get_tensor(shape=(1 + n_cdf, n_chains, n_rounded))
        chord_post_probs = self._la.get_tensor(shape=(1 + n_cdf, n_chains))
        chord_prop_probs = self._la.get_tensor(shape=(1 + n_cdf, 1 + n_cdf, n_chains))
        pert_post_probs = self.potential(y)

        if return_data:
            chord_data = self._la.get_tensor(shape=(1 + n_cdf, n_chains, len(self.data_id)))
            pert_post_probs, data = pert_post_probs
            chord_data[0] = data[:n_chains]

        chord_post_probs[0] = pert_post_probs[:n_chains]  # ordering of the samples from the PDF does not matter for inverse sampling
        chain_selector = self._la.arange(n_chains)

        y = y[: n_chains, :]
        chord_ys[0] = y

        n_tot = n_burn + n * thinning_factor
        pre_sample_batch = min(pre_sample_batch, n_tot)
        return_what = (n_cdf == 1) and (chord_proposal in self.SYMMETRIC_PROPOSALS)  # only useful for symmetrical proposals and n_cdf==1
        # TODO I think for n_cdf and symmetrical proposals, we dont need to compute anything
        perturb_kwargs = dict(
            batch_shape=(pre_sample_batch, n_chains),
            n_cdf=n_cdf,
            chord_proposal=chord_proposal,
            chord_std=chord_std,
            xch_proposal=xch_proposal,
            xch_std=xch_std,
            return_what=return_what,
        )
        pbar = tqdm.tqdm(total=n_tot, ncols=100)
        i = 0
        # try:
        while i < n_tot:
            pert_ys = self.perturb_particles(y, i, **perturb_kwargs)
            if return_what:
                pert_ys, pert_prop_probs = pert_ys
                # two lines below only hold for symmetric proposals, otherwise we need to compute exactly
                chord_prop_probs[0, 1] = pert_prop_probs
                chord_prop_probs[1, 0] = pert_prop_probs

            chord_ys[1:] = pert_ys
            pert_post_probs = self.potential(pert_ys)

            if self.potentype == 'approx':
                raise NotImplementedError('reject ABC proposals if all are too distant!')

            if return_data:
                pert_post_probs, data = pert_post_probs
                chord_data[1:] = data

            chord_post_probs[1:] = pert_post_probs

            if not return_what:
                chord_prop_probs = self.compute_proposal_prob(
                    old_particles=chord_ys,
                    new_particles=chord_ys,
                    chord_proposal=chord_proposal,
                    chord_std=chord_std,
                    xch_proposal=xch_proposal,
                    xch_std=xch_std,
                    old_is_new=True,
                )
            accept_idx = self.accept_reject(i, chord_post_probs, chord_prop_probs, pre_sample_batch, peskunize)
            accepted_probs = chord_post_probs[accept_idx, chain_selector]
            chord_post_probs[0] = accepted_probs  # set the log-probs of the current sample
            y = chord_ys[accept_idx, chain_selector]
            # print(chord_post_probs)
            # print(accept_idx)
            # print()
            chord_ys[0] = y  # set the log-probs of the current sample
            if return_data:
                data = chord_data[accept_idx, chain_selector]
                chord_data[0] = data

            j = i - n_burn
            pbar.update(1)
            if j > 0:
                if n_cdf > 1:
                    accept_idx[accept_idx > 0] = 1
                accept_rate += accept_idx
                avg_rate = self._la.mean(accept_rate / j)
                if j % 50 == 0:
                    pbar.set_postfix(avg_acc=avg_rate.item())

            if (j % thinning_factor == 0) and (j > -1):
                k = j // thinning_factor
                post_probs[k] = accepted_probs
                chains[k] = y
                if return_data:
                    sim_data[k] = data
            i += 1
        try:
            pass
        except Exception as e:
            if e is not KeyboardInterrupt:
                print(e)
        finally:
            pbar.close()
            chains = self.map_chains_2_theta(chains)

            if not return_az:
                return chains

            posterior_predictive = None
            if return_data:
                posterior_predictive = {
                    'data': self._la.transax(sim_data, dim0=1, dim1=0)
                }

            attrs = {
                'potentype': potentype,
                'evaluate_prior': str(evaluate_prior),
                'potential_kwargs': [(k, v if not isinstance(v, bool) else str(v)) for k, v in potential_kwargs.items()],
                'n_burn': n_burn,
                'acceptance_rate': self._la.tonp(accept_rate) / j,
                'thinning_factor': thinning_factor,
                'n_cdf': n_cdf,
                'line_kernel': chord_proposal,
                'line_variance': self._la.tonp(chord_std),
                'xch_kernel': xch_proposal,
                'xch_variance': self._la.tonp(xch_std),
                'running_time': pbar.format_dict['elapsed'],
                'pbar_n': pbar.format_dict['n'],
                'peskunize': int(peskunize),
            }
            if self.true_theta is not None:
                attrs['true_theta'] = self._la.tonp(self._true_theta)
                attrs['true_theta_id'] = self._true_theta_id

            n_obs = potential_kwargs.get('n_obs', 0)
            dims, coords = self._format_dims_coords(n_obs=n_obs if self.potentype == 'approx' else 0)
            return az.from_dict(
                posterior={
                    'theta': self._la.transax(chains, dim0=1, dim1=0)  # chains x draws x param
                },
                dims=dims,
                coords=coords,
                observed_data={
                    'observed_data': self.measurements.values
                },
                sample_stats={
                    'lp': post_probs.T  # chains x draws
                },
                posterior_predictive=posterior_predictive,
                attrs=attrs
            )


class SMC(_BaseBayes):
    # https://www.annualreviews.org/doi/pdf/10.1146/annurev-ecolsys-102209-144621
    #  https://jblevins.org/notes/smc-intro
    #  https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.pdf
    _CHECK_TRANSFORM = False

    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            prior: _BasePrior = None,
            boundary_observation_model: BoundaryObservationModel = None,
            num_processes=0,
    ):
        if self._CHECK_TRANSFORM:  # can be switched off if we want to compare to ABC on raw simplex data
            for labelling_id, obsmod in mdv_observation_models.items():
                if obsmod._transformation is None:
                    raise ValueError(f'Observationmodel {obsmod} does not have a transformation and therefore '
                                     f'euclidian distance is not defined (data lies on simplices)')
        super(SMC, self).__init__(model, substrate_df, mdv_observation_models, prior, boundary_observation_model)
        self._num_processes = num_processes
        self._dss = DataSetSim(
            model=model,
            substrate_df=self._substrate_df,
            mdv_observation_models=self._obmods,
            boundary_observation_model=self._bom,
            num_processes=num_processes,
        )
        # this is so that we can make compute_distances use the right simulation function
        self.__call__ = partial(self._dss.__call__, close_pool=False)

    def _calculate_new_log_weights(
            self,
            new_particles,
            potential,
            old_particles,
            old_log_weights,
            chord_proposal,
            chord_std,
            xch_proposal,
            xch_std,
            evaluate_prior=True,
            population_batch=1000,

    ):
        population_batch = min(old_particles.shape[0], population_batch)
        prop_probs = self._la.get_tensor(shape=(*old_particles.shape[:-1], *new_particles.shape[:-1]))

        for i in range(0, old_particles.shape[0], population_batch):
            prop_probs[i: i + population_batch] = self.compute_proposal_prob(
                old_particles[i: i + population_batch],
                new_particles,
                chord_proposal,
                chord_std,
                xch_proposal,
                xch_std,
                old_is_new=False
            )
        log_weighted_sum = self._la.logsumexp(old_log_weights[:, None] + prop_probs, 0)  # computes importance weights

        if evaluate_prior:
            prior_log_probs = self._prior.log_prob(new_particles)
        else:
            prior_log_probs = -0.1  # for a uniform prior, dont need to evaluate

        if self._potentype == 'approx':
            potential[:] = 0.0  # for ABC, we filter by distance and do not assign a probability

        return potential + prior_log_probs - log_weighted_sum

    def _sample_next_population(
            self,
            particles,
            log_weights,
            epsilon: float,
            kernel_variance_scale=1.0,
            chord_proposal='gauss',
            xch_proposal='gauss',
            population_batch=1000,
            n_cdf=1,
            return_data=True,
            evaluate_prior=False,
    ):
        """Return particles, weights and distances of new population."""

        new_particles = []
        new_log_weights = []
        new_distances = []
        new_data = []

        m = 0
        n = particles.shape[0]
        population_batch = min(n, population_batch)

        chord_cov, xch_std = None, None
        if (chord_proposal == 'gauss') or (xch_proposal == 'gauss'):
            theta_cov = self.mvn_kernel_variance(
                particles,
                weights=self._la.exp(log_weights),
                samples_per_dim=500,
                kernel_std_scale=kernel_variance_scale,
                prev_cov=True,
            )
            chord_cov = theta_cov[:self._K, :self._K]
            if self._nx > 0:
                xch_std = self._la.sqrt(self._la.diagonal(theta_cov[-self._nx:, -self._nx:]))

        pbar = tqdm.tqdm(total=n, ncols=100)
        pbar.set_postfix(epsilon=float(epsilon))
        try:
            sofar = 0
            while m < n:
                # Sample from previous population and perturb.
                sample_indices = self._la.multinomial(population_batch, self._la.exp(log_weights))
                sampled_particles = particles[sample_indices]

                perturbed_particles = self.perturb_particles(
                    theta=sampled_particles,
                    i=0,
                    batch_shape=(1, population_batch),
                    n_cdf=n_cdf,
                    chord_proposal=chord_proposal,
                    chord_std=chord_cov,
                    xch_proposal=xch_proposal,
                    xch_std=xch_std,
                    return_what=0
                ).squeeze(0)

                dist = self.potential(perturbed_particles, epsilon, show_progress=False)
                if return_data:
                    dist, data = dist

                is_accepted = ~self._la.isnan(dist)  # this is to filter approximations!
                num_accepted_batch = is_accepted.sum()

                if num_accepted_batch > 0:
                    n_batch = self._la.tonp(num_accepted_batch)
                    if sofar + n_batch > n:
                        n_batch = n - sofar
                    sofar += n_batch
                    pbar.update(n_batch)
                    accepted_particles = perturbed_particles[is_accepted]
                    new_particles.append(accepted_particles)
                    new_distances.append(dist[is_accepted])
                    new_log_weights.append(
                        self._calculate_new_log_weights(
                            new_particles=accepted_particles,
                            potential=dist[is_accepted],
                            old_particles=particles,
                            old_log_weights=log_weights,
                            evaluate_prior=evaluate_prior,
                            chord_proposal=chord_proposal,
                            chord_std=chord_cov,
                            xch_proposal=xch_proposal,
                            xch_std=xch_std,
                            population_batch=population_batch,
                        )
                    )
                    if return_data:
                        new_data.append(data[is_accepted])
                    m += num_accepted_batch
        except Exception as e:
            print(e)
        finally:
            pbar.close()
            self._totime.append(pbar.format_dict['elapsed'])

        # collect lists of tensors into tensors
        new_distances   = self._la.cat(new_distances)
        sort_idx        = self._la.argsort(new_distances)

        new_distances   = new_distances[sort_idx][:n]
        new_particles   = self._la.cat(new_particles)[sort_idx][:n]
        new_log_weights = self._la.cat(new_log_weights)[sort_idx][:n]
        if return_data:
            new_data    = self._la.cat(new_data)[sort_idx][:n]

        # normalize the new weights
        new_log_weights -= self._la.logsumexp(new_log_weights, dim=0)

        return (
            new_particles,
            new_log_weights,
            new_distances,
            new_data,
        )


    def _arviz_2_partial_mdv(self, result: az.InferenceData):
        # this renames stuff so that the data is now partial MDVs and the log-ratio data is called lr_data

        result.posterior_predictive = result.posterior_predictive.rename({'data': 'lr_data', 'data_id': 'lr_data_id'})
        result.prior_predictive = result.prior_predictive.rename({'data': 'lr_data', 'data_id': 'lr_data_id'})
        result.observed_data = result.observed_data.rename({'data_id': 'lr_data_id', 'observed_data': 'lr_observed_data'})

        post_pred = result.posterior_predictive.lr_data.values
        prior_pred = result.prior_predictive.lr_data.values
        observed = result.observed_data.lr_observed_data.values
        post_pred = self._la.tonp(self.to_partial_mdvs(post_pred, pandalize=False))
        prior_pred = self._la.tonp(self.to_partial_mdvs(prior_pred, pandalize=False))
        observed = self._la.tonp(self.to_partial_mdvs(observed, pandalize=False))

        cols = self.to_partial_mdvs(post_pred[:, 0, 0, :], pandalize=True)
        cols = [f'{i[0]}: {i[1]}' for i in cols.columns.tolist()]

        post_data = xr.DataArray(
            data=post_pred,
            dims=['chain', 'draw', 'obs_idx', 'data_id'],
            coords={
                'chain': np.arange(post_pred.shape[0]),
                'draw': np.arange(post_pred.shape[1]),
                'obs_idx': np.arange(post_pred.shape[2]),
                'data_id': cols
            },
        )
        result.posterior_predictive['data'] = post_data

        prior_data = xr.DataArray(
            data=prior_pred,
            dims=['chain', 'draw', 'obs_idx', 'data_id'],
            coords={
                'chain': np.arange(prior_pred.shape[0]),
                'draw': np.arange(prior_pred.shape[1]),
                'obs_idx': np.arange(prior_pred.shape[2]),
                'data_id': cols
            },
        )
        result.prior_predictive['data'] = prior_data

        observed = xr.DataArray(
            data=observed,
            dims=['measurement_id', 'data_id'],
            coords={
                'measurement_id': result.observed_data.measurement_id.values,
                'data_id': cols
            },
        )
        result.observed_data['observed_data'] = observed
        return result

    def run(
            self,
            n_smc_steps=3,
            n=100,
            n_obs=5,
            n0_multiplier=2,
            population_batch=1000,
            distance_based_decay=True,
            epsilon_decay=0.8,
            kernel_std_scale=1.0,
            evaluate_prior=False,
            potentype='approx',
            return_data=True,
            potential_kwargs={},
            metric='rmse',
            chord_proposal='gauss',
            xch_proposal='gauss',
            xch_std=0.4,
            return_all_populations=False,
            return_az=True,
            debug=False,
    ):
        if potentype == 'exact':
            # we dont evaluate the prior in log_probs since we do this is computing the weights!
            potential_kwargs['evaluate_prior'] = False

        xch_std = self._la.get_tensor(values=np.array([xch_std]))

        self._set_potential(potentype, **dict(n_obs=n_obs, metric=metric, return_data=return_data, **potential_kwargs))
        if self._potentype != 'approx':
            raise NotImplementedError(
                'think about what it means for non-approximate potential '
                'where we do not need to reject stuff below epsilon'
            )

        data = None
        self._totime = []
        if n_smc_steps < 2:
            raise ValueError
        try:
            for i in range(n_smc_steps):
                if i == 0:
                    prior_theta = self._prior.sample(sample_shape=(int(n * n0_multiplier), ))
                    if self._la.backend != 'torch':
                        prior_theta = self._prior._fcm._la.tonp(prior_theta)

                    dist, prior_time = self.potential(prior_theta, epsilon=-float('inf'), show_progress=True, return_time=True)
                    self._totime.append(prior_time)
                    if return_data:
                        dist, data = dist
                        prior_data = data

                    sortidx = self._la.argsort(dist)
                    particles = prior_theta[sortidx][:n]
                    dist = dist[sortidx][:n]
                    epsilon = dist[-1]
                    log_weights = self._la.log(1 / n * self._la.ones(n))

                    if return_all_populations:
                        all_particles = [particles]
                        all_log_weights = [log_weights]
                        all_distances = [dist]
                        all_epsilons = [epsilon]
                        if return_data:
                            all_data = [data[sortidx][:n]]
                else:
                    if distance_based_decay:
                        # Quantile of last population
                        epsidx = self.quantile_indices(dist, quantiles=1 - epsilon_decay)
                        epsilon = dist[epsidx]
                    else:
                        # Constant decay.
                        epsilon *= epsilon_decay

                    particles, log_weights, dist, data = self._sample_next_population(
                        particles=particles,
                        log_weights=log_weights,
                        epsilon=epsilon,
                        population_batch=population_batch,
                        kernel_variance_scale=kernel_std_scale,
                        chord_proposal=chord_proposal,
                        xch_proposal=xch_proposal,
                        return_data=return_data,
                        evaluate_prior=evaluate_prior,
                    )
                    if return_all_populations:
                        all_particles.append(particles)
                        all_log_weights.append(log_weights)
                        all_distances.append(dist)
                        all_epsilons.append(epsilon)
                        if return_data:
                            all_data.append(data)
        except Exception as e:
            print(e)
        finally:
            if return_all_populations:
                particles = self._la.stack(all_particles, 0)
                log_weights = self._la.stack(all_log_weights, 0)
                dist = self._la.stack(all_distances, 0)
                epsilon = self._la.stack(all_epsilons, 0)
                if return_data:
                    data = self._la.stack(all_data, 0)
            else:
                # add the 'chains' dimension
                particles = particles[None, ...]
                log_weights = log_weights[None, ...]
                dist = dist[None, ...]
                if return_data:
                    data = data[None, ...]

            if not return_az:
                return particles

            posterior_predictive, prior_predictive = None, None
            if return_data:
                posterior_predictive = {
                    'data': data
                }
                prior_predictive = {
                    'data': prior_data[None, ...],  # add the 'chains' dimension
                }

            attrs = {
                'potentype': potentype,
                'population_batch': population_batch,
                'epsilons': self._la.tonp(epsilon),
                'evaluate_prior': str(evaluate_prior),
                'n_smc_steps': n_smc_steps,
                'potential_kwargs': [(k, v if not isinstance(v, bool) else str(v)) for k, v in potential_kwargs.items()],
                'n_obs': n_obs,
                'kernel_variance_scale': kernel_std_scale,
                'n0_multiplier': n0_multiplier,
                'distance_based_decay': str(distance_based_decay),
                'metric': metric,
                'epsilon_decay': epsilon_decay,
                'line_kernel': chord_proposal,
                'xch_kernel': xch_proposal,
                'xch_variance': self._la.tonp(xch_std),
                'running_time': np.array(self._totime),
            }
            if self.true_theta is not None:
                attrs['true_theta'] = self._la.tonp(self._true_theta)
                attrs['true_theta_id'] = self._true_theta_id

            dims, coords = self._format_dims_coords(n_obs=n_obs if self.potentype == 'approx' else 0)

            return az.from_dict(
                posterior={
                    'theta': particles  # chains x draws x param
                },
                prior={
                    'theta': prior_theta[None, ...],  # add the 'chains' dimension
                },
                dims=dims,
                coords=coords,
                observed_data={
                    'observed_data': self.measurements.values,
                },
                sample_stats={
                    'log_weights': log_weights,
                    'distances': dist,
                },
                posterior_predictive=posterior_predictive,
                prior_predictive=prior_predictive,
                attrs=attrs
            )


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=500)

    from sbmfi.models.small_models import spiro, multi_modal
    from sbmfi.models.build_models import build_e_coli_anton_glc, _bmid_ANTON
    from sbmfi.inference.priors import UniNetFluxPrior, ProjectionPrior
    from sbmfi.inference.complotting import PlotMonster
    from sbmfi.core.polytopia import FluxCoordinateMapper, PolytopeSamplingModel, sample_polytope, fast_FVA
    import pickle
    from sbmfi.core.observation import MVN_BoundaryObservationModel

    model, kwargs = multi_modal(backend='torch')
    mcmc = MCMC(
        model=model,
        substrate_df=kwargs['substrate_df'],
        mdv_observation_models=kwargs['basebayes']._obmods,
        prior=kwargs['basebayes']._prior,
        boundary_observation_model=kwargs['basebayes']._bom
    )
    mcmc.set_measurement(x_meas=kwargs['measurements']) # [:, :-3]
    mcmc.set_true_theta(theta=kwargs['true_theta'])
    res = mcmc.run(
        n=4000, n_burn=0, thinning_factor=1, n_cdf=1, n_chains=1, chord_std=0.3, peskunize=True,
        chord_proposal='gauss', xch_proposal='gauss', xch_std=0.4
    )
    # pm = PlotMonster(model._fcm._sampler.basis_polytope, res)
    import holoviews as hv

    # hv.save(pm.grand_theta_plot(var1_id='R_a_in', var2_id='R_v4'), 'ding', fmt='html', backend='bokeh')

    print(
        (res.posterior['theta'].values[:,:,-1] > 0.).sum() / len(res.posterior['chain'])
    )

    # hdf = r"C:\python_projects\sbmfi\spiro_flow_lowsigma.h5"
    # did = 'sims'
    #
    # model, kwargs = spiro(
    #     backend='torch', v2_reversible=True, v5_reversible=False, build_simulator=True, which_measurements='com',
    #     which_labellings=['A', 'B'], transformation='ilr'
    # )
    # dss = DataSetSim(
    #     model=model,
    #     substrate_df=kwargs['substrate_df'],
    #     mdv_observation_models=kwargs['basebayes']._obmods,
    #     boundary_observation_model=kwargs['basebayes']._bom,
    #     num_processes=0,
    # )
    # ilr_data = dss.read_hdf(hdf, dataset_id=did, what='data', stop=1000)
    # theta = dss.read_hdf(hdf=hdf, dataset_id=did, what='theta', stop=1000)
    # mdv_data = dss.to_partial_mdvs(ilr_data, pandalize=False)
    #
    # model, kwargs = spiro(
    #     backend='torch', v2_reversible=True, v5_reversible=False, build_simulator=True, which_measurements='com',
    #     which_labellings=['A', 'B'], transformation=None
    #
    # )
    # up = UniNetFluxPrior(model.flux_coordinate_mapper, cache_size=100000)
    #
    # bom = MVN_BoundaryObservationModel(model, kwargs['basebayes']._bom._bound_id.values, biomass_std=0.01, boundary_std=0.03)
    #
    # dss = DataSetSim(
    #     model=model,
    #     substrate_df=kwargs['substrate_df'],
    #     mdv_observation_models=kwargs['basebayes']._obmods,
    #     boundary_observation_model=kwargs['basebayes']._bom,
    #     num_processes=0,
    # )
    # mcmc = MCMC(model, dss._substrate_df, dss._obmods, prior=up, boundary_observation_model=dss._bom)
    #
    #
    # idx = 265
    # x = mdv_data[[idx], 0]
    # y = theta[[idx]]
    # mcmc.set_measurement(x_meas=x) # [:, :-3]
    # mcmc.set_true_theta(theta=pd.Series(y[0].numpy(), mcmc.theta_id))
    # res = mcmc.run(
    #     n=10000, n_burn=0, thinning_factor=1, n_cdf=1, n_chains=4, chord_std=0.0000000001, peskunize=True,
    #     chord_proposal='gauss', xch_proposal='gauss', xch_std=0.4
    # )


    # model, kwargs = build_e_coli_anton_glc(
    #     backend='numpy',
    #     auto_diff=False,
    #     build_simulator=True,
    #     ratios=False,
    #     batch_size=25,
    #     which_measurements='tomek',
    #     which_labellings=['20% [U]Glc', '[1]Glc'],
    #     measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
    #     seed=1,
    # )
    #
    # sdf = kwargs['substrate_df']
    # dss = kwargs['basebayes']
    # simm = dss._obmods
    # bom = dss._bom
    # up = UniNetFluxPrior(model, cache_size=2000)
    #
    # smc = SMC(
    #     model=model,
    #     substrate_df=sdf,
    #     mdv_observation_models=simm,
    #     boundary_observation_model=bom,
    #     prior=up,
    #     num_processes=0,
    # )
    # smc.set_measurement(x_meas=kwargs['measurements'])
    # smc.set_true_theta(theta=kwargs['theta'])
    #
    # result = az.from_netcdf("C:/python_projects/sbmfi/SMC_e_coli_glc_tomek_obsmod_copy_NEW.nc")
    #
    # result = smc._arviz_2_partial_mdv(result)
    # # model, kwargs = build_e_coli_anton_glc()
    #
    # # model, kwargs = spiro(
    # #     seed=None,
    # #     batch_size=50,
    # #     backend='torch', v2_reversible=True, ratios=False, build_simulator=True,
    # #     which_measurements='lcms', which_labellings=['A', 'B'], v5_reversible=True
    # # )
    # # sdf = kwargs['substrate_df']
    # # bb = kwargs['basebayes']
    # # up = UniNetFluxPrior(model._fcm)
    # # smc = SMC(model, sdf, bb._obmods, prior=up, boundary_observation_model=bb._bom, num_processes=0)
    # # smc.set_measurement(x_meas=kwargs['measurements'])
    # # smc.set_true_theta(theta=kwargs['theta'])
    # #
    # # res = smc.run(n=5000, n_smc_steps=8, epsilon_decay=0.4, return_all_populations=True, )
    # # az.to_netcdf(res, 'spiro_TEST_SMC.nc')
    #
    # # model2, kwargs2 = spiro(
    # #     seed=9, batch_size=2,
    # #     backend='torch', v2_reversible=True, ratios=False, build_simulator=True,
    # #     which_measurements='com', which_labellings=['C', 'D'], v5_reversible=True, include_bom=True
    # # )
    # # up = UniNetFluxPrior(model2._fcm)
    # # bb2 = kwargs2['basebayes']
    # # sdf = kwargs2['substrate_df']
    # # mcmc = MCMC(model2, sdf, bb2._obmods, prior=up, boundary_observation_model=bb2._bom)
    # # mcmc.set_measurement(x_meas=kwargs2['measurements'])
    # mcmc.set_true_theta(theta=kwargs2['theta'])
    # # res = mcmc.run(
    # #     n=20, n_burn=0, thinning_factor=1, n_cdf=1, n_chains=2, chord_std=0.6, peskunize=True,
    # #     chord_proposal='gauss', xch_proposal='gauss', xch_std=0.4
    # # )
    # # print(profile2.print_stats())
    # # mcmc.simulate_data(res, n=10000)
    # # az.to_netcdf(res, 'spiro_TEST_MCMC.nc')


