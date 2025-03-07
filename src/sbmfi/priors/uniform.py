import psutil
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import inspect
from torch.distributions.constraints import Constraint, _Dependent, _Interval
from torch.types import _size

from sbmfi.core.model import LabellingModel
from sbmfi.core.linalg import LinAlg
from sbmfi.core.polytopia import (
    LabellingPolytope,
    PolytopeSamplingModel,
    sample_polytope,
    compute_volume,
    get_rounded_polytope,
)
from sbmfi.core.coordinater import FluxCoordinateMapper, make_theta_polytope
from typing import Iterable, Union, List, Dict
from torch.distributions import constraints
from torch.distributions import Distribution
import math
import tqdm
#   https://math.stackexchange.com/questions/4484178/computing-barycentric-coordinates-for-convex-n-dimensional-polytope-that-is-not

def sampling_tasks(
        polytope: LabellingPolytope, # this is a basis polytope that will be modified using b_constraint_df and A_constraint_dct
        kernel_id ='svd',
        counts: Union[int, pd.Series] = 20,  # number of fluxes to sample from yielded polytope
        A_constraint_df: pd.DataFrame = None, # this should have a multiindex with level 1 being names, and 2 being constraint_names
        S_constraint_df: pd.DataFrame = None, # this should have a multiindex with level 1 being names, and 2 being constraint_names
        b_constraint_df: pd.DataFrame = None,
        n_burn: int = 100,
        thinning_factor: int = 5,
        n_chains: int = 4,
        sampling_function = sample_polytope,
        linalg: LinAlg = None,
        return_psm = False,
        return_what = 'rounded',
):
    if (A_constraint_df is None) and (S_constraint_df is None) and (b_constraint_df is None):
        raise ValueError
    if (S_constraint_df is not None) and (polytope.S is None):
        raise ValueError

    for thing in [A_constraint_df, S_constraint_df, b_constraint_df]:
        if thing is not None:
            index = thing.index
            if isinstance(index, pd.MultiIndex):  # for S and A constraint dfs
                index = index.levels[0]

    if isinstance(counts, int):
        counts = pd.Series(counts, index=index)
    if A_constraint_df is None:
        A_constraint_df = pd.DataFrame(None, index=index)
    if S_constraint_df is None:
        S_constraint_df = pd.DataFrame(None, index=index)
    if b_constraint_df is None:
        b_constraint_df = pd.DataFrame(None, index=index)

    func_kwargs = inspect.getfullargspec(sampling_function).args
    for name, row in b_constraint_df.iterrows():
        pol = polytope.copy()
        if row.size > 0:
            pol.b.loc[b_constraint_df.columns] = row

        A = A_constraint_df.loc[name]
        if A.size > 0:
            pol.A.loc[A.index, A.columns] = A

        S = S_constraint_df.loc[name]
        if S.size > 0:
            pol.S.loc[S.index, S.columns] = S

        if counts is None:
            count = name
        else:
            count = counts.loc[name]

        kwargs = {  # these are the kwargs to coordinate_hit_and_run
            'model': pol,
            'n': count,
            'n_burn': n_burn,
            'initial_points': None,
            'thinning_factor': thinning_factor,
            'n_chains': n_chains,
            'new_initial_points': False,
            'return_psm': return_psm,
            'phi': None,
            'linalg': linalg,
            'kernel_id': kernel_id,
            'density': None,
            'n_cdf': 1,
            'return_arviz': False,
            'return_what': return_what,
        }

        kwargs = {key: kwargs.get(key) for key in func_kwargs}
        yield tuple(kwargs.values())  # because cannot pickle dict_values... god I hate dict_keys and dict_values


def volume_tasks(
        models: Union[PolytopeSamplingModel, LabellingPolytope],
        n: int = -1,
        n0_multiplier: int = 5,
        thinning_factor: int = 1,
        epsilon: float = 1.0,
        enumerate_vertices: bool = False,
        return_all_ratios: bool = False,
        quadratic_program: bool = False,
):
    func_kwargs = inspect.getfullargspec(compute_volume).args
    for i, model in enumerate(models):
        kwargs = {
            'model': model,
            'n': n,
            'n0_multiplier': n0_multiplier,
            'thinning_factor': thinning_factor,
            'epsilon': epsilon,
            'enumerate_vertices': enumerate_vertices,
            'return_all_ratios': return_all_ratios,
            'quadratic_program': quadratic_program,
        }
        kwargs = {key: kwargs.get(key) for key in func_kwargs}
        yield tuple(kwargs.values())  # because cannot pickle dict_values... god I hate dict_keys and dict_values


def sizifier(fn):
    def inner():
        return
    return fn


class _CannonicalPolytopeSupport(_Dependent):  #
    def __init__(
            self,
            polytope: LabellingPolytope,
            validation_tol = 1e-6,
    ):
        if polytope.S is not None:
            raise ValueError('only for cannonical polytopes, Av <= b')

        self._constraint_id = polytope.A.columns
        self._A = torch.from_numpy(polytope.A.values)
        self._b = torch.atleast_2d(torch.from_numpy(polytope.b.values + validation_tol)).T
        super().__init__(is_discrete=False, event_dim=self._A.shape[1])

    def to(self, *args, **kwargs):
        self._A = self._A.to(*args, **kwargs)
        self._b = self._b.to(*args, **kwargs)

    @property
    def constraint_id(self) -> pd.Index:
        return self._constraint_id.copy()

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype != self._A.dtype:
            value = value.to(self._A.dtype)
        vape = value.shape
        if len(vape) > 2:
            value = value.view((math.prod(vape[:-1]), vape[-1]))
        value = value[:, :self._A.shape[1]]
        valid   = (self._A @ value.T <= self._b).T
        return valid.view(*vape[:-1], self._A.shape[0])


class _BasePrior(Distribution):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            num_processes: int = 0,
    ):
        # prior sampling variables
        if isinstance(model, LabellingModel):
            model = model.flux_coordinate_mapper
        if model._la.backend != 'torch':
            linalg = LinAlg('torch', seed=model._la._backwargs['seed'])
            model = model.to_linalg(linalg)

        self._fcm = model
        self._la = model._la

        if num_processes < 0:
            num_processes = psutil.cpu_count(logical=False)
        self._num_processes = num_processes

        self._mp_pool = None
        if num_processes > 0:
            self._mp_pool = mp.Pool(self._num_processes)

        # passing validate_args={} will trigger support checking
        super().__init__(event_shape=torch.Size((self.n_theta,)), validate_args={})

    def __getstate__(self):
        if self._mp_pool is not None:
            self._mp_pool.terminate()
            self._mp_pool.join()
            self._mp_pool = None
        dct = self.__dict__
        return dct

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._num_processes > 0:
            self._mp_pool = mp.Pool(self._num_processes)

    @property
    def n_theta(self):
        # number of theta elements, depends on coordinate system for fluxes or number of ratios
        raise NotImplementedError

    @property
    def theta_id(self):
        raise NotImplementedError

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {}

    def to(self, *args, **kwargs):
        # this is useful for when we would like to sample on GPU
        raise NotImplementedError

    def sample_pandalize(self, n: int):
        return pd.DataFrame(self._la.tonp(self.sample_n(n)), columns=self.theta_id)


class BaseRoundedPrior(_BasePrior):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            num_processes: int = 0,
    ):
        self._initial_points = None
        super(BaseRoundedPrior, self).__init__(model, num_processes)

    # NB event_dim=1 means that the right-most dimension defines an event!
    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        supp = _CannonicalPolytopeSupport(polytope=get_rounded_polytope(self._fcm.sampler))
        # supp.to(dtype=torch.float32)  # TODO maybe pass dtype as a kwarg or maybe always enforce float32
        return supp

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        n = sample_shape.numel()
        n_burn = 500 if self._initial_points is None else 0  # dont need burnin if we already sampled
        results = sample_polytope(
            model=self._fcm._sampler, initial_points=self._initial_points, n=n, n_burn=n_burn, new_initial_points=True,
            thinning_factor=3, n_chains=6, return_what='rounded'
        )
        self._initial_points = results['new_initial_points']
        return results['rounded']

    @property
    def theta_id(self) -> pd.Index:
        return self._fcm.theta_id()

    @property
    def n_theta(self):
        return len(self._fcm.theta_id())


class _BaseXchFluxPrior(Distribution):
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
    ):
        # we do not have a support for these priors, since they are checked by the support of
        #   UniFluxPrior anyways
        if isinstance(model, LabellingModel):
            model = model.flux_coordinate_mapper
        if model._la.backend != 'torch':
            linalg = LinAlg('torch', seed=model._la._backwargs['seed'])
            model = model.to_linalg(linalg)

        if model._nx == 0:
            raise ValueError('no boundary fluxes')

        self._fcm = model
        self._la = model._la
        self._rho_bounds = model._rho_bounds
        super().__init__(event_shape=torch.Size((model._nx, )), validate_args={})

    @property
    def theta_id(self) -> pd.Index:
        return self._fcm.xch_theta_id()
    
    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return constraints.interval(self._rho_bounds[:, 0], self._rho_bounds[:, 1])


class XchFluxPrior(_BaseXchFluxPrior):  # TODO rename
    def __init__(
            self,
            model: Union[FluxCoordinateMapper, LabellingModel],
            mu_sigma: pd.DataFrame=None,  # for truncated normal sampling
    ):
        self._which = 'unif'
        self._mu = 0.5
        self._std = 0.2
        super().__init__(model)
        if mu_sigma is not None:
            self._which = 'gauss'
            raise NotImplementedError('this should signal that we sample from a truncated normal!')

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        result = self._la.sample_bounded_distribution(
            shape=sample_shape,
            lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1],
            mu=self._mu, std=self._std, which=self._which
        )
        return result.view(self._extended_shape(sample_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # we do not check any support here, since that has been done in UniFluxPrior
        if self._validate_args:
            self._validate_sample(value)
        if self._which == 'gauss':
            return self._la.trunc_norm_log_pdf(
                value, lo=self._rho_bounds[:, 0], hi=self._rho_bounds[:, 1], mu=self._mu, std=self._std
            )
        return torch.zeros((*value.shape[:-1], 1))


class UniformRoundedFleXchPrior(BaseRoundedPrior):
    def __init__(
            self,
            model,
            xch_prior: _BaseXchFluxPrior = None,
    ):
        # TODO not parallelized currently; not necessary in any of my applications, so back burner type thing
        #    would be necessary when sampling large flux models for instance
        super(UniformRoundedFleXchPrior, self).__init__(model, num_processes=0)
        if (self._fcm._nx > 0) and (xch_prior is None):
            xch_prior = XchFluxPrior(self._fcm)
        self._xch_prior = xch_prior

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        polytope = make_theta_polytope(self._fcm)
        supp = _CannonicalPolytopeSupport(polytope=polytope)
        # supp.to(dtype=torch.float32)  # TODO maybe pass dtype as a kwarg or maybe always enforce float32
        return supp

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        rounded_xch = super(UniformRoundedFleXchPrior, self).rsample(sample_shape)
        if self._fcm._nx > 0:
            xch_samples = self._xch_prior.sample(sample_shape)
            rounded_xch = self._la.cat([rounded_xch, xch_samples], dim=-1)
        return rounded_xch

    def log_prob(self, value):
        # log prob for uniform distribution is log(1 / vol(polytope))
        # for non-uniform xch flux distribution, return log(1 / vol(net_polytope)) + log_prob(xch_flux)
        if self._validate_args:
            self._validate_sample(value)
        # place-holder until we can compute polytope volumes
        log_prob = torch.zeros((*value.shape[:-1], 1))
        if self._fcm._nx > 0:
            if isinstance(self._xch_prior, XchFluxPrior) and (self._xch_prior._which != 'unif'):
                xch_fluxes = value[..., -self._fcm._nx:]
                log_prob += self._xch_prior.log_prob(xch_fluxes)
        return log_prob



if __name__ == "__main__":
    import pickle, os
    from sbmfi.settings import MODEL_DIR, BASE_DIR
    from sbmfi.models.build_models import build_e_coli_anton_glc, build_e_coli_tomek
    from sbmfi.models.small_models import spiro
    # from equilibrator_api import *

    # model, kwargs = spiro()
    # fcm = FluxCoordinateMapper(model)
    # pickle.dump(fcm, open('spiro_fcm.p', 'wb'))
    #
    model, kwargs = spiro(
        backend='numpy', add_biomass=True, ratios=False, build_simulator=False, v2_reversible=True, v5_reversible=True,
        kernel_id='rref',
    )
    fcm = FluxCoordinateMapper(
        model,
        kernel_id='svd',
    )
    xchp = UniformRoundedFleXchPrior(fcm)
    s = xchp.sample((10,))
    # print(s)
    # print(xchp._la.scale(torch.tensor(0.0), lo=-2.0, hi=2.0))
    # print(pd.DataFrame(s.numpy(), columns=xchp.theta_id))
    # print(fcm.theta_id)
    # up = UniformNetPrior(fcm, cache_size=50)
    # up.sample((20, ))
    # projected_fluxes = kwargs['measured_boundary_fluxes'][1:]
    # up = ProjectionPrior(model, projected_fluxes=projected_fluxes, cache_size=2000, number_type='fraction',
    #                      num_processes=0)
    # up._fill_caches(n=10, )
    # pickle.dump(up, open('spiro_projection_prior_w_volumes.p', 'wb'))
    # up = pickle.load(open('spiro_projection_prior_w_volumes.p', 'rb'))

    # model, kwargs = build_e_coli_anton_glc(backend='torch', which_measurements=None)
    # projected_fluxes = kwargs['measured_boundary_fluxes']
    # model.reactions.get_by_id('EX_glc__D_e').bounds = (-12.0, 0.0)
    # for rid in projected_fluxes:
    #     r = model.reactions.get_by_id(rid)
    #     print(r.bounds, r)
    # fcm = FluxCoordinateMapper(model, kernel_id='svd', coordinate_id='rounded', free_reaction_id=projected_fluxes)
    # up = ProjectionPrior(model, projected_fluxes=projected_fluxes, cache_size=2000, number_type='fraction', num_processes=0)
    # up._fill_caches(n=5000, )
    # pickle.dump(up, open('anton_glc_projection_prior_w_volumes.p', 'wb'))
    # up = pickle.load(open('anton_glc_projection_prior_w_volumes.p', 'rb'))
