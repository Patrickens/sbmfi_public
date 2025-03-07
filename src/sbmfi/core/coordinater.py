import pandas as pd
import numpy as np
from typing import Union, List
import math
import copy
from PolyRound.api import Polytope
from scipy.spatial import Delaunay
from scipy.optimize import root
from sbmfi.core.reaction import LabellingReaction
from sbmfi.core.linalg import LinAlg
from sbmfi.core.polytopia import (
    PolytopeSamplingModel,
    extract_labelling_polytope,
    thermo_2_net_polytope,
    LabellingPolytope,
    get_rounded_polytope,
)
from scipy.linalg import helmert


def make_net_theta_id(
        psm: PolytopeSamplingModel,
        coordinate_id='rounded',
        hemi=False,
        sep_radius=True
):
    if coordinate_id == 'transformed':
        return psm._transformed_id.rename('net_theta_id')
    elif coordinate_id == 'rounded':
        return psm.rounded_id
    elif coordinate_id == 'ball':
        basis_str = 'B' if not hemi else 'HB'
        return pd.Index(
            [f'{basis_str}_{psm._ker_id}_{i}' for i in range(psm.dimensionality)] +
            ['R'], name='net_theta_id'
        )
    elif coordinate_id == 'cylinder':
        basis_str = 'C' if not hemi else 'HC'
        index = ['phi'] + [f'{basis_str}_{psm._ker_id}_{i}' for i in range(psm.dimensionality - 2)]
        if sep_radius:
            index += ['R']
        return pd.Index(index, name='net_theta_id')
    else:
        raise ValueError(f'{coordinate_id}')


def map_theta_2_tokens(
        psm: PolytopeSamplingModel,
):
    raise NotImplementedError('this is to prepare the data for transformers like in Simformer (https://arxiv.org/abs/2404.09636)')


def map_ball_2_polar(
        psm: PolytopeSamplingModel,
        ball: Union[pd.DataFrame, np.ndarray, 'torch.Tensor'],
        pandalize=False,
):
    # polar coordinates is another option...
    raise NotImplementedError
    coords = np.asarray(coords)
    n = len(coords)

    # Calculate polar angles
    angles = []
    for i in range(n - 1):
        norm = np.linalg.norm(coords[i:])
        if norm == 0:
            angle = 0
        else:
            angle = np.arccos(coords[i] / norm)
        angles.append(angle)
    return np.array(angles)


def map_polar_2_ball(
        psm: PolytopeSamplingModel,
        polar: Union[pd.DataFrame, np.ndarray, 'torch.Tensor'],
        pandalize=False,
):
    raise NotImplementedError
    n = len(polar) + 1
    coords = []
    angles = np.asarray(polar)
    n = len(angles) + 1

    coords = np.zeros(n)
    product = 1.0
    for i in range(n):
        if i < n - 1:
            coords[i] = product * np.cos(angles[i])
            product *= np.sin(angles[i])
        else:
            coords[i] = product
    return coords


def map_rounded_2_ball(
        psm: PolytopeSamplingModel,
        rounded: Union[pd.DataFrame, np.ndarray, 'torch.Tensor'],
        hemi: bool = False,
        alpha_root: float = None,
        sep_radius: bool = True,
        jacobian: bool = False,
        pandalize: bool = False
):
    """
    Maps a point `rounded` (in ℝ^K) into a ball coordinate system.

    For sep_radius=True, the transformation is defined as:
      norm = ||rounded||
      directions = rounded / norm
      alpha_frac = (norm / alpha_max(directions))^(alpha_root)
      result = [directions, alpha_frac]

    When jacobian=True, the function returns (result, J) where J has shape:
       - (n_samples, K+1, K) when sep_radius is True
       - (n_samples, K,   K) when sep_radius is False.

    The Jacobian for the scalar component (last row) now includes the additional term
    accounting for the dependence of alpha_max on the directions.
    """
    index = None
    if isinstance(rounded, pd.DataFrame):
        index = rounded.index
        rounded = psm._la.get_tensor(values=rounded.loc[:, psm.rounded_id].values)

    # Ensure rounded is 2D. If it's 1D, add a batch dimension.
    if len(rounded.shape) == 1:
        rounded = rounded[None, :]

    K = psm.dimensionality  # expected input dimension
    if jacobian:
        out_dim = K + 1 if sep_radius else K
        J = psm._la.get_tensor(shape=(*rounded.shape[:-1], out_dim, K))
        if hemi:
            raise NotImplementedError("Jacobian for hemi=True is not implemented.")
        if len(rounded.shape) > 2:
            raise NotImplementedError(f"{rounded.shape} not supported; expected (n_samples x K) tensor.")

    # ----- Forward computation -----
    norm = psm._la.norm(rounded, 2, -1, keepdims=True)  # shape: (n_samples, 1)
    directions = rounded / norm  # shape: (n_samples, K)
    if hemi:
        signs = psm._la.sign(directions[..., [0]])
        directions = directions * signs

    # Compute α_max from the linear constraints:
    allpha = psm._h.T / psm._la.tensormul_T(psm._G, directions)
    alpha_max = psm._la.min_pos_max_neg(
        allpha,
        return_what=0 if hemi else 1,
        keepdims=True,
        return_indices=jacobian
    )
    if jacobian:
        alpha_max, active_constraints = alpha_max
        active_G = psm._G[active_constraints]  # shape: (n_samples, 1, K)
        denom = psm._la.tensormul_T(active_G, directions[..., None, :])  # shape: (n_samples, 1, 1)

    if hemi:
        # (For hemi=True, a slightly different formula is used; not detailed here.)
        alpha_min, alpha_max = alpha_max
        first_el = directions[..., [0]]
        alpha = (rounded[..., [0]] - (alpha_min * first_el)) / first_el
        alpha_frac = alpha / (alpha_max - alpha_min)
    else:
        # Here t = norm/α_max is the untransformed radius fraction.
        alpha_frac = norm / alpha_max  # shape: (n_samples, 1)

    if alpha_root is None:
        alpha_root = K
    alpha_frac = psm._la.float_power(alpha_frac, 1.0 / alpha_root)

    if sep_radius:
        result = psm._la.cat([directions, alpha_frac], dim=-1)  # shape: (n_samples, K+1)
    else:
        result = directions * alpha_frac  # shape: (n_samples, K)

    if jacobian:
        batch_shape = rounded.shape[:-1]  # e.g. (n_samples,)

        # Compute d(directions)/dx = I/norm - (rounded rounded^T)/(norm^3)
        I = psm._la.eye(K, device=rounded.device)
        I_expanded = I.view(*((1,) * len(batch_shape)), K, K).expand(*batch_shape, K, K)
        d_directions_dx = I_expanded / norm[..., None] - psm._la.einsum('bi,bj->bij', rounded, rounded) / (norm[..., None] ** 3)

        # --- FIX: Derivative of alpha_frac = (norm/α_max)^(α_root) ---
        # The derivative of x^(α_root) is α_root * x^(α_root - 1). Thus:
        term1 = rounded / (norm ** 2)  # derivative of norm (ignoring α_max dependence)
        # Correction for the dependence of α_max on directions:
        term2 = psm._la.einsum('bmn,bnk->bmk', active_G, d_directions_dx).squeeze(1) / (denom.squeeze(-1))
        # Total derivative:
        d_alpha_frac_dx = alpha_frac * (1.0 / alpha_root) * (term1 + term2)  # shape: (n_samples, K)

        if sep_radius:
            J[..., :K, :] = d_directions_dx
            J[..., K, :] = d_alpha_frac_dx
        else:
            J[..., :, :] = (alpha_frac.unsqueeze(-1) * d_directions_dx +
                            psm._la.einsum('bi,bj->bij', directions, d_alpha_frac_dx))
    if pandalize:
        net_theta_id = make_net_theta_id(psm, coordinate_id='ball', hemi=hemi, sep_radius=sep_radius)
        result = pd.DataFrame(psm._la.tonp(result), index=index, columns=net_theta_id)
        result.index.name = 'samples_id'
    if jacobian:
        return result, J
    return result


def map_ball_2_rounded(
        psm: PolytopeSamplingModel,
        ball: Union[pd.DataFrame, np.ndarray, 'torch.Tensor'],
        hemi: bool = False,
        alpha_root: float = None,
        sep_radius: bool=True,
        jacobian: bool=False,
        pandalize: bool=False,
):
    """
    Maps a point `ball` (either a DataFrame or a tensor) into a "rounded" coordinate.

    For sep_radius=True, the input is assumed to be of the form:
         ball = [directions, radius_fraction]
    with directions in ℝ^K and radius_fraction a scalar in ℝ.

    When sep_radius=False, the entire input is interpreted as a ball point:
         directions = ball / ||ball||, and radius_fraction = ||ball||

    In both cases, the transformation is defined as:
         t' = (radius_fraction)^(1/alpha_root)
         α = t' * α_max(u)
         rounded = u * α
    where u = directions and α_max(u) is computed from linear constraints.

    If jacobian==True, the function returns (rounded, J) where J is the Jacobian:
      - When sep_radius=True, J has shape (n_samples, K, K+1).
      - When sep_radius=False, J has shape (n_samples, K, K).
    """

    index = None
    if isinstance(ball, pd.DataFrame):
        index = ball.index
        columns = make_net_theta_id(psm, coordinate_id='ball', hemi=hemi, sep_radius=sep_radius)
        ball = psm._la.get_tensor(values=ball.loc[:, columns].values)

    # Ensure ball is 2D; if 1D, add a batch dimension.
    if ball.dim() == 1:
        ball = ball.unsqueeze(0)

    K = psm.dimensionality

    dim = ball.shape[-1]
    if dim == K:
        Kk = K
    elif dim == K + 1:
        Kk = K + 1
    else:
        raise ValueError

    if jacobian:
        # For sep_radius=True, ball is assumed to have sampler.dimensionality+1 columns.
        # For sep_radius=False, we require ball.shape[-1] to match sampler.dimensionality.
        diags = psm._la.arange(K)
        # Preallocate a tensor for J in the sep_radius==True branch.
        J = psm._la.get_tensor(shape=(*ball.shape[:-1], K, Kk))
        J[..., diags, diags] = 1  # insert identity block
        if hemi:
            raise NotImplementedError("Jacobian for hemi=True is not implemented.")
        if len(ball.shape) > 2:
            raise NotImplementedError(f"{ball.shape}: expected (n_samples x n_ball) matrix only.")

    # --- Extract input parts ---
    if sep_radius:
        # ball = [directions, radius_fraction]
        directions = ball[..., :-1]  # shape: (n_samples, K)
        alpha_frac = ball[..., -1:]  # shape: (n_samples, 1)
    else:
        if ball.shape[-1] != K:
            raise ValueError('crack')
        norm = psm._la.norm(ball, 2, -1, keepdims=True)  # shape: (n_samples, 1)
        directions = ball / norm  # shape: (n_samples, K)
        alpha_frac = norm  # initially, use the norm

    if alpha_root is None:
        alpha_root = K

    alpha_frac = psm._la.float_power(alpha_frac, alpha_root)  # shape: (n_samples, 1)

    # --- Compute α_max from the constraints ---
    allpha = psm._h.T / psm._la.tensormul_T(psm._G, directions)
    alpha_max = psm._la.min_pos_max_neg(
        allpha, return_what=0 if hemi else 1, keepdims=True, return_indices=jacobian
    )
    if jacobian:
        alpha_max, active_constraints = alpha_max  # Expecting alpha_max: (n, 1)
    # --- Compute scalar multiplier α ---
    if hemi:
        alpha_min, alpha_max = alpha_max
        alpha = alpha_frac * (alpha_max - alpha_min) + alpha_min
    else:
        alpha = alpha_frac * alpha_max  # shape: (n_samples, 1)

    # --- Compute final rounded coordinate ---
    rounded = directions * alpha  # shape: (n_samples, K)

    # --- Jacobian computation ---
    if jacobian:
        if sep_radius:
            # --- Jacobian for sep_radius==True (unchanged snippet) ---
            root_correction = alpha_root * alpha_frac / ball[..., -1:]
            J[..., :, -1] = alpha_max * ball[..., :-1] * root_correction
            active_G = psm._G[active_constraints]
            denom = psm._la.tensormul_T(active_G, directions[..., None, :])
            num = psm._la.einsum("bi,bj->bij", directions, active_G.squeeze(-2))
            J[..., :, :-1] = psm._la.unsqueeze(alpha, -1) * (J[..., :, :-1] - num / denom)
        else:
            # --- Jacobian for sep_radius==False ---
            # Here, rounded = u * F, with u = ball/||ball|| and F = alpha_frac * alpha_max.
            # norm = psm._la.norm(ball, 2, -1, keepdims=True)  # shape: (n, 1) # TODO THIS IS COMPUTED TWICE
            d_u_dx = J / norm[..., None] - psm._la.einsum('bi,bj->bij', ball, ball) / (norm[..., None] ** 3)
            d_alpha_frac_dx = alpha_root * psm._la.float_power(norm, alpha_root - 2) * ball  # (n, K)
            active_G = psm._G[active_constraints[..., 0]]  # shape: (n, K)
            denom = (active_G * directions).sum(dim=-1, keepdim=True)  # shape: (n, 1)
            d_alpha_max_du = -alpha_max / denom * active_G  # (n, K)
            d_alpha_max_dx = psm._la.einsum(
                'bmn,bnk->bmk', psm._la.unsqueeze(d_alpha_max_du, 1), d_u_dx
            ).squeeze(1)
            dF_dx = alpha_max * d_alpha_frac_dx + alpha_frac * d_alpha_max_dx  # (n, K)
            J = psm._la.unsqueeze(alpha, -1) * d_u_dx + psm._la.einsum('bi,bj->bij', directions, dF_dx)  # (n, K, K)

    if pandalize:
        rounded = pd.DataFrame(psm._la.tonp(rounded), index=index, columns=psm.rounded_id)
        rounded.index.name = 'samples_id'
    if jacobian:
        return rounded, J
    return rounded


def map_ball_2_cylinder(
        psm: PolytopeSamplingModel,
        ball: Union[pd.DataFrame, np.ndarray, 'torch.Tensor'],
        hemi:bool=False,
        rescale_val:float=1.0,
        jacobian:bool=False,
        pandalize:bool=False
):
    if jacobian:
        raise NotImplementedError
    index = None
    if isinstance(ball, pd.DataFrame):
        index = ball.index
    if ball.shape[-1] < 2:
        raise ValueError('not possible for polytopes K<2')
    output = psm._la.vecopy(ball)
    for i in reversed(range(2, ball.shape[-1] - 1)):
        output[..., :i] /= psm._la.sqrt(1.0 - output[..., [i]] ** 2)

    atan = psm._la.arctan2(output[..., [0]], output[..., [1]])

    if rescale_val is not None:
        # this scales atan to [-1, 1]
        # atan is [0, pi] if _hemi else [-pi, pi]
        minb = 0.0 if hemi else -math.pi
        atan = -1 + 2 * (atan - minb) / (math.pi - minb)
        # R in [0, 1], so we scale to [-1, 1]
        output[..., -1] = -1 + 2 * output[..., -1]

    cylinder = atan
    if ball.shape[-1] > 2:
        cylinder = psm._la.cat([atan, output[..., 2:]], dim=-1)

    if (rescale_val is not None) and (rescale_val != 1):
        # scales to [-_bound, _bound]
        cylinder = -rescale_val + 2 * rescale_val * (cylinder + 1.0) / 2

    if pandalize:
        net_theta_id = make_net_theta_id(psm, coordinate_id='cylinder', hemi=hemi)
        cylinder = pd.DataFrame(psm._la.tonp(cylinder), index=index, columns=net_theta_id)
        cylinder.index.name = 'samples_id'
    return cylinder


def map_cylinder_2_ball(
        psm: PolytopeSamplingModel,
        cylinder: Union[pd.DataFrame, np.ndarray, 'torch.Tensor'],
        hemi=False,
        rescale_val=1.0,
        jacobian=False,
        pandalize=False,
):
    index = None
    if isinstance(cylinder, pd.DataFrame):
        index = cylinder.index
        net_theta_id = make_net_theta_id(psm, coordinate_id='cylinder', hemi=hemi)
        cylinder = psm._la.get_tensor(values=cylinder.loc[:, net_theta_id].values)

    cylinder = cylinder + 0.0  # copy data, since we modify in-place

    K = psm.dimensionality
    if jacobian:
        J_polar_rescale = psm._la.get_tensor(shape=(*cylinder.shape[:-1], K + 1, K))
        J_ball_cylinder = psm._la.get_tensor(shape=(*cylinder.shape[:-1], K + 1, K + 1))
        diags = psm._la.arange(K + 1)
        J_ball_cylinder[..., diags, diags] = 1

        cyl_rescale = 1
        atan_rescale = 1
        r_rescale = 1

    if (rescale_val is not None) and (rescale_val != 1):
        # scales cylinder to [-1, 1]
        cyl_rescale = 2 * rescale_val / 2
        cylinder = (cylinder + rescale_val) / (2 * rescale_val) * 2 - 1

    atan = cylinder[..., [0]]
    if rescale_val is not None:
        # y in [c, d], x in [a, b], dy/dx = (d-c)/(b-a)

        # scales atan is [0, pi] if _hemi else [-pi, pi]
        minb = 0.0 if hemi else -math.pi
        atan = (atan + 1) / 2 * (math.pi - minb) + minb
        atan_rescale = (math.pi - minb) / 2

        # scales R back to [0, 1]
        cylinder[..., -1] = (cylinder[..., -1] + 1) / 2
        r_rescale = 1 / 2

    sin = psm._la.sin(atan)
    cos = psm._la.cos(atan)
    if jacobian:
        J_polar_rescale[..., diags[:-1] + 1, diags[:-1]] = 1 * cyl_rescale  # all the cylinder coordinates and radius
        J_polar_rescale[..., -1, -1] *= r_rescale  # rescaling radius r
        J_polar_rescale[..., [0], 0] = cos * cyl_rescale * atan_rescale  # rescaling atan and d x_1 \ d theta = d sin(theta) / d theta = cos(theta)
        J_polar_rescale[..., [1], 0] = -sin * cyl_rescale * atan_rescale  # rescaling atan and d x_2 \ d theta = d cos(theta) / d theta = -sin(theta)

    ball = psm._la.cat([sin, cos, cylinder[..., 1:]], dim=-1)
    for i in range(2, ball.shape[-1] - 1):
        sqrt_1_r2 = psm._la.sqrt(1.0 - ball[..., [i]] ** 2)
        if jacobian:
            J_ball_cylinder[..., :i, i] = (ball[..., :i] * -ball[..., [i]]) / sqrt_1_r2
            J_ball_cylinder[..., diags[:i], diags[:i]] *= sqrt_1_r2
        ball[..., :i] *= sqrt_1_r2

    if jacobian:
        for i in range(2, K - 1):
            J_ball_cylinder[..., :i, i] *= J_ball_cylinder[..., i, [i]]

    if pandalize:
        net_theta_id = make_net_theta_id(psm, coordinate_id='ball', hemi=hemi)
        ball = pd.DataFrame(psm._la.tonp(ball), index=index, columns=net_theta_id)
        ball.index.name = 'samples_id'
    if jacobian:
        J = psm._la.tensormul_T(psm._la.transax(J_polar_rescale), J_ball_cylinder)
        return ball, J
    return ball


def barycentrics_from_simplex(x: np.ndarray, simplex, vertices):
    T = vertices[simplex]
    A = (T[1:] - T[0]).T
    b = x - T[0]
    lambdas_simplex = np.linalg.solve(A, b)
    lambdas_simplex = np.concatenate(([1 - np.sum(lambdas_simplex)], lambdas_simplex))
    return lambdas_simplex, simplex


def map_cartesian_2_delauney(x: np.ndarray, vertices: np.ndarray):
    tri = Delaunay(vertices)
    simplex_index = tri.find_simplex(x)
    if simplex_index == -1:
        raise ValueError("x is not inside the convex hull")
    simplex = tri.simplices[simplex_index]
    lambdas_simplex, simplex = barycentrics_from_simplex(x, simplex, vertices)
    full_lambdas = np.zeros(len(vertices))
    for i, idx in enumerate(simplex):
        full_lambdas[idx] = lambdas_simplex[i]
    return full_lambdas


class FluxCoordinateMapper(object):
    def __init__(
            self,
            model: 'LabellingModel',
            pr_verbose = False,
            kernel_id ='svd',  # basis for null-space of simplified polytope
            linalg: LinAlg = None,
            **kwargs
    ):
        # this is if we rebuild model and set new free reactions
        # free_reaction_id = [] if free_reaction_id is None else list(free_reaction_id)
        # if len(model._labelling_reactions) == 0:
        if not model._is_built:
            raise ValueError('build the model first')

        self._la = linalg if linalg else model._la

        self._F  = extract_labelling_polytope(model, 'labelling')
        self._Ft = extract_labelling_polytope(model, 'thermo')
        self._Fn = thermo_2_net_polytope(self._Ft, pr_verbose)
        self._n_lr = len(self.labelling_fluxes_id)

        self._sampler = PolytopeSamplingModel(self._Fn, pr_verbose, kernel_id, self._la, **kwargs)
        self._tsampler = None

        self._fwd_id = pd.Index(self._Ft.mapper.keys())
        self._only_rev = model._only_rev
        self._fwd_idx = self._la.get_tensor(
            values=np.array([self._Ft.A.columns.get_loc(rid) for rid in self._fwd_id]),
            dtype=np.int64
        )
        self._rev_idx = self._la.get_tensor(
            values=np.array([self._Ft.A.columns.get_loc(rid) for rid in self._Ft.mapper.values()]),
            dtype=np.int64
        )
        self._only_rev_idx = self._la.get_tensor(
            values=np.array([self._F.A.columns.get_loc(rid) for rid in self._only_rev.keys()]),
            dtype=np.int64
        )
        self._nx = len(self._fwd_id)
        self._rho_bounds = self._la.zeros((self._nx, 2))
        for i, rid in enumerate(self._fwd_id):
            reaction = model.labelling_reactions.get_by_id(rid)
            self._rho_bounds[i, 0] = reaction.rho_min
            self._rho_bounds[i, 1] = reaction.rho_max

        self._samples_id = self._la._batch_size

    @property
    def sampler(self):
        return self._sampler

    @property
    def samples_id(self):
        if isinstance(self._samples_id, int):
            return pd.RangeIndex(stop=self._samples_id)
        return self._samples_id.copy()

    @property
    def fwd_id(self):
        return self._fwd_id

    @property
    def xch_theta_id(self):
        if len(self._fwd_id) == 0:
            return pd.Index(name='xch_theta_id')
        return (self._fwd_id + '_xch').rename(name='xch_theta_id')

    def theta_id(self, coordinate_id='rounded'):
        net_theta_id = make_net_theta_id(self._sampler, coordinate_id)
        if self._nx > 0:
            xch_theta_id = self.xch_theta_id
            return net_theta_id.append(xch_theta_id).rename('theta_id')
        return net_theta_id

    @property
    def fluxes_id(self):
        return self._F.A.columns.copy()

    @property
    def labelling_fluxes_id(self):
        return self.fluxes_id[len(self._F.non_labelling_reactions):]

    @property
    def thermo_fluxes_id(self):
        return self._Ft.A.columns.copy()

    def map_theta_2_ball(self, rounded_xch):
        # TODO this function converts both net and xch coordinates to a ball
        if self._nx == 0:
            return map_rounded_2_ball(self._sampler, rounded_xch, sep_radius=False)
        if self._tsampler is None:
            theta_pol = make_theta_polytope(self)
            self._tsampler = PolytopeSamplingModel(theta_pol, False, 'svd', self._la)
        return map_rounded_2_ball(self._tsampler, rounded_xch, sep_radius=False)

    def map_net_theta_2_net_fluxes(
            self, net_theta: pd.DataFrame, coordinate_id='rounded', jacobian=False, pandalize=False
    ):
        index = None
        if isinstance(net_theta, pd.DataFrame):
            index = net_theta.index
            net_theta_id = make_net_theta_id(self._sampler, coordinate_id)
            net_theta = self._la.get_tensor(values=net_theta.loc[:, net_theta_id].values)

        J_rb, J_bc = None, None
        if coordinate_id != 'transformed':
            if coordinate_id == 'cylinder':
                net_theta = map_cylinder_2_ball(self._sampler, cylinder=net_theta, rescale_val=1.0, jacobian=jacobian) # = ball
                if jacobian:
                    net_theta, J_bc = net_theta
            if coordinate_id != 'rounded':
                net_theta = map_ball_2_rounded(self._sampler, ball=net_theta, jacobian=jacobian)  # = rounded
                if jacobian:
                    net_theta, J_rb = net_theta
            net_fluxes = self._sampler.map_rounded_2_fluxes(rounded=net_theta, jacobian=jacobian)  # = fluxes
            if jacobian:
                net_fluxes, J_fr = net_fluxes
                if J_rb is not None:
                    J_ft = J_fr @ J_rb
                if J_bc is not None:
                    J_ft = J_fr @ J_bc
                else:
                    J_ft = J_fr
        else:
            net_fluxes = self._la.tensormul_T(self._sampler._T, net_theta) + self._sampler._tau.T  # = transformed
            if jacobian:
                J_ft = self._sampler._T[None, ...]

        if pandalize:
            net_fluxes = pd.DataFrame(self._la.tonp(net_fluxes), index=index, columns=self._sampler.reaction_id)
            net_fluxes.index.name = 'samples_id'

        if jacobian:
            return net_fluxes, J_ft
        return net_fluxes

    def map_net_fluxes_2_net_theta(
            self, net_fluxes: pd.DataFrame, coordinate_id='rounded', pandalize=False
    ):
        index = None
        if isinstance(net_fluxes, pd.DataFrame):
            index = net_fluxes.index
            net_fluxes = self._la.get_tensor(values=net_fluxes.loc[:, self._sampler.reaction_id].values)

        net_theta = self._la.tensormul_T(self._sampler._T_1, net_fluxes - self._sampler._tau.T)  # = transformed

        if coordinate_id != 'transformed':
            net_theta = self._la.tensormul_T(self._sampler._E_1, net_theta - self._sampler._epsilon.T) # = rounded
            if coordinate_id != 'rounded':
                net_theta = map_rounded_2_ball(self._sampler, net_theta)  # = ball
                if coordinate_id != 'ball':
                    net_theta = map_ball_2_cylinder(self._sampler, net_theta, rescale_val=1.0)  # = cylinder

        if pandalize:
            net_theta_id = make_net_theta_id(self._sampler, coordinate_id=coordinate_id)
            net_theta = pd.DataFrame(self._la.tonp(net_theta), index=index, columns=net_theta_id)
            net_theta.index.name = 'samples_id'
        return net_theta

    def rescale_xch(self, xch_fluxes, rescale_val=1.0, to_rescale_val=True):
        if rescale_val is None:
            raise ValueError
        if to_rescale_val:
            old_lo, old_hi = self._rho_bounds[:, 0], self._rho_bounds[:, 1]
            new_lo, new_hi = -rescale_val, rescale_val
        else:
            old_lo, old_hi = -rescale_val, rescale_val
            new_lo, new_hi = self._rho_bounds[:, 0], self._rho_bounds[:, 1]
        zero_one_scale = self._la.scale(xch_fluxes, lo=old_lo, hi=old_hi, rev=False)
        return self._la.scale(zero_one_scale, lo=new_lo, hi=new_hi, rev=True)

    def frame_fluxes(self, labelling_fluxes: Union[pd.DataFrame, pd.Series, np.array], samples_id=None, trim=True):
        if isinstance(labelling_fluxes, pd.Series):
            # needed to have correct dimensions
            labelling_fluxes = labelling_fluxes.to_frame(name=labelling_fluxes.name).T

        if isinstance(labelling_fluxes, pd.DataFrame):
            samples_id = labelling_fluxes.index  # this means that the passed samples_id is ignored!
            labelling_fluxes = self._la.get_tensor(values=labelling_fluxes.loc[:, self._F.A.columns].values)

        labelling_fluxes = self._la.atleast_2d(labelling_fluxes)

        if samples_id is None:
            self._samples_id = labelling_fluxes.shape[0]
        else:
            self._samples_id = pd.Index(samples_id)
            if len(samples_id) != labelling_fluxes.shape[0]:
                raise ValueError('batch-size does not match samples_id size')
            elif self._samples_id.duplicated().any():
                raise ValueError('non-unique sample ids')
        if trim:
            labelling_fluxes = labelling_fluxes[..., len(self._F.non_labelling_reactions):]
        if labelling_fluxes.shape[-1] != self._n_lr:
            raise ValueError(f'wrong shape brahh, last dim should be {self._n_lr}, is {labelling_fluxes.shape}, maybe wrong trim?')
        return labelling_fluxes

    def map_thermo_2_fluxes(self, thermo_fluxes: pd.DataFrame, jacobian=False, pandalize=False):
        index = None
        if isinstance(thermo_fluxes, pd.DataFrame):
            index = thermo_fluxes.index
            thermo_fluxes = self._la.get_tensor(values=thermo_fluxes.loc[:, self.thermo_fluxes_id].values)

        fluxes = self._la.vecopy(thermo_fluxes)

        if self._nx > 0:
            xch = fluxes[..., self._rev_idx]
            net = fluxes[..., self._fwd_idx]

            abs_net = abs(net)
            rev = (abs_net * xch) / (1.0 - xch)
            fwd = rev + abs_net
            wherrev = net < 0.0
            remember = rev[wherrev]
            rev[wherrev] = fwd[wherrev]
            fwd[wherrev] = remember

            fluxes[..., self._rev_idx] = rev
            fluxes[..., self._fwd_idx] = fwd

        if len(self._only_rev) > 0:
            fluxes[..., self._only_rev_idx] *= -1
        if pandalize:
            fluxes = pd.DataFrame(self._la.tonp(fluxes), index=index, columns=self.fluxes_id)
            fluxes.index.name = 'samples_id'
        return fluxes

    def map_fluxes_2_thermo(self, fluxes: pd.DataFrame, jacobian=False, pandalize=False):
        index = None
        if jacobian:
            raise NotImplementedError
        if isinstance(fluxes, pd.DataFrame):
            index = fluxes.index
            fluxes = self._la.get_tensor(values=fluxes.loc[:, self._F.A.columns].values)

        thermo_fluxes = self._la.vecopy(fluxes)

        if len(self._only_rev) > 0:
            thermo_fluxes[..., self._only_rev_idx] *= -1

        if self._nx > 0:
            rev = thermo_fluxes[..., self._rev_idx]
            fwd = thermo_fluxes[..., self._fwd_idx]

            net = fwd - rev
            xch = rev / fwd
            wherrev = net < 0.0
            xch[wherrev] = 1.0 / xch[wherrev]
            thermo_fluxes[..., self._rev_idx] = xch
            thermo_fluxes[..., self._fwd_idx] = net
        if pandalize:
            thermo_fluxes = pd.DataFrame(self._la.tonp(thermo_fluxes), index=index, columns=self.thermo_fluxes_id)
            thermo_fluxes.index.name = 'samples_id'
        return thermo_fluxes

    def map_theta_2_fluxes(
            self, theta: pd.DataFrame, coordinate_id='rounded', return_thermo=True, pandalize=False
    ):
        index = None
        if isinstance(theta, pd.DataFrame):
            index = theta.index
            theta = self._la.get_tensor(values=theta.loc[:, self.theta_id(coordinate_id)].values)
        else:
            theta = self._la.vecopy(theta)  # theta is modified in-place in some map function, so we need to copy here
        if self._nx > 0:
            net_theta = theta[..., :-self._nx]  # this selects the net-variables
            xch_fluxes = theta[..., -self._nx:]
            if coordinate_id == 'cylinder':
                xch_fluxes = self.rescale_xch(xch_fluxes, to_rescale_val=False)
        else:
            net_theta = theta

        thermo_fluxes = self.map_net_theta_2_net_fluxes(net_theta, coordinate_id)  # should be in linalg form already
        if self._nx > 0:
            thermo_fluxes = self._la.cat([thermo_fluxes, xch_fluxes], dim=-1)
        if return_thermo:
            if pandalize:
                thermo_fluxes = pd.DataFrame(self._la.tonp(thermo_fluxes), index=index, columns=self.thermo_fluxes_id)
                thermo_fluxes.index.name = 'samples_id'
            return thermo_fluxes

        labelling_fluxes = self.map_thermo_2_fluxes(thermo_fluxes, pandalize=pandalize)
        if pandalize:
            if index is not None:
                labelling_fluxes.index = index
            labelling_fluxes.index.name = 'samples_id'
        return labelling_fluxes

    def map_fluxes_2_theta(
            self, fluxes: pd.DataFrame, coordinate_id='rounded', is_thermo=True, pandalize=False
    ):
        if coordinate_id not in ['transformed', 'rounded', 'ball', 'cylinder']:
            raise ValueError(f'{coordinate_id} not a valid basis coordinate system')

        index = None
        if isinstance(fluxes, pd.DataFrame):
            index = fluxes.index
            if is_thermo:
                cols = self._Ft.A.columns
            else:
                cols = self._F.A.columns
            fluxes = self._la.get_tensor(values=fluxes.loc[:, cols].values)

        thermo_fluxes = fluxes
        if not is_thermo:
            thermo_fluxes = self.map_fluxes_2_thermo(thermo_fluxes)

        if self._nx > 0:
            xch_fluxes = thermo_fluxes[..., self._rev_idx]
            if coordinate_id == 'cylinder':
                xch_fluxes = self.rescale_xch(xch_fluxes, to_rescale_val=True)

            net_fluxes = thermo_fluxes[..., :-self._nx]
            net_theta = self.map_net_fluxes_2_net_theta(net_fluxes, coordinate_id)
            theta = self._la.cat([net_theta, xch_fluxes], dim=1)
        else:
            theta = self.map_net_fluxes_2_net_theta(thermo_fluxes, coordinate_id)

        if pandalize:
            theta = pd.DataFrame(self._la.tonp(theta), index=index, columns=self.theta_id(coordinate_id))
            theta.index.name = 'samples_id'
        return theta

    def to_linalg(self, linalg: LinAlg):
        new = copy.copy(self)
        new._la = linalg
        new._sampler = self._sampler.to_linalg(linalg)
        for kwarg in ['_fwd_idx', '_rev_idx', '_only_rev_idx', '_rho_bounds',]:
            value = new.__dict__[kwarg]
            new.__dict__[kwarg] = linalg.get_tensor(values=value)
        return new


def expit_xch(fcm: FluxCoordinateMapper, xch_fluxes):
    return fcm._la.scale(
        fcm._la.expit(xch_fluxes), lo=fcm._rho_bounds[:, 0], hi=fcm._rho_bounds[:, 1], rev=True
    )


def logit_xch(fcm: FluxCoordinateMapper, xch_fluxes):
    return fcm._la.logit(fcm._la.scale(
        xch_fluxes, lo=fcm._rho_bounds[:, 0], hi=fcm._rho_bounds[:, 1], rev=False
    ))


def map_thermo_2_gibbs(
        fcm:FluxCoordinateMapper,
        thermo_fluxes: pd.DataFrame,
        pandalize=False
):
    if fcm._nx == 0:
        raise ValueError('no reversible reactions!')

    index = None
    if isinstance(thermo_fluxes, pd.DataFrame):
        index = thermo_fluxes.index
        thermo_fluxes = fcm._la.get_tensor(values=thermo_fluxes.loc[:, fcm._Ft.A.columns].values)

    xch = thermo_fluxes[..., fcm._rev_idx]
    net = thermo_fluxes[..., fcm._fwd_idx]

    xch[xch == 0.0] = 1.0
    exponent = fcm._la.ones(net.shape)
    exponent[net < 0.0] = -1
    T = LabellingReaction.T
    R = LabellingReaction._R
    dgibbsr = R * T * fcm._la.log(xch) ** exponent
    if LabellingReaction._KILOJOULE:
        dgibbsr /= 1000.0
    if pandalize:
        dgibbsr = pd.DataFrame(fcm._la.tonp(dgibbsr), index=index, columns=fcm._fwd_id + '_xch')
        dgibbsr.index.name = 'samples_id'
    return dgibbsr


def map_gibbs_2_xch_fluxes(
        fcm: FluxCoordinateMapper,
        dgibbsr: pd.DataFrame
):
    if fcm._nx == 0:
        raise ValueError('no reversible reactions!')
    if not isinstance(dgibbsr, pd.DataFrame):
        raise ValueError('needs to be a dataframe, since we need to .loc columns')

    dgibbsr = dgibbsr.loc[:, fcm._fwd_id]
    if LabellingReaction._KILOJOULE:
        dgibbsr = dgibbsr * 1000

    T = LabellingReaction.T
    R = LabellingReaction._R
    exponent = np.ones(dgibbsr.shape)
    exponent[dgibbsr > 0.0] = -1.0

    xch_fluxes = np.exp(dgibbsr / (R * T)) ** exponent
    return pd.DataFrame(xch_fluxes, index=dgibbsr.index, columns=dgibbsr.columns)


def map_labelling_jac_2_rounded_jac(
        psm: FluxCoordinateMapper,
        labelling_jacobian: Union['torch.Tensor', np.array],  # this is a jacobian of state w.r.t. fluxes, we might want to differentiate further to free variables
        thermo_fluxes: pd.DataFrame,
):
    # raise NotImplementedError
    # TODO this is a complex function that propagates the jacobian all the
    #  way from labelling jacobian to free variables jacobian
    # TODO: need to fix fwd and reverse mapping for bi-directional fluxes
    # TODO deal with the fact that cofactor fluxes are included in the coordinate mapper!!!!


    if fcm._J_lt is None:
        # labelling fluxes w.r.t. thermo fluxes
        n = len(fcm.thermo_fluxes_id)
        fcm._J_lt = fcm._la.get_tensor(shape=(thermo_fluxes.shape[0], n, n))
        fcm._J_lt[...] = fcm._la.eye(n)[None, :, :]
        if len(fcm._only_rev) > 0:
            fcm._J_lt[..., fcm._only_rev_idx, fcm._only_rev_idx] = -1.0

    if fcm._nx > 0:
        xch = thermo_fluxes[..., fcm._rev_idx]
        net = thermo_fluxes[..., fcm._fwd_idx]

        drev_dnet = (xch / (1.0 - xch))
        fcm._J_lt[..., fcm._fwd_idx, fcm._rev_idx] = drev_dnet
        fcm._J_lt[..., fcm._fwd_idx, fcm._fwd_idx] = drev_dnet + 1.0

        drev_dxch = net / (1.0 - xch) ** 2
        fcm._J_lt[..., fcm._rev_idx, fcm._rev_idx] = drev_dxch
        fcm._J_lt[..., fcm._rev_idx, fcm._fwd_idx] = drev_dxch

    if fcm._J_tt is None:
        # thermo fluxes w.r.t. theta
        n = 1
        if fcm._logxch:
            n = thermo_fluxes.shape[0]
        fcm._J_tt = fcm._la.get_tensor(shape=(n, len(fcm.theta_id), len(fcm.labelling_fluxes_id)))
        fcm._J_tt[:, :len(fcm.make_net_theta_id), :-fcm._nx] = fcm._mapper.to_fluxes_transform[0].T[None, :, :]
        if not fcm._logxch and (fcm._nx > 0):
            fcm._J_tt[:, -fcm._nx, -fcm._nx] = fcm._la.ones(fcm._nx)

    return fcm._J_tt @ fcm._J_lt @ labelling_jacobian


def make_theta_polytope(fcm: FluxCoordinateMapper):
    net_polytope = get_rounded_polytope(fcm._sampler)
    if fcm._nx == 0:
        return net_polytope
    xch_id = fcm.xch_theta_id()
    xch_A = pd.DataFrame(0.0, columns=xch_id, index=net_polytope.A.index)
    A = pd.concat([net_polytope.A, xch_A], axis=1)
    ub_idx = fcm._fwd_id + '_xch|ub'
    lb_idx = fcm._fwd_id + '_xch|lb'
    A_xch = pd.DataFrame(0.0, columns=A.columns, index=ub_idx.append(lb_idx))
    A_xch.loc[ub_idx, xch_id] =  np.eye(fcm._nx)
    A_xch.loc[lb_idx, xch_id] = -np.eye(fcm._nx)
    A_xch[A_xch == -0.0] = 0.0
    bounds = fcm._la.tonp(fcm._rho_bounds)
    b_xch = pd.Series(np.concatenate([bounds[:, 1], bounds[:, 0]]), index=A_xch.index)
    return Polytope(A=pd.concat([A, A_xch], axis=0), b=pd.concat([net_polytope.b, b_xch]))


def map_simplex_2_ilr(simplex, linalg=LinAlg('torch'), H=None, eps=1e-12, jacobian=False):
    if H is None:
        H = linalg.get_tensor(values=helmert(simplex.shape[-1]), dtype=simplex.dtype)

    if len(simplex.shape) < 2:
        simplex = simplex[None, :]

    safe_simplex = linalg.clip(simplex, eps, 1)

    ilr = (H @ linalg.log(safe_simplex.T)).T
    if jacobian:
        J = linalg.diag_embed(1 / safe_simplex)
        J = H @ J
        return ilr, J
    return ilr


def map_ilr_2_simplex(z, linalg=LinAlg('torch'), H=None, jacobian=False):
    if H is None:
        H = linalg.get_tensor(values=helmert(z.shape[-1] + 1), dtype=z.dtype, device=z.device)

    if len(z.shape) < 2:
        z = z[None, :]

    logx = (H.T @ z.T).T  # shape (B, D)

    x = linalg.exp(logx)
    x = x / x.sum(axis=1, keepdims=True)

    if jacobian:
        dx_dv = linalg.diag_embed(x) - linalg.einsum('ni,nj->nij', x, x)
        J =  dx_dv @ H.T
        return x, J
    return x


def map_rounded_2_max_entropy(x: np.ndarray, vertices: np.ndarray, tolerance=1e-10):
    if not isinstance(x, np.ndarray):
        raise ValueError('only for numpy so far')

    def jacobian_map_rounded_2_max_entropy(mu) -> np.ndarray:
        dot_products = vertices.dot(mu)  # shape (n,)
        # Subtract maximum to improve stability
        max_dot = np.max(dot_products)
        exp_dot = np.exp(dot_products - max_dot)  # shape (n,)
        Z = np.sum(exp_dot)
        lambdas = exp_dot / Z  # shape (n,)

        # Compute weighted outer products in a stable manner.
        J = (vertices * lambdas[:, None]).T @ vertices

        # Compute the weighted average of vertices
        v_bar = np.sum(lambdas[:, None] * vertices, axis=0)
        J -= np.outer(v_bar, v_bar)
        return J

    def f(mu, x):
        dot_products = vertices.dot(mu)  # shape (n,)
        max_dot = np.max(dot_products)
        exp_dot = np.exp(dot_products - max_dot)
        Z = np.sum(exp_dot)
        lambdas = exp_dot / Z  # shape (n,)
        return np.dot(lambdas, vertices) - x  # shape (d,)

    # Ensure x is at least 2-dimensional (each row is a point)
    if len(x.shape) < 2:
        x = x[None, :]
    elif len(x.shape) != 2:
        raise NotImplementedError("x must be 1D or 2D.")

    results = []
    K = x.shape[1]
    mu0 = np.zeros(K)
    for xi in x:
        # We use a lambda to fix xi in the function f.
        sol = root(lambda mu: f(mu, xi), mu0, tol=tolerance, jac=jacobian_map_rounded_2_max_entropy)
        mu = sol.x
        dot_products = vertices.dot(mu)
        max_dot = np.max(dot_products)
        exp_dot = np.exp(dot_products - max_dot)
        Z = np.sum(exp_dot)
        lambdas = exp_dot / Z
        results.append(lambdas)
    return np.array(results)


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    from sbmfi.core.polytopia import sample_polytope
    from torch.autograd.functional import jacobian
    import torch



    m,k = spiro(backend='torch')
    m.build_model()
    fcm = FluxCoordinateMapper(m)
    rounded = sample_polytope(fcm.sampler, n=50, n_burn=0)['rounded']
    K = fcm.sampler.dimensionality

    alpha_root = K



    # sep_radius = True
    #
    # b, J = map_rounded_2_ball(fcm.sampler, rounded[:3], jacobian=True, alpha_root=alpha_root, sep_radius=sep_radius)
    # # print(b)
    # # print(b.norm(2,-1))
    # f = lambda x: map_ball_2_rounded(fcm.sampler, x, alpha_root=alpha_root, sep_radius=sep_radius)
    # autojac = jacobian(f, b[0], strict=True)
    #
    # r, J = map_ball_2_rounded(fcm.sampler, b, jacobian=True, alpha_root=alpha_root, sep_radius=sep_radius)
    #
    # print(rounded[:3])
    # print(r)
    # print(autojac)
    # print(J[0])


    # f, J = fcm.map_net_theta_2_net_fluxes(res['rounded'], jacobian=True)


    #
    # sep_radius = False
    # ball, j1 = map_rounded_2_ball(fcm.sampler, res['rounded'], jacobian=True, sep_radius=sep_radius)
    # ding = lambda x: map_rounded_2_ball(fcm.sampler, x, sep_radius=sep_radius)
    # JJ = jacobian(ding, res['rounded'][0])
    # print(j1[0])
    # print(JJ)
    # #
    # r, j = map_ball_2_rounded(fcm.sampler, ball, jacobian=True, sep_radius=sep_radius)
    # ding = lambda x: map_ball_2_rounded(fcm.sampler, x, sep_radius=sep_radius)
    # JJ = jacobian(ding, ball[0])
    #
    #
    # print(j[0])
    # print(JJ)

    # cyl_pol = fcm.map_rounded_2_net_theta(res['rounded'], coordinate_id='cylinder', pandalize=False)
    # cyl, J = fcm.Jacobian_cylinder_polar_cylinder(cyl_pol)
    # ball, J2 = fcm.Jacobian_ball_cylinder(cyl)
    # r1, J3 = fcm.Jacobian_rounded_ball(cyl)
    # ball, J = fcm.map_cylinder_2_ball(cyl_pol, jacobian=True)
    # rounded, J4 = fcm.map_ball_2_rounded(cyl, jacobian=True)
    # from torch.autograd.functional import jacobian
    #
    # jongehh = lambda x: fcm.map_ball_2_rounded(fcm.map_cylinder_2_ball(x)[None, :])
    #
    # jackie = jacobian(jongehh, flow_samples[0])
    # jackie
    #
    # print(r1)
    # print(rounded)