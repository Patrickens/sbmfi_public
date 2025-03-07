import numpy as np
from typing import Union, List
import math
import scipy
import pandas as pd

import tqdm
from cobra import Reaction
import functools
from sympy import nsimplify, Matrix
from sympy.core.numbers import One
import cvxpy as cp
import cdd
import cdd.gmp
from sbmfi.core.util import _optlang_reverse_id_rex, _rho_constraints_rex, _net_constraint_rex, \
    _rev_reactions_rex, _xch_reactions_rex
from sbmfi.core.linalg import LinAlg
import copy
from PolyRound.api import PolyRoundApi, Polytope, PolyRoundSettings
from PolyRound.static_classes.lp_utils import ChebyshevFinder
from PolyRound.static_classes.rounding.maximum_volume_ellipsoid import MaximumVolumeEllipsoidFinder
from fractions import Fraction
from importlib.metadata import version
from scipy.spatial import ConvexHull
# import pypoman

class LabellingPolytope(Polytope):
    tolerance = 1e-12
    def __init__(
            self,
            A: pd.DataFrame,
            b: pd.Series,
            S: pd.DataFrame = None,
            h: pd.Series = None,
            mapper: dict = None,
            objective: dict = None,
            non_labelling_reactions: pd.Index = None
    ):
        if b is not None:
            b.name = 'ub'
        if h is not None:
            h.name = 'eq'
        Polytope.__init__(self, A=A, b=b, S=S, h=h)
        self._mapper: dict = mapper if mapper else {}
        self._objective: dict = objective if objective else {}
        self._nlr = non_labelling_reactions if non_labelling_reactions is not None else pd.Index([])

    @property
    def mapper(self):
        return self._mapper.copy()

    @property
    def non_labelling_reactions(self):
        return self._nlr.copy()

    @property
    def objective(self):
        return self._objective.copy()

    @objective.setter
    def objective(self, val: dict):
        for k in val.keys():
            if k not in self.A.columns:
                raise ValueError(f'{k} not in the polytope columns')
        self._objective = val

    @staticmethod
    def generate_cvxpy_LP(polytope, objective:dict=None, solve=False):
        # objective_reactions = cobra.util.solver.linear_reaction_coefficients(model)
        # polytope = polytope.copy()
        n = polytope.A.shape[1]

        objective = np.zeros(n)
        obj_dct = None
        if objective is not None:
            obj_dct = objective
        elif hasattr(polytope, '_objective'):
            obj_dct = polytope._objective

        if obj_dct is not None:
            for rid, coef in polytope._objective.items():
                objective[polytope.A.columns.get_loc(rid)] = coef

        objective = cp.Parameter(shape=objective.shape, value=objective)

        n_ineq = polytope.A.shape[0]

        v_cp = cp.Variable(n, name='fluxes')
        A_cp = cp.Parameter(polytope.A.shape, name='A', value=polytope.A.values)
        b_cp = cp.Parameter(n_ineq, name='b', value=polytope.b.values)
        constraints = [A_cp @ v_cp <= b_cp]

        cvx_result = {}
        if polytope.S is not None:
            n_met = polytope.S.shape[0]
            S_cp = cp.Parameter(polytope.S.shape, name='S', value=polytope.S.values)
            h_cp = cp.Parameter(n_met, name='h', value=polytope.h.values)
            constraints.append(S_cp @ v_cp == h_cp)
            cvx_result['S'] = S_cp
            cvx_result['h'] = h_cp

        problem = cp.Problem(objective=cp.Maximize(objective @ v_cp), constraints=constraints)

        cvx_result.update({
            'v': v_cp,  # easiest way to change bounds in the cvxpy problem
            'A': A_cp,  # easiest way to change bounds in the cvxpy problem
            'b': b_cp,
            'constraints': constraints,
            'problem': problem,
            'objective': objective,
            'polytope': polytope,
        })

        if solve and len(polytope._objective) > 0:
            problem.solve(solver=cp.GUROBI, verbose=False)
            cvx_result['solution'] = pd.Series(v_cp.value, index=polytope.A.columns, name=f'optimum', dtype=np.float64)
            cvx_result['optimum'] = problem.value
        return cvx_result


def fast_FVA(polytope: Polytope, full=False):
    cvx_result = LabellingPolytope.generate_cvxpy_LP(polytope)
    problem = cvx_result['problem']
    objective = cvx_result['objective']
    polytope = cvx_result['polytope']
    objective.value[:] = 0.0

    result = {}
    for i, reaction_id in zip(range(objective.value.shape[0]), polytope.A.columns):
        objective.value[i] = 1.0
        problem.solve(solver=cp.GUROBI, ignore_dpp=True)
        if problem.status != 'optimal':
            # raise ValueError(f'{reaction_id}: {problem.status}')
            print(f'{reaction_id}: {problem.status}')
        reac_max = round(problem.value, 4)
        objective.value[i] = -1.0
        problem.solve(solver=cp.GUROBI, ignore_dpp=True)
        reac_min = round(problem.value * -1, 4)
        objective.value[i] = 0.0
        if full:
            # TODO store the full flux vector at optimum instead of only the optimum
            raise NotImplementedError
        else:
            result[reaction_id] = (reac_min, reac_max)
    return pd.DataFrame(result, index=['min', 'max']).T


def _set_cdd_lib(number_type='float'):
    global _CDD
    if number_type == 'float':
        _CDD = cdd
    elif number_type == 'fraction':
        _CDD = cdd.gmp


def _cdd_mat_pol(ineq, eq=None, return_polyhedron=True, number_type='float', reptype=cdd.RepType.INEQUALITY, canonicalize=True):
    cdd_version = [int(_) for _ in version('pycddlib').split('.')]

    (A, b) = ineq
    if b.ndim < 2:
        b = b[:, None]
    np_arr = np.hstack([b, -A])

    lin_set = ()
    if eq is not None:
        (S, h) = eq
        if h.ndim < 2:
            h = h[:, None]
        if cdd_version[0] > 2:
            eq_arr = np.hstack([h, -S])  # TODO S or -S not sure
            np_arr = np.vstack([np_arr, eq_arr])
            lin_set = np.arange(b.shape[0], np_arr.shape[0])
    if cdd_version[0] > 2:  # changes from version 3:  https://pycddlib.readthedocs.io/en/stable/changes.html#version-3-0-0-4-october-2024
        _set_cdd_lib(number_type)
        if number_type == 'fraction':
            fractionater = np.vectorize(lambda x: Fraction(x))
            np_arr = fractionater(np_arr)
        matrix = _CDD.matrix_from_array(np_arr, rep_type=reptype, lin_set=lin_set)
        if canonicalize:
            sets = _CDD.matrix_canonicalize(matrix)  # TODO for some reason lin_set is set here????
            matrix.lin_set = ()  # TODO THIS IS WEIRD
        return _CDD.polyhedron_from_matrix(matrix) if return_polyhedron else matrix
    else:
        matrix = cdd.Matrix(np_arr, number_type=number_type)
        matrix.rep_type = reptype
        matrix.lin_set = lin_set
        if canonicalize:
            matrix = matrix.canonicalize()
        return cdd.Polyhedron(matrix) if return_polyhedron else matrix


def _cdd_dual(polyhedron: cdd.Polyhedron | cdd.gmp.Polyhedron, to_V_rep=True):
    cdd_version = [int(_) for _ in version('pycddlib').split('.')]
    if cdd_version[0] > 2:
        return _CDD.copy_output(polyhedron)
    else:
        if to_V_rep:
            return polyhedron.get_generators()
        return polyhedron.get_inequalities()


def _cdd_mat_ar(mat: cdd.Matrix | cdd.gmp.Matrix):
    cdd_version = [int(_) for _ in version('pycddlib').split('.')]
    if cdd_version[0] > 2:
        mat = np.array(mat.array).astype(float)
    return np.array(mat)


# function copied from https://github.com/stephane-caron/pypoman
def project_polyhedron(proj, ineq, eq=None, canonicalize=True, number_type='float'):
    # raise ValueError('does not yet work after update of the pycddlib API for version >= 3.x.x')

    P = _cdd_mat_pol(
        ineq, eq, return_polyhedron=True, number_type=number_type,
        reptype=cdd.RepType.INEQUALITY, canonicalize=canonicalize
    )
    # OLD code
    # linsys = cdd.Matrix(np.hstack([b, -A]), number_type=number_type)
    # linsys.rep_type = cdd.RepType.INEQUALITY

    # the input [d, -C] to cdd.Matrix.extend represents (d - C * x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    # Convert from H- to V-representation
    generators = _cdd_dual(P)
    if generators.lin_set:
        print("Generators have linear set: {}".format(generators.lin_set))
    V = _cdd_mat_ar(generators)

    # Project output wrenches to 2D set
    (E, f) = proj
    vertices, rays = [], []
    free_coordinates = []
    for i in range(V.shape[0]):
        if generators.lin_set and i in generators.lin_set:
            free_coordinates.append(list(V[i, 1:]).index(1.))
        elif V[i, 0] == 1:  # vertex
            vertices.append(np.dot(E, V[i, 1:]) + f)
        else:  # ray
            rays.append(np.dot(E, V[i, 1:]))

    # print(13, vertices, rays)
    return vertices, rays


def compute_polytope_vertices(ineq, number_type='float'):
    P = _cdd_mat_pol(ineq, None, True, number_type, cdd.RepType.INEQUALITY)
    g = _cdd_dual(P)
    V = _cdd_mat_ar(g)
    vertices = []
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            raise Exception("Polyhedron is not a polytope")
        elif i not in g.lin_set:
            vertices.append(V[i, 1:])
    return vertices


def simplify_vertices(vertices: pd.DataFrame, tolerance=1e-10):
    vertices = vertices.drop_duplicates()
    diff = (vertices.values[None, ...] - vertices.values[:, None, ...])
    norm_tol = np.linalg.norm(diff, 2, 2) < tolerance
    n = vertices.shape[0]
    norm_tol[np.tril_indices(n)] = False
    rows, cols = np.where(norm_tol)
    selecta = np.ones(n, dtype=bool)
    selecta[cols] = False
    return vertices.loc[selecta]


def V_representation(polytope: Polytope, number_type='float', vertices_tol=1e-10):
    if polytope.S is not None:
        raise ValueError('must be in cannonical form!')
    vertices = compute_polytope_vertices(ineq=(polytope.A.values, polytope.b.values), number_type=number_type)
    # # TODO try bretl, somehow give it some equality constraints...
    # vertices = compute_polytope_vertices(A=polytope.A.values, b=polytope.b.values, number_type='float')
    # pypoman.duality.compute_polytope_vertices(A=self.A.values, b=self.b.values)
    vertices = pd.DataFrame(vertices, columns=polytope.A.columns)
    if number_type == 'float':
        vertices = vertices.astype(float)
    vertices[abs(vertices) < vertices_tol] = 0.0
    if vertices_tol > 0.0:
        vertices = simplify_vertices(vertices, vertices_tol)
    return vertices


def H_representation(vertices: List[np.array], number_type='float', halfspace_tol=1e-10):
    """
    Compute the halfspace representation (H-rep) of a polytope defined as
    convex hull of a set of vertices:

    .. math::

        A x \\leq b
        \\quad \\Leftrightarrow \\quad
        x \\in \\mathrm{conv}(\\mathrm{vertices})

    Parameters
    ----------
    vertices : list of arrays
        List of polytope vertices.

    Returns
    -------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.
    """

    if isinstance(vertices, pd.DataFrame):
        V = vertices.values
    elif isinstance(vertices, list):
        V = np.vstack(vertices)
    t = np.ones((V.shape[0], 1))  # first column is 1 for vertices
    P = _cdd_mat_pol((-V, t), None,True, number_type, cdd.RepType.GENERATOR)
    bA = _cdd_dual(P, to_V_rep=False)
    bA = _cdd_mat_ar(bA)
    if bA.shape == (0,):  # bA == []
        return bA
    # the polyhedron is given by b + A x >= 0 where bA = [b|A]
    b, A = np.array(bA[:, 0]), -np.array(bA[:, 1:])
    b[abs(b) < halfspace_tol] = 0.0
    A[abs(A) < halfspace_tol] = 0.0
    return A, b


def project_polytope(
        polytope: Polytope,
        P: pd.DataFrame,
        p: pd.Series = None,
        return_vertices=False,
        vertices_tol=1e-10,
        number_type='float'
):
    # raise ValueError('not yet tested after pycddlib API change with version >= 3.x.x')
    # computes the projection/ infinite shadow of the labelling-polytope onto the exchange flux dimensions
    # https://github.com/stephane-caron/pypoman
    # https://pycddlib.readthedocs.io/en/latest/index.html
    P = P.loc[:, polytope.A.columns]
    if not P.index.isin(polytope.A.columns).all():
        raise ValueError('projection fluxes not in polytope fluxes')
    if p is None:
        p = pd.Series(0.0, index=P.index)
    ineq = (polytope.A.values, polytope.b.values)
    proj = (-P.values, p.values)
    if polytope.S is None:
        eq = None
    else:
        eq = (polytope.S.values, polytope.h.values)
    vertices, _ = project_polyhedron(proj=proj, ineq=ineq, eq=eq, number_type=number_type)
    vertices = pd.DataFrame(vertices, columns=P.index).drop_duplicates()  # TODO maybe clean up a bit
    if vertices_tol > 0.0:
        vertices = simplify_vertices(vertices, vertices_tol)
    if return_vertices:
        return vertices
    # A, b = pypoman.duality.compute_polytope_halfspaces(vertices.values)  # NOTE could also be done with scipy ConvHull
    A, b = H_representation(vertices)  # NOTE could also be done with scipy ConvHull
    A[abs(A) < vertices_tol] = 0.0
    # TODO make A have the right columns and come up with some index names
    return Polytope(A=pd.DataFrame(A, columns=P.index), b=pd.Series(b), mapper=None)


def thermo_2_net_polytope(polytope: LabellingPolytope, verbose=False):
    if len(polytope.mapper) == 0: # NB this is already a net_pol
        if verbose:
            print('already net')
        return polytope

    if (_xch_reactions_rex.search(list(polytope.mapper.values())[0]) is None):
        raise ValueError('this is not a thermo_pol')

    A = polytope.A.loc[:, ~polytope.A.columns.isin(list(polytope.mapper.values()))]
    has_coeff = np.linalg.norm(A, axis=1) != 0.0
    A = A.loc[has_coeff]
    b = polytope.b.loc[has_coeff]

    S = None
    if polytope.S is not None:
        S = polytope.S.loc[:, A.columns]

    return LabellingPolytope(A=A, b=b, S=S, h=polytope.h, mapper=None, objective=polytope.objective)


def extract_labelling_polytope(
        model: 'LabellingModel',
        coordinate_id ='labelling',
        zero_tol    = 1e-10,
        inf_bound   = 1e5,
) -> LabellingPolytope:
    # TODO test this thing with RatioMixins that have optlang ratio constraints! Should work, since we iterate over constraints
    if coordinate_id not in ['thermo', 'labelling']:
        raise ValueError('not a valid coordinate system')

    S_rows = {}
    h_rows = {}

    A_rows = {}
    b_rows = {}

    variables_id = dict((var.name, _optlang_reverse_id_rex.sub('_rev', var.name)) for var in model.variables)
    for constraint in model.constraints:
        # constraint.xb ∈ {value, inf, None}
        lb = constraint.lb
        ub = constraint.ub

        lb = lb if ((lb is None) or (abs(lb) != math.inf)) else inf_bound
        ub = ub if ((ub is None) or (abs(ub) != math.inf)) else inf_bound

        equality = False
        if (lb is not None) and (ub is not None) and ((ub - lb) < zero_tol):  # we know ub >= lb
            equality = True

        coefs = {
            variables_id[key.name]: val for key, val in
            constraint.get_linear_coefficients(constraint.variables).items()
        }
        if equality:
            S_rows[constraint.name] = coefs
            h_rows[constraint.name] = ub
        else:
            A_rows[constraint.name] = coefs
            b_rows[constraint.name] = [lb, ub]

    S = pd.DataFrame(S_rows, dtype=np.double).fillna(value=0.0).T  # metabolite and ratio equalities
    h = pd.Series(h_rows, name='eq')  # equalities

    A = pd.DataFrame(A_rows, index=S.columns, dtype=np.double).fillna(value=0.0).T  # these are the rho, net and ratio constraints
    b = pd.DataFrame(b_rows, index=['lb', 'ub']).T  # these are bounds on constraints in A

    bvar = {}
    non_labelling_reactions = []
    for reaction in model.reactions:
        if reaction.bounds == (0.0, 0.0):
            continue
        if type(reaction) == Reaction:
            non_labelling_reactions.append(reaction.id)
        fwd_var = reaction.forward_variable
        rev_var = reaction.reverse_variable
        if hasattr(reaction, '_rho_max') and (reaction.rho_max > 0.0):
            if coordinate_id == 'labelling':
                bvar[fwd_var.name] = (fwd_var.lb, fwd_var.ub)
                bvar[variables_id[rev_var.name]] = (rev_var.lb, rev_var.ub)
            elif coordinate_id == 'thermo':
                bvar[fwd_var.name] = reaction.bounds
                bvar[variables_id[rev_var.name]] = (reaction.rho_min, reaction.rho_max)
        # this causes a reaction that runs reverse to its 'definition' to be named reaction_rev in the polytope, which is not desirable
        elif (reaction.upper_bound <= 0.0) and (coordinate_id == 'labelling'):
            bvar[variables_id[rev_var.name]] = (-reaction.upper_bound, -reaction.lower_bound)
        else:
            bvar[reaction.id] = reaction.bounds

    non_labelling_reactions = pd.Index(non_labelling_reactions)

    bvar = pd.DataFrame(bvar, index=['lb', 'ub']).T
    A = A.loc[:, bvar.index]
    S = S.loc[:, bvar.index]

    if (coordinate_id == 'thermo') and len(b) > 0:
        wherho  = b.index.str.contains(_rho_constraints_rex)
        whernet = b.index.str.contains(_net_constraint_rex)
        A = A.loc[~(wherho | whernet)]
        b = b.loc[~(wherho | whernet)]

    n = bvar.shape[0]
    A_index = bvar.index

    wherrev = A_index.str.contains(_rev_reactions_rex)

    if hasattr(model, '_only_rev'):
        only_rev = model._only_rev
    else:
        raise NotImplementedError('this is useful for pure cobra models')

    if coordinate_id == 'thermo':
        xchid = A_index[wherrev].str.replace(_rev_reactions_rex, '_xch', regex=True)
        mapper = dict([(k, v) for k, v in zip(A_index[wherrev], xchid)])
        A_index = A_index.map(lambda x: mapper[x] if x in mapper else x)
    else:
        fwdid = A_index[wherrev].str.replace(_rev_reactions_rex, '', regex=True)
        mapper = dict([(k, v) for v, k in zip(A_index[wherrev], fwdid) if v not in only_rev])

    Avar = pd.DataFrame(np.eye(n, n), index=A_index, columns=bvar.index)
    Avar_1 = Avar * -1
    Avar_1.index = Avar.index + '|lb'
    Avar.index = Avar.index + '|ub'

    if len(A_rows) == 0:  # for models without reverse reactions
        A.index = A.index.astype('str')

    A_1 = A * -1
    A_1.index = A.index + '|lb'
    A.index = A.index + '|ub'
    A = pd.concat([Avar, Avar_1, A, A_1], axis=0)

    # construct final b
    b = pd.concat([
        bvar.loc[:, 'ub'],
        -bvar.loc[:, 'lb'],
        b.loc[:, 'ub'],
        -b.loc[:, 'lb'],
    ], names='ub')

    non_nan_constraints = ~b.isna().values
    b = b.loc[non_nan_constraints]
    A = A.loc[non_nan_constraints]
    b.index = A.index

    fluxes_id = model.labelling_fluxes_id
    if coordinate_id == 'thermo':
        fluxes_id = fluxes_id.map(lambda x: only_rev[x] if x in only_rev else x)

    exclude = slice(None)
    fluxes_id = non_labelling_reactions.append(fluxes_id)
    A = A.loc[:, fluxes_id]
    S = S.loc[exclude, fluxes_id]
    h = h.loc[exclude]

    # row_norm = 0 happens when we have a rho_min = 0.0
    ineq_coef = np.linalg.norm(A, axis=1) != 0.0
    A = A.loc[ineq_coef, :]

    A.sort_index(axis=0, inplace=True)  # readability
    b = b.loc[A.index]

    eq_coef = np.linalg.norm(S, axis=1) != 0.0
    S = S.loc[eq_coef, :]
    h = h.loc[eq_coef]

    if A.index.duplicated().any():
        print(A.index[A.index.duplicated()])
        raise ValueError('constraints need to have unique names!')

    # cosmetics
    A[A == -0.0] = 0.0
    b[b == -0.0] = 0.0

    if coordinate_id == 'thermo':
        S.loc[:, mapper.keys()] = 0.0
        S.rename(mapper, axis=1, inplace=True)
        A.rename(mapper, axis=1, inplace=True)
        mapper = dict([(_rev_reactions_rex.sub('', k), v) for k, v in mapper.items()])

    objective = {}
    objective_expression = model.solver.objective.expression
    coefficients = objective_expression.as_coefficients_dict()
    # from cobra.util import linear_reaction_coefficients
    for var, coef in coefficients.items():
        if isinstance(var, One):
            # NB this means that no objective has been set I think...
            break
        rid = variables_id[var.name]
        if rid in A.columns:
            objective[rid] = coef

    return LabellingPolytope(A=A, b=b, S=S, h=h, mapper=mapper, objective=objective, non_labelling_reactions=non_labelling_reactions)


def rref_null_space(S: pd.DataFrame, tolerance=1e-10):
    # nsimplify changes matrix from floats to rationals, this avoids numerical issues, makes it too slow for some models
    f = functools.partial(nsimplify, **{
        'constants': (),
        'tolerance': tolerance,
        'full': False,
        'rational': True,
        'rational_conversion': 'base10',
    })
    M = Matrix(S.values).applyfunc(f)
    reduced, pivots = M.rref(simplify=True, normalize_last=True)

    free_vars = np.array([i for i in range(M.cols) if i not in pivots], dtype=np.int64)
    basis = []

    for free_var in free_vars:
        vec = [M.zero] * M.cols
        vec[free_var] = M.one
        for piv_row, piv_col in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]
        basis.append(vec)

    NS = np.array([M._new(M.cols, 1, b) for b in basis]).astype(np.double).squeeze().T
    NS[abs(NS) < tolerance] = 0.0
    return pd.DataFrame(NS, index=S.columns, columns=S.columns[free_vars]), free_vars


def svd_null_space(S: pd.DataFrame, tolerance=1e-10):
    u, s, vh = np.linalg.svd(S.values)
    s = np.array(s.tolist())
    vh = np.array(vh.tolist())
    null_mask = s <= tolerance
    null_mask = np.append(null_mask, True)
    null_ind = np.argmax(null_mask)
    null = vh[null_ind:, :]
    freedex = [f'svd{i}' for i in range(null.shape[0])]
    return pd.DataFrame(np.transpose(null), index=S.columns, columns=freedex)


def simplify_polytope(
        polytope: Polytope,
        settings = PolyRoundSettings(),
        normalize = True
):
    return PolyRoundApi.simplify_polytope(polytope, settings, normalize)


def transform_polytope_keep_transform(
    polytope: Polytope,
    settings: PolyRoundSettings = PolyRoundSettings(),
    kernel_id ='svd',
) -> (Polytope, pd.DataFrame, pd.DataFrame, pd.Series):
    # PolyRoundApi.transform_polytope()
    if polytope.inequality_only:
        raise ValueError("Polytope already transformed (only contains inequality constraints)")

    polytope = polytope.copy()
    x, dist = ChebyshevFinder.chebyshev_center(polytope, settings)

    if polytope.border_distance(x) <= 0:
        raise ValueError("Chebyshev center outside polytope before transforming")

    if settings.verbose:
        print("chebyshev distance is : " + str(dist))
        pre_b_dist = polytope.border_distance(x)
        print("border distance pre-transformation is: " + str(pre_b_dist))

    if settings.verbose:
        x_0 = np.zeros(x.shape)
        b_dist_at_zero = polytope.border_distance(x_0)
        print("border distance zero-transformation is: " + str(b_dist_at_zero))

    cols = polytope.A.columns
    stoichiometry = polytope.S

    if kernel_id == 'svd':
        T = svd_null_space(stoichiometry, tolerance=settings.numerics_threshold)
        T_1 = T.T

        # put x at zero! # TODO is this correct
        polytope.apply_shift(x)
    elif kernel_id == 'rref':
        T, free_vars = rref_null_space(stoichiometry, tolerance=settings.numerics_threshold)
        T_1 = pd.DataFrame(0.0, index=T.columns, columns=T.index)
        T_1.loc[T.columns, T.columns] = np.eye(len(free_vars))

        x_star = T_1 @ x
        tau = (x - (T @ x_star))
        tau[abs(tau) < settings.numerics_threshold] = 0.0
        polytope.apply_shift(tau.values)
    else:
        raise ValueError

    tau = polytope.shift.to_frame()

    polytope.transformation.columns = cols
    polytope.apply_transformation(T)
    polytope.A.columns = T.columns
    if settings.verbose:
        u = np.zeros((T.shape[1], 1))
        norm_check = np.linalg.norm(np.matmul(stoichiometry.values, T))
        print("norm of the null space is: " + str(norm_check))
        b_dist = polytope.border_distance(u)
        print("border distance after transformation is: " + str(b_dist))
        # test if we can reproduce the original x
        trans_x = polytope.back_transform(u)
        x_rec_diff = np.max(trans_x - np.squeeze(tau.values))
        print("the deviation of the back transform is: " + str(x_rec_diff))
    # if isinstance(polytope, LabellingPolytope):
    #     polytope._mapper = {}
    #     polytope._objective = {}
    return polytope, T, T_1, tau


def round_polytope_keep_ellipsoid(
        polytope: Polytope,
        settings: PolyRoundSettings = PolyRoundSettings()
) -> (Polytope, pd.DataFrame, pd.DataFrame, pd.Series):
    polytope = polytope.copy()
    cols = polytope.A.columns
    bool = False
    bool += np.isinf(polytope.A.values).any()
    bool += np.isinf(polytope.b.values).any()
    if bool:
        raise ValueError("Polytope assigned for rounding contains inf")

    blank_polytope = Polytope(polytope.A, polytope.b)
    MaximumVolumeEllipsoidFinder.iterative_solve(blank_polytope, settings)
    # MaximumVolumeEllipsoidFinder.iterative_solve(
    #     o_polytope, backend, hp_flags=hp_flags, verbose=verbose, sgp=sgp
    # )
    # check if the transformation is full dimensional
    _, s, _ = np.linalg.svd(blank_polytope.transformation)
    if not np.min(s) > settings.thresh / settings.accepted_tol_violation:
        raise ValueError("Rounding transformation not full dimensional")
    # check if 0 is a solution
    if not blank_polytope.b.min() > 0:
        raise ValueError("Zero point not inside rounded polytope")

    E = blank_polytope.transformation
    epsilon = blank_polytope.shift.to_frame()

    E.columns = 'R_' + cols
    E.index = cols

    polytope.apply_shift(epsilon.values)
    polytope.apply_transformation(E.values)
    E_1 = pd.DataFrame(np.linalg.inv(E), index=E.columns, columns=E.index)
    polytope.A.columns = E.columns
    polytope.transformation.columns = E.columns
    return polytope, E, E_1, epsilon


def project_batch_onto_polytope(A, b, X):
    """
    Projects each point in a batch onto a polytope defined by A x <= b.

    Parameters
    ----------
    A : np.ndarray, shape (m, d)
        Matrix of inequality coefficients.
    b : np.ndarray, shape (m,)
        Right-hand side vector for the inequalities.
    X : np.ndarray, shape (n, d)
        Batch of points (each row is a point) to be projected.

    Returns
    -------
    projections : np.ndarray, shape (n, d)
        The closest point in the polytope for each input point.
    distances : np.ndarray, shape (n,)
        The Euclidean distance from each input point to its projection.
    """
    n, d = X.shape
    projections = []
    distances = []

    # For each point in the batch, solve a QP to compute its projection.
    for i in range(n):
        x = X[i]
        z = cp.Variable(d)
        objective = cp.Minimize(cp.sum_squares(z - x))
        constraints = [A @ z <= b]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        proj = z.value
        projections.append(proj)
        distances.append(np.linalg.norm(x - proj))

    return np.array(projections), np.array(distances)


class PolytopeSamplingModel(object):
    # combine stuff from labelling polytope and mapping things
    # this one is meant to be used by
    # TODO: it is better to refactor this into a class for the sampling model and a class for the ball and cylinder
    def __init__(
            self,
            polytope: Polytope,
            pr_verbose = False,
            kernel_id ='svd',
            linalg: LinAlg = None,
            **kwargs
    ):
        if kernel_id not in ['rref', 'svd']:
            raise ValueError(f'{kernel_id} not a valid basis kernel basis')

        if polytope.A.columns.str.contains(_xch_reactions_rex).any() or \
                polytope.A.columns.str.contains(_rev_reactions_rex).any():
            print(
                'This is not a net-polytope, you sure about this? '
                'Volume computation should not include xch fluxes and is unreliable for polytope over '
                '20 dimensions such as labelling polytopes due to the algorithm implementation'
            )

        self._pr_settings = PolyRoundSettings(**{'verbose': pr_verbose, **kwargs})
        self._ker_id = kernel_id # if transform_type in ['svd', 'rref']

        normalize = kernel_id != 'rref'

        F_simp = polytope
        if F_simp.S is not None:
            F_simp = simplify_polytope(polytope, settings=self._pr_settings, normalize=normalize)
            F_trans, self._T, self._T_1, self._tau = transform_polytope_keep_transform(
                F_simp, self._pr_settings, kernel_id
            )
        else:
            F_trans = F_simp
            self._T = pd.DataFrame(np.eye(F_simp.A.shape[1]), index=F_simp.A.columns, columns=F_simp.A.columns)
            self._T_1 = self._T.copy()
            self._tau = np.zeros(F_simp.A.shape[1])

        F_round, self._E, self._E_1, self._epsilon = round_polytope_keep_ellipsoid(F_trans, self._pr_settings)
        self._log_det_E = np.log(np.linalg.eig(self._E)[0]).sum()

        self._rounded_id = self._E_1.index
        self._transformed_id = self._T_1.index
        self._reaction_id = polytope.A.columns.tolist()

        if linalg == None:
            linalg = LinAlg(backend='numpy')

        self._la = linalg
        self._G = F_round.A.values
        self._h = F_round.b.values[:, np.newaxis]
        self._Q = F_round.transformation.values
        self._q = F_round.shift.values[:, np.newaxis]
        new = self.to_linalg(linalg)
        self.__dict__.update(new.__dict__)

    @property
    def log_det_E(self):
        return self._log_det_E

    @property
    def kernel_id(self):
        return self._ker_id

    @property
    def dimensionality(self) -> int:
        return self._G.shape[1]

    def map_fluxes_2_rounded(self, net_fluxes: pd.DataFrame, pandalize=False):
        index = None
        if isinstance(net_fluxes, pd.DataFrame):
            index = net_fluxes.index
            net_fluxes = self._la.get_tensor(values=net_fluxes.loc[:, self._rounded_id].values)

        transformed = self._la.tensormul_T(self._T_1, net_fluxes - self._tau.T)
        rounded = self._la.tensormul_T(self._E_1, transformed - self._epsilon.T)
        if pandalize:
            rounded = pd.DataFrame(self._la.tonp(rounded), index=index, columns=self._rounded_id)
            rounded.index.name = 'samples_id'
        return rounded

    def map_rounded_2_fluxes(self, rounded: pd.DataFrame, jacobian=True, pandalize=False):
        index = None
        if isinstance(rounded, pd.DataFrame):
            index = rounded.index
            rounded = self._la.get_tensor(values=rounded.loc[:, self._rounded_id].values)

        fluxes = self._la.tensormul_T(self._Q, rounded) + self._q.T
        if jacobian:
            J = self._Q[None, ...]  # need to add a dimension for batches of samples

        if pandalize:
            fluxes = pd.DataFrame(self._la.tonp(fluxes), index=index, columns=self.reaction_id)
            fluxes.index.name = 'samples_id'
        if  jacobian:
            return fluxes, J
        return fluxes

    def get_initial_points(self, num_points: int):
        # UniformSamplingModel.get_initial_points(self, num_points)
        distances = self._h / self._la.norm(self._G, ord=2, axis=1)  # the arguments are ord and axis
        radius = distances.min()
        samples = self._la.sample_unit_hyper_sphere_ball(shape=(num_points, self.dimensionality), ball=True)
        return samples * radius

    @property
    def reaction_id(self):
        """Gets the IDs of the reactions in the model."""
        return self._reaction_id

    @property
    def rounded_id(self):
        return self._rounded_id.copy(name='net_theta_id')

    def to_linalg(self, linalg: LinAlg):
        new = copy.copy(self)
        new._la = linalg
        for kwarg in ['_T', '_T_1', '_tau', '_E', '_E_1', '_epsilon', '_G', '_h', '_Q', '_q']:
            value = new.__dict__[kwarg]
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                value = value.values
            new.__dict__[kwarg] = linalg.get_tensor(values=value)
        return new


def get_rounded_polytope(psm: PolytopeSamplingModel):
    A = pd.DataFrame(psm._la.tonp(psm._G), columns=psm.rounded_id)
    b = pd.Series(psm._la.tonp(psm._h)[:, 0], name='ub')
    return Polytope(A=A, b=b)


class MarkovTransition():
    def __init__(
            self,
            model: PolytopeSamplingModel,
            target_density: 'torch.distributions.Distribution',
            n_cdf=7,
            proposal_id='gauss',
            chord_std=0.4,
            transition_id='peskun',
            return_log_prob_pi=True,  # if we need to compute Z, we should save the log_probs for all proposals
    ):
        self._la = model._la

        if proposal_id not in ['gauss', 'unif']:
            raise ValueError('not a valid proposal_id')
        self._unif = proposal_id == 'unif'

        if not hasattr(target_density, 'log_prob'):
            raise ValueError('cannot evaluate target density')
        self._pi = target_density

        self._n_cdf = n_cdf
        self._retlp = return_log_prob_pi

        if transition_id not in ['barker', 'peskun']:
            raise ValueError(f'not a valid transition transition_id: {transition_id} not in (barker, peskun)')
        self._barker = transition_id == 'barker'

        self._non_isotropic = False
        if not isinstance(chord_std, float):
            if (chord_std.ndim != 2) or (chord_std.shape[0] != model.dimensionality):
                raise ValueError('check shape of covariance matrix')
            try:
                self._la.LU(chord_std)
            except:
                raise ValueError('not a valid covariance matrix')
            self._non_isotropic = True
        self._chord_std = self._la.get_tensor(values=chord_std, dtype=np.float64)

        self._n_chains = 0
        self._selecta = None
        self._line_xs = None
        self._log_prob_pi = None
        self._axept = None
        self._tot = 0
        self._alpha = None

    def proposal(self, x, direction, alpha_min, alpha_max):
        log_prob = None

        if self._unif:
            alpha = self._la.sample_bounded_distribution(  # dont need log_probs, since they cancel out
                shape=(self._n_cdf, ), lo=alpha_min, hi=alpha_max
            )
        else:
            if self._non_isotropic:  # std in the direction of direction
                chord_std = self._la.sqrt(self._la.sum(((direction @ self._chord_std) * direction), -1))
            else:
                chord_std = self._chord_std
            alpha = self._la.sample_bounded_distribution(  #  mu = 0.0, since the current alpha = 0.0
                shape=(self._n_cdf, ), lo=alpha_min, hi=alpha_max, std=chord_std,
                which='gauss', return_log_prob=self._barker
            )
            if self._barker:
                alpha, log_prob = alpha
            else:
                self._alpha[1:] = alpha
                log_prob = self._la.bounded_distribution_log_prob(
                    x=self._alpha, lo=alpha_min, hi=alpha_max, mu=self._alpha, std=chord_std,
                    which='gauss', old_is_new=True, k=1
                    )
        return x + alpha[..., None] * direction, log_prob

    def __call__(self, x, direction, alpha_min, alpha_max):
        # THIS PUBLICATION HAS AN EASY PESKUN EXPLANATION FFS...
        # On parallelizable Markov chain Monte Carlo algorithms with waste-recycling
        # THIS ONE HAS THE SAME RESULTS
        # A general construction for parallelizing Metropolis−Hastings algorithms

        n_chains, n_dim = x.shape
        if self._n_chains != n_chains:
            self._selecta = self._la.arange(n_chains)
            self._line_xs = self._la.get_tensor(shape=(1 + self._n_cdf, n_chains, n_dim))
            self._log_prob_pi = self._la.get_tensor(shape=(1 + self._n_cdf, n_chains))
            self._log_prob_pi[0] = self._pi.log_prob(x)
            self._axept = self._la.zeros(n_chains, dtype=int)
            if not self._unif:
                self._alpha = self._la.zeros((self._n_cdf + 1, n_chains))
            self._n_chains = n_chains

        self._line_xs[0] = x

        self._line_xs[1:], log_q = self.proposal(x, direction, alpha_min, alpha_max)
        self._log_prob_pi[1:] = self._pi.log_prob(self._line_xs[1:])

        log_probs = self._log_prob_pi
        if log_q is not None:
            if self._barker:
                log_probs = self._log_prob_pi + 0.0  # copy data
                log_probs[1:] += log_q
            else:
                log_probs = self._log_prob_pi + log_q.sum(1)

        log_probs = log_probs - self._la.max(log_probs, dim=0)
        probs = self._la.exp(log_probs)

        if self._barker:
            probs = probs / probs.sum(0)
        else:
            probs[1:] = (probs[1:] / probs[0])  # R = pi(x_i)K(x/i | x_i) / pi(x_0)K(x/0 | x_0)
            probs[1:][probs[1:] > 1.0] = 1.0    # min(1, R)
            probs[1:] *= 1 / self._n_cdf        # 1 / m
            probs[0] = 1.0 - self._la.sum(probs[1:], 0)  # A[i,i] = 1 - sum(A[i,j]) j!=i

        accept_idx = self._la.multinomial(1, p=probs.T)[:, 0]
        new_x = self._line_xs[accept_idx, self._selecta]
        self._log_prob_pi[0] = self._log_prob_pi[accept_idx, self._selecta]

        self._tot += 1
        self._axept += accept_idx > 0
        if self._retlp:
            return new_x, self._log_prob_pi[0]
        return new_x


def sample_polytope(
        model: Union[PolytopeSamplingModel, Polytope],
        n: int = 2000,
        n_burn: int = 100,
        initial_points = None,
        thinning_factor = 3,
        n_chains: int = 2,
        new_initial_points=False,
        return_psm = False,
        phi: float = None,
        linalg: LinAlg = None,
        kernel_id: str = 'svd',
        markov_transition: MarkovTransition=None,
        return_what='rounded',
        show_progress=False,
):
    # TODO just use the function MCMC from sbmfi.estimate.simulator!
    r"""
    Hit and run sampler from uniform sampling points from a polytope,
    described via inequality constraints A*x<=b.

    Args:
        A: A Tensor describing inequality constraints
            so that all samples satisfy Ax<=b.
        b: A Tensor describing the inequality constraints
            so that all samples satisfy Ax<=b.
        x0: A `d`-dim Tensor representing a starting point of the chain
            satisfying the constraints.
        n: The number of resulting samples kept in the output.
        n_burn: The number of burn-in samples. The chain will produce
            n+n0 samples but the first n0 samples are not saved.
        seed: The seed for the sampler. If omitted, use a random seed.

    Returns:
        (n, d) dim Tensor containing the resulting samples.
    """

    if return_what not in ('all', 'fluxes', 'rounded', 'chains'):
        raise ValueError(f'return_what: {return_what} not in [all, fluxes, rounded, chains]')

    result = {}
    if isinstance(model, Polytope):
        model = PolytopeSamplingModel(model, kernel_id=kernel_id, linalg=linalg)
        if return_psm:
            result['psm'] = model
        result['log_det_E'] = model.log_det_E

    if (phi is not None) and (phi < 1.0):
        raise ValueError('c`est ne pas possiblementenete')

    K = model.dimensionality

    if initial_points is not None:
        n_burn = 0

    n_per_chain = math.ceil(n / n_chains)
    n_tot = n_burn + n_per_chain * thinning_factor
    chains = model._la.get_tensor(shape=(n_per_chain, n_chains, K))  # use for PSRF computation

    pbar = range(n_tot)
    if show_progress:
        pbar = tqdm.tqdm(pbar, ncols=100)

    if initial_points is None:
        x = model.get_initial_points(num_points=n_chains)
    else:
        if not initial_points.shape[0] == n_chains:
            raise ValueError
        x = initial_points

    if markov_transition is not None:
        if not model._la == markov_transition._la:
            raise ValueError(f'unequal backends')
        if markov_transition._retlp:
            chain_log_probs = model._la.get_tensor(shape=(n_per_chain, n_chains))

    biatch = min(5000, n_tot)  # batching this makes it a bit faster
    for i in pbar:
        # given x, the next point in the chain is x+alpha*r
        #             # it also satisfies A(x+alpha*r)<=b which implies A*alpha*r<=b-Ax
        #             # so alpha<=(b-Ax)/ar for ar>0, and alpha>=(b-Ax)/ar for ar<0.
        #             # b - A @ x is always >= 0, clamping for numerical tolerances

        if i % biatch == 0:
            # pre-sample samples from hypersphere
            # uniform samples from unit ball in d dims
            sphere_samples = model._la.sample_unit_hyper_sphere_ball(shape=(biatch, n_chains, K))
            # batch compute distances to all planes
            A_dist = model._la.tensormul_T(model._G, sphere_samples)
            if markov_transition is None:
                rands = model._la.randu((biatch, n_chains), dtype=model._G.dtype)

        sphere_sample = sphere_samples[i % biatch]
        ar = A_dist[i % biatch]
        dist = model._h.T - model._la.tensormul_T(model._G, x)
        dist[dist < 0.0] = 0.0
        allpha = dist / ar

        alpha_min, alpha_max = model._la.min_pos_max_neg(allpha, return_what=0)

        if phi is not None:
            # this is ellipsoid aware sampling for volume computation, meaning that
            # we choose the next step to be in the intersection of the polytope and a ball of radius rho
            # a = 1  # length of ball(1)-vector is 1...
            b = (sphere_sample * x).sum(1) * 2
            c = (x * x).sum(1) - phi ** 2   # elements of ax**2 + bx + c = 0
            sqrt = model._la.sqrt(b ** 2 - 4 * c)

            phi_max = (-b + sqrt) / 2
            phi_min = (-b - sqrt) / 2

            alpha_max = model._la.minimum(phi_max, alpha_max)
            alpha_min = model._la.maximum(phi_min, alpha_min)

        if markov_transition is None:
            # this means we do vanilla hit-and-run with uniform proposal along the line
            rnd = rands[i % biatch]
            alpha = alpha_min + rnd * (alpha_max - alpha_min)
            x = x + alpha[:, None] * sphere_sample
        else:
            # construct points along the line-segment and compute the empirical CDF from which we select the next step
            x = markov_transition(x, sphere_sample, alpha_min, alpha_max)
            if markov_transition._retlp:
                x, log_probs = x

        j = i - n_burn
        if (j > -1) & (j % thinning_factor == 0):
            chains[j // thinning_factor] = x
            if markov_transition is not None:
                if j == 0:
                    markov_transition._tot = 0  # only count after warm-up
                    markov_transition._axept[:] = 0  # only count after warm-up
                if markov_transition._retlp:
                    chain_log_probs[j // thinning_factor] = log_probs

    if show_progress:
        pbar.close()

    if return_what != 'chains':
        chains = model._la.view(chains, (n_chains * n_per_chain, K))[:n, :]
        if new_initial_points:
            new_points_idx = model._la.choice(n_chains, n)
            result['new_initial_points'] = chains[new_points_idx, :]

        if return_what in ('rounded', 'all'):
            result['rounded'] = chains

        if return_what in ('fluxes', 'all'):
            result['fluxes'] = model.map_rounded_2_fluxes(chains)
    else:
        result['chains'] = chains
        if markov_transition is not None:
            if markov_transition._retlp:
                result['log_probs'] = chain_log_probs
            result['acceptanced'] = markov_transition._axept
            result['tot_steps'] = markov_transition._tot
    return result

def compute_volume(
        model: Union[PolytopeSamplingModel, Polytope],
        n: int = -1,
        n0_multiplier: int = 5,
        thinning_factor: int = 1,
        epsilon: float = 1.0,
        enumerate_vertices: bool = False,
        return_all_ratios: bool = False,
        verbose: bool = False,
        use_scipy=False,
        kernel_id='svd'
):
    """
    a quadratic programme to find phi_max is maximizing a convex optimization problem,
    which is non-convex and  generally NP-hard
    """
    # raise NotImplementedError('refactored stuff, look at this')
    # raise NotImplementedError
    if isinstance(model, Polytope):
        model = PolytopeSamplingModel(model, kernel_id=kernel_id)

    if any([_xch_reactions_rex.search(rid) is not None for rid in model.reaction_id]):
        raise ValueError('This is a thermodynamic model that includes xch fluxes which lie in a hyper-rectangle, '
                         'it is much faster to compute the volume of the net polytope and the huper-rectangle separately!')

    K = model.dimensionality

    if n < 0:  # this is taken from the paper
        n = min(int(400 * epsilon ** -2 * K * np.log(K)), 100000)

    if enumerate_vertices or use_scipy:
        if (K > 6) and use_scipy:
            raise ValueError('this polytope is more than 6 dimensional, '
                             'thus making the ConvexHull algorithms prohibitively slow')
        vertices = V_representation(model._F_round)
        n_vertices = vertices.shape[0]
        if use_scipy:
            conv_hull = ConvexHull(vertices)
            return dict(scipy_volume=conv_hull.volume, n_vertices=n_vertices)
        else:
            if verbose:
                print('vertices done')
            phis = model._la.norm(vertices, 2, 1)
            phi_max = model._la.max(phis)
    else:
        sampling_result = sample_polytope(
            model=model, n=n * n0_multiplier, thinning_factor=thinning_factor, return_what='basis'
        )
        basis_samples = sampling_result['basis']
        phis = model._la.norm(basis_samples, 2, 1)
        phi_max = model._la.max(phis)  # this is the radius of the max ball that almost fully encloses the polytope

    beta = math.ceil(K * np.log(phi_max))
    ball_phis = np.array([np.exp(i / K) for i in range(0, beta + 1)])

    result = dict(
        log_B0_vol=np.log(np.pi ** (K / 2) / (scipy.special.gamma(K / 2 + 1))),
        K=K, N=n, phi_max=phi_max, beta=beta, log_det_E=model.log_det_E,
    )
    if enumerate_vertices:
        result['n_vertices'] = n_vertices

    ratios = np.zeros(ball_phis.size -1 )

    for i, phi_i_1 in enumerate(ball_phis[1:]):
        samples = sample_polytope(model, n=n, thinning_factor=thinning_factor, phi=phi_i_1, return_what='rounded')['rounded']
        sample_phis = model._la.norm(samples, 2, 1)
        n_ball_i = (sample_phis <= ball_phis[i]).sum()
        ratios[i] = n / n_ball_i

    if return_all_ratios:
        result['ratios'] = ratios

    result['log_ratio'] = np.log(ratios).sum()
    if verbose:
        print('VOLDONE')
    return result


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    import pickle
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    model, kwargs = spiro(
        backend='torch',
        # device='cuda:0',
        v2_reversible=False,
        device='cpu',
        build_simulator=True,
        which_measurements='lcms',
    )
    psm = PolytopeSamplingModel(model.flux_coordinate_mapper._Fn, linalg=model._la)


    import torch
    from sbmfi.priors.mog import MixtureOfGaussians

    n = 3
    gen = torch.Generator().manual_seed(3)
    K = psm.dimensionality

    if n > K:
        raise ValueError('Come on now.')

    means = torch.eye(psm.dimensionality) * 1.015
    means = torch.cat([means, -means])

    perm = torch.randperm(means.shape[0], generator=gen)
    which_means = perm[:n]
    means = means[which_means]

    weights = torch.randint(low=1, high=5, size=torch.Size((n,)), dtype=torch.double, generator=gen)
    weights /= weights.sum()

    covs = torch.eye(means.shape[-1])
    covs = torch.stack([covs] * means.shape[0]) * weights[:, None, None] / 8  # means that the distributions with less weight are more concentrated, good for plotting

    target = MixtureOfGaussians(means=means, covariances=covs, weights=weights)

    markov = MarkovTransition(psm, target_density=target, n_cdf=3)
    res = sample_polytope(psm, n_burn=0, n_chains=1, n=100, show_progress=True, markov_transition=markov)

