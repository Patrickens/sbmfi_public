
from cobra.util.context import get_context
from cobra import Model, Reaction, Metabolite, DictList, Object
import math
from sbmfi.core.linalg import LinAlg
from sbmfi.core.util   import (
    _rev_reactions_rex,
)
from sbmfi.core.polytopia import (
    extract_labelling_polytope,
    thermo_2_net_polytope,
    fast_FVA
)
from sbmfi.core.coordinater import FluxCoordinateMapper
from itertools import repeat
from typing import Iterable, Union
from sbmfi.core.util import _get_dictlist_idxs, _read_atom_map_str_rex, _find_biomass_rex, _strip_bigg_rex
import pandas as pd
from cobra import Reaction
from math import isinf
from collections.abc import Iterable
import operator
from copy import copy, deepcopy
from sbmfi.lcmsanalysis.formula import Formula
import numpy as np
import re
from abc import abstractmethod
import pickle


class LabelledMetabolite(Metabolite):
    """
    Contains information on a metabolite and 13C labelling states

    Parameters
    ----------
    id : str
        BiGG identifier to associate with the metabolite
    sym : bool
        Whether the metabolite has a rotational symmetry of 180°; e.g. succinate
    formula : str
        Chemical formula (e.g. H2O)
    name : str
        A human readable name.
    charge : float
       The charge number of the metabolite
    compartment: str or None
       Compartment of the metabolite.
    """
    def __init__(
            self,
            idm = None,
            symmetric: bool = False,
            formula: str = '',
            name: str = '',
            charge: int = 0,
            compartment: str = None,
            total_intensity = None,  # either a number or a distribution from which total intensities are sampled
    ):
        if isinstance(idm, LabelledMetabolite):
            raise NotImplementedError
        elif isinstance(idm, Metabolite):  # only if metabolite
            self.__dict__.update(idm.__dict__)
            self.formula = formula if formula else self.__dict__.pop('formula')
        elif isinstance(idm, str) or (idm is None):  # None for consistent copying behavior
            Metabolite.__init__(
                self, id=idm, name=name, formula=formula, charge=charge, compartment=compartment
            )
        else:
            raise ValueError
        self.symmetric = symmetric

    def __getstate__(self):
        state = super(LabelledMetabolite, self).__getstate__()
        state['_formula'] = self.formula
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.formula = state['_formula']
        self._init_state()

    @property
    def weight(self):
        return self._formula['C']

    @property
    def formula(self):
        return self._formula.to_chnops() # added to_chnops, hope this does not break stuff!

    @formula.setter
    def formula(self, val: str):
        formula = Formula(formula=val).no_isotope()
        if '-' in formula:
            raise ValueError('Parse charge separately!')
        self._formula = formula
        self._init_state()
        if self._model is not None:
            self._model._is_built = False

    @property
    def elements(self):
        return self._formula

    def remove_from_model(self, destructive=False):
        raise NotImplementedError

    @property
    def formula_weight(self):
        return self._formula.mass(ion=False)

    @abstractmethod
    def _init_state(self):
        pass


class IsoCumo(Object):
    _label_check = re.compile('^[01]+$')
    def __init__(self, metabolite: LabelledMetabolite, label:str, name:str=None):
        Object.__init__(self, id=metabolite.id + '/' + label, name=name)
        self.metabolite = metabolite
        self.label = label

    def __getstate__(self):
        state = Object.__getstate__(self)
        state['metabolite'] = None
        return state

    @property
    def label(self):
        return ''.join(self._label.astype(int).astype(str))

    @property
    def weight(self):
        return self._label.sum()

    @property
    def int10(self):
        return int(self.label, 2)

    @label.setter
    def label(self, val: str):
        if not self._label_check.match(val):
            raise ValueError('Label should only have 0s and 1s')
        val = np.array(list(val), dtype=int).astype(bool)
        if not self.metabolite.elements['C'] == val.shape[0]:
            raise ValueError('Label does not match number of carbons')
        self._label = val

    @property
    def formula(self):
        return (self.metabolite._formula.add_C13(nC13=self.weight)).to_chnops()


class EMU_Metabolite(LabelledMetabolite):
    def __getstate__(self):
        state = super(EMU_Metabolite, self).__getstate__()
        state['emus'] = None
        state['convolvers'] = None
        return state

    def _init_state(self):
        self.emus = dict([(weight, DictList()) for weight in range(1, self.elements['C']+1)])
        self.convolvers = DictList()
        if self.elements['C'] > 0:
            self_emu = EMU(metabolite=self, positions=np.arange(self.elements['C']))
            self.emus[self.elements['C']].append(self_emu)

    def get_emu(self, positions:np.array):
        emu = EMU(metabolite=self, positions=positions)
        if emu in self.emus[emu.weight]:
            emu = self.emus[emu.weight].get_by_id(id=emu.id)
        else:
            self.emus[emu.weight].append(emu)
        return emu

    def get_convolved_emu(self, emus):
        convolemu = ConvolutedEMU(emus=emus)
        if convolemu in self.convolvers:
            convolemu = self.convolvers.get_by_id(id=convolemu.id)
        else:
            for emu in emus:
                if convolemu not in emu.metabolite.convolvers:
                    emu.metabolite.convolvers.append(convolemu)
        return convolemu


class EMU(Object):
    def __init__(self, metabolite:LabelledMetabolite, positions:np.array, name:str=None):
        self.metabolite = metabolite
        self.positions = positions
        Object.__init__(self, id=metabolite.id + '|[' + ','.join(self.positions.astype(str)) + ']', name=name)

    def __getstate__(self):
        state = Object.__getstate__(self)
        state['metabolite'] = None
        return state

    @property
    def weight(self):
        return self._positions.shape[0]

    @property
    def positions(self):
        return self._positions.copy()
    @positions.setter
    def positions(self, val: np.array):
        positions = np.array(val)
        positions.sort()
        if any(positions < 0):
            raise ValueError(f'cannot deal with negative positions {positions}')
        if not np.unique(positions).shape[0] == positions.shape[0]:
            raise ValueError(f'non-unique positions {positions}')
        if not all(positions < self.metabolite.elements['C']) or (len(positions) > self.metabolite.elements['C']):
            raise ValueError(f'positions {positions} longer than number of carbons {self.metabolite.elements["C"]}')
        self._positions = positions

    def getmu(self):
        return {self}


def _key(x):
    return x.id
class ConvolutedEMU(Object):
    def __init__(self, emus:list, name=None):
        emus = list(emus)
        emus.sort(key=_key)  # absolutely necessary in order to test equality between two ConvolutedEMU objects!
        Object.__init__(self, id=' ∗ '.join(emu.id for emu in emus), name=name)
        self._emus = emus

    @property
    def weight(self):
        return sum(emu.weight for emu in self._emus)

    def getmu(self):
        return {self} | set(self._emus)


class LabellingReaction(Reaction):
    """Base class for reactions in either cumomer or EMU simulation algorithms
    bi-directional refers to net-flux proceeding in both directions and
    # TODO reaction.forward_variable and reaction.reverse_variable do now work with _rev_reactions
    # TODO nee to fix when to set _is_built = False, we only want to update fcm and the like, not necessarily all matrices for EMU simulation
    Attributes
    ----------
    atom_map : collections.OrderedDict
        Extends the cobrapy Reaction.metabolites with atom transition information.
    """
    _RHO_MAX  = 0.999   # closest we will simulate to equilibrium! corresponds to -0.... kj/mol, this is necessary for numerical reasons, corresponds to a dGr ~ 0.025 kJ
    _RHO_MIN  = 0.001  # if rho_max < _RHO_MIN, we consider the reaction to be uni-directional and set self.rho_max = 0.0; corresponds to a dGr ~ -23.7 kJ
    _RHO_ATOL = 0.005  # the difference between _rho_min and _rho_max above which we add separate constraints for both
    _KILOJOULE = True  # whether Gibbs reaction energies are given in kiloJoule, if false we use Joules
    T = 310.15  # K, temperature
    _R = 8.31446261815324  # J K-1 mol-1, gas constant
    _TYPE_METABOLITE = LabelledMetabolite
    def __init__(
            self,
            idr=None,
            name: str = '',
            subsystem: str = '',
            lower_bound: float = 0.0,
            upper_bound: float = None,
            rho_min: float = 0.0,
            rho_max: float = 0.0,
            tau: float = 1.0,
            pseudo: bool = False,
    ):
        if type(idr) == Reaction:
            self.__dict__.update(idr.__dict__)
        elif type(idr) == LabellingReaction:
            raise NotImplementedError
        elif isinstance(idr, str) or (idr is None):
            Reaction.__init__(
                self, id=idr, name=name, subsystem=subsystem,
                lower_bound=lower_bound, upper_bound=upper_bound
            )

        self._pseudo = pseudo

        self._atom_map = {}  # {Met: (stoich, [tuple('atoms'),...] ) } required to be ordered for the
        self._rect_prod_map: np.array = None  # mapping all reactant atoms to (present) product atoms
        self._rho_min = 0.0  # minimal fraction of flux going in the reverse direction; has to do with Gibbs free energy change
        self._rho_max = 0.0
        self._tau  = 1.0  # tau \geq 1; constant is used to automatically scale rho_min to account for path dependency
        self._dgibbsr  = 0.0  # the currently set dGr

        # this default selection is important when initializing with a id_or_reaction=Reaction
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.tau = tau

        # rev_reac is a unidirectional reaction in the opposite direction of self
        self._rev_reaction = None
        self._initialize_rev_reaction()

    def _initialize_rev_reaction(self):
        if not self._pseudo:  # we dont need a rev_reaction for pseudo-reactions
            rid = self.id
            if rid is None:
                rid = ''
            self._rev_reaction: LabellingReaction = type(self)(
                idr=rid + '_rev', lower_bound=0.0, upper_bound=1.0, rho_max=0.0, pseudo=True
            )
            self._rev_reaction._rev_reaction = self  # the reverse of the reverse reaction is self

    def __setstate__(self, state):
        Reaction.__setstate__(self, state=state)
        for x in state['_atom_map']:
            x._model = self._model
            x._reaction.add(self)
        self._initialize_rev_reaction()

    def __getstate__(self):
        state = super(LabellingReaction, self).__getstate__()
        state['_rev_reaction'] = None
        return state

    def __imul__(self, coefficient): # TODO: __iadd__ and __sub__ are impossible since they wouldnt have atom mappings
        if float(abs(coefficient)) != 1.0:
            raise ValueError('cannot multiply by coefficients other than [-]1, '
                             'since this would not have an atom mapping')
        Reaction.__imul__(self, coefficient=coefficient) # TODO: test this; this used to be super!
        self.set_atom_map(
            atom_map=dict([(met, (-stoich, atoms)) for met, (stoich, atoms) in self._atom_map.items()])
        )
        if self.rho_max > 0.0:
            print('watch out, thermo')
        return self

    def __iadd__(self, other):
        raise ValueError('impossible, since we cannot add atom mappings')

    def build_reaction_string(self, show_atoms=True):
        """Generate a human readable reaction string"""

        reactant_bits = []
        product_bits = []

        def format(number):
            return "" if number == 1 else str(number).rstrip(".") + " "

        is_condensed = not set(
            [met.id for met in self._atom_map.keys()]
        ).issubset(
            set(
                [met.id for met in self.metabolites.keys()]
            )
        )

        metabolites = self.metabolites.copy()

        def process_map():
            for atom_met in self._atom_map:
                stoich, atoms = self._atom_map[atom_met]
                for atom in atoms:
                    if atom is None:
                        atom = '.'
                        stoich_str = format(abs(stoich))
                    else:
                        stoich_str = ''
                    bit = f'{stoich_str}{atom_met.id}/{"".join(atom)}'

                    if stoich > 0.0:
                        product_bits.append(bit)
                    else:
                        reactant_bits.append(bit)
                for met_met in self.metabolites:
                    if met_met.id == atom_met.id:
                        metabolites.pop(met_met)

        if self._atom_map and show_atoms and not is_condensed:
            process_map()

        for met, stoich in metabolites.items():
            bit = format(abs(stoich)) + met.id
            if stoich > 0.0:
                product_bits.append(bit)
            else:
                reactant_bits.append(bit)

        reactant_string = ' + '.join(reactant_bits)
        product_string = ' + '.join(product_bits)

        if is_condensed:
            reactant_bits = []
            product_bits = []
            process_map()
            reactant_string += ' (' + ' + '.join(reactant_bits) + ')'
            product_string += ' (' + ' + '.join(product_bits) + ')'

        if self.rho_max == 0.0:
            if self.lower_bound < 0 and self.upper_bound <= 0:
                arrow = '<--'
            else:
                arrow = '-->'
        elif (self.rho_max > 0.0):
            if (self.lower_bound < 0) and (self.upper_bound <= 0):
                arrow = '<=='
            elif (self.lower_bound >= 0) and (self.upper_bound > 0):
                arrow = '==>'
            else:
                arrow = '<=>'

        if self.bounds == (0.0, 0.0):
            arrow = '!=!'

        return f'{reactant_string} {arrow} {product_string}'

    def _change_constraint(self, id: str, ub=None, lb=None, variables: dict = None):
        constraint = self.model.constraints.get(id, None)

        if constraint is not None:
            if variables is None:
                self.model.remove_cons_vars(constraint)
                return
            else:
                # perhaps first check if get_linear_coefficients(variables.keys()) == variables ?
                constraint.set_linear_coefficients(variables)
            if (lb is not None) and (lb != constraint.lb):
                constraint.lb = lb
            if (ub is not None) and (ub != constraint.ub):
                constraint.ub = ub
        elif variables is not None:
            constraint = self.model.problem.Constraint(
                sum([variable * coef for variable, coef in variables.items()]),
                lb=lb, ub=ub, name=id
            )
            self.model.add_cons_vars(constraint)
        return constraint

    def update_variable_bounds(self):
        if self.model is None:
            return

        # this is necessary for when we update self.bounds, but do not explcitly change rho
        self._rho_min, self._rho_max = self._check_rho_bounds(rho_min=self._rho_min, rho_max=self._rho_max)

        if self._rho_max > 0.0: # this is for A ==> B and A <=> B reactions
            net_lb = self._lower_bound
            net_ub = self._upper_bound

            # this way we dont have to change bounds to respect directionality
            if self._dgibbsr < 0.0:
                net_lb = 0.0
            elif self._dgibbsr > 0.0:
                net_ub = 0.0

            if isinf(net_lb) or isinf(net_ub):
                raise ValueError('net upper bound needs to be finite')

            max_bound = max(abs(net_ub), abs(net_lb))
            max_bound = max_bound / (1.0 - self._rho_max)  # this means that rho_max has to be set to the correct value beforehand!

            self.forward_variable.set_bounds(lb=0.0, ub=max_bound)
            self.reverse_variable.set_bounds(lb=0.0, ub=max_bound)
            self._change_constraint(  # constraint on net flux
                id=f'{self.id}_net', lb=net_lb, ub=net_ub,
                variables={self.forward_variable: 1.0, self.reverse_variable: -1.0}
            )

            if self.reversibility and (self._dgibbsr == 0.0):
                # reversibility implies we do not know anything about the direction and
                #  thus neither about the magnitudes of bi-directional fluxes
                for name in ['', '_min', '_max']:
                    self._change_constraint(id=f'{self.id}_rho{name}')
            else:
                # for reactions that are not reversible, but that do carry bi-directional fluxes
                forward, reverse = (self.forward_variable, self.reverse_variable) if \
                    net_ub > 0.0 else (self.reverse_variable, self.forward_variable)

                if np.isclose(self._rho_max, self._rho_min, atol=self._RHO_ATOL):
                    # TODO this means that the upper bound
                    ub = 0.0
                    constraint_id = f'{self.id}_rho'
                    for name in ['_min', '_max']:
                        self._change_constraint(id=f'{self.id}_rho{name}')
                else:
                    # only add a second bound if the two bounds are not extremely close
                    constraint_id = f'{self.id}_rho_min'
                    ub = None
                    self._change_constraint(id=f'{self.id}_rho')
                    self._change_constraint(
                        id=f'{self.id}_rho_max', ub=0.0, variables={reverse: 1.0, forward: -self._rho_max}
                    )
                if self._rho_min > 0.0:
                    # this makes sure that we do not add a useless constraint
                    self._change_constraint(
                        id=constraint_id, lb=0.0, variables={reverse: 1.0, forward: -self._rho_min}
                    )

        elif self._rho_max == 0.0: # this is for A --> B reactions
            if self.reversibility:
                # happens if we change a lower bound without first changing rho_max explicitly as happens in pta
                self.rho_max = self._RHO_MAX
            else:
                for name in ['', '_net', '_min', '_max']:
                    self._change_constraint(id=f'{self.id}_rho{name}')
                if (self.model is not None) and not self._pseudo:
                    Reaction.update_variable_bounds(self)  # standard cobrapy behavior

        if self._model is not None:
            self._model._is_built = False  # needs to rerun to update model._fcm and the like
            # this is to make sure that all changes in rho_max and bounds are reflected in labellingreactions
            self._model._labelling_reactions = DictList()

    @property
    def pseudo(self):
        return self._pseudo

    @pseudo.setter
    def pseudo(self, val: bool):
        # this is necessary for setattr() when building the model!
        if val == True:
            self.rho_max = 0.0
            self.bounds = (0.0, 0.0)
            self._pseudo = val
        elif val == False:
            self._pseudo = val
        else:
            raise ValueError

    @property
    def atom_map(self):
        return self._atom_map.copy()

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, val:float):
        if not val >= 1.0:
            raise ValueError(f'tau: {val}!')
        self._tau = val
        if self._dgibbsr != 0.0:
            pass
            # self.set_dGr(dGr=self._dGr, update_bounds=True) # TODO update this

    def _check_rho_bounds(self, rho_min=None, rho_max=None):
        if rho_min is None:
            rho_min = self._rho_min
        if rho_max is None:
            rho_max = self._rho_max

        if self.reversibility and (self._dgibbsr == 0.0):
            # if reversible, and not in thermodynamic context, return extremes
            rho_min, rho_max = 0.0, self._RHO_MAX
        elif self.bounds == (0.0, 0.0):
            rho_min, rho_max = 0.0, 0.0
        else:
            if rho_max > self._RHO_MAX:
                rho_max = self._RHO_MAX
            elif rho_max < self._RHO_MIN:
                # this means we consider the reaction uni-directional!
                rho_min, rho_max = 0.0, 0.0
            if rho_min > self._RHO_MAX:
                rho_min = self._RHO_MAX
        self._check_bounds(rho_min, rho_max)
        return rho_min, rho_max

    @property
    def rho_max(self): return self._rho_max

    @rho_max.setter
    def rho_max(self, val:float):
        rho_min, rho_max = self._check_rho_bounds(rho_max=val)
        if (self.model is not None):
            if (self._rho_max == 0.0) and (rho_max > 0.0):
                self._model._is_built = False  # extra reaction as free reaction
                self._model._labelling_reactions = DictList()
        self._rho_max = rho_max
        self.update_variable_bounds()

    @property
    def rho_min(self): return self._rho_min

    @rho_min.setter
    def rho_min(self, val:float):
        rho_min, rho_max = self._check_rho_bounds(rho_min=val)
        self._rho_min = rho_min
        self.update_variable_bounds()

    @property
    def dgibbsr(self):
        if self._KILOJOULE:
            return self._dgibbsr / 1000.0
        return self._dgibbsr

    @dgibbsr.setter
    def dgibbsr(self, val):
        self.set_dgibbsr(dgibbsr=val, update_constraints=True)

    def set_dgibbsr(self, dgibbsr: float, update_constraints=False, thermo_consistency_check=True):
        dgibbsr = dgibbsr # we have to copy value when applying to pandas dataframe
        if self._KILOJOULE:
            dgibbsr *= 1e3
        # dGr = np.clip(dGr, -709.78, 709.78)  # TODO prevent overflow perhaps?

        if thermo_consistency_check and \
                (((dgibbsr > 0.0) and (self._lower_bound >= 0.0)) or
                 ((dgibbsr < 0.0) and (self._upper_bound <= 0.0))):
            raise ValueError('bounds and thermodynamics do not match; TFS should not have sampled this orthant!')

        self._dgibbsr = dgibbsr
        if dgibbsr == 0.0:  # this resets stuff to defaults!
            rho_min, rho_max = 0.0, self._RHO_MAX
        else:
            rho_min = np.exp(dgibbsr / (self._R * self.T))
            if self._tau > 1.0:
                rho_max = np.exp((dgibbsr / self._tau) / (self._R * self.T))
            else:
                rho_max = rho_min
            if dgibbsr > 0.0:
                # chose to keep the min(v_fwd, v_rev) frame of reference!
                rho_min, rho_max = 1.0 / rho_min, 1.0 / rho_max
            rho_min, rho_max = self._check_rho_bounds(rho_min=rho_min, rho_max=rho_max)

        if update_constraints:
            self._rho_min = rho_min
            self.rho_max  = rho_max  # here update_bounds is called!
        return rho_min, rho_max

    def gettants(self, reactant=True):
        op = operator.lt if reactant else operator.gt
        return [met for met, (stoich, atoms) in self._atom_map.items() for atom in atoms if op(stoich, 0.0)]

    def set_atom_map(self, atom_map: dict):
        """
        The concatenated reactant labels map onto the concatenated product labels as follows
        [abcdef] -> [abdcef], thus: self._rect_prod_map = [0,1,3,2,4,5]

        Parameters
        ----------
        atom_map : OrderedDict
            with atom mapping of susbtrate molecules to product molecules e.g. A/abc + A/def -> B/abd + C/cef
            OrderedDict({Met(A) : (-2,[('a','b','c'),('d','e','f')]),  Met(B) : (1,[('a','b','d')]), Met(C) : (1,[('c','e','f')])})
        """
        if not atom_map:
            return
        if not self.metabolites:
            if self.pseudo:
                # add_metabolites adds constraints to model
                self._metabolites = {met: stoich for met, (stoich, atoms) in atom_map.items()}
            else:
                self.add_metabolites(
                    metabolites_to_add={met: stoich for met, (stoich, atoms) in atom_map.items()}, combine=False,
                )

        map = {}
        for metabolite, (stoich, atoms) in atom_map.items():
            if not isinstance(metabolite, LabelledMetabolite):
                raise ValueError(f'{self.id} atom_map contains non-LabelledMetabolite object: {metabolite.id}')

            if self.model is not None:
                if metabolite in self.model.metabolites:
                    model_met = self.model.metabolites.get_by_id(metabolite.id)
                elif metabolite in self.model.pseudo_metabolites:
                    model_met = self.model.pseudo_metabolites.get_by_id(metabolite.id)
                else:
                    model_met = None
                if (metabolite is not model_met) or (metabolite._model is not self._model):
                    raise ValueError(f'first use model.fix_metabolite_reference_mess(...) {metabolite.id}!')

            for met_met, stoich_met in list(self._metabolites.items()):
                if metabolite.id == met_met.id:
                    if (stoich != stoich_met) or ((atoms[0] is not None) and (abs(stoich) != len(atoms))):
                        raise ValueError(
                            f'{self.id}: for {metabolite.id} stoichiometry and atom mapping are inconsistent'
                        )
                    if metabolite is not met_met:
                        self._metabolites[metabolite] = self._metabolites.pop(met_met)

            # cleaning up the reference mess
            metabolite._reaction.add(self)
            for reaction in list(metabolite._reaction):
                if (reaction.id == self.id) and (reaction is not self):
                    metabolite._reaction.remove(reaction)

            for atom in atoms:
                if (atom is not None) and (metabolite.elements['C'] != len(atom)):
                    raise ValueError(f'{self.id}: for {metabolite.id} different number of carbons in '
                          f'formula: {metabolite.formula}, than atoms in atom mapping: C{len(atom)}')
            map[metabolite] = (stoich, np.array(atoms))

        self._atom_map = map

        if all(stoich <= 0.0 for metabolite, (stoich, atoms) in self._atom_map.items()) and not self.boundary:
            # catches biomass
            return

        if not self.boundary:
            cumul_rect_atoms = np.concatenate([
                atom for met, (stoich, atoms) in self._atom_map.items() for atom in atoms if stoich < 0.0
            ])
            cumul_prod_atoms = np.concatenate([
                atom for met, (stoich, atoms) in self._atom_map.items() for atom in atoms if stoich > 0.0
            ])
            if not (np.unique(cumul_rect_atoms).shape[0] == cumul_rect_atoms.shape[0]) \
                   and (np.unique(cumul_prod_atoms).shape[0] == cumul_prod_atoms.shape[0]):
                raise ValueError(f'non-unique atom mapping {self.id}')
            if not np.setdiff1d(cumul_prod_atoms, cumul_rect_atoms).size == 0:
                raise ValueError(f'product atoms do not occur in substrate {self.id}')
            if not self._pseudo:
                if cumul_prod_atoms.shape[0] != cumul_rect_atoms.shape[0]:
                    raise ValueError(f'cannot have a reverse reaction for an unbalanced forward reaction {self.id}')
            self._rect_prod_map = np.where(cumul_prod_atoms[:, None] == cumul_rect_atoms[None, :])[1]

        if not self._pseudo:
            self._rev_reaction.set_atom_map(atom_map=dict([
                (met, (-stoich, atoms)) for met, (stoich, atoms) in self._atom_map.items()
            ]))

    def build_atom_map_from_string(self, atom_map_str: str, metabolite_kwargs: dict = None):
        # TODO: make it possible to build entire reaction from string with co-factors and all
        if metabolite_kwargs is None:
            metabolite_kwargs = {}

        rects, arrow, prods = _read_atom_map_str_rex.findall(string=atom_map_str)[0]
        is_biomass = _find_biomass_rex.search(rects) is not None

        if ((arrow == '<=>') and not self.reversibility) or \
                (('==' in arrow) and self.reversibility) or \
                (('--' in arrow) and (self.reversibility or self.rho_max != 0.0)):
            print(f'wrong arrow or bounds {self.id}')

        if ('=' in arrow) and (self.rho_max == 0.0):
            # for when we dont pass rho_max as an explicit argument
            self.rho_max = self._RHO_MAX

        atom_map = {}

        if is_biomass:
            # when setting biomass, it is important that all metabolites are already in LabelledMetabolite form!
            if (self.rho_max != 0.0) or (self.lower_bound < 0.0):
                raise ValueError('biomass has wrong bounds')
            elif not self.metabolites:
                raise ValueError('First add_metabolites to biomass reaction before building atom_map!')
            for metabolite, stoich in self.metabolites.items():
                if isinstance(metabolite, LabelledMetabolite):
                    if (not stoich < 0.0) and (metabolite.elements.get('C', False)):
                        raise ValueError('biomass is producing LabelledMetabolites!')
                    atom_map[metabolite] = (stoich, [None])
        else:
            intermediates = (rects + '+' + prods).split('+')
            rects = [rect.split('/')[0].strip() for rect in rects.split('+')]
            prods = [prod.split('/')[0].strip() for prod in prods.split('+')]

            met_mets = DictList(self.metabolites)
            created_mets = DictList()  # necessary for when no model is associated with this reaction
            for intermediate in intermediates:
                intermediate = intermediate.strip()

                if intermediate == '∅':  # deals with boundary reactions of the pysumo model
                    continue

                met_id, atoms = intermediate.split('/')
                atoms_arr = tuple(atoms)

                comparmented_kwargs = metabolite_kwargs.get(met_id, {})
                compartment = _strip_bigg_rex.search(met_id)
                if compartment is not None:
                    compartment = compartment.group()[1:]
                compartment = comparmented_kwargs.get('compartment', compartment)

                no_compartment = _strip_bigg_rex.sub('', met_id)
                compartmentless_kwargs = metabolite_kwargs.get(no_compartment, {})
                formula = compartmentless_kwargs.get('formula', 'C' + str(len(atoms_arr)))

                all_kwargs = {**compartmentless_kwargs, **comparmented_kwargs}

                if met_id in created_mets:
                    metabolite = created_mets.get_by_id(id=met_id)
                elif (self.model is not None) and (met_id in self.model.metabolites):
                    metabolite = self.model.metabolites.get_by_id(id=met_id)
                elif (self.model is not None) and (met_id in self.model._pseudo_metabolites):
                    metabolite = self.model._pseudo_metabolites.get_by_id(id=met_id)
                elif met_id in met_mets:
                    metabolite = met_mets.get_by_id(id=met_id)
                else:
                    metabolite = self._TYPE_METABOLITE(idm=met_id, formula=formula, compartment=compartment)
                    created_mets.append(metabolite)

                if not isinstance(metabolite, LabelledMetabolite):
                    # this means that met._reaction is also copied, thus not breaking references
                    metabolite = self._TYPE_METABOLITE(idm=metabolite)
                    created_mets.append(metabolite)

                for kwarg, value in all_kwargs.items():
                    if not hasattr(metabolite, kwarg):
                        raise ValueError(f'Faulty metabolite kwargs {met_id}: {kwarg}')
                    setattr(metabolite, kwarg, value)

                if metabolite not in atom_map:
                    atom_map[metabolite] = (0, [])

                stoich, atoms = atom_map[metabolite]

                if met_id in rects:
                    stoich -= 1
                elif met_id in prods:
                    stoich += 1
                atom_map[metabolite] = (stoich, atoms)
                atoms.append(atoms_arr)
        return atom_map

    def add_metabolites(self, metabolites_to_add, combine=True, reversibly=True):
        Reaction.add_metabolites(self, metabolites_to_add, combine=combine, reversibly=reversibly)
        if not self._pseudo:
            self._rev_reaction.add_metabolites(
                metabolites_to_add={m: -s for m, s in metabolites_to_add.items()},
                combine=combine, reversibly=reversibly
            )

    def subtract_metabolites(self, metabolites: dict, combine: bool = True, reversibly: bool = True):
        # NB we need this for pta.tfs I believe, since remove_reactions is called a bunch of times
        Reaction.subtract_metabolites(self, metabolites=metabolites, combine=combine, reversibly=reversibly)
        self._atom_map = {}
        if not self._pseudo:
            self._rev_reaction._atom_map = {}
            self._rev_reaction.subtract_metabolites(
                metabolites={m: -s for m, s in metabolites.items()}, combine=combine, reversibly=reversibly
            )

    def copy(self):
        model = self._model
        self._model = None
        for i in self._metabolites:
            i._model = None
        for i in self._atom_map:
            i._model = None
        for i in self._genes:
            i._model = None
        # now we can copy
        # TODO: test whether this does what I want! I think it does
        #   looks like the new metabolites and emus all point to the same object
        new_reaction = deepcopy(self)
        # restore the references
        self._model = model
        for i in self._metabolites:
            i._model = model
        for i in self._atom_map:
            i._model = model
        for i in self._genes:
            i._model = model
        return new_reaction

    @abstractmethod
    def build_tensors(self): raise NotImplementedError
    @abstractmethod
    def pretty_tensors(self, weight: int): raise NotImplementedError
    @abstractmethod
    def map_reactants_products(self, **kwargs): raise NotImplementedError


class LabellingModel(Model):
    """Model that allows for sampling of the flux-space defined by the null-space
    of the stoichiometric matrix defined in the model, upper/lower bounds on reactions
    and bounds on a flux-ratio of interest.

    Attributes
    ----------


    Parameters
    ----------


    Notes
    -----
    None

    References
    ----------
        [1] Wolfgang Wiechert,  Michael Möllney,  Nichole Isermann, Michael Wurzel, Albert A. de Graaf
        Bidirectional reaction steps in metabolic networks: III.
        Explicit solution and analysis of isotopomer labeling systems
        Biotech. and Bioeng.  (2000)
        https://doi.org/10.1002/(SICI)1097-0290(1999)66:2<69::AID-BIT1>3.0.CO;2-6

        [2] Maria Kogadeeva, Nicola Zamboni
        SUMOFLUX: A Generalized Method for Targeted 13C Metabolic Flux Ratio Analysis
        PloS Comp. Biol. (2016)
        <https://doi.org/10.1371/journal.pcbi.1005109>
    """
    _TYPE_REACTION = LabellingReaction
    def __init__(
            self,
            linalg: LinAlg,
            id_or_model = None,
            name: str = None,
    ):
        if isinstance(id_or_model, LabellingModel):
            raise NotImplementedError
        super(LabellingModel, self).__init__(id_or_model, name)
        self._la = linalg

        # flags
        self._is_built = False  # signals that the all the variables and matrices have not been built yet

        # flux variables
        self._fluxes = None
        self._fcm: FluxCoordinateMapper = None  # map fluxes in some coordinate system and get back fwd_rev fluxes
        self._only_rev = {}  # ids of always net reverse reactions for easy mapping of flux coordonates

        # tolerances
        self.tolerance = 1e-9  # needed to have decent flux sampling results; default tol=1e-6

        # input labelling variables
        self._input_labelling = {}
        self._labelling_id: str = None
        self._labelling_repo: dict = {}  # repository of all labellings that we encountered

        # collections of metabolites
        self._measurements = DictList()  # these are the metabolites/ EMUs that we simulate labelling for since they are measured
        self._pseudo_metabolites = DictList()  # all the products of pseudo reactions e.g. all amino acids

        # collections of reactions of various sorts
        self._biomass_id: str = None
        self.pseudo_reactions     = DictList()  # used to simulate the labelling of products of linear pathways e.g. amino acids
        self._labelling_reactions = DictList()  # reactions for which all reactants and products are present and carry carbon
        self._free_reaction_id = []

        self._initialize_state()  # sets even more attributes; function is reused when building the model

        self.groups = DictList() # TODO: no functionality has been implemented or tested for groups

    def __setstate__(self, state):
        super(LabellingModel, self).__setstate__(state)
        for r in self.reactions:
            # NB for some reason, all bounds get scrambled during pickling
            #   this is an optlang issue that I do not know how to resolve
            #   many reactions end up with net-constraints where ub == lb == -1000.0
            r.update_variable_bounds()
            if isinstance(r, LabellingReaction):
                fixed_map = self._fix_metabolite_reference_mess(r, r._atom_map)
                r.set_atom_map(atom_map=fixed_map)

        pseudo_reactions = state.get('pseudo_reactions')
        if pseudo_reactions is not None:
            for r in pseudo_reactions:
                r._model = self
        input_labelling = state.get('_input_labelling')
        if input_labelling is not None:
            self.set_input_labelling(input_labelling=input_labelling)
        measurements = state.get('_measurements')
        if measurements is not None:
            self._measurements = DictList()
            self.set_measurements(measurement_list=measurements)
        self._labelling_reactions = DictList()  # gets set in metabolites_in_state; which calls labelling_fluxes_id
        linalg = state.get('_la')
        if linalg is not None:
            self._initialize_state()

    def __getstate__(self):
        odict = super(LabellingModel, self).__getstate__()

        odict['_only_rev'] = {}
        odict['_fcm'] = None
        odict['_is_built'] = False

        odict['_s'] = None
        odict['_sum'] = None
        odict['_dsdv'] = None
        odict['_jacobian'] = None

        # the attributes below are stored in a format where __setstate__ can set them
        odict['_input_labelling'] = self.input_labelling
        odict['_labelling_repo'] = {}  # TODO, I think storing this would be too much ugly code
        odict['_measurements'] = self._measurements.list_attr('id')
        odict['_metabolites_in_state'] = None

        odict['_pseudo_metabolites'] = DictList()
        odict['_labelling_reactions'] = DictList()
        return odict

    def _initialize_state(self):
        # state and jacobian variables
        self._s = self._la.get_tensor(shape=(0,))  # state vector
        self._sum = self._la.get_tensor(shape=(0,))  # sums metabolites to 1
        self._dsdv = self._la.get_tensor(shape=(0,))  # ds / dvi, vector that stores sensitivity of state wrt some reaction
        self._jacobian = self._la.get_tensor(shape=(0,))  # dim(reaction x output variabless)

    @property
    def is_built(self):
        return self._is_built

    @property
    def biomass_id(self):
        if self._biomass_id is None:
            return ''
        return self._biomass_id[:]

    @property
    def labelling_id(self):
        if self._labelling_id is None:
            return ''
        return self._labelling_id[:]

    @property
    def labelling_fluxes_id(self) -> pd.Index:
        return pd.Index(self.labelling_reactions.list_attr('id'), name='labelling_fluxes_id')

    @property
    def state_id(self) -> pd.Index:
        # this assumes that we return MDVs; therefore cumomers reimplement this
        return pd.Index([
            '+'.join(tup)
                for met in self.measurements
                    for tup in zip(repeat(met.id), map(str, range(met.weight + 1)))
        ], name='mdv_id')

    @property
    def state(self):
        if not self._is_built:
            raise ValueError('MUST BUILD')
        state = np.atleast_2d(self._la.tonp(self._format_return(s=self._s)))
        return pd.DataFrame(state, index=self._fcm.samples_id, columns=self.state_id).round(decimals=3)

    @property
    def jacobian(self):
        if not self._is_built:
            raise ValueError('MUST BUILD')
        jac = self._la.tonp(self._jacobian)
        framed_jacs = [pd.DataFrame(sub_jac, index=self.labelling_fluxes_id, columns=self.state_id) for sub_jac in jac]
        return pd.concat(framed_jacs, keys=self._fcm._samples_id)

    @property
    def input_labelling(self):
        """entity can be IsoCumo or EMU"""
        return pd.Series(dict((isocumo.id, frac) for isocumo, frac in self._input_labelling.items()),
                         name=self._labelling_id, dtype=np.float64).round(4)

    @property
    def input_metabolites(self):
        return DictList(set([entity.metabolite for entity in self._input_labelling.keys()]))

    @property
    def measurements(self):
        if not self._measurements:
            self._measurements = self.metabolites_in_state + self.pseudo_metabolites  # basically errthangg
        return self._measurements

    @property
    def labelling_reactions(self):
        if self._labelling_reactions:
            return self._labelling_reactions

        self._only_rev = {}  # irreversible reactions whose net flux is always negative
        for reaction in self.reactions:
            lb, ub = reaction.bounds
            # if isinstance(reaction, LabellingReaction) and (not reaction.pseudo) and ((lb, ub) != (0.0, 0.0)):
            if isinstance(reaction, LabellingReaction) and ((lb, ub) != (0.0, 0.0)):
                if reaction.rho_max > 0.0:
                    self._labelling_reactions.append(reaction)
                    self._labelling_reactions.append(reaction._rev_reaction)
                elif lb >= 0.0:
                    self._labelling_reactions.append(reaction)
                elif ub <= 0.0:
                    self._labelling_reactions.append(reaction._rev_reaction)
                    self._only_rev[reaction._rev_reaction.id] = reaction.id

        self._jacobian = self._la.get_tensor(
            shape=(self._la._batch_size, len(self._labelling_reactions), self.state_id.shape[0])
        )
        return self._labelling_reactions

    @property
    def flux_coordinate_mapper(self) -> FluxCoordinateMapper:
        if not self._is_built:
            raise ValueError('build the model first!')
        return self._fcm

    def set_fluxes(self, labelling_fluxes: Union[pd.DataFrame, np.array], samples_id=None, trim=True):
        if not self._is_built:
            raise ValueError('MUST BUILD')
        labelling_fluxes = self._fcm.frame_fluxes(labelling_fluxes, samples_id, trim)
        if len(labelling_fluxes.shape) > 2:
            raise ValueError('can only deal with 2D stratified fluxes!')
        if self._la._auto_diff:
            labelling_fluxes.requires_grad_(True)
        if labelling_fluxes.shape[0] != self._la._batch_size:
            raise ValueError(f'batch_size = {self._la._batch_size}; fluxes.shape[0] = {labelling_fluxes.shape[0]}')
        self._fluxes = labelling_fluxes

    def set_input_labelling(self, input_labelling: pd.Series):
        self._input_labelling = {}
        self._labelling_id = input_labelling.name
        for isotopomer_str, frac in input_labelling.items():
            if frac == 0.0:
                continue
            met_id, label = isotopomer_str.rsplit('/')
            metabolite = self.metabolites.get_by_id(id=met_id)
            if hasattr(metabolite, 'isotopomers'):
                isotopomer = metabolite.isotopomers.get_by_id(isotopomer_str)
            else:
                isotopomer = IsoCumo(metabolite=self.metabolites.get_by_id(id=met_id), label=label)
            self._input_labelling[isotopomer] = frac

        fractions = np.fromiter(self._input_labelling.values(), dtype=np.double)
        if any(fractions < 0.0) or any(fractions > 1.0):
            raise ValueError('Negative or over 1 value in input labelling')

        isotopomers = np.array(list(self._input_labelling.keys()))
        input_metabolites = np.array([ic.metabolite for ic in isotopomers])
        for metabolite in set(input_metabolites):
            sum_met = fractions[input_metabolites == metabolite].sum()
            if not math.isclose(a=sum_met, b=1.0, abs_tol=1e-4):
                raise ValueError(f'Input labeling fractions of metabolite {metabolite.id} do not sum up to 1.0')
            fractions[input_metabolites == metabolite] /= sum_met  # makes sum closer to 1
        self._input_labelling = dict((key, frac) for key, frac in zip(isotopomers, fractions))

        input_reactions = DictList()
        for metabolite in set(self.input_metabolites):
            for reaction in metabolite.reactions:
                if reaction.boundary and isinstance(reaction, LabellingReaction) and not reaction.pseudo:
                    if not reaction.rho_max == 0.0:
                        raise ValueError(f'input reaction is illegaly reversible {reaction.id}')
                    if reaction.lower_bound >= 0.0:
                        input_reactions.append(reaction)
                    elif reaction.upper_bound <= 0.0:
                        input_reactions.append(reaction._rev_reaction)
                    else:
                        raise ValueError(f'input reaction {reaction.id} '
                                         f'for metabolite {metabolite.id} has (0, 0) bounds')
            if not any([reaction in input_reactions for reaction in metabolite.reactions]):
                raise ValueError(f'metabolite {metabolite.id} has no input reactions')

        self._labelling_repo[input_labelling.name] = dict(_input_labelling=self._input_labelling)

    def _parse_measurement(self, all_metabolites:DictList, measurement_id:str):
        if measurement_id in all_metabolites:
            return all_metabolites.get_by_id(id=measurement_id)
        return None

    def set_measurements(self, measurement_list, verbose=False):
        all_metabolites = self.metabolites_in_state + self.pseudo_metabolites
        self._measurements = DictList()
        unsimulable = []
        for meas_id in measurement_list:
            if hasattr(meas_id, 'id'):
                meas_id = meas_id.id
            measurement = self._parse_measurement(all_metabolites=all_metabolites, measurement_id=meas_id)
            if measurement is None:
                unsimulable.append(meas_id)
            else:
                self._measurements.append(measurement)
        if verbose and unsimulable:
            string = ', '.join(unsimulable)
            print(f'Cannot simulate {string}')

    def _set_free_reactions(self, free_reaction_id: Iterable = None):
        if free_reaction_id is None:
            free_reaction_id = []
        if len(free_reaction_id) == 0:
            free_reaction_id = self._free_reaction_id

        free_reaction_id = list(free_reaction_id)

        # this is because we typically have measurements for input/bm/boundary reactions!
        bm = DictList()
        if (self._biomass_id is not None) and (self._biomass_id not in free_reaction_id):
            bm.append(self.labelling_reactions.get_by_id(self._biomass_id))

        user_chosen = DictList()
        zero_facet = DictList()
        boundary = DictList()
        fwd = DictList()
        rev = DictList()
        for reaction in self.labelling_reactions - bm:  # self.labelling_reactions is triggered here
            revr = reaction._rev_reaction
            if reaction.pseudo and (reaction.id not in self._only_rev):
                rev.append(reaction)
            elif (abs(reaction.upper_bound - reaction.lower_bound) < self._tolerance) or \
                    (reaction.id in self._only_rev and (abs(revr.upper_bound - revr.lower_bound) < self._tolerance)):
                zero_facet.append(reaction)
            elif (reaction.id in free_reaction_id) or (self._only_rev.get(reaction.id) in free_reaction_id):
                user_chosen.append(reaction)
            elif reaction.boundary:
                # TODO make input reactions work!
                boundary.append(reaction)
            else:
                fwd.append(reaction)
        user_chosen.sort(key=lambda x: \
            free_reaction_id.index(_rev_reactions_rex.sub('', x.id)) if x.id not in free_reaction_id else x.id
                         )
        self._free_reaction_id = user_chosen.list_attr('id')
        self._labelling_reactions = fwd + boundary + bm + user_chosen + zero_facet + rev

    def _fix_metabolite_reference_mess(self, reaction, atom_map):
        if not isinstance(reaction, LabellingReaction):
            raise ValueError('only meant for LabellingReaction')

        fixed_atom_map = {}
        for metabolite, (stoich, atoms) in atom_map.items():
            if not isinstance(metabolite, LabelledMetabolite):
                raise ValueError('atom_map should only contain LabelledMetabolite')

            if metabolite in self.metabolites:
                model_metabolite = self.metabolites.get_by_id(metabolite.id)
                if type(model_metabolite) == Metabolite:
                    # creates a lot of difficulties; I dont know how else to fix this...
                    # takes over full __dict__ of model_metabolite with annotation and correct formula and such
                    metabolite = self._TYPE_REACTION._TYPE_METABOLITE(
                        idm=model_metabolite, symmetric=metabolite.symmetric, formula=metabolite.formula
                    )
                elif isinstance(model_metabolite, LabelledMetabolite):
                    metabolite = model_metabolite
                else:
                    raise NotImplementedError

                if reaction.pseudo and (stoich > 0):
                    self.metabolites.remove(metabolite.id)
                    self.remove_cons_vars([self.solver.constraints[metabolite.id]])
                    self._pseudo_metabolites.append(metabolite)    # throws error if already present!
                    # raise ValueError(f'{metabolite.id} is pseudo and has more than one reaction producing')
                elif model_metabolite is not metabolite:
                    # happens if we created a LabelledMetablolite above!
                    self.metabolites._replace_on_id(new_object=metabolite)
            elif metabolite in self._pseudo_metabolites:
                metabolite = self._pseudo_metabolites.get_by_id(metabolite.id)
            elif reaction.pseudo and (stoich > 0):
                self._pseudo_metabolites.append(metabolite)  # throws error if already present!
                metabolite._model = self
            else:
                self.add_metabolites(metabolite_list=[metabolite])

            fixed_atom_map[metabolite] = (stoich, atoms)

            n_pseudo = 0
            is_pseudo = metabolite in self._pseudo_metabolites
            for met_reaction in list(metabolite._reaction):
                for met_met, met_stoich in list(met_reaction._metabolites.items()):
                    if met_stoich > 0 and is_pseudo:
                        n_pseudo += 1
                        if n_pseudo > 1:
                            raise ValueError('multiple pseudo-reactions producing a single pseudo_metabolite!')
                    if (met_met.id == metabolite.id) and (met_met is not metabolite):
                        # harmonize objects in atom_map and metabolites
                        met_reaction._metabolites[metabolite] = met_reaction._metabolites.pop(met_met)
        return fixed_atom_map

    def add_reaction_labelling(  # refactor this
            self,
            reaction_list: Iterable = None,
            metabolite_kwargs: dict = None,
            reaction_kwargs: dict = None
    ):
        context = get_context(self)
        if context:
            raise NotImplementedError

        reaction_kwargs = {} if reaction_kwargs is None else reaction_kwargs
        # maybe make sure that the reactions in reaction_list are not in self.reactions...
        reaction_list = DictList() if reaction_list is None else DictList(reaction_list)
        reac_kwargs = dict(zip(reaction_list.list_attr('id'), repeat({})))
        reac_kwargs.update(reaction_kwargs)

        # these properties will be recalculated accordingly when they are called!
        self._labelling_reactions = DictList()

        for reac_id, kwargs in reac_kwargs.items():
            if reac_id in self.reactions:
                reaction = self.reactions.get_by_id(id=reac_id)
                self.reactions.remove(reac_id)
            elif reac_id in reaction_list:
                reaction = reaction_list.get_by_id(reac_id)
                reaction_list.remove(reac_id)
            elif reac_id in self.pseudo_reactions:
                reaction = self.pseudo_reactions.get_by_id(id=reac_id)
            else:
                reaction = Reaction(id=reac_id, lower_bound=0.0, upper_bound=0.0)

            # this is to make sure that upper_bound is set before lower_bound
            # TODO: also set arbitrary kwargs (not in list below)!
            for kwarg in ['name', 'bounds', 'upper_bound', 'lower_bound', 'subsystem', 'gene_reaction_rule']:
                val = kwargs.get(kwarg)
                if (kwarg == 'upper_bound') and (val is not None) and (val < reaction.lower_bound):
                    lval = kwargs.get('lower_bound')
                    if lval is not None:
                        reaction.lower_bound = lval
                if val is not None:
                    setattr(reaction, kwarg, val)

            if (type(reaction) == Reaction) and ('atom_map_str' in kwargs):
                reaction = self._TYPE_REACTION(idr=reaction)
                for metabolite in reaction._metabolites:
                    for met_reaction in metabolite._reaction:
                        if (met_reaction.id == reaction.id) and (met_reaction is not reaction):
                            metabolite._reaction.remove(met_reaction)
                            metabolite._reaction.add(reaction)

            if isinstance(reaction, LabellingReaction):
                for kwarg in ['tau', 'dgibbsr', 'rho_max', 'rho_min', 'pseudo', '_sigma_dgibbsr',]:
                    val = kwargs.get(kwarg)
                    if val is not None:
                        setattr(reaction, kwarg, val)
                if reaction.pseudo and (reaction not in self.pseudo_reactions):
                    reaction._model = self
                    if reaction in self.reactions:
                        raise NotImplementedError
                    self.pseudo_reactions.append(reaction)
                elif reaction not in reaction_list:
                    reaction_list.append(reaction)
            elif isinstance(reaction, Reaction):
                reaction_list.append(reaction)
        Model.add_reactions(self, reaction_list=reaction_list)

        for reac_id, kwargs in reac_kwargs.items():
            atom_map_str = kwargs.get('atom_map_str')
            if atom_map_str is None:
                continue
            reactants = _read_atom_map_str_rex.findall(string=atom_map_str)[0][0]
            is_biomass = _find_biomass_rex.search(reactants) is not None
            if is_biomass:
                if (self._biomass_id is not None) and (self._biomass_id != reac_id):
                    raise ValueError(f'watch out, more than one biomass reaction in reac_kwargs! '
                                     f'self._biomass_id = {self._biomass_id}, reac_id = {reac_id}')
                self._biomass_id = reac_id
                continue
            if reac_id in self.pseudo_reactions:
                reaction = self.pseudo_reactions.get_by_id(reac_id)
            else:
                reaction = self.reactions.get_by_id(id=reac_id)
            atom_map = reaction.build_atom_map_from_string(atom_map_str=atom_map_str, metabolite_kwargs=metabolite_kwargs)
            fixed_atom_map = self._fix_metabolite_reference_mess(reaction=reaction, atom_map=atom_map)
            reaction.set_atom_map(atom_map=fixed_atom_map)

        if self._biomass_id is not None:
            reaction = self.reactions.get_by_id(self._biomass_id)
            atom_map = reaction.build_atom_map_from_string(atom_map_str='biomass --> ∅', metabolite_kwargs=metabolite_kwargs)
            # TODO where did fixed_atom_map go?
            fixed_atom_map = self._fix_metabolite_reference_mess(reaction=reaction, atom_map=atom_map)
            reaction.set_atom_map(atom_map=fixed_atom_map)
        self._is_built = False

    def make_sbml_writable(self):
        # we need to do this since there are a bunch of things that writing to sbml does not like if I remember correctly
        # TODO: maybe include this in __setstate__ and __getstate__?
        # TODO: deal with pseudo_reactions
        raise NotImplementedError
        # new = Model(id_or_model=self.id, name=self.name)
        # new.notes = deepcopy(self.notes)
        # new.annotation = deepcopy(self.annotation)
        # new.add_reactions(reaction_list=self.reactions + self.pseudo_reaction)
        # return new

    def remove_reactions(self, reactions: list, remove_orphans=False):
        for reaction in reactions:
            if (hasattr(reaction, 'id') and (reaction.id == self._biomass_id)) or (reaction == self._biomass_id):
                self._biomass_id = None
        Model.remove_reactions(self, reactions=reactions, remove_orphans=remove_orphans)
        if remove_orphans:
            # necessary because a metabolite migh still be associated with only rev_reactions
            to_remove = []
            for met in self.metabolites:
                if not any(reac in self.reactions for reac in met._reaction):
                    to_remove.append(met)
            self.remove_metabolites(metabolite_list=to_remove)
        # since these are set by the properties, we can just reset it like this
        self._labelling_reactions = DictList()
        self._is_built = False

    def remove_metabolites(self, metabolite_list: Iterable, destructive=False):
        if not hasattr(metabolite_list, "__iter__"):
            metabolite_list = [metabolite_list]

        remove_measurements = []
        for metabolite in metabolite_list:
            if metabolite in self._measurements:
                self._measurements.remove(metabolite)
            if metabolite in self.input_metabolites:
                print('removing input metabolite for which labelling is set!')
                self._input_labelling = {}
            if not destructive:
                # NB this is necessary for condensed reactions where a
                #   metabolite appears in the atom_map but not in metabolites
                for reaction in metabolite._reaction:
                    reaction._metabolites[metabolite] = 0.0

            for measurement in self._measurements:
                if hasattr(measurement, 'metabolite') and (metabolite is measurement.metabolite):
                    remove_measurements.append(measurement)

        for measurement in remove_measurements:
            self._measurements.remove(measurement)

        Model.remove_metabolites(self, metabolite_list=metabolite_list, destructive=destructive)
        self._pseudo_metabolites = DictList()  # need to recompute this
        self._is_built = False

    def add_groups(self, group_list):
        raise NotImplementedError

    def remove_groups(self, group_list):
        raise NotImplementedError

    def copy(self):
        # NB this will delete all things associated with build_simulator, but keeps polytope
        return pickle.loads(pickle.dumps(self))

    def reset_state(self):
        # TODO do all of this with self._la.set_to(...)
        self._dsdv[:] = 0.0
        self._jacobian[:] = 0.0

    def dsdv(self, reaction_i: LabellingReaction):
        self._dsdv[:] = 0.0

        if self._fluxes is None:
            raise ValueError('no fluxes')

        if self._la._auto_diff:
            # very circumspect, but I see no other (readable) way at the moment
            reaction_idx = self.labelling_reactions.index(reaction_i)
            jacobian = self._la.diff(inputs=self._fluxes, outputs=self._format_return(s=self._s))
            return jacobian[:, reaction_idx, :]

    def compute_jacobian(self, dept_reactions_idx: np.array = None):
        if self._fluxes is None:
            raise ValueError('no fluxes')

        if self._la._auto_diff:
            self._jacobian = self._la.diff(inputs=self._fluxes, outputs=self._format_return(s=self._s))
            return self._jacobian

        if dept_reactions_idx is None:
            dept_reactions_idx = range(len(self._labelling_reactions))

        for i in dept_reactions_idx:
            reaction = self._labelling_reactions[i]
            self._jacobian[:, i, :] = self.dsdv(reaction_i=reaction)

        return self._jacobian

    @abstractmethod
    def _format_return(self, s): raise NotImplementedError

    @abstractmethod
    def _set_state(self): raise NotImplementedError

    @abstractmethod
    def _initialize_tensors(self): raise NotImplementedError

    @property
    def metabolites_in_state(self):
        metabolites_in_state = DictList()
        polytope = extract_labelling_polytope(model=self, coordinate_id='thermo')

        unbalanced = (polytope.S > 0.0).all(1) | (polytope.S < 0.0).all(1)
        if (unbalanced).any():
            raise ValueError(f'Unbalanced metabolites {polytope.S.index[unbalanced].values}')

        for mid in polytope.S.index:
            if mid in self.metabolites:
                metabolite = self.metabolites.get_by_id(mid)
                if isinstance(metabolite, LabelledMetabolite):
                    metabolites_in_state.append(metabolite)
        return metabolites_in_state

    @property
    def pseudo_metabolites(self):
        if self._pseudo_metabolites:
            return self._pseudo_metabolites
        metabolites_in_state = self.metabolites_in_state
        self._pseudo_metabolites = DictList()
        for pseudo_reaction in self.pseudo_reactions:
            for metabolite, coeff in pseudo_reaction._metabolites.items():
                if coeff > 0:
                    self._pseudo_metabolites.append(metabolite)
                else:
                    if metabolite not in metabolites_in_state:
                        raise ValueError(f'Cannot simulate {pseudo_reaction.id} since {metabolite.id} not in state')
        return self._pseudo_metabolites

    def prepare_polytopes(self, free_reaction_id=None, verbose=False):
        if len(self._input_labelling) == 0:
            raise ValueError('set labelling input first!')  # need to have set labelling before generating system!

        # TODO: why did we implement this again; I think it was because otherwise cobra and optlang dont like it
        thermo_pol = extract_labelling_polytope(self, coordinate_id='thermo')
        net_pol = thermo_2_net_polytope(thermo_pol, verbose)
        fva_df = fast_FVA(polytope=net_pol)
        never_net = (abs(fva_df) < self.tolerance).all(axis=1)
        never_net_rids = never_net.index[never_net].str.replace(_rev_reactions_rex, '', regex=True)
        for rid in never_net_rids:
            self.reactions.get_by_id(rid).bounds = (0.0, 0.0)

        # TODO change the bounds for the other fluxes to the fva ones, this basically finds 0-facets that we need to deal with!

        self._labelling_reactions = DictList()  # since we reset a bunch of reactions to 0 bounds
        self._pseudo_metabolites  = DictList()  # this way we make sure it is recomputed with updated metabolites_in_state

        if never_net.any() and verbose:
            string = ", ".join([f'{i}' for i in never_net_rids])
            print(f'These reactions never carry a net flux and therefore now have 0 bounds: \n{string}\n')

        # this way we autmoatically filter the unsimulable metabolites, TODO DOES NOT WORK CURRENTLY!
        self.set_measurements(measurement_list=self._measurements, verbose=verbose)
        self.solver.update()  # this is to filter out the unsimulable metabolites
        self._set_free_reactions(free_reaction_id=free_reaction_id)

    @abstractmethod
    def build_model(self, free_reaction_id=None, verbose=False):
        self._initialize_state()
        self.prepare_polytopes(free_reaction_id, verbose)
        self._is_built = True
        self._fcm = FluxCoordinateMapper(
            model=self,
            pr_verbose=verbose,
            linalg=self._la,
        )
        self._is_built = False  # set True by the child class again after  build-steps are completed successfully
        self._set_state()

    @abstractmethod
    def cascade(self, pandalize=False): raise NotImplementedError

    @abstractmethod
    def pretty_cascade(self, weight: int): raise NotImplementedError


if __name__ == "__main__":

    reaction_kwargs = {
        'a_in': {
            'lower_bound': 10.0, 'upper_bound': 10.0,
            'atom_map_str': '∅ --> A/ab'
        },
        # 'a_in': {
        #     'lower_bound': -10.0, 'upper_bound': -10.0,
        #     'atom_map_str': 'A/ab --> ∅'
        # },
        'd_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'D/abc --> ∅'
        },
        'f_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'F/a --> ∅'
        },
        'h_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'H/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab --> B/ab'
        },
        'v2': {
            'lower_bound': 0.0, 'upper_bound': 100.0,
            # 'rho_min': 0.1, 'rho_max': 0.8,
            'atom_map_str': 'B/ab --> E/ab'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab + E/cd --> C/abcd'
        },
        'v4': {
            'upper_bound': 100.0,  # 'lower_bound': -10.0,
            'atom_map_str': 'E/ab --> H/ab'
        },
        # 'v5': {
        #     'upper_bound': 100.0,
        #     'atom_map_str': 'C/abcd --> F/a + D/bcd'
        # },
        'v5': {  # NB this is an always reverse reaction!
            'lower_bound': -100.0,  # 'upper_bound': 100.0
            'atom_map_str': 'F/a + D/bcd  <-- C/abcd',  # <--  ==>
            # 'atom_map_str': 'F/a + D/bcd  <=> C/abcd',  # <--  ==>
        },
        'v6': {
            'upper_bound': 100.0,
            'atom_map_str': 'D/abc --> E/ab + F/c'
        },
        'v7': {
            'upper_bound': 100.0,
            'atom_map_str': 'F/a + F/b --> H/ab'
        },
        'vp': {
            'lower_bound': 0.0,  # 'upper_bound': 100.0,
            'pseudo': True,
            'atom_map_str': 'C/abcd + D/efg + H/hi --> L/abgih'
        },
    }
    metabolite_kwargs = {
        'A': {'formula': 'C2H4O5'},
        'B': {'formula': 'C2HPO3'},
        'C': {'formula': 'C4H6N4OS'},
        'D': {'formula': 'C3H2'},
        'E': {'formula': 'C2H4O5'},
        'F': {'formula': 'CH2'},
        'G': {'formula': 'CH2'},  # not used
        'H': {'formula': 'C2H2'},
        'L': {'formula': 'C5KNaSH'},  # pseudo-metabolite
        'L|[1,2]': {'formula': 'C2H2O7'},  # pseudo-metabolite
        'P': {'formula': 'C2H'},
    }
    linalg = LinAlg(backend='torch', batch_size=1, device='cpu', )
    model = LabellingModel(linalg=linalg, name='niks')
    model.add_reaction_labelling(
        reaction_kwargs=reaction_kwargs,
        metabolite_kwargs=metabolite_kwargs
    )