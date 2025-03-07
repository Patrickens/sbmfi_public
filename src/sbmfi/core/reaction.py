from sbmfi.core.metabolite import LabelledMetabolite, EMU_Metabolite, EMU, ConvolutedEMU
from sbmfi.core.util import _get_dictlist_idxs, _read_atom_map_str_rex, _find_biomass_rex, _strip_bigg_rex
import numpy as np
import pandas as pd
from cobra import Reaction, DictList, Metabolite
from math import isinf
from collections.abc import Iterable
from abc import abstractmethod
import operator
from copy import copy, deepcopy


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
            reaction:Reaction,
            rho_min: float = 0.0,
            rho_max: float = 0.0,
            pseudo: bool = False,
    ):
        if isinstance(reaction, LabellingReaction):
            raise NotImplementedError
        if isinstance(reaction, Reaction):
            self.__dict__.update(reaction.__dict__)
        else:
            raise ValueError(f'need to instantiate with a cobra Reaction object, got {type(reaction)}')

        self._pseudo = pseudo

        self._atom_map = {}  # {Met: (stoich, [tuple('atoms'),...] ) } required to be ordered for the
        self._rect_prod_map: np.array = None  # mapping all reactant atoms to (present) product atoms
        self._rho_min = 0.0  # minimal fraction of flux going in the reverse direction; has to do with Gibbs free energy change
        self._rho_max = 0.0
        self._dgibbsr  = 0.0  # the currently set dGr

        if not self._pseudo:
            # this default selection is important when initializing with a id_or_reaction=Reaction
            self.rho_max = rho_max
            self.rho_min = rho_min

            # rev_reac is a unidirectional reaction in the opposite direction of self
            self._rev_reaction = None
            self._initialize_rev_reaction()

    def _initialize_rev_reaction(self):
        if not self._pseudo:  # we dont need a rev_reaction for pseudo-reactions
            rid = self.id
            if rid is None:
                rid = ''
            reaction = Reaction(id=rid + '_rev', lower_bound=0.0, upper_bound=1.0)
            reaction._metabolites = {k: -v for k, v in self._metabolites.items()}  # add_metabolites also changes bounds
            reaction._model = self._model
            self._rev_reaction = type(self)(reaction=reaction, pseudo=True)
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
        if self._model is None:
            raise ValueError('this only works once the reaction is part of a model')
        if not self.metabolites:
            raise ValueError('the reaction needs to be defined already')

        map = {}
        for metabolite, (stoich, atoms) in atom_map.items():
            if not isinstance(metabolite, LabelledMetabolite):
                raise ValueError(f'{self.id} atom_map contains non-LabelledMetabolite object: {metabolite.id}')

            if metabolite in self.model.metabolites:
                model_met = self.model.metabolites.get_by_id(metabolite.id)

            if (metabolite is not model_met) or (metabolite._model is not self._model):
                raise ValueError(f'first use model.repair(), the references are messed up for: {metabolite.id}!')

            for met_met, stoich_met in list(self._metabolites.items()):
                if metabolite.id == met_met.id:
                    if (stoich != stoich_met) or ((atoms[0] is not None) and (abs(stoich) != len(atoms))):
                        raise ValueError(
                            f'{self.id}: for {metabolite.id} stoichiometry and atom mapping are inconsistent'
                        )
                    if metabolite is not met_met:
                        raise ValueError('metabolite in atom_map is not equal to metabolite in self.metabolites')

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

    def build_atom_map_from_string(self, atom_map_str: str):
        # TODO: make it possible to build entire reaction from string with co-factors and all

        if self._model is None:
            raise ValueError('this only works once the reaction is part of a model')

        rects, arrow, prods = _read_atom_map_str_rex.findall(string=atom_map_str)[0]
        is_biomass = _find_biomass_rex.search(rects) is not None

        if ((arrow == '<=>') and not self.reversibility) or \
                (('==' in arrow) and self.reversibility) or \
                (('--' in arrow) and (self.reversibility or self.rho_max != 0.0)):
            print(f'wrong arrow or bounds {self.id} {atom_map_str}, {self.bounds}, {self.rho_min, self.rho_min}')

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

            for intermediate in intermediates:
                intermediate = intermediate.strip()

                if intermediate == '∅':  # deals with boundary reactions of the pysumo model
                    continue
                met_atoms = intermediate.split('/')
                if len(met_atoms) == 1:
                    # this indicates that there is a cofactor in the atom_map_str
                    continue
                met_id, atoms = met_atoms
                atoms_arr = tuple(atoms)

                if met_id in self.model.metabolites:
                    metabolite = self.model.metabolites.get_by_id(id=met_id)
                else:
                    raise ValueError('metabolite not part of model')

                if metabolite not in atom_map:
                    atom_map[metabolite] = (0, [])

                stoich, atoms = atom_map[metabolite]

                if met_id in rects:
                    stoich -= 1
                elif met_id in prods:
                    stoich += 1
                atom_map[metabolite] = (stoich, atoms)
                atoms.append(atoms_arr)
        return atom_map, is_biomass

    def add_metabolites(self, metabolites_to_add, combine=True, reversibly=True):
        raise NotImplementedError('Please finish all cobra Reaction objects in terms of metabolites and genes'
                                  'before turning them into LabellingReaction objects; '
                                  'LabellingReactions are equipped with atom_maps, which would also have to be '
                                  'redefined, which we chose to handle in one fell swoop at instantiation.')

    def subtract_metabolites(self, metabolites: dict, combine: bool = True, reversibly: bool = True):
        raise NotImplementedError('Please finish all cobra Reaction objects in terms of metabolites and genes'
                                  'before turning them into LabellingReaction objects; '
                                  'LabellingReactions are equipped with atom_maps, which would also have to be '
                                  'redefined, which we chose to handle in one fell swoop at instantiation.')

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


class EMU_Reaction(LabellingReaction):
    _TYPE_METABOLITE = EMU_Metabolite
    def __init__(
            self,
            reaction: Reaction,
            rho_min: float = 0.0,
            rho_max: float = 0.0,
            pseudo = False,
    ):
        LabellingReaction.__init__(**locals())
        self.A_elements = {}
        self.B_elements = {}

        self.A_tensors = {}
        self.B_tensors = {}

    def __getstate__(self):
        state = super(EMU_Reaction, self).__getstate__()
        state['A_elements'] = {}
        state['B_elements'] = {}
        state['A_tensors'] = {}
        state['B_tensors'] = {}
        return state

    def _find_reactant_emus(self, product_emu: EMU, substrate_metabolites: Iterable, n_eq_EMU=1, eq_EMU=None) -> list:
        if not product_emu.metabolite in self.gettants(reactant=False):
            raise ValueError('pjurre kenss')
        prod_stoich, all_prod_atoms = self._atom_map[product_emu.metabolite]
        if eq_EMU is None:
            eq_EMU = product_emu

        for product_atoms in all_prod_atoms:
            convolvers = []
            tot_weight = 0
            emu_atoms = product_atoms[product_emu.positions]
            for metabolite, (stoich, reactant_atoms) in self._atom_map.items():
                if stoich > 0:
                    continue
                for rect_atoms in reactant_atoms:
                    positions = np.where(emu_atoms[:, None] == rect_atoms[None, :])[1]
                    if positions.size:
                        reactant_emu = metabolite.get_emu(positions=positions)
                        if (reactant_emu.weight < product_emu.weight) or (metabolite in substrate_metabolites):
                            convolvers.append(reactant_emu)
                            tot_weight += reactant_emu.weight
                            if tot_weight < product_emu.weight:
                                continue
                            elif len(convolvers) > 1:
                                reactant_emu = convolvers[0].metabolite.get_convolved_emu(emus=convolvers)
                            stoich = -1.0 / n_eq_EMU
                            AorB = self.B_elements.setdefault(product_emu.weight, [])
                        else:
                            stoich = 1.0 / n_eq_EMU
                            AorB = self.A_elements.setdefault(product_emu.weight, [])

                        # TODO: if build_tensors is called twice it sucks!
                        A_element = (-1.0, eq_EMU, eq_EMU) # test this for A + B -> 2
                        A = self.A_elements.setdefault(product_emu.weight, [])
                        # if (A_element not in A) or (len(self._atom_map[eq_EMU.metabolite][1]) != A.count(A_element)):
                        if (A_element not in A):  # TODO check whether this performs as desired!!! leu_syn is a good example since it produces 2 co2s
                            A.append(A_element)

                        element = (stoich, eq_EMU, reactant_emu)
                        if element not in AorB:
                            AorB.append(element)

    def _product_elements(self, product_emu):
        allements = []
        for dct in [self.A_elements, self.B_elements]:
            for weight, elements in dct.items():
                for element in elements:
                    if element[1] == product_emu:
                        allements.append(element)
        return allements

    def map_reactants_products(self, product_emu: EMU, substrate_metabolites: Iterable):
        if not product_emu.metabolite.symmetric:
            self._find_reactant_emus(product_emu=product_emu, substrate_metabolites=substrate_metabolites)
            return self._product_elements(product_emu=product_emu)

        product_metabolite = product_emu.metabolite
        full_emu_positions = product_metabolite.emus[product_metabolite.elements['C']][0].positions
        sym_emu_positions = full_emu_positions[::-1][product_emu.positions]
        sym_emu_positions.sort()

        if all(sym_emu_positions == product_emu.positions):
            self._find_reactant_emus(product_emu=product_emu, substrate_metabolites=substrate_metabolites)
            return self._product_elements(product_emu=product_emu)

        sym_product_emu = product_metabolite.get_emu(positions=sym_emu_positions)
        self._find_reactant_emus(product_emu=product_emu, substrate_metabolites=substrate_metabolites, n_eq_EMU=2)
        elements = self._product_elements(product_emu=product_emu)
        self._find_reactant_emus(
            product_emu=product_emu, substrate_metabolites=substrate_metabolites, n_eq_EMU=2, eq_EMU=sym_product_emu
        )
        return elements + self._product_elements(product_emu=sym_product_emu)

    def build_tensors(self):
        if self.model is None:
            raise ValueError('no model')

        xemus = self.model._xemus
        yemus = self.model._yemus

        # NOTE: cannot reinitialize A_elements and B_elements here!
        self.A_tensors = {}
        self.B_tensors = {}

        for weight, A_elem in self.A_elements.items():
            A_elem = np.array(A_elem)
            A_indices = _get_dictlist_idxs(xemus[weight], A_elem[:, 1:])
            A_values = A_elem[:, 0].astype(np.double)
            self.A_tensors[weight] = self.model._la.get_tensor(
                shape=(len(xemus[weight]), len(xemus[weight])),
                indices=A_indices, values=A_values
            )

            B_elem = self.B_elements.get(weight, None)
            if B_elem is not None:
                B_elem = np.array(B_elem)
                B_xidx = _get_dictlist_idxs(xemus[weight], B_elem[:, 1])
                B_yidx = _get_dictlist_idxs(yemus[weight], B_elem[:, 2])
                B_indices = np.concatenate((B_xidx[:, None], B_yidx[:, None]), axis=1)
                B_values = B_elem[:, 0].astype(np.double)
                self.B_tensors[weight] = self.model._la.get_tensor(
                    shape=(len(xemus[weight]), len(yemus[weight])),
                    indices=B_indices, values=B_values
                )

    def pretty_tensors(self, weight: int):
        if self.model is None:
            raise ValueError('no model')
        elif not self.model.is_built:
            raise ValueError('model not built')
        result = {}
        if weight == 0:
            return result
        A = self.A_tensors.get(weight, None)
        B = self.B_tensors.get(weight, None)

        adx = self.model._xemus[weight].list_attr('id')
        bdx = self.model._yemus[weight].list_attr('id')

        if A is not None:
            result['A'] = pd.DataFrame(self._model._la.tonp(A), index=adx, columns=adx)
        if B is not None:
            result['B'] = pd.DataFrame(self._model._la.tonp(B), index=adx, columns=bdx)
        return result


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro, multi_modal

    model, kwargs = spiro(build_simulator=True)
    model.reactions[4].pretty_tensors(0)

