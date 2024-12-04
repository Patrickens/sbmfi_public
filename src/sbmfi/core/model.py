# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # this is to avoid some weird stuff
from cobra.util.context import get_context
from cobra import Model, Reaction, Metabolite, DictList
import math
import numpy as np
import pandas as pd
from sbmfi.core.linalg import LinAlg
from sbmfi.core.util   import (
    _read_atom_map_str_rex,
    _find_biomass_rex,
    _rev_reactions_rex,
)
from sbmfi.core.polytopia import (
    FluxCoordinateMapper,
    LabellingPolytope,
    extract_labelling_polytope,
    thermo_2_net_polytope,
    fast_FVA
)
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction
from sbmfi.core.metabolite  import LabelledMetabolite, ConvolutedEMU, EMU, IsoCumo
from itertools import repeat
from typing import Iterable, Union
from abc import abstractmethod
from copy import copy, deepcopy
import pickle


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
        self._pseudo_metabolites = DictList()

        # collections of reactions of various sorts
        self._biomass_id: str = None
        self.pseudo_reactions     = DictList()
        self._labelling_reactions = DictList()  # reactions for which all reactants and products are present and carry carbon
        self._chosen_rid = []

        self._fcm_kwargs = {}
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
        return pd.Index(self.labelling_reactions.list_attr('id'), name='fluxes_id')

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
            raise ValueError('build the simulator first!')
        return self._fcm

    def set_fluxes(self, fluxes: Union[pd.DataFrame, np.array], samples_id=None, trim=True):
        if not self._is_built:
            raise ValueError('MUST BUILD')
        fluxes = self._fcm.frame_fluxes(fluxes, samples_id, trim)
        if len(fluxes.shape) > 2:
            raise ValueError('can only deal with 2D stratified fluxes!')
        if self._la._auto_diff:
            fluxes.requires_grad_(True)
        if fluxes.shape[0] != self._la._batch_size:
            raise ValueError(f'batch_size = {self._la._batch_size}; fluxes.shape[0] = {fluxes.shape[0]}')
        self._fluxes = fluxes

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
            free_reaction_id = self._chosen_rid
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
        self._chosen_rid = user_chosen.list_attr('id')
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

    def add_reactions(
            self,
            reaction_list: Iterable = None,
            metabolite_kwargs: dict = None,
            reaction_kwargs: dict = None
    ):
        """ A function that is used to instantiate a SUMod object with an existing cobra.Model
        object. To do so, arguments such as atom-mappings need to be passed to the relevant
        reactions and symmetry information needs to be passed to metabolites.
        TODO: fix contexts
        TODO: fix genes and groups, that sucks now! might still be referring to cobra Reaction
            objects that are made obsolete by SUReaction objects
        TODO: maybe add reaction_str for the reaction_from_string method of cobra.Reaction??
        Parameters
        ----------
        reaction_list : Iterable, optional
            ...
        metabolite_kwargs : dict, optional
            When instantiating SUMod with an existing Model or SUMod object, this dictionary
            specifies which cobra.Metabolite objects are turned into SUMet objects.
            Labelling is only computed for SUMet objects.
        reaction_kwargs : dict, optional
            When instantiating SUMod with an existing Model or SUMod object, this dictionary
            specifies which cobra.Reaction objects are turned into SUReac objects.
            Only SUReac objects influence the labelling state of the system.
        """

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
                    raise ValueError('watch out, more than one biomass reaction in reac_kwargs!')
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

        if self._is_built:
            # TODO respect previously set free fluxes I guess?
            self.build_simulator(**self._fcm.fcm_kwargs)

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
        polytope = extract_labelling_polytope(model=self, coordinates='thermo')

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

    @abstractmethod
    def prepare_polytopes(self, free_reaction_id=None, verbose=False):
        if len(self._input_labelling) == 0:
            raise ValueError('set labelling input first!')  # need to have set labelling before generating system!

        # TODO: why did we implement this again; I think it was because otherwise cobra and optlang dont like it
        thermo_pol = extract_labelling_polytope(self, coordinates='thermo')
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
    def build_simulator(
            self,
            free_reaction_id=None,
            kernel_basis='svd',
            basis_coordinates='rounded',
            logit_xch_fluxes=True,
            hemi_sphere=False,
            scale_bound=None,
            verbose=False,
    ):
        self._initialize_state()
        self._fcm = FluxCoordinateMapper(
            model=self,
            kernel_basis=kernel_basis,
            basis_coordinates=basis_coordinates,
            free_reaction_id=free_reaction_id,
            logit_xch_fluxes=logit_xch_fluxes,
            pr_verbose=verbose,
            linalg=self._la,
            hemi_sphere=hemi_sphere,
            scale_bound=scale_bound,
        )
        self._fcm_kwargs = self._fcm.fcm_kwargs
        self._set_state()

    @abstractmethod
    def cascade(self, pandalize=False): raise NotImplementedError

    @abstractmethod
    def pretty_cascade(self, weight: int): raise NotImplementedError


class RatioMixin(LabellingModel):
    """
    This is a mixin that defines all the stuff to do with flux-ratios
    """
    _RATIO_ATOL = 1e-3  # if the difference between lb and ub for a ratio is below this; we consider it an equality
    
    def __getstate__(self):
        odict = super(RatioMixin, self).__getstate__()
        odict['_ratio_repo'] = self.ratio_repo
        odict['_ratio_num_sum'] = None
        odict['_ratio_den_sum'] = None
        return odict

    def __setstate__(self, state):
        super(RatioMixin, self).__setstate__(state)
        ratio_repo = state.get('_ratio_repo', None)
        if ratio_repo is not None:
            self.set_ratio_repo(ratio_repo)

    @property
    def ratios_id(self) -> pd.Index:
        return pd.Index(list(self._ratio_repo.keys()), name='ratios_id')

    @property
    def ratio_reactions(self) -> DictList:
        ratio_reactions = {}  # the keys are an ordered set
        for vals in self._ratio_repo.values():
            for reac_id, coeff in {**vals['numerator'], **vals['denominator']}.items():
                reac = self.labelling_reactions.get_by_id(id=reac_id)
                ratio_reactions.setdefault(reac, None)
        return DictList(ratio_reactions.keys())

    def compute_ratios(self, fluxes, tol=1e-10, pandalize=True) -> pd.DataFrame:
        # TODO this is not the right place for this...
        index = None
        if isinstance(fluxes, pd.DataFrame):
            index = fluxes.index
            fluxes = self._la.get_tensor(values=fluxes.loc[:, self.labelling_reactions.list_attr('id')].values)

        num = self._ratio_num_sum @ fluxes.T
        den = self._ratio_den_sum @ fluxes.T
        den[den == 0.0] += tol

        with np.errstate(invalid='ignore'):
            ratios = self._la.divide(num, den).T

        if pandalize:
            return pd.DataFrame(self._la.tonp(ratios), index=index, columns=self.ratios_id)
        return ratios

    def _initialize_state(self):
        super(RatioMixin, self)._initialize_state()
        self._ratio_num_sum = self._la.get_tensor(shape=(0,))
        self._ratio_den_sum = self._la.get_tensor(shape=(0,))
        self._ratio_repo = {}  # repository of all flux-ratios (with names as keys) that we are interested in in a particular model

    @staticmethod
    def _sum_getter(key, ratio_repo: dict, linalg: LinAlg, index: pd.Index):
        # TODO THIS FUNCTION IS CURRENTLY WRONG!
        if not ratio_repo:
            return linalg.get_tensor(shape=(0,))
        indices = []
        coeffs = []
        # essential to be ratio_repo because _ratio_repo is condensed
        for i, (name, ratio) in enumerate(ratio_repo.items()):
            for frackey, values in ratio.items():
                if (frackey != key) and (key == 'numerator'):
                    continue
                for reac_id, coeff in values.items():
                    reac_idx = index.get_loc(reac_id)
                    indices.append((i, reac_idx))
                    coeffs.append(coeff)
        return linalg.get_tensor(
            shape=(i + 1, len(index)), indices=np.array(indices), values=np.array(coeffs, dtype=np.double)
        )

    @property
    def ratio_repo(self):
        repo = {}
        for ratio_id, vals in self._ratio_repo.items():
            numerator = {}
            denominator = {}
            for reac_id, coeff in {**vals['numerator'], **vals['denominator']}.items():
                net_id = _rev_reactions_rex.sub('', reac_id)
                if reac_id != net_id:
                    coeff *= -1
                denominator[net_id] = coeff
                if reac_id in vals['numerator']:
                    numerator[net_id] = coeff
            repo[ratio_id] = {
                'numerator': numerator,
                'denominator': {**numerator, **denominator},
            }
        return repo

    def set_ratio_repo(self, ratio_repo: dict):
        # condensed means that we pass the ratio_repo with disjoint sets for numerator and denominator
        # TODO only accept ratios defined in net-coordinate system
        # TODO update this with the self._always_rev attribute functionality
        repo = {}
        for ratio_id, vals in ratio_repo.items():
            num = vals['numerator']
            den = vals['denominator']
            numerator = {}
            denominator = {}
            for reac_id, coeff in {**num, **den}.items():

                if reac_id not in self.reactions:
                    raise ValueError(f'Not in reactions {reac_id}')

                reaction = self.reactions.get_by_id(id=reac_id)
                always_rev = (reaction.upper_bound <= 0.0) and (reaction.rho_max == 0.0)

                if (reac_id not in self.labelling_reactions) or \
                        (always_rev and (reaction._rev_reaction.id in self.labelling_reactions)):
                    raise ValueError(f'First add reactions and atom_mappings; {reac_id} not in labelling_reactions')

                if reac_id in num:
                    element = numerator
                    if reac_id in den:
                        if coeff != den.get(reac_id):
                            raise ValueError('different numerator and denominator coefficients!')
                elif reac_id in den:
                    element = denominator

                if not always_rev:
                    element[reac_id] = coeff

                if (reaction.rho_max > 0.0) or always_rev:
                    element[reaction._rev_reaction.id] = -coeff
            repo[ratio_id] = {'numerator': numerator, 'denominator': denominator}

        self._ratio_repo = repo  # condensed representation!
        self._ratio_num_sum = self._sum_getter('numerator', repo, self._la, self.labelling_fluxes_id)
        self._ratio_den_sum = self._sum_getter('denominator', repo, self._la, self.labelling_fluxes_id)

    def prepare_polytopes(self, free_reaction_id=None, verbose=False):
        if free_reaction_id is None:
            free_reaction_id = []
        free_reaction_id = self.ratio_reactions.list_attr('id') + list(free_reaction_id)
        ratio_repo = self.ratio_repo
        LabellingModel.prepare_polytopes(self, free_reaction_id, verbose)
        self.set_ratio_repo(ratio_repo=ratio_repo)  # NB this is necessary, because ratio_repo is reset in _initialize_state

    def get_free_and_dependent_indices(self):
        raise NotImplementedError
        ratio_NS = self._fcm.null_space.loc[
            self.ratio_reactions.list_attr('id')]  # null-space that contributes to ratio reactions
        # _free_idx are the reactions that contribute to ratio_reactions
        self._ratio_free_idx = self._la.get_tensor(values=np.where(~(ratio_NS == 0.0).all(0))[0])
        # _dept_idx are dependent reactions that contrubute to the free reactions that contribute to the ratio reactions
        self._ratio_dept_idx = np.where(self._la.tonp(abs(self._fcm._NS[:, self._ratio_free_idx]).sum(1) > 0.0))[0]


class EMU_Model(LabellingModel):
    """
    DINGON
    """
    _TYPE_REACTION = EMU_Reaction

    def __getstate__(self):
        odict = super(EMU_Model, self).__getstate__()

        odict['_ns'] = None
        odict['_xemus'] = {}
        odict['_yemus'] = {}
        odict['_emu_indices'] = {}
        odict['_A_tot'] = {}
        odict['_LUA'] = {}
        odict['_B_tot'] = {}
        odict['_X'] = {}
        odict['_Y'] = {}
        odict['_dXdv'] = {}
        odict['_dYdv'] = {}
        return odict

    def _initialize_state(self):
        super(EMU_Model, self)._initialize_state()
        
        # state-objects
        self._xemus = {}  # the EMUs that make up the X matrices ordered by weight
        self._yemus = {}  # the EMUs that make up the Y matrices ordered by weight
        self._emu_indices = {}  # maps EMUs from X and Y to vector self._s

        # state-variables
        self._A_tot = {}  # rhs matrix in the the EMU equation to be inverted
        self._LUA = {}  # LU factored A matrix for jacobian calculations
        self._B_tot = {}
        self._X = {}  # emu state matrix
        self._Y = {}  # emu input matrix

        # jacobian stuff
        self._dXdv = {}
        self._dYdv = {}

    def _set_state(self):
        num_el_s = 0
        sum_indices = []
        for i, metabolite in enumerate(self.measurements):
            if type(metabolite) == EMU:
                met_emu = metabolite
                met_weight = metabolite.positions.shape[0]
                metabolite = met_emu.metabolite
            else:
                met_weight = metabolite.elements['C']
                met_emu = metabolite.emus[met_weight][0]

            sum_indices.extend([(i, j) for j in range(num_el_s, num_el_s + met_weight + 1)])
            num_el_s += met_weight + 1

            if metabolite in self.input_metabolites:
                emus = self._yemus
            else:
                emus = self._xemus
            for i in range(1, met_weight + 1):
                self._xemus.setdefault(i, DictList())
                self._yemus.setdefault(i, DictList())
            emus[met_weight].append(met_emu)
        self._ns   = num_el_s
        self._s    = self._la.get_tensor(shape=(self._la._batch_size, num_el_s))
        self._dsdv = self._la.get_tensor(shape=(self._la._batch_size, num_el_s))
        self._sum  = self._la.get_tensor(
            shape=(len(self.measurements), len(self.state_id)),
            indices=np.array(sum_indices, dtype=np.int64),
            values=np.ones(len(sum_indices), dtype=np.double)
        )

    def reset_state(self):
        # N.B. absolutely has to be done in-place! Otherwise matrix in self._emu_indices is not valid anymore
        super().reset_state()
        self._s[:] = 0.0
        for weight, A in self._A_tot.items():
            A[:] = 0.0
            self._B_tot[weight][:] = 0.0
            self._X[weight][:]     = 0.0
            self._dXdv[weight][:]  = 0.0
            self._dYdv[weight][:]  = 0.0
            # NB Y is modified in-place and does not need reinitialization

    def set_input_labelling(self, input_labelling: pd.Series):
        labelling_id = input_labelling.name
        settings = self._labelling_repo.get(labelling_id, None)
        if settings is None:
            super().set_input_labelling(input_labelling=input_labelling)
            if len(self._yemus) > 0:
                self._initialize_Y()
        else:
            self._labelling_id = labelling_id
            self._input_labelling = settings['_input_labelling']
            Y = settings.get('_Y', None)
            if Y is None:
                # this occurs if we rebuild with different batch_size
                self._initialize_Y()
                Y = settings.get('_Y', None)
            if (len(Y) == 0):
                raise ValueError
            self._Y = Y
            for weight, yek in Y.items():
                yemus = self._yemus[weight]
                for yemu in yemus:
                    # this is when a built model gets new input labelling
                    if type(yemu) == ConvolutedEMU:
                        continue
                    if yemu in self._emu_indices:
                        matrix, dmdv, row = self._emu_indices[yemu]
                        self._emu_indices[yemu] = yek, dmdv, row

    def _parse_measurement(self, all_metabolites:DictList, measurement_id:str):
        if '|[' in measurement_id: # this indicates its an EMU
            measurement_id = measurement_id.replace(' ', '')
            metabolite_id, positions = measurement_id.split('|')
            if metabolite_id in all_metabolites:
                metabolite = all_metabolites.get_by_id(id=metabolite_id)
                emu = metabolite.get_emu(positions=eval(positions))
                return emu
        return super(EMU_Model, self)._parse_measurement(all_metabolites, measurement_id)

    def _initialize_emu_split(self):
        # TODO: check if every emu.metabolite is in self.metabolites_in_state?
        input_metabolites = self.input_metabolites
        state_reactions = self.labelling_reactions + self.pseudo_reactions
        for weight, xemus in reversed(self._xemus.items()):
            for product_emu in xemus:
                for reaction in product_emu.metabolite.reactions:
                    if (type(reaction) != EMU_Reaction) or (reaction not in state_reactions) or \
                            (not reaction.gettants()) or (not reaction.gettants(reactant=False)):
                        continue
                    if product_emu.metabolite in reaction.gettants(reactant=False):  # map product
                        emu_reaction_elements = reaction.map_reactants_products(
                            product_emu=product_emu, input_metabolites=input_metabolites
                        )
                        for (stoich, prod, rect) in emu_reaction_elements:
                            for emu in rect.getmu():
                                if isinstance(emu, ConvolutedEMU) or (emu.metabolite in input_metabolites):
                                    if emu not in self._yemus[emu.weight]:
                                        self._yemus[emu.weight].append(emu)
                                else:
                                    if emu not in self._xemus[emu.weight]:
                                        self._xemus[emu.weight].append(emu)

        both = sorted(list(set(self._xemus.keys()) | set(self._yemus.keys())))
        self._xemus = dict([(weight, self._xemus[weight]) for weight in both])
        self._yemus = dict([(weight, self._yemus[weight]) for weight in both])

    def _initialize_Y(self):
        # deepcopy is necessary, otherwise the previous labelling state is modified in place!
        self._Y = deepcopy(self._Y)
        for weight, yemus in self._yemus.items():
            Y_values, Y_indices = [], []
            for i, yemu in enumerate(yemus):
                if type(yemu) == ConvolutedEMU:
                    continue
                for isocumo, fraction in self._input_labelling.items():
                    if isocumo.metabolite == yemu.metabolite:
                        emu_label = isocumo._label[yemu.positions]
                        M_plus = emu_label.sum()
                        for j in range(self._la._batch_size):
                            Y_values.append(fraction)
                            Y_indices.append((j, i, M_plus))

            Y_indices = np.array(Y_indices, dtype=np.int64)
            Y_values = np.array(Y_values, dtype=np.double)
            # TODO we can also create this via tiling!
            Y = self._la.get_tensor(shape=(self._la._batch_size, len(yemus), weight + 1), indices=Y_indices, values=Y_values)
            self._Y[weight] = Y

            for yemu in yemus:
                # this is when a built model gets new input labelling
                if type(yemu) == ConvolutedEMU:
                    continue
                if yemu in self._emu_indices:
                    matrix, dmdv, row = self._emu_indices[yemu]
                    self._emu_indices[yemu] = Y, dmdv, row

        res = self._labelling_repo[self._labelling_id]
        if '_Y' not in res:
            res['_Y'] = self._Y

    def _initialize_tensors(self):
        for (weight, xemus), yemus in zip(self._xemus.items(), self._yemus.values()):
            self._A_tot[weight] = self._la.get_tensor(shape=(self._la._batch_size, len(xemus), len(xemus)))
            self._B_tot[weight] = self._la.get_tensor(shape=(self._la._batch_size, len(xemus), len(yemus)))
            self._X[weight]     = self._la.get_tensor(shape=(self._la._batch_size, len(xemus), weight + 1))
            self._dXdv[weight]  = self._la.get_tensor(shape=(self._la._batch_size, len(xemus), weight + 1))
            self._dYdv[weight]  = self._la.get_tensor(shape=(self._la._batch_size, len(yemus), weight + 1))
        self._initialize_Y()

    def _initialize_emu_indices(self):
        # TODO: this might go wrong after pickling?? This is why we rebuild
        for (weight, xemus), yemus in zip(self._xemus.items(), self._yemus.values()):
            for emu in (yemus + xemus):
                if isinstance(emu, ConvolutedEMU):
                    continue
                elif emu.metabolite in self.input_metabolites:
                    matrix = self._Y[emu.weight]
                    dmdv = self._dYdv[emu.weight]
                    row = self._yemus[emu.weight].index(emu)
                else:
                    matrix = self._X[emu.weight]
                    dmdv = self._dXdv[emu.weight]
                    row = self._xemus[emu.weight].index(emu)
                self._emu_indices[emu] = matrix, dmdv, row

    def build_simulator(
            self,
            free_reaction_id=None,
            kernel_basis='svd',
            basis_coordinates='rounded',
            logit_xch_fluxes=True,
            hemi_sphere=False,
            scale_bound=None,
            verbose=False,
    ):
        super().build_simulator(
            free_reaction_id, kernel_basis, basis_coordinates, logit_xch_fluxes, hemi_sphere, scale_bound, verbose
        )
        self._initialize_emu_split()

        for reaction in self.labelling_reactions + self.pseudo_reactions:
            reaction._model = self  # NOTE: this is to ensure that reverse_reactions recognize this model
            reaction.build_tensors()

        # necessary if we change i.e. batch_size; will be rebuilt when set_labelling is called
        for v in self._labelling_repo.values():
            Y = v.pop('_Y', None)

        self._initialize_tensors()
        self._initialize_emu_indices()
        self._is_built = True

    def _build_Y(self, weight):
        # TODO slowest function!
        for i, yemu in enumerate(self._yemus[weight]):
            if isinstance(yemu, EMU):
                continue  # skip input emus, only deal with convolvedEMUs
            for j, xemu in enumerate(yemu._emus):  # this way we can convolve more than 2 mdvs!
                tensor, _, emu_row = self._emu_indices[xemu]

                mdv_xemu = tensor[:, emu_row, :]

                if j == 0:
                    mdv = mdv_xemu
                else:
                    # slowest part of slowest function
                    mdv = self._la.convolve(a=mdv, v=mdv_xemu)
            self._Y[weight][:, i, :] = mdv
        return self._Y[weight]

    def _add_to_system(self, weight, reaction, v=None):
        A = reaction.A_tensors.get(weight)
        if A is None:
            return
        A = A[None, :]
        if v is not None:
            A = v[:, None, None] * A
        self._A_tot[weight] += A

        B = reaction.B_tensors.get(weight)
        if B is None:
            return
        B = B[None, :]
        if v is not None:
            B = v[:, None, None] * B
        self._B_tot[weight] += B

    def _build_A_B_by_weight(self, weight):
        for reaction, v in zip(self.labelling_reactions, self._fluxes.T):
            self._add_to_system(weight=weight, reaction=reaction, v=v)

        # assumes that pseudo-metabolites are only made by 1 reaction!!!
        for pseudo_reaction in self.pseudo_reactions:
            self._add_to_system(weight=weight, reaction=pseudo_reaction)

        return self._A_tot[weight], self._B_tot[weight]

    def _format_return(self, s, derivative=False):
        num_el_s = 0
        for metabolite in self.measurements:
            if type(metabolite) == EMU:
                metemu = metabolite
            else:
                metemu = metabolite.emus[metabolite._formula['C']][0]

            state, dmdv, emu_row = self._emu_indices[metemu]

            if derivative:
                tensor = dmdv
            else:
                tensor = state

            s[:, num_el_s: num_el_s + metemu.weight + 1] = tensor[:, emu_row, :]
            num_el_s += metemu.weight + 1
        return s

    def pretty_cascade(self, weight: int):
        if not self._is_built:
            raise ValueError
        adx = self._xemus[weight].list_attr('id')
        bdx = self._yemus[weight].list_attr('id')

        def batch_corrector(values, index, columns=None):
            batches = {}
            for sid, sub_vals in zip(self._fcm.samples_id, values):
                batches[sid] = pd.DataFrame(sub_vals, index=index, columns=columns)
            return pd.concat(batches.values(), keys=batches.keys())

        As = batch_corrector(values=self._la.tonp(self._A_tot[weight]), index=adx, columns=adx)
        Bs = batch_corrector(values=self._la.tonp(self._B_tot[weight]), index=adx, columns=bdx)
        Xs = batch_corrector(values=self._la.tonp(self._X[weight]), index=adx)
        Ys = batch_corrector(values=self._la.tonp(self._Y[weight]), index=bdx)
        return {'A': As, 'B': Bs, 'X': Xs, 'Y': Ys}

    def cascade(self, pandalize=False):
        if not (self._is_built and (self._fluxes is not None)):
            raise ValueError('first build model and set fluxes!')

        self.reset_state()

        for weight, X in self._X.items():
            A, B = self._build_A_B_by_weight(weight=weight)
            if A.shape[1] == 0:
                continue
            Y = self._build_Y(weight=weight)
            LU = self._la.LU(A=A)
            self._LUA[weight] = LU  # NOTE: store for computation of Jacobian
            A_B = self._la.solve(LU=LU, b=B)
            X = A_B @ Y
            self._X[weight] += X
        state = self._format_return(s=self._s, derivative=False)
        if pandalize:
            state = pd.DataFrame(self._la.tonp(state), index=self._fcm.samples_id, columns=self.state_id)
        return state

    def _build_dYdvi(self, weight):
        if weight == next(iter(self._dYdv.keys())): # NOTE: all 0s anyways
            return self._dYdv[weight]

        self._dYdv[weight][:] = 0.0
        for i, yemu in enumerate(self._yemus[weight]):
            if isinstance(yemu, EMU):
                continue
            for j, xemu_a in enumerate(yemu._emus): # NOTE: this way we can convolve more than 2 mdvs!
                _, dmdv_a, emu_row_a = self._emu_indices[xemu_a]
                dmdvdv_a = dmdv_a[:, emu_row_a, :] # NOTE: selecting the derivative of an emu
                mdv = None
                for k, xemu_b in enumerate(yemu._emus):
                    if k == j:
                        continue
                    tensor_b, _, emu_row_b = self._emu_indices[xemu_b]
                    mdv_b = tensor_b[:, emu_row_b, :]
                    if mdv is None:
                        mdv = mdv_b
                    else:
                        mdv = self._la.convolve(a=mdv, v=mdv_b)
                self._dYdv[weight][:, i, :] += self._la.convolve(a=mdv, v=dmdvdv_a)
        return self._dYdv[weight]

    def dsdv(self, reaction_i: EMU_Reaction):
        super().dsdv(reaction_i=reaction_i)

        for weight, X in self._X.items():
            dBdvi = reaction_i.B_tensors.get(weight)
            dAdvi = reaction_i.A_tensors.get(weight)

            dYdvi = self._build_dYdvi(weight=weight)
            dXdvi = self._dXdv[weight]

            B = self._B_tot[weight]
            Y = self._Y[weight]

            if dBdvi is None:
                dBdv_Y = self._la.get_tensor(shape=X.shape)
            else:
                dBdv_Y = dBdvi @ Y

            B_dYdvi = B @ dYdvi

            if dAdvi is None:
                dAdv_X = 0.0 * X
            else:
                dAdv_X = dAdvi @ X

            lhs = dBdv_Y + B_dYdvi - dAdv_X
            LU = self._LUA.get(weight)

            if LU is not None:
                dXdvi[:] = self._la.solve(LU=LU, b=lhs)
        return self._format_return(s=self._dsdv, derivative=True)


# NB this class is needed to make pickling attribute lookup work!
class RatioEMU_Model(EMU_Model, RatioMixin): pass


if __name__ == "__main__":
    # from pta.sampling.tfs import sample_drg
    from sbmfi.settings import BASE_DIR
    from sbmfi.inference.priors import *
    from sbmfi.models.build_models import build_e_coli_tomek, build_e_coli_anton_glc
    from sbmfi.models.small_models import spiro

    import pickle

    model, kwargs = build_e_coli_anton_glc(batch_size=2)
    sdf = kwargs['substrate_df'].loc[['[1]Glc']]
    adf = kwargs['anton']['annotation_df']
    free_id = ['EX_glc__D_e', 'EX_ac_e', 'biomass_rxn']
    model.build_simulator(free_reaction_id=free_id)
    f = pd.read_excel(r"C:\python_projects\pysumo\src\sumoflux\estimate\f2.xlsx", index_col=0).iloc[:2]
    model.set_fluxes(f)
    model.cascade()
    s = model.state


    # fcm = FluxCoordinateMapper(model=m)
    # up = UniFluxPrior(fcm)
    # t, f = up.sample_dataframes(n=n)
    # pickle.dump((t,f), open('tf.p', 'wb'))
    #
    # t,f = pickle.load(open('tf.p', 'rb'))
    # weight = 3
    # t = t.iloc[:n]
    # f = f.iloc[:n]
    #
    # m.set_fluxes(f)
    # m.cascade()
    # pp1 = m.pretty_cascade(weight)
    # s1 = m.state
    #
    # torch_fcm = FluxCoordinateMapper(model=m, linalg=LinAlg(backend='torch', auto_diff=True))
    # theta = torch.from_numpy(t.values).requires_grad_(True)
    # ft = torch_fcm.map_theta_2_fluxes(theta, return_thermo=True)
    #
    # m.compute_jacobian()
    # jacuito = torch_fcm.free_jacobian(jacobian=m._jacobian, thermo_fluxes=ft)
    # jacuito = m._la.tonp(jacuito)
    # framed_jacs = [pd.DataFrame(sub_jac, index=m._fcm.theta_id, columns=m.state_id) for sub_jac in jacuito]
    # jacuito = pd.concat(framed_jacs, keys=m._fcm._samples_id)
    #
    # ftt = pd.DataFrame(ft.detach().numpy(), columns=m._fcm.thermo_fluxes_id)
    # f3 = torch_fcm.map_thermo_2_fluxes(ft)
    #
    # t_ft = m._la.diff(inputs=theta, outputs=ft)
    # # t_ft = m._la.tonp(t_ft)
    # # framed_jacs = [pd.DataFrame(sub_jac, index=m._fcm.theta_id, columns=m._fcm.thermo_fluxes_id) for sub_jac in t_ft]
    # # t_ft = pd.concat(framed_jacs, keys=m._fcm._samples_id)
    #
    # ft = torch.from_numpy(ftt.values).requires_grad_(True)
    # f3 = m._fcm.map_thermo_2_fluxes(ft)
    # ft_f = m._la.diff(inputs=ft, outputs=f3)
    # # ft_f = m._la.tonp(ft_f)
    # # framed_jacs = [pd.DataFrame(sub_jac, index=m._fcm.thermo_fluxes_id, columns=m._fcm.fluxes_id) for sub_jac in ft_f]
    # # ft_f = pd.concat(framed_jacs, keys=m._fcm._samples_id)
    #
    #
    # m.set_fluxes(f3)
    # m.cascade()
    # # pp3 = m._pretty_cascade_at_weight(weight)
    # # s3 = m.state
    #
    # jac = m._la.diff(inputs=f3, outputs=m._format_return(s=m._s))
    # # jac = m._la.tonp(jac)
    # # framed_jacs = [pd.DataFrame(sub_jac, index=m._fcm.fluxes_id, columns=m.state_id) for sub_jac in jac]
    # # jac = pd.concat(framed_jacs, keys=m._fcm._samples_id)
    #
    # jac = t_ft @ ft_f @ jac
    # jac = m._la.tonp(jac)
    # framed_jacs = [pd.DataFrame(sub_jac, index=m._fcm.theta_id, columns=m.state_id) for sub_jac in jac]
    # jac = pd.concat(framed_jacs, keys=m._fcm._samples_id)
    #
    #
    # jac2 = m._la.diff(inputs=theta, outputs=m._format_return(s=m._s))
    # jac2 = m._la.tonp(jac2)
    # framed_jacs = [pd.DataFrame(sub_jac, index=m._fcm.theta_id, columns=m.state_id) for sub_jac in jac2]
    # jac2 = pd.concat(framed_jacs, keys=m._fcm._samples_id)
