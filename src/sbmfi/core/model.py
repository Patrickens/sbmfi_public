from cobra.util.context import get_context
from cobra import Model, Reaction, Metabolite, DictList
import numpy as np
import math
import sys
import operator
import pandas as pd
from sbmfi.core.linalg import LinAlg
from sbmfi.core.util   import (
    _read_atom_map_str_rex,
    _find_biomass_rex,
    _rev_reactions_rex,
    _biomass_coeff_rex,
)
from sbmfi.core.polytopia import (
    extract_labelling_polytope,
    thermo_2_net_polytope,
    fast_FVA
)
from sbmfi.core.coordinater import FluxCoordinateMapper
from sbmfi.core.reaction import LabellingReaction, EMU_Reaction
from sbmfi.core.metabolite  import LabelledMetabolite, ConvolutedEMU, EMU, IsoCumo
from sbmfi.lcmsanalysis.formula import Formula
from itertools import repeat
from typing import Iterable, Union, Optional
from abc import abstractmethod
from copy import deepcopy
import pickle


def create_full_metabolite_kwargs(
        reaction_kwargs: dict,
        metabolite_kwargs: dict,
        infer_formula=True,
        add_cofactors=False,
):
    metabolite_kwargs = metabolite_kwargs.copy()
    for reac_id, kwargs in reaction_kwargs.items():
        reaction_string = kwargs.get('atom_map_str', None)
        if reaction_string is None:
            reaction_string = kwargs.get('reaction_str', None)

        if reaction_string is not None:
            rects, arrow, prods = _read_atom_map_str_rex.findall(string=reaction_string)[0]
            all_mets = [inter.strip().split('/') for inter in (rects + '+' + prods).split('+')]
            for met_atom in all_mets:
                if len(met_atom) == 1:
                    met_id = met_atom[0]
                    if met_id in ['∅', 'biomass']:
                        continue
                    if add_cofactors:
                        metabolite_kwargs[met_id] = metabolite_kwargs.get(met_id, {})
                    continue
                met_id, atoms = met_atom
                if met_id in metabolite_kwargs:
                    met_kwargs = metabolite_kwargs.get(met_id)
                    if 'formula' in met_kwargs:
                        met_formula = Formula(met_kwargs.get('formula'))
                        if met_formula['C'] != len(atoms):
                            raise ValueError
                    elif infer_formula:
                        met_kwargs['formula'] = f'C{len(atoms)}'
                else:
                    metabolite_kwargs[met_id] = {}
                    if infer_formula:
                        metabolite_kwargs[met_id]['formula'] = f'C{len(atoms)}'
    return metabolite_kwargs


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
            model: Model,
    ):
        if isinstance(model, LabellingModel):
            raise NotImplementedError
        elif not isinstance(model, Model):
            raise ValueError('Need to instantiate with an existing cobra model')
        super(LabellingModel, self).__init__(id_or_model=model)
        self._la = linalg

        # flags
        self._is_built = False  # signals that the all the variables and matrices have not been built yet

        # flux variables
        self._fluxes = None
        self._fcm: FluxCoordinateMapper = None  # map fluxes in some coordinate system and get back fwd_rev fluxes
        self._only_rev = {}  # ids of always net reverse reactions for easy mapping of flux coordonates

        # tolerances
        self.tolerance = 1e-9  # needed to have decent flux sampling results; default tol=1e-6

        # substrate labelling variables
        self._substrate_labelling = {}
        self._labelling_id: str = None
        self._labelling_repo: dict = {}  # repository of all labellings that we encountered

        # collections of metabolites
        self._measurements = DictList()  # these are the metabolites/ EMUs that we simulate labelling for since they are measured
        self.pseudo_metabolites = DictList()  # all the products of pseudo reactions e.g. all amino acids

        # collections of reactions of various sorts
        self._biomass_id: str = None
        self.pseudo_reactions     = DictList()  # used to simulate the labelling of products of linear pathways e.g. amino acids
        self._labelling_reactions = DictList()  # reactions for which all reactants and products are present and carry carbon
        self._free_reaction_id = []

        self._initialize_state()  # sets even more attributes; function is reused when building the model

        self.groups = DictList() # TODO: no functionality has been implemented or tested for groups

    def __setstate__(self, state):
        # needed for initialization of LabellingModel with Model, since it calls __setstate__
        self.pseudo_reactions = DictList()
        self.pseudo_metabolites = DictList()

        super(LabellingModel, self).__setstate__(state)

        for r in self.reactions:
            # not sure why, but pickling messes up the whole solver...
            r.update_variable_bounds()

        self.repair()
        substrate_labelling = state.get('_substrate_labelling', None)
        if substrate_labelling is not None:
            self.set_substrate_labelling(substrate_labelling=substrate_labelling)
        measurements = state.get('_measurements', None)
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
        odict['_substrate_labelling'] = self.substrate_labelling
        odict['_labelling_repo'] = {}  # TODO, I think storing this would be too much ugly code
        odict['_measurements'] = self._measurements.list_attr('id')
        odict['_metabolites_in_state'] = None

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
    def substrate_labelling(self):
        """entity can be IsoCumo or EMU"""
        return pd.Series(dict((isocumo.id, frac) for isocumo, frac in self._substrate_labelling.items()),
                         name=self._labelling_id, dtype=np.float64).round(4)

    @property
    def substrate_metabolites(self):
        return DictList(set([entity.metabolite for entity in self._substrate_labelling.keys()]))

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
            if isinstance(reaction, LabellingReaction) and ((lb, ub) != (0.0, 0.0)):
                if reaction.rho_max > 0.0:
                    self._labelling_reactions.append(reaction)
                    self._labelling_reactions.append(reaction._rev_reaction)
                elif lb >= 0.0:
                    self._labelling_reactions.append(reaction)
                elif ub <= 0.0:
                    self._labelling_reactions.append(reaction._rev_reaction)
                    self._only_rev[reaction._rev_reaction.id] = reaction.id
                    for metabolite in reaction._metabolites:
                        # this has to happen here, since this is where _only_rev is created!
                        if reaction in metabolite._reaction:
                            metabolite._reaction.remove(reaction)
                        metabolite._reaction.add(reaction._rev_reaction)

        self._jacobian = self._la.get_tensor(
            shape=(self._la._batch_size, len(self._labelling_reactions), self.state_id.shape[0])
        )
        return self._labelling_reactions

    @property
    def flux_coordinate_mapper(self) -> FluxCoordinateMapper:
        if not self._is_built:
            raise ValueError('build the model first!')
        return self._fcm

    def set_fluxes(self, labelling_fluxes: Union[pd.DataFrame, np.array, 'torch.Tensor'], samples_id=None, trim=True):
        if not self._is_built:
            raise ValueError('MUST BUILD')
        labelling_fluxes = self._fcm.frame_fluxes(labelling_fluxes, samples_id, trim)
        if len(labelling_fluxes.shape) > 2:
            raise ValueError('can only deal with 2D stratified fluxes!')
        if labelling_fluxes.shape[0] != self._la._batch_size:
            raise ValueError(f'batch_size = {self._la._batch_size}; fluxes.shape[0] = {labelling_fluxes.shape[0]}')
        self._fluxes = labelling_fluxes

    def set_substrate_labelling(self, substrate_labelling: pd.Series):
        self._substrate_labelling = {}
        self._labelling_id = substrate_labelling.name
        for isotopomer_str, frac in substrate_labelling.items():
            if frac == 0.0:
                continue
            met_id, label = isotopomer_str.rsplit('/')
            metabolite = self.metabolites.get_by_id(id=met_id)
            if hasattr(metabolite, 'isotopomers'):
                isotopomer = metabolite.isotopomers.get_by_id(isotopomer_str)
            else:
                isotopomer = IsoCumo(metabolite=self.metabolites.get_by_id(id=met_id), label=label)
            self._substrate_labelling[isotopomer] = frac

        fractions = np.fromiter(self._substrate_labelling.values(), dtype=np.double)
        if any(fractions < 0.0) or any(fractions > 1.0):
            raise ValueError('Negative or over 1 value in substrate labelling')

        isotopomers = np.array(list(self._substrate_labelling.keys()))
        substrate_metabolites = np.array([ic.metabolite for ic in isotopomers])
        for metabolite in set(substrate_metabolites):
            sum_met = fractions[substrate_metabolites == metabolite].sum()
            if not math.isclose(a=sum_met, b=1.0, abs_tol=1e-4):
                raise ValueError(f'substrate labeling fractions of metabolite {metabolite.id} do not sum up to 1.0')
            fractions[substrate_metabolites == metabolite] /= sum_met  # makes sum closer to 1
        self._substrate_labelling = dict((key, frac) for key, frac in zip(isotopomers, fractions))

        substrate_reactions = DictList()
        for metabolite in set(self.substrate_metabolites):
            for reaction in metabolite.reactions:
                if reaction.boundary and isinstance(reaction, LabellingReaction) and not reaction.pseudo:
                    if not reaction.rho_max == 0.0:
                        raise ValueError(f'substrate reaction is illegaly reversible {reaction.id}')
                    if reaction.lower_bound >= 0.0:
                        substrate_reactions.append(reaction)
                    elif reaction.upper_bound <= 0.0:
                        substrate_reactions.append(reaction._rev_reaction)
                    else:
                        raise ValueError(f'substrate reaction {reaction.id} '
                                         f'for metabolite {metabolite.id} has (0, 0) bounds')
            if not any([reaction in substrate_reactions for reaction in metabolite.reactions]):
                raise ValueError(f'metabolite {metabolite.id} has no substrate reactions')

        self._labelling_repo[substrate_labelling.name] = dict(_substrate_labelling=self._substrate_labelling)

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
            elif (abs(reaction.upper_bound - reaction.lower_bound) < self._tolerance) and (reaction.id not in self._only_rev):
                zero_facet.append(reaction)
            elif (reaction.id in self._only_rev and (abs(revr.upper_bound - revr.lower_bound) < self._tolerance)):
                zero_facet.append(reaction)
            elif (reaction.id in free_reaction_id) or (self._only_rev.get(reaction.id) in free_reaction_id):
                user_chosen.append(reaction)
            elif reaction.boundary:
                boundary.append(reaction)
            else:
                fwd.append(reaction)
        user_chosen.sort(key=lambda x: \
            free_reaction_id.index(_rev_reactions_rex.sub('', x.id)) if x.id not in free_reaction_id else x.id
                         )
        self._free_reaction_id = user_chosen.list_attr('id')
        self._labelling_reactions = fwd + boundary + bm + user_chosen + zero_facet + rev

    def repair(
        self, rebuild_index: bool = True, rebuild_relationships: bool = True
    ) -> None:
        super(LabellingModel, self).repair(rebuild_index, rebuild_relationships)
        if rebuild_relationships:
            for metabolite in self.pseudo_metabolites:
                metabolite._reaction.clear()
                metabolite._model = self

            for reaction in self.pseudo_reactions:
                reaction._model = self
                for metabolite in reaction._metabolites:
                    metabolite._reaction.add(reaction)

            for reaction in self.reactions:
                if isinstance(reaction, LabellingReaction) and (reaction.rho_max > 0.0):
                    rev_reaction = reaction._rev_reaction
                    rev_reaction._model = self
                    for metabolite in rev_reaction._metabolites:
                        metabolite._reaction.add(rev_reaction)
                        # during picking, we completely delete _rev_reaction and after its recreation
                        #    thats why we need to put the atom_map back in here
                        reaction._rev_reaction.set_atom_map(atom_map=dict([
                            (met, (-stoich, atoms)) for met, (stoich, atoms) in reaction._atom_map.items()
                        ]))

    def add_labelling_kwargs(self, reaction_kwargs, metabolite_kwargs):
        self._labelling_reactions = DictList()  # dynamically recomputed
        self._is_built = False

        # In a first step we convert all cobra metabolites to LabellingMetabolite objecte
        metabolite_kwargs = create_full_metabolite_kwargs(
            reaction_kwargs, metabolite_kwargs, infer_formula=True, add_cofactors=False
        )
        for met_id, kwargs in metabolite_kwargs.items():
            if met_id not in self.metabolites:
                raise ValueError(f'All metabolites should be in the model before processing labelling info {met_id}')
            # we allow for formula to be changed for instance for CoA which has 28 carbons, but 1 participating in
            #   labelling reactions; the rest of the attributes we assume to have been set in the model at instantiation
            #   of this model
            metabolite = self.metabolites.get_by_id(met_id)
            self.metabolites._replace_on_id(new_object=self._TYPE_REACTION._TYPE_METABOLITE(
                metabolite=metabolite, symmetric=kwargs.get('symmetric', False), formula=kwargs.get('formula', None)
            ))

        # Next, we convert cobra Reactions to LabellingReactions and do all the atom mapping and split of pseudo reactions
        for reac_id, kwargs in reaction_kwargs.items():
            if reac_id not in self.reactions:
                raise ValueError(f'All reactions should be in the model before processing labelling info: {reac_id}')
            reaction = self.reactions.get_by_id(reac_id)
            if 'upper_bound' in kwargs:
                ub = kwargs['upper_bound']
                if ub != reaction.upper_bound:
                    reaction.upper_bound = ub
            if 'lower_bound' in kwargs:
                lb = kwargs['lower_bound']
                if lb != reaction.lower_bound:
                    reaction.lower_bound = lb

            new_metabolites = {}
            for metabolite, stoich in reaction.metabolites.items():
                new_metabolites[self.metabolites.get_by_id(metabolite.id)] = stoich
            reaction._metabolites = new_metabolites
            if 'atom_map_str' in kwargs:
                reaction = self._TYPE_REACTION(
                    reaction, rho_min=kwargs.get('rho_min', None),
                    rho_max=kwargs.get('rho_max', None), pseudo=kwargs.get('pseudo', False)
                )
                atom_map, is_biomass = reaction.build_atom_map_from_string(kwargs['atom_map_str'])
                if is_biomass:
                    if self._biomass_id is not None:
                        raise ValueError('more than 1 biomass in the atom_mapt_str of reaction_kwargs')
                    self._biomass_id = reac_id
                reaction.set_atom_map(atom_map)
                if reaction.pseudo:
                    products = reaction.products
                    cons_vars = [reaction.forward_variable, reaction.reverse_variable]
                    for metabolite in products:
                        if metabolite in self.pseudo_metabolites:
                            raise ValueError('more than 1 pseudo_reaction producing this metabolite'
                                             'by definition impossible')
                        cons_vars.append(self.solver.constraints[metabolite.id])
                        self.pseudo_metabolites.append(metabolite)
                        self.metabolites.remove(metabolite.id)
                    self.remove_cons_vars(cons_vars)
                    self.reactions.remove(reac_id)
                    self.pseudo_reactions.append(reaction)
                else:
                    self.reactions._replace_on_id(new_object=reaction)

        self.solver.update()  # due to the cons_vars
        self.repair()

    def add_reactions(self, reaction_list: Iterable[Reaction]) -> None:
        for reaction in reaction_list:
            if hasattr(reaction, '_pseudo') and reaction.pseudo:
                raise ValueError('This is a pseudo reaction')
        super(LabellingModel, self).add_reactions(reaction_list)

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
            if metabolite in self.substrate_metabolites:
                print('removing substrate metabolite for which labelling is set!')
                self._substrate_labelling = {}
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

        super(LabellingModel, self).remove_metabolites(metabolite_list=metabolite_list, destructive=destructive)
        self._is_built = False

    def merge(
        self,
        right: "Model",
        prefix_existing: Optional[str] = None,
        inplace: bool = True,
        objective: str = "left",
    ) -> "Model":
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def add_groups(self, group_list):
        raise NotImplementedError

    def remove_groups(self, group_list):
        raise NotImplementedError

    def copy(self) -> 'LabellingModel':
        # NB this will delete all things associated with build_simulator, but keeps polytope intact
        return pickle.loads(pickle.dumps(self))

    def reset_state(self):
        self._dsdv[:] = 0.0
        self._jacobian[:] = 0.0

    def dsdv(self, reaction_i: LabellingReaction):
        self._dsdv[:] = 0.0
        if self._fluxes is None:
            raise ValueError('no fluxes')

    def compute_jacobian(self, dept_reactions_idx: np.array = None):
        if self._fluxes is None:
            raise ValueError('no fluxes')

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
        if not self._labelling_reactions:
            self.labelling_reactions  # necessary to fill _rev_reactions, which otherwise trip up the line below
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

    def prepare_polytopes(self, free_reaction_id=None, verbose=False):
        if len(self._substrate_labelling) == 0:
            raise ValueError('set substrate labelling first!')  # need to have set labelling before generating system!

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

            if metabolite in self.substrate_metabolites:
                emus = self._yemus
            else:
                emus = self._xemus
            for i in range(1, met_weight + 1):
                self._xemus.setdefault(i, DictList())
                self._yemus.setdefault(i, DictList())
            emus[met_weight].append(met_emu)
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

    def set_substrate_labelling(self, substrate_labelling: pd.Series):
        labelling_id = substrate_labelling.name
        settings = self._labelling_repo.get(labelling_id, None)
        if settings is None:
            super().set_substrate_labelling(substrate_labelling=substrate_labelling)
            if len(self._yemus) > 0:
                self._initialize_Y()
        else:
            self._labelling_id = labelling_id
            self._substrate_labelling = settings['_substrate_labelling']
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
        substrate_metabolites = self.substrate_metabolites
        state_reactions = self.labelling_reactions + self.pseudo_reactions
        for weight, xemus in reversed(self._xemus.items()):
            for product_emu in xemus:
                for reaction in product_emu.metabolite.reactions:
                    if (type(reaction) != EMU_Reaction) or (reaction not in state_reactions) or \
                            (not reaction.gettants()) or (not reaction.gettants(reactant=False)):
                        continue
                    if product_emu.metabolite in reaction.gettants(reactant=False):  # map product
                        emu_reaction_elements = reaction.map_reactants_products(
                            product_emu=product_emu, substrate_metabolites=substrate_metabolites
                        )
                        for (stoich, prod, rect) in emu_reaction_elements:
                            for emu in rect.getmu():
                                if isinstance(emu, ConvolutedEMU) or (emu.metabolite in substrate_metabolites):
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
                for isocumo, fraction in self._substrate_labelling.items():
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
                # this is when a built model gets new substrate labelling
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
        for (weight, xemus), yemus in zip(self._xemus.items(), self._yemus.values()):
            for emu in (yemus + xemus):
                if isinstance(emu, ConvolutedEMU):
                    continue
                elif emu.metabolite in self.substrate_metabolites:
                    matrix = self._Y[emu.weight]
                    dmdv = self._dYdv[emu.weight]
                    row = self._yemus[emu.weight].index(emu)
                else:
                    matrix = self._X[emu.weight]
                    dmdv = self._dXdv[emu.weight]
                    row = self._xemus[emu.weight].index(emu)
                self._emu_indices[emu] = matrix, dmdv, row

    def build_model(self, free_reaction_id=None, verbose=False):
        super().build_model(free_reaction_id, verbose)
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
                continue  # skip substrate emus, only deal with convolvedEMUs
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


def model_builder_from_dict(
        reaction_kwargs: dict,
        metabolite_kwargs: dict,
        model_id='model',
        name=None,
) -> Model:
    reaction_kwargs = reaction_kwargs.copy()
    model = Model(id_or_model=model_id, name=name)
    biomass_kwargs = reaction_kwargs.pop('biomass', None)
    metabolite_kwargs = create_full_metabolite_kwargs(reaction_kwargs, metabolite_kwargs, add_cofactors=True)
    # now metabolite_kwargs contains all metabolites that should be tranformed into labelling metabolites
    metabolite_list = DictList()
    for met_id, kwargs in metabolite_kwargs.items():
        metabolite_list.append(
            Metabolite(
                met_id, formula=kwargs.get('formula'), name=kwargs.get('name', None),
                charge=kwargs.get('charge', None), compartment=kwargs.get('compartment', 'c'),
            )
        )
    model.add_metabolites(metabolite_list=metabolite_list)

    def count_items(dct, lst, add=True):
        oprat = operator.add if add else operator.sub
        for item in lst:
            if item in ['∅', 'biomass']:
                continue
            item = metabolite_list.get_by_id(item)
            dct[item] = oprat(dct.get(item, 0), 1)
        return dct

    reaction_list = DictList()
    for reac_id, kwargs in reaction_kwargs.items():
        reaction_string = kwargs.get('atom_map_str', None)
        if reaction_string is None:
            reaction_string = kwargs.get('reaction_str', None)
        rects, arrow, prods = _read_atom_map_str_rex.findall(string=reaction_string)[0]
        rects = [rect.split('/')[0].strip() for rect in rects.split('+')]
        prods = [prod.split('/')[0].strip() for prod in prods.split('+')]
        metabolites = {}
        count_items(metabolites, rects, False)
        count_items(metabolites, prods)
        reac = Reaction(
            reac_id, name=kwargs.get('name', None), subsystem=kwargs.get('name', None),
            lower_bound=kwargs.get('lower_bound', 0.0), upper_bound=kwargs.get('upper_bound', 1000.0),
        )
        reac.add_metabolites(metabolites_to_add=metabolites)
        reaction_list.append(reac)

    if biomass_kwargs is not None:
        biomass_coeff = _biomass_coeff_rex.findall(biomass_kwargs['reaction_str'])
        biomass_coeff = {model.metabolites.get_by_id(k): -float(v) for v, k in biomass_coeff}
        bm_reac = Reaction(
            'biomass', name=biomass_kwargs.get('name', None), subsystem=biomass_kwargs.get('name', None),
            lower_bound=biomass_kwargs.get('lower_bound', 0.0), upper_bound=biomass_kwargs.get('upper_bound', 1000.0),
        )
        bm_reac.add_metabolites(metabolites_to_add=biomass_coeff)
        reaction_list.append(bm_reac)

    model.add_reactions(reaction_list=reaction_list)
    return model


if __name__ == "__main__":
    # from pta.sampling.tfs import sample_drg
    from sbmfi.settings import BASE_DIR
    # from sbmfi.priors.uniform import *
    # from sbmfi.models.build_models import build_e_coli_tomek, build_e_coli_anton_glc
    # from sbmfi.models.small_models import spiro

    import pickle

    pd.set_option('display.max_columns', None)

    v5_reversible = True
    v2_reversible = True
    add_biomass = True
    add_cofactor = True
    C_symmetric = True
    which_labellings = ['A','B']
    L_12_omega = 1.0
    measured_boundary_fluxes = ('h_out', )
    ratios=False
    build_simulator=True

    if v5_reversible:
        v5_atom_map_str = 'F/a + D/bcd  <== C/abcd'
    else:
        v5_atom_map_str = 'F/a + D/bcd  <-- C/abcd'

    reaction_kwargs = {
        'biomass': {
            'reaction_str': '0.3H + 0.6B + 0.5E + 0.1C --> ∅',
            'atom_map_str': 'biomass --> ∅',
        },
        'a_in': {
            'lower_bound': 10.0, 'upper_bound': 10.0,
            'atom_map_str': '∅ --> A/ab'
        },
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
        'cof_out': {
            'upper_bound': 100.0,
            'reaction_str': 'cof --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/ab --> B/ab'
        },
        'v2': {
            'lower_bound': 0.0, 'upper_bound': 100.0,
            'rho_min': 0.1, 'rho_max': 0.8,
            'atom_map_str': 'B/ab ==> E/ab'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab + E/cd --> C/abcd'
        },
        'v4': {
            'upper_bound': 100.0,  # 'lower_bound': -10.0,
            'atom_map_str': 'E/ab --> H/ab'
        },
        'v5': {  # NB this is an always reverse reaction!
            'lower_bound': -100.0, 'upper_bound': 0.0,
            'atom_map_str': v5_atom_map_str,  # <--  ==>
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
            'lower_bound': 0.0,
            'pseudo': True,
            'atom_map_str': 'C/abcd + D/efg + H/hi --> L/abgih'
        },
    }
    metabolite_kwargs = {
        'A': {'formula': 'C2H4O5'},
        'B': {'formula': 'C2HPO3'},
        'C': {'formula': 'C4H6N4OS', 'symmetric': C_symmetric},
        'D': {'formula': 'C3H2'},
        'E': {'formula': 'C2H4O5'},
        'F': {'formula': 'CH2'},
        'G': {'formula': 'CH2'},  # unused metabolite
        'H': {'formula': 'C2H2'},
        'L': {'formula': 'C5KNaSH'},  # pseudo-metabolite
        'L|[1,2]': {'formula': 'C2H2O7'},  # pseudo-metabolite
    }
    ratio_repo = {
        'E|v2': {
            'numerator': {'v2': 1},
            'denominator': {'v2': 1, 'v6': 1}
        },
        'H|v4': {
            'numerator': {'v4': 1},
            'denominator': {'v7': 1, 'v4': 1}
        },
        # 'denominator': {'v6': 1, 'v4': 1}},  # make ratios correlated
    }

    if not add_biomass:
        reaction_kwargs.pop('biomass')

    if not v2_reversible:
        reaction_kwargs['v2'] = {
            'lower_bound': 0.0, 'upper_bound': 100.0,
            'atom_map_str': 'B/ab --> E/ab'
        }
        ratio_repo['E|v2'] = {
            'numerator': {'v2': 1},
            'denominator': {'v2': 1, 'v6': 1},
        }
    if add_cofactor:
        reaction_kwargs['v3']['atom_map_str'] = 'B/ab + E/cd --> C/abcd + cof'
        reaction_kwargs['cof_out'] = {'reaction_str': 'cof --> ∅', 'upper_bound': 100.0}

    substrate_df = pd.DataFrame([
        [0.2, 0.0, 0.0, 0.8],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.8, 0.2],
    ], columns=['A/00', 'A/01', 'A/10', 'A/11'], index=list('ABCDE'))

    if which_labellings is not None:
        substrate_df = substrate_df.loc[which_labellings]

    annotation_df = pd.DataFrame([
        ('H', 1, 'M-H', 3.0, 1.0, 0.01, None, 3e3),
        ('H', 0, 'M-H', 2.0, 1.0, 0.01, None, 3e3),

        ('H', 1, 'M+F', 5.0, 1.0, 0.03, None, 3e3),
        ('H', 1, 'M+Cl', 88.0, 1.0, 0.03, None, 2e3),
        ('H', 0, 'M+F', 4.0, 1.0, 0.03, None, 3e3),  # to indicate that da_df is not yet in any order!
        ('H', 0, 'M+Cl', 89.0, 1.0, 0.03, None, 2e3),

        ('Q', 1, 'M-H', 3.7, 3.0, 0.02, None, 2e3),  # an annotated metabolite that is not in the model
        ('Q', 2, 'M-H', 4.7, 3.0, 0.02, None, 2e3),
        ('Q', 3, 'M-H', 5.7, 3.0, 0.02, None, 2e3),

        ('C', 0, 'M-H', 1.5, 4.0, 0.02, None, 7e5),
        ('C', 3, 'M-H', 4.5, 4.0, 0.02, None, 7e5),
        ('C', 4, 'M-H', 5.5, 4.0, 0.02, None, 7e5),

        ('D', 2, 'M-H', 12.0, 5.0, 0.01, None, 1e5),
        ('D', 0, 'M-H', 9.0, 5.0, 0.01, None, 1e5),
        ('D', 3, 'M-H', 13.0, 5.0, 0.01, None, 1e5),

        ('L|[1,2]', 0, 'M-H', 14.0, 6.0, 0.01 * L_12_omega, L_12_omega, 4e4),  # a scaling factor other than 1.0
        ('L|[1,2]', 1, 'M-H', 15.0, 6.0, 0.01 * L_12_omega, L_12_omega, 4e4),

        ('L', 0, 'M-H', 14.0, 6.0, 0.01, None, 4e5),
        ('L', 1, 'M-H', 15.0, 6.0, 0.01, None, 4e5),
        ('L', 2, 'M-H', 16.0, 6.0, 0.01, None, 4e5),
        ('L', 5, 'M-H', 19.0, 6.0, 0.01, None, 4e5),
    ], columns=['met_id', 'nC13', 'adduct_name', 'mz', 'rt', 'sigma', 'omega', 'total_I'])
    formap = {k: v['formula'] for k, v in metabolite_kwargs.items()}
    annotation_df['formula'] = annotation_df['met_id'].map(formap)

    biomass_id = 'biomass' if add_biomass else None
    if add_biomass:
        measured_boundary_fluxes = list(measured_boundary_fluxes)
        measured_boundary_fluxes.append(biomass_id)

    model = model_builder_from_dict(reaction_kwargs, metabolite_kwargs, model_id='spiro', name='spiralus')
    linalg = LinAlg(
        backend='torch', batch_size=1, solver='lu_solve', device='cpu',
        fkwargs=None, seed=2
    )
    if ratios:
        model_type = RatioEMU_Model
    else:
        model_type = EMU_Model

    model = model_type(linalg=linalg, model=model)
    model.add_labelling_kwargs(
        reaction_kwargs=reaction_kwargs,
        metabolite_kwargs=metabolite_kwargs
    )
    if (ratio_repo is not None) and ratios:
        model.set_ratio_repo(ratio_repo=ratio_repo)
    model.set_substrate_labelling(substrate_labelling=substrate_df.iloc[0])
    model.set_measurements(measurement_list=annotation_df['met_id'].unique())
    # if build_simulator:
    #     model.build_model(free_reaction_id=measured_boundary_fluxes)

    if add_biomass:
        fluxes = {
            'a_in': 10.00,
            'd_out': 0.00,
            'f_out': 0.00,
            'h_out': 7.60,
            'v1': 10.00,
            'v2': 1.80,
            'v2_rev': 0.90,
            'v3': 8.20,
            'v4': 0.00,
            'v5': 0.05,
            'v5_rev': 8.10,
            'v6': 8.05,
            'v7': 8.05,
            'biomass': 1.50,
        }
        bm = model.reactions.get_by_id('biomass')
        model.objective = {bm: 1}
    else:
        fluxes = {
            'a_in': 10.0,
            'd_out': 1.0,
            'f_out': 1.0,
            'h_out': 8.0,
            'v1': 10.0,
            'v2': 7.0,
            'v2_rev': 3.5,
            'v3': 7.0,
            'v4': 2.0,
            'v5': 3.0,
            'v5_rev': 10.0,
            'v6': 6.0,
            'v7': 6.0,
        }
        model.objective = {model.reactions.get_by_id('h_out'): 1}

    if not v2_reversible:
        fluxes['v2'] = fluxes['v2'] - fluxes.pop('v2_rev')

    if not v5_reversible:
        fluxes['v5_rev'] = fluxes['v5_rev'] - fluxes.pop('v5')

    if add_cofactor:
        fluxes['cof_out'] = fluxes['v3']
    fluxes = pd.Series(fluxes, name='v')

    model.build_model(free_reaction_id=measured_boundary_fluxes)
    model.set_fluxes(labelling_fluxes=fluxes)
    res = model.cascade(pandalize=True)
    # print(model.labelling_reactions)
    print('NOW COPYING')
    mm = model.copy()
    mm.build_model()
    mm.set_fluxes(labelling_fluxes=fluxes)
    res2 = mm.cascade(pandalize=True)
