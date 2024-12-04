import pandas as pd
from collections import OrderedDict
from sbmfi.core.model import LabellingModel, EMU_Model, RatioEMU_Model
from sbmfi.inference.bayesian import _BaseBayes
from sbmfi.inference.priors import UniNetFluxPrior
from sbmfi.core.observation import ClassicalObservationModel, LCMS_ObservationModel, MVN_BoundaryObservationModel, MDV_ObservationModel
from sbmfi.core.linalg import LinAlg
from sbmfi.models.build_models import simulator_factory, _correct_base_bayes_lcms
from sbmfi.settings import MODEL_DIR, SIM_DIR
from sbmfi.lcmsanalysis.util import _strip_bigg_rex
import sys, os
import cobra
from cobra.io import read_sbml_model
from cobra import Reaction, Metabolite, DictList, Model
from sbmfi.lcmsanalysis.formula import Formula

def spiro(
        backend='numpy',
        auto_diff=False,
        batch_size=1,
        add_biomass=True,
        v2_reversible=False,
        ratios=True,
        build_simulator=False,
        add_cofactors=True,
        which_measurements=None,
        seed=2,
        which_labellings=None,
        include_bom=True,
        v5_reversible=False,
        measured_boundary_fluxes = ('h_out', ),
        n_obs=0,
        kernel_basis='svd',
        basis_coordinates='rounded',
        logit_xch_fluxes=False,
        L_12_omega = 1.0,
        clip_min=None,
        transformation='ilr',
):
    # NOTE: this one has 2 interesting flux ratios!
    # NOTE this has been parametrized to exactly match the Wiechert fml file: C:\python_projects\pysumo\src\sumoflux\models\fml\spiro.fml
    #   which is a slightly modified version of the one found on: https://github.com/modsim/FluxML/blob/master/examples/models/spirallus_model_level_1.fml
    # for m in M.metabolites: # NOTE: needed to store as sbml
    #     m.compartment = 'c'
    # sbml_file = os.path.join(MODEL_DIR, 'sbml', 'spiro.xml')
    # json_file = os.path.join(MODEL_DIR, 'escher_input', 'model', 'spiro.json')
    # cobra.io.write_sbml_model(cobra_model=M, filename=sbml_file)
    # cobra.io.save_json_model(model=M, filename=json_file)
    if (which_measurements is not None) and not build_simulator:
        raise ValueError

    if v5_reversible:
        v5_atom_map_str = 'F/a + D/bcd  <== C/abcd'
    else:
        v5_atom_map_str = 'F/a + D/bcd  <-- C/abcd'

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
            'rho_min': 0.1, 'rho_max': 0.8,
            'atom_map_str': 'B/ab ==> E/ab'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab + E/cd --> C/abcd'
        },
        'v4': {
            'upper_bound': 100.0, # 'lower_bound': -10.0,
            'atom_map_str': 'E/ab --> H/ab'
        },
        # 'v5': {
        #     'upper_bound': 100.0,
        #     'atom_map_str': 'C/abcd --> F/a + D/bcd'
        # },
        'v5': {  # NB this is an always reverse reaction!
            'lower_bound': -100.0, # 'upper_bound': 100.0
            'atom_map_str': v5_atom_map_str,  # <--  ==>
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
            'lower_bound': 0.0, # 'upper_bound': 100.0,
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

    substrate_df = pd.DataFrame([
        [0.2, 0.0, 0.0, 0.8],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.8, 0.2],
    ], columns=['A/00', 'A/01', 'A/10', 'A/11'], index=list('ABCDE'))

    if which_labellings is not None:
        substrate_df = substrate_df.loc[which_labellings]

    ratio_repo = {
        'E|v2': {
            'numerator': OrderedDict({'v2': 1}),
            'denominator': OrderedDict({'v2': 1, 'v6': 1})},
        'H|v4': {
            'numerator': OrderedDict({'v4': 1}),
            'denominator': OrderedDict({'v7': 1, 'v4': 1})},
            # 'denominator': OrderedDict({'v6': 1, 'v4': 1})},  # make ratios correlated
    }

    if not v2_reversible:
        reaction_kwargs['v2'] = {
            'lower_bound': 0.0, 'upper_bound': 100.0,
            'atom_map_str': 'B/ab --> E/ab'
        }
        ratio_repo['E|v2']= {
            'numerator': OrderedDict({'v2': 1}),
            'denominator': OrderedDict({'v2': 1, 'v6': 1})
        }

    annotation_df = pd.DataFrame([
        ('H', 1, 'M-H', 3.0, 1.0, 0.01, None, 3e3),
        ('H', 0, 'M-H', 2.0, 1.0, 0.01, None, 3e3),

        ('H', 1, 'M+F', 5.0,   1.0, 0.03, None, 3e3),
        ('H', 1, 'M+Cl', 88.0, 1.0, 0.03, None, 2e3),
        ('H', 0, 'M+F', 4.0,   1.0, 0.03, None, 3e3),  # to indicate that da_df is not yet in any order!
        ('H', 0, 'M+Cl', 89.0, 1.0, 0.03, None, 2e3),

        ('P', 1, 'M-H', 3.7, 3.0, 0.02, None, 2e3),  # an annotated metabolite that is not in the model
        ('P', 2, 'M-H', 4.7, 3.0, 0.02, None, 2e3),
        ('P', 3, 'M-H', 5.7, 3.0, 0.02, None, 2e3),

        ('C', 0, 'M-H', 1.5, 4.0, 0.02, None, 7e5),
        ('C', 3, 'M-H', 4.5, 4.0, 0.02, None, 7e5),
        ('C', 4, 'M-H', 5.5, 4.0, 0.02, None, 7e5),

        ('D', 2, 'M-H', 12.0, 5.0, 0.01, None, 1e5),
        ('D', 0, 'M-H', 9.0,  5.0, 0.01, None, 1e5),
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

    biomass_id = 'bm' if add_biomass else None
    if add_biomass:
        measured_boundary_fluxes = list(measured_boundary_fluxes)
        measured_boundary_fluxes.append(biomass_id)
        measured_boundary_fluxes = tuple(measured_boundary_fluxes)

    model = simulator_factory(
        id_or_file_or_model='spiro',
        backend=backend,
        auto_diff=auto_diff,
        metabolite_kwargs=metabolite_kwargs,
        reaction_kwargs=reaction_kwargs,
        input_labelling=substrate_df.iloc[0],
        ratio_repo=ratio_repo,
        measurements=annotation_df['met_id'].unique(),
        batch_size=batch_size,
        ratios=ratios,
        build_simulator=build_simulator,
        seed=seed,
        free_reaction_id=measured_boundary_fluxes,
        kernel_basis=kernel_basis,
        basis_coordinates=basis_coordinates,
        logit_xch_fluxes=logit_xch_fluxes,
    )

    if add_biomass:
        bm = Reaction(id='bm', lower_bound=0.05, upper_bound=1.5)
        bm.add_metabolites(metabolites_to_add={
            model.metabolites.get_by_id('H'): -0.3,
            model.metabolites.get_by_id('B'): -0.6,
            model.metabolites.get_by_id('E'): -0.5,
            model.metabolites.get_by_id('C'): -0.1,
        })
        model.add_reactions(reaction_list=[bm], reaction_kwargs={bm.id: {'atom_map_str': 'biomass --> ∅'}})
        fluxes = {
            'a_in':   10.00,
            # 'a_in_rev':   10.00,
            'd_out':  0.00,
            'f_out':  0.00,
            'h_out':  7.60,
            'v1':     10.00,
            'v2':     1.80,
            'v2_rev': 0.90,
            'v3':     8.20,
            'v4':     0.00,
            'v5':     0.05,
            'v5_rev': 8.10,
            'v6':     8.05,
            'v7':     8.05,
            'bm':     1.50,
        }
        bm = model.reactions.get_by_id('bm')
        model.objective = {bm: 1}
    else:
        fluxes = {
            'a_in': 1.0,
            'd_out': 0.1,
            'f_out': 0.1,
            'h_out': 0.8,
            'v1': 1.0,
            'v2': 0.7,
            'v2_rev': 0.35,
            'v3': 0.7,
            'v4': 0.2,
            'v5': 0.3,
            'v5_rev': 1.0,
            'v6': 0.6,
            'v7': 0.6,
        }
        model.objective = {model.reactions.get_by_id('h_out'): 1}

    if not v2_reversible:
        fluxes['v2'] = fluxes['v2'] - fluxes.pop('v2_rev')

    if not v5_reversible:
        fluxes['v5_rev'] = fluxes['v5_rev'] - fluxes.pop('v5')

    if add_cofactors:
        cof = Metabolite('cof', formula='H2O')
        v3 = model.reactions.get_by_id('v3')
        v3.add_metabolites({cof: 1})
        ex_cof = Reaction('EX_cof', lower_bound=0.0, upper_bound=1000.0)
        ex_cof.add_metabolites({cof: -1})
        model.add_reactions([ex_cof])
        fluxes['EX_cof'] = fluxes['v3']

    fluxes = pd.Series(fluxes, name='v')
    if (batch_size == 1) and build_simulator:
        model.set_fluxes(fluxes=fluxes)

    observation_df = MDV_ObservationModel.generate_observation_df(model, annotation_df)
    annotation_df['mz'] = 0.0
    annotation_df.loc[observation_df['annot_df_idx'], 'mz'] = observation_df['isotope_decomposition'].apply(lambda x: Formula(x).mass()).values

    labelling_specific_annots = {
        'A': ['C+0', 'C+3', 'C+4', 'D+0', 'D+2', 'D+3', 'H+0', 'H+1', 'L+0', 'L+1', 'L+2', 'L+5', 'L|[1,2]+0', 'L|[1,2]+1'],
        'B': ['C+0', 'C+3', 'D+0', 'D+2', 'H+0', 'H+1', 'H_{M+Cl}+0', 'H_{M+Cl}+1', 'L|[1,2]+0', 'L|[1,2]+1'],
        'C': None,
        'D': None,
        'E': None,
    }
    measurements, basebayes, true_theta = None, None, None
    if which_measurements is not None:

        annotation_dfs = {}
        for labelling_id in substrate_df.index:
            labelling_ion_ids = labelling_specific_annots[labelling_id]
            if labelling_ion_ids is None:
                labelling_annot_df = annotation_df.copy()
            else:
                labelling_obs_df = observation_df.loc[labelling_specific_annots[labelling_id]].copy()
                labelling_annot_df = annotation_df.iloc[labelling_obs_df['annot_df_idx'].values].copy()
            annotation_dfs[labelling_id] = labelling_annot_df

        if which_measurements == 'lcms':
            total_intensities = observation_df.drop_duplicates('ion_id').set_index('ion_id')['total_I']
            if clip_min is None:
                clip_min = 750.0
            obsmods = LCMS_ObservationModel.build_models(
                model, annotation_dfs, total_intensities=total_intensities, clip_min=clip_min,
                transformation=transformation
            )
        elif which_measurements == 'com':
            sigma_ii = observation_df['sigma']
            omegas = observation_df.drop_duplicates('ion_id').set_index('ion_id')['omega']
            com_annotation_dfs = {labelling_id: (adf, sigma_ii, omegas) for labelling_id, adf in annotation_dfs.items()}
            if clip_min is None:
                clip_min = 1e-5
            elif clip_min > 1.0:
                raise ValueError('not a valid clip_min for the the classical observation model')
            obsmods = ClassicalObservationModel.build_models(
                model, com_annotation_dfs, clip_min=clip_min, transformation=transformation
            )
        else:
            raise ValueError

        bom = None
        if include_bom:
            bom = MVN_BoundaryObservationModel(model, measured_boundary_fluxes, biomass_id)

        up = UniNetFluxPrior(model._fcm, cache_size=1000)

        basebayes = _BaseBayes(model, substrate_df, obsmods, up, bom)

        true_theta = model._fcm.map_fluxes_2_theta(fluxes.to_frame().T, pandalize=True)
        basebayes.set_true_theta(true_theta.iloc[0])
        if which_measurements == 'lcms':
            _correct_base_bayes_lcms(basebayes, total_intensities=total_intensities, clip_min=clip_min)
        measurements = basebayes.simulate_true_data(n_obs=n_obs).iloc[[0]]

    kwargs = {
        'annotation_df': annotation_dfs,
        'substrate_df': substrate_df,
        'measured_boundary_fluxes': measured_boundary_fluxes,
        'measurements': measurements,
        'fluxes': fluxes,
        'true_theta': true_theta,
        'basebayes': basebayes,
    }

    return model, kwargs


def heteroskedastic(
        backend='numpy', auto_diff=False, return_type='mdv', batch_size=1, build_simulator=True
):
    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0, 'lower_bound' : 10.0,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> B/ab + co2/c'
        },
        'v2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> C/bc + co2/a'
        },
        'v3': { # TODO: maybe make this one reversible to display thermodynamic effects
            'upper_bound': 100.0,'lower_bound': 4.0, # 'rho_min':0.0, 'rho_max':0.95,
            'atom_map_str': 'B/ab --> D/ab'
        },
        'v4': {
            'upper_bound': 100.0,'lower_bound': 2.0,
            'atom_map_str': 'C/ab --> D/ab'
        },
        'v5': {
            'upper_bound': 100.0,
            'atom_map_str': 'D/ab --> E/ab',
        },
        'v6': {
            'upper_bound': 100.0,
            'atom_map_str': 'C/ab --> E/ab'
        },
    }
    input_labelling = OrderedDict({
        'A/110': 1.0,
    })
    ratio_repo = {
        'θ1': {
            'numerator': OrderedDict({'v3': 1}),
            'denominator': OrderedDict({'v3': 1, 'v4': 1})
        },
        'θ2': {
            'numerator': OrderedDict({'v5': 1}),
            'denominator': OrderedDict({'v5': 1, 'v6': 1})
        },
    }
    measurements = ['E'] # , 'D'
    model = simulator_factory(
        id_or_file_or_model='heteroskedastic',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        measurements=measurements,
        batch_size=batch_size,
        build_simulator=build_simulator,
    )
    e_out = model.reactions.get_by_id('e_out')
    model.objective = {e_out: 1}
    return model


def multi_modal(
        backend='numpy',
        auto_diff=False,
        batch_size=1,
        ratios=False,
        which_measurements='com',
        clip_min=None,
        transformation=None,
        include_bom=True,
        a_in_lb = 5.0,
        a_in = 7.5,
        kernel_basis = 'rref',
        top_frac = 0.3,
        basis_coordinates = 'rounded',
        include_D = False,
        n_obs=1,
):
    if a_in_lb > 10.0:
        raise ValueError
    elif a_in_lb < 10.0:
        if (a_in < a_in_lb) or (a_in > 10.0):
            raise ValueError

    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0, 'lower_bound': a_in_lb,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> B/ab + D/c'
        },
        'v2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> C/bc + D/a'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab + D/c --> E/ac + co2/b'
        },
        'v4': {
            'upper_bound': 100.0,
            'atom_map_str': 'C/ab + D/c --> E/cb + co2/a'
        },
    }
    metabolite_kwargs = {
        'E': {'formula': 'C2H4O2'},
        'D': {'formula': 'CH4'},
    }
    input_met = 'A'

    input_labelling = pd.Series(OrderedDict({
        f'{input_met}/011': 1.0,
    }), name='A')

    ratio_repo = {
        'r1': {
            'numerator': OrderedDict({'v3': 1}),
            'denominator': OrderedDict({'v3': 1, 'v4': 1})
        },
    }

    free_reaction_id = ['v4']
    if a_in_lb < 10.0:
        free_reaction_id.insert(0, 'a_in')

    annotation_df = pd.DataFrame([
        ['E', 'M-H', 0, 0.01, 1e4, None],
        ['E', 'M-H', 1, 0.01, 1e4, None],
        ['E', 'M-H', 2, 0.01, 1e4, None],
        ['D', 'M-H', 0, 0.01, 1e5, None],
        ['D', 'M-H', 1, 0.01, 1e5, None],
    ], columns=['met_id', 'adduct_name', 'nC13', 'sigma', 'total_I', 'omega'])
    formap = {k: v['formula'] for k, v in metabolite_kwargs.items()}
    annotation_df['formula'] = annotation_df['met_id'].map(formap)
    if not include_D:
        annotation_df = annotation_df.iloc[:3]

    model = simulator_factory(
        id_or_file_or_model='multi_modal',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        metabolite_kwargs=metabolite_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        free_reaction_id=free_reaction_id,
        measurements=annotation_df['met_id'].unique(),
        batch_size=batch_size,
        ratios=ratios,
        kernel_basis=kernel_basis,
        basis_coordinates=basis_coordinates,
        build_simulator=True
    )
    substrate_df = model.input_labelling.to_frame().T

    observation_df = MDV_ObservationModel.generate_observation_df(model, annotation_df)
    annotation_df['mz'] = 0.0
    annotation_df.loc[observation_df['annot_df_idx'], 'mz'] = observation_df['isotope_decomposition'].apply(
        lambda x: Formula(x).mass()).values

    top_flux = top_frac * 10.0
    bot_flux = 10.0 - top_flux
    fluxes = {
        'a_in': 10.0,
        'co2_out': 10.0,
        'e_out': 10.0,
        'v1': top_flux,
        'v2': bot_flux,
        'v3': top_flux,
        'v4': bot_flux,
    }
    if a_in_lb < 10.0:
        top_flux = top_frac * a_in
        bot_flux = a_in - top_flux
        fluxes = {
            'a_in': a_in,
            'co2_out': a_in,
            'e_out': a_in,
            'v1': top_flux,
            'v2': bot_flux,
            'v3': top_flux,
            'v4': bot_flux,
        }
    fluxes = pd.Series(fluxes, name='v')
    if batch_size == 1:
        model.set_fluxes(fluxes=fluxes)

    measurements, basebayes, true_theta = None, None, None
    if which_measurements is not None:
        if which_measurements == 'lcms':
            annotation_dfs = {'A': annotation_df}
            total_intensities = observation_df.drop_duplicates('ion_id').set_index('ion_id')['total_I']
            if clip_min is None:
                clip_min = 750.0
            obsmods = LCMS_ObservationModel.build_models(
                model, annotation_dfs, total_intensities=total_intensities, clip_min=clip_min,
                transformation=transformation
            )
        elif which_measurements == 'com':
            sigma_ii = observation_df['sigma']
            omegas = observation_df.drop_duplicates('ion_id').set_index('ion_id')['omega']
            annotation_dfs = {'A': (annotation_df, sigma_ii, omegas)}
            if clip_min is None:
                clip_min = 1e-5

            obsmods = ClassicalObservationModel.build_models(
                model, annotation_dfs, clip_min=clip_min, transformation=transformation
            )
        else:
            raise ValueError

        bom = None
        if include_bom:
            bom = MVN_BoundaryObservationModel(model, ['a_in'], None)

        up = UniNetFluxPrior(model._fcm, cache_size=1000)

        basebayes = _BaseBayes(model, substrate_df, obsmods, up, bom)
        true_theta = model._fcm.map_fluxes_2_theta(fluxes.to_frame().T, pandalize=True)
        basebayes.set_true_theta(true_theta.iloc[0])
        if which_measurements == 'lcms':
            _correct_base_bayes_lcms(basebayes, clip_min=clip_min)
        measurements = basebayes.simulate_true_data(n_obs=n_obs).iloc[[0]]

    kwargs = {
        'true_theta': true_theta,
        'measurements': measurements,
        'substrate_df': substrate_df,
        'hdfile': os.path.join(SIM_DIR, 'multi_modal.h5'),
        'annotation_df': annotation_df,
        'basebayes': basebayes,
    }
    return model, kwargs

def polytope_volume(algorithm='emu', backend='numpy', auto_diff=False, return_type='mdv', batch_size=1):
    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> E/ab + co2/c'
        },
        # 'v2': {
        #     'upper_bound': 100.0,
        #     'atom_map_str': 'A/abc --> E/ac + co2/b'
        # },
        'v2': {
            'upper_bound': 0.0, 'lower_bound': -100.0,
            'atom_map_str': 'E/ac + co2/b --> A/abc'
        },
        'v3': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> E/bc + co2/a'
        },
    }
    input_labelling = pd.Series(OrderedDict({
        'A/011': 1.0,
    }), name='l')
    ratio_repo = {
        'θ1': {
            'numerator': OrderedDict({'v1': 1}),
            'denominator': OrderedDict({'v1': 1, 'v2': 1, 'v3': 1})
        },
    }

    measurements=['E']
    model = simulator_factory(
        id_or_file_or_model='polytope_volume',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        measurements=measurements,
        batch_size=batch_size,
        build_simulator=False,
    )
    model.set_input_labelling(input_labelling=input_labelling)
    model.objective = {model.reactions[4]: 1}
    return model


def thermodynamic(algorithm='emu', backend='numpy', auto_diff=False, return_type='mdv', batch_size=1, v5_reversible=False):
    # TODO: should demonstrate why including thermodynamics matters
    reaction_kwargs = {
        'a_in': {
            'upper_bound': 10.0, 'lower_bound': 10.0,
            'atom_map_str': '∅ --> A/abc'
        },
        'co2_out': {
            'upper_bound': 100.0,
            'atom_map_str': 'co2/a --> ∅'
        },
        'e_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'E/ab --> ∅'
        },
        'd_out': {
            'upper_bound': 10.0,
            'atom_map_str': 'D/ab --> ∅'
        },
        'v1': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> B/ab + co2/c'
        },
        'v2': {
            'upper_bound': 100.0,
            'atom_map_str': 'A/abc --> C/bc + co2/a'
        },
        'v3': {
            'upper_bound': 100.0, 'lower_bound': -100.0,  # 'rho_min':0.0, 'rho_max':0.95,
            'dgibbsr': -6.0, '_sigma_dgibbsr': 4.0,
            'atom_map_str': 'B/ab <=> D/ab'
        },
        'v4': {
            'upper_bound': 100.0, 'lower_bound': 0.0,
            'atom_map_str': 'C/ab --> D/ab'
        },
        'v5': {
            'upper_bound': 100.0,
            'atom_map_str': 'B/ab --> E/ab',
        },
        'v6': {
            'upper_bound': 100.0,
            'atom_map_str': 'C/ab --> E/ab'
        },
    }

    input_labelling = pd.Series(OrderedDict({
        'A/011': 1.0,
    }), name='l')
    ratio_repo = {
        'θ1': {
            'numerator': OrderedDict({'v3': 1}),
            'denominator': OrderedDict({'v3': 1, 'v4': 1})
        },
        'θ2': {
            'numerator': OrderedDict({'v5': 1}),
            'denominator': OrderedDict({'v5': 1, 'v6': 1})
        },
    }

    if v5_reversible:
        reaction_kwargs['v5'] = {  # this would yield crazy stuff!
            'upper_bound': 100.0, 'lower_bound': -100.0,
            'dgibbsr': -10.0, '_sigma_dGr': 4.0,
            'atom_map_str': 'B/ab --> E/ab',
        }
        ratio_repo['θ2'] = {
            'numerator': OrderedDict({'v5': 1, 'v5_rev': -1}),
            'denominator': OrderedDict({'v5': 1, 'v5_rev': -1, 'v6': 1})
        }


    measurements=['E']
    M = simulator_factory(
        id_or_file_or_model='thermodynamic',
        backend=backend,
        auto_diff=auto_diff,
        reaction_kwargs=reaction_kwargs,
        input_labelling=input_labelling,
        ratio_repo=ratio_repo,
        measurements=measurements,
        batch_size=batch_size,
    )
    M.set_input_labelling(input_labelling=input_labelling)
    return M


if __name__ == "__main__":
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # model, kwargs = spiro(
    #     backend='torch', add_biomass=True, v2_reversible=True, v5_reversible=True,
    #     batch_size=1, which_measurements='lcms', build_simulator=True, which_labellings=list('CD'),
    #     kernel_basis='rref',
    # )

    model, kwargs = spiro(
        backend='torch',
        auto_diff=False,
        batch_size=1,
        add_biomass=True,
        v2_reversible=True,
        ratios=True,
        build_simulator=True,
        add_cofactors=True,
        which_measurements='lcms',
        seed=2,
        which_labellings=['A', 'B'],
        include_bom=True,
        v5_reversible=False,
        n_obs=0,
        kernel_basis='svd',
        basis_coordinates='rounded',
        logit_xch_fluxes=False,
        L_12_omega=1.0,
        clip_min=None,
        transformation='ilr',
    )
    # model, kwargs = spiro(
    #     seed=9, batch_size=1,
    #     backend='torch', v2_reversible=True, ratios=False, build_simulator=True,
    #     which_measurements='com', which_labellings=['A', 'B'], v5_reversible=True, include_bom=False, compute_mz=True
    # )
    #
    # print(kwargs['annotation_df'])
    #
    # dss = kwargs['basebayes']
    # data = kwargs['measurements']
    # inv = dss.to_partial_mdvs(data)


    # model.set_input_labelling(kwargs['substrate_df'].loc['C'])
    # model.set_input_labelling(kwargs['substrate_df'].loc['D'])

    # datasetsim = kwargs['datasetsim']
    # fluxes = kwargs['fluxes'].to_frame().T
    # thermo = model._fcm.map_fluxes_2_thermo(fluxes, pandalize=True)
    # theta = model._fcm.map_fluxes_2_theta(fluxes, pandalize=True)
    # ttheta = model._fcm.map_fluxes_2_theta(thermo, is_thermo=True, pandalize=True)
    # measurements = datasetsim.simulate(fluxes, n_obs=0, pandalize=True)
    # print(fluxes)
    # print(thermo)
    # print(kwargs['theta'])
    # print(theta)
    # print(ttheta)
    # print(measurements)