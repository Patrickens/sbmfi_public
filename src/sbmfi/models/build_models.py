import pandas as pd
import numpy as np
from sbmfi.core.model import LabellingModel, EMU_Model, RatioEMU_Model
from sbmfi.core.observation import LCMS_ObservationModel, MVN_BoundaryObservationModel, ClassicalObservationModel, MDV_ObservationModel
from sbmfi.core.reaction import LabellingReaction
from sbmfi.core.linalg import LinAlg
from sbmfi.core.util import make_multidex
from sbmfi.inference.bayesian import _BaseBayes
from sbmfi.priors.uniform import UniformRoundedFleXchPrior
from sbmfi.settings import MODEL_DIR
from sbmfi.core.polytopia import transform_polytope_keep_transform, simplify_polytope
from PolyRound.api import PolyRoundSettings
import os
import cobra
from cobra.io import read_sbml_model
from cobra import Reaction, Metabolite, DictList, Model
# from pta import ConcentrationsPrior
import pickle
from sbmfi.core.polytopia import (
    extract_labelling_polytope,
    rref_null_space,
    thermo_2_net_polytope
)
from sbmfi.core.coordinater import FluxCoordinateMapper
from sbmfi.lcmsanalysis.formula import Formula, isotopologues
from sbmfi.lcmsanalysis.util import build_correction_matrix
import cvxpy as cp
from typing import Iterable
import copy
from sbmfi.lcmsanalysis.nist_mass import _nist_mass
import itertools
from math import comb

_bmid_GAM = 'BIOMASS_Ecoli_core_w_GAM'
_bmid_ANTON = "biomass_rxn"
_metabolite_kwargs = {
    'succ': {'symmetric': True},
    'fum':  {'symmetric': True},
    '26dap_LL':  {'symmetric': True},

    'accoa': {'formula': 'C2H3O'},
    'succoa': {'formula': 'C4H6O4'},
    'methf': {'formula': 'C1'},
    '10fthf': {'formula': 'C1'},
    '5mthf': {'formula': 'C1'},
    'mlthf': {'formula': 'C1'},
}
_extra_model_kwargs = {
    "BIOMASS_Ecoli_core_w_GAM": {
        "atom_map_str": "biomass --> ∅"
    },

    # transport reactions
    "EX_fru_e": {
        "atom_map_str": "fru_e/abcdef --> ∅"
    },
    "FRUpts2": {
        "atom_map_str": "fru_e/abcdef + pep_c/ghi --> f6p_c/abcdef + pyr_c/ghi"
    },
    "EX_dha_e": {
        "atom_map_str": "dha_e/abc --> ∅"
    },
    "DHAt": {
        "atom_map_str": "dha_e/abc --> dha_c/abc"
    },
    "EX_glyc_e": {
        "atom_map_str": "glyc_e/abc --> ∅"
    },
    "GLYCt": {
        "atom_map_str": "glyc_e/abc --> glyc_c/abc"
    },
    "EX_pyr_e": {
        "atom_map_str": "pyr_e/abc --> ∅"
    },
    "PYRt2": {
        "atom_map_str": "pyr_e/abc --> pyr_c/abc"
    },
    "EX_lac__D_e": {
        "atom_map_str": "lac__D_e/abc --> ∅"
    },
    "D_LACt2": {
        "atom_map_str": "lac__D_e/abc --> lac__D_c/abc"
    },
    "EX_for_e": {
        "atom_map_str": "for_e/a --> ∅"
    },
    "FORt": {
        "atom_map_str": "for_e/a --> for_c/a", 'upper_bound': 0.0,
    },
    "EX_acald_e": {
        "atom_map_str": "acald_e/ab --> ∅"
    },
    "ACALDt": {
        "atom_map_str": "acald_e/ab --> acald_c/ab"
    },
    "EX_etoh_e": {
        "atom_map_str": "etoh_e/ab --> ∅"
    },
    "ETOHt2r": {
        "atom_map_str": "etoh_e/ab --> etoh_c/ab"
    },
    "EX_gln__L_e": {
        "atom_map_str": "gln__L_e/abcde --> ∅"
    },
    "GLNabc": {
        "atom_map_str": "gln__L_e/abcde --> gln__L_c/abcde"
    },
    "EX_glu__L_e": {
        "atom_map_str": "glu__L_e/abcde --> ∅"
    },
    "GLUt2r": {
        "atom_map_str": "glu__L_e/abcde --> glu__L_c/abcde"
    },
    "EX_akg_e": {
        "atom_map_str": "akg_e/abcde --> ∅"
    },
    "AKGt2r": {
        "atom_map_str": "akg_e/abcde --> akg_c/abcde"
    },
    "EX_succ_e": {
        "atom_map_str": "succ_e/abcd --> ∅"
    },
    "SUCCt2_2": {
        "atom_map_str": "succ_e/abcd --> succ_c/abcd"
    },
    "SUCCt3": {
        "atom_map_str": "succ_c/abcd --> succ_e/abcd"
    },
    "EX_fum_e": {
        "atom_map_str": "fum_e/abcd --> ∅"
    },
    "FUMt2_2": {
        "atom_map_str": "fum_e/abcd --> fum_c/abcd"
    },
    "EX_mal__L_e": {
        "atom_map_str": "mal__L_e/abcd --> ∅"
    },
    "MALt2_2": {
        "atom_map_str": "mal__L_e/abcd --> mal__L_c/abcd"
    },
    "EX_xyl__D_e": {
        "atom_map_str": "xyl__D_e/abcde --> ∅"
    },
    "XYLK": {
        "atom_map_str": "xyl__D_e/abcde --> xu5p__D_c/abcde"
    },

    # glutamine and glutamate uptake
    "GLUSy": {
        "atom_map_str": "akg_c/abcde + gln__L_c/fghij --> glu__L_c/abcde + glu__L_c/fghij"
    },
    "GLUN": {
        "atom_map_str": "gln__L_c/abcde --> glu__L_c/abcde"
    },
    "GLYK": {
        "atom_map_str": "glyc_c/abc --> glyc3p_c/abc"
    },

    # glycerol uptake
    "G3PD5": {
        "atom_map_str": "glyc3p_c/abc ==> dhap_c/abc"
    },
    "G3PD2": {
        "atom_map_str": "glyc3p_c/abc <=> dhap_c/abc"
    },
    "GLYCDx": {
        "atom_map_str": "glyc_c/abc --> dha_c/abc"
    },
    "DHAPT": {
        "atom_map_str": "dha_c/abc + pep_c/def ==> dhap_c/abc + pyr_c/def"
    },
    "F6PA": {
        "atom_map_str": "f6p_c/abcdef <=> dha_c/cba + g3p_c/def"
    },

    # mixed acid production
    "LDH_D": {
        "atom_map_str": "lac__D_c/abc <=> pyr_c/abc"
    },
    "ACALD": {
        "atom_map_str": "acald_c/ab <=> accoa_c/ab"
    },
    "ALCD2x": {
        "atom_map_str": "etoh_c/ab <=> acald_c/ab"
    },

    # glycolysis
    "FBP": {
        "atom_map_str": "fdp_c/abcdef --> f6p_c/abcdef"
    },

    # lower glycolysis
    "PPS": {
        "atom_map_str": "pyr_c/abc --> pep_c/abc"
    },
    "PFL": {
        "atom_map_str": "pyr_c/abc --> for_c/a + accoa_c/bc"
    },

    # pentose-phosphate pathway
    "TKT1": {
        "atom_map_str": "xu5p__D_c/abcde + r5p_c/fghij <=> g3p_c/cde + s7p_c/abfghij"
    },
    "TKT2": {
        "atom_map_str": "xu5p__D_c/abcde + e4p_c/fghi <=> g3p_c/cde + f6p_c/abfghi"
    },
    "TALA": {
        "atom_map_str": "s7p_c/abcdefg + g3p_c/hij <=> e4p_c/defg + f6p_c/abchij"
    },
    "FRD7": {
        "atom_map_str": "fum_c/abcd --> succ_c/abcd"
    },

    # # 1-carbon extra and inorganic phosphate
    # 'FTHFLi': {},  # consumes formate from only PFL to generate 10thf_c
    #
    # 'PFK_3': {},
    # 'FBA3': {},
}
_concentrations_priors = [
    'M9_aerobic',
    'M9_anaerobic',
    'ecoli_M9_ac',
    'ecoli_M9_fru',
    'ecoli_M9_gal',
    'ecoli_M9_glc',
    'ecoli_M9_glcn',
    'ecoli_M9_glyc',
    'ecoli_M9_pyr',
    'ecoli_M9_specific',
    'ecoli_M9_succ',
    'ecoli_M9_unspecific',
]

# general glucose condition inputs
_gluc_ratio_repo = {
        'pyr_c|glycolysis': {  # Glycolysis/PPP
            'numerator': {'PGI': 1.0},  # this is the netflux of the pgi reaction
            'denominator': {'PGI': 1.0, 'PGL': 1.0}
        },

        'pyr_c|EDD': {  # Pyruvate from ED
            'numerator': {'EDD': 1.0},
            'denominator': {'EDD': 1.0, 'PYK': 1.0, 'ME1': 1.0, 'ME2': 1.0}
        },

        'pyr_c|MAE': {  # Pyruvate from malic enzyme
            'numerator': {'ME1': 1.0, 'ME2': 1.0},
            'denominator': {'ME1': 1.0, 'ME2': 1.0, 'PYK': 1.0, 'EDD': 1.0}
        },

        'pep_c|PCK': {  # gluconeogenesis (PEP from oxaloacetate)
            'numerator': {'PPCK': 1.0},
            'denominator': {'PPCK': 1.0, 'ENO': 1.0}
        },

        'oaa_c|PPC': {  # anaplerosis (OAA from pyruvate) # NOTE: this ratio has a weird definition in SUMOFLUX...
            'numerator': {'PPC': 1.0},
            'denominator': {'PPC': 1.0, 'MDH': 1.0}
        },

        'mal__L_c|MALS': {  # glyoxylate shunt
            'numerator': {'MALS': 1.0},
            'denominator': {'MALS': 1.0, 'FUM': 1.0}
        },
    }
_anton_biomass_dct = {
    'ala__L_c': -0.488,
    'arg__L_c': -0.281,
    'asn__L_c': -0.229,
    'asp__L_c': -0.229,
    'cys__L_c': -0.087,
    'glu__L_c': -0.25 * 1.0,
    'gln__L_c': -0.25 * 1.0,
    'gly_c'   : -0.582,
    'his__L_c': -0.09,
    'ile__L_c': -0.276,
    'leu__L_c': -0.428,
    'lys__L_c': -0.326,
    'met__L_c': -0.146,
    'phe__L_c': -0.176,
    'pro__L_c': -0.21,
    'ser__L_c': -0.205,
    'thr__L_c': -0.241,
    'trp__L_c': -0.054,
    'tyr__L_c': -0.131,
    'val__L_c': -0.402,
    'g6p_c': -0.205,
    'f6p_c': -0.071,
    'r5p_c': -0.754,
    'g3p_c': -0.129,
    '3pg_c': -0.619,
    'pep_c': -0.051,
    'pyr_c': -0.083,
    'accoa_c': -2.51 * 1.0,
    'akg_c': -0.087,
    'oaa_c': -0.34,
    'mlthf_c': -0.443,
    'atp_c': -33.247 * 1.0,
    'nadph_c': -5.363 * 1.0,
    'nadh_c': 1.455 * 1.0,
    # 'bm': 39.68, # we dont want an extra outlet for biomass...
}  # this has a norm(vanton/vgam) ~ 7.52
_biomass_cofactor_correction = {
    'h2o_c': _anton_biomass_dct['atp_c'],
    'adp_c': -_anton_biomass_dct['atp_c'],
    'pi_c': -_anton_biomass_dct['atp_c'],
    'h_c': -_anton_biomass_dct['atp_c'],
    'nadp_c': -_anton_biomass_dct['nadph_c'],

    'nad_c': -_anton_biomass_dct['nadh_c'],
    'coa_c': -_anton_biomass_dct['accoa_c'],
    'thf_c': -_anton_biomass_dct['mlthf_c'],
}
_anton_biomass_dct = {**_anton_biomass_dct, **_biomass_cofactor_correction}
_anton_model_kwargs = {
    # 11 free net, 23 free xch fluxes
    # from file 1-s2.0-S1096717615000038-mmc1.xlsx with some extra stuff to make bigg model feasible
    "Glycolysis": {
        "v1": {
            "fluxml_id": "Glc_upt___1",
            "anton_str": "Gluc.ext (abcdef) + PEP (ghi) --> G6P (abcdef) + Pyr (ghi)",
            "bigg_kwargs": {
                "EX_glc__D_e": {
                    "lower_bound": -10.0,
                    "upper_bound": -10.0,
                    "coeff": -1,
                    "atom_map_str": "glc__D_e/abcdef --> ∅"
                },
                "GLCpts": {
                    "atom_map_str": "glc__D_e/abcdef + pep_c/ghi --> g6p_c/abcdef + pyr_c/ghi"
                },
            },
        },
        "v2": {
            "fluxml_id": "EMP2_v2",
            "anton_str": "G6P (abcdef) <=> F6P (abcdef)",
            "bigg_kwargs": {
                "PGI": {
                    # "lower_bound": 0.0,
                    "atom_map_str": "g6p_c/abcdef <=> f6p_c/abcdef"
                },
            },
        },
        "v3": {
            "fluxml_id": "EMP3_v3",
            "anton_str": "F6P (abcdef) + ATP --> FBP (abcdef)",
            "bigg_kwargs": {
                "PFK": {
                    "atom_map_str": "f6p_c/abcdef --> fdp_c/abcdef"
                },
            },
        },
        "v4": {
            "fluxml_id": "EMP4_v4",
            "anton_str": "FBP (abcdef) <=> DHAP (cba) + GAP (def)",
            "bigg_kwargs": {
                "FBA": {
                    "atom_map_str": "fdp_c/abcdef <=> g3p_c/def + dhap_c/cba"
                },
            },
        },
        "v5": {
            "fluxml_id": "EMP5_v5",
            "anton_str": "DHAP (abc) <=> GAP (abc)",
            "bigg_kwargs": {
                "TPI": {
                    "atom_map_str": "dhap_c/abc <=> g3p_c/abc"
                },
            },
        },
        "v6": {
            "fluxml_id": "EMP6_v6",
            "anton_str": "GAP (abc) <=> 3PG (abc) + ATP + NADH",
            "bigg_kwargs": {
                "GAPD": {
                    "atom_map_str": "g3p_c/abc <=> 13dpg_c/abc"
                },
            },
        },
        "v7": {
            "fluxml_id": "EMP7_v7",
            "anton_str": "3PG (abc) <=> PEP (abc)",
            "bigg_kwargs": {
                "PGK": {
                    "coeff": -1,
                    "atom_map_str": "3pg_c/abc <=> 13dpg_c/abc"
                },
                "PGM": {
                    "coeff": -1,
                    "atom_map_str": "2pg_c/abc <=> 3pg_c/abc"
                },
                "ENO": {
                    "atom_map_str": "2pg_c/abc <=> pep_c/abc"
                },
            },
        },
        "v8": {
            "fluxml_id": "EMP1_v8",
            "anton_str": "PEP (abc) --> Pyr (abc) + ATP",
            "bigg_kwargs": {
                "PYK": {
                    "atom_map_str": "pep_c/abc --> pyr_c/abc"
                },
            },
        }
    },
    "Pentose Phosphate Pathway": {
        "v9": {
            "fluxml_id": "PPP1_v9___1",
            "anton_str": "G6P (abcdef) --> 6PG (abcdef) + NADPH",
            "bigg_kwargs": {
                "G6PDH2r": {
                    "lower_bound": 0.0,
                    "atom_map_str": "g6p_c/abcdef --> 6pgl_c/abcdef"
                }
            },
        },
        "v10": {
            "fluxml_id": "PPP2_v10___1",
            "anton_str": "6PG (abcdef) --> Ru5P (bcdef) + CO2 (a) + NADPH",
            "bigg_kwargs": {
                "PGL": {
                    "atom_map_str": "6pgl_c/abcdef --> 6pgc_c/abcdef"
                },
                "GND": {
                    "atom_map_str": "6pgc_c/abcdef --> co2_c/a + ru5p__D_c/bcdef"
                },
            },
        },
        "v11": {
            "fluxml_id": "PPP3_v11___1",
            "anton_str": "Ru5P (abcde) <=> X5P (abcde)",
            "bigg_kwargs": {
                "RPE": {
                    "atom_map_str": "ru5p__D_c/abcde <=> xu5p__D_c/abcde"
                },
            },
        },
        "v12": {
            "fluxml_id": "PPP4_v12___1",
            "anton_str": "Ru5P (abcde) <=> R5P (abcde)",
            "bigg_kwargs": {
                "RPI": {
                    "coeff": -1,
                    "atom_map_str": "r5p_c/abcde <=> ru5p__D_c/abcde"
                },
            },
        },
        "v13": {
            "fluxml_id": "PPP5_v13___1",
            "anton_str": "X5P (abcde) <=> TK-C2 (ab) + GAP (cde)",
            "bigg_kwargs": {
                "TKTa": {
                    "upper_bound": 1000.0,
                    "lower_bound": -1000.0,
                    "metabolite_dct": {
                        "xu5p__D_c": -1,
                        "TKT_c": 1,
                        "g3p_c": 1,
                    },
                    "atom_map_str": "xu5p__D_c/abcde <=> TKT_c/ab + g3p_c/cde"
                },
            },
        },
        "v14": {
            "fluxml_id": "PPP6_v14___1",
            "anton_str": "F6P (abcdef) <=> TK-C2 (ab) + E4P (cdef)",
            "bigg_kwargs": {
                "TKTb": {
                    "upper_bound": 1000.0,
                    "lower_bound": -1000.0,
                    "metabolite_dct": {
                        "f6p_c": -1,
                        "TKT_c": 1,
                        "e4p_c": 1,
                    },
                    "atom_map_str": "f6p_c/abcdef <=> TKT_c/ab + e4p_c/cdef"
                },
            },
        },
        "v15": {
            "fluxml_id": "PPP7_v15___1",
            "anton_str": "S7P (abcdefg) <=> TK-C2 (ab) + R5P (cdefg)",
            "bigg_kwargs": {
                "TKTc": {
                    "upper_bound": 1000.0,
                    "lower_bound": -1000.0,
                    "metabolite_dct": {
                        "s7p_c": -1,
                        "TKT_c": 1,
                        "r5p_c": 1,
                    },
                    "atom_map_str": "s7p_c/abcdefg <=> TKT_c/ab + r5p_c/cdefg"
                 },
            },
        },
        "v16": {
            "fluxml_id": "PPP8_v16___1",
            "anton_str": "F6P (abcdef) <=> TA-C3 (abc) + GAP (def)",
            "bigg_kwargs": {
                "TALAa": {
                    "upper_bound": 1000.0,
                    "lower_bound": -1000.0,
                    "metabolite_dct": {
                        "f6p_c": -1,
                        "TALA_c": 1,
                        "g3p_c": 1,
                    },
                    "atom_map_str": "f6p_c/abcdef <=> TALA_c/abc + g3p_c/def"
                },
            },
        },
        "v17": {
            "fluxml_id": "PPP9_v17___1",
            "anton_str": "S7P (abcdefg) <=> TA-C3 (abc) + E4P (defg)",
            "bigg_kwargs": {
                "TALAb": {
                    "upper_bound": 1000.0,
                    "lower_bound": -1000.0,
                    "metabolite_dct": {
                        "s7p_c": -1,
                        "TALA_c": 1,
                        "e4p_c": 1,
                    },
                    "atom_map_str": "s7p_c/abcdefg <=> TALA_c/abc + e4p_c/defg"
                },
            },
        }
    },
    "Entner-Doudoroff Pathway": {
        "v18": {
            "fluxml_id": "PPP10_v18___1",
            "anton_str": "6PG (abcdef) --> KDPG (abcdef)",
            "bigg_kwargs": {
                "EDD": {
                    "atom_map_str": "6pgc_c/abcdef --> 2ddg6p_c/abcdef"
                },
            },
        },
        "v19": {
            "fluxml_id": "PPP11_v19___1",
            "anton_str": "KDPG (abcdef) --> Pyr (abc) + GAP (def)",
            "bigg_kwargs": {
                "EDA": {
                    "atom_map_str": "2ddg6p_c/abcdef --> pyr_c/abc + g3p_c/def"
                },
            },
        }
    },
    "TCA Cycle": {
        "v20": {
            "fluxml_id": "TCA1_v20",
            "anton_str": "Pyr (abc) --> AcCoA (bc) + CO2 (a) + NADH",
            "bigg_kwargs": {
                "PDH": {
                    "atom_map_str": "pyr_c/abc --> co2_c/a + accoa_c/bc"
                },
            },
        },
        "v21": {
            "fluxml_id": "TCA2_v21",
            "anton_str": "OAC (abcd) + AcCoA (ef) --> Cit (dcbfea)",
            "bigg_kwargs": {
                "CS": {
                    "atom_map_str": "oaa_c/abcd + accoa_c/ef --> cit_c/dcbfea"
                },
            },
        },
        "v22": {
            "fluxml_id": "TCA3_v22___1",
            "anton_str": "Cit (abcdef) <=> ICit (abcdef)",
            "bigg_kwargs": {
                "ACONTa": {
                    "atom_map_str": "cit_c/abcdef <=> acon_C_c/abcdef"
                },
                "ACONTb": {
                    "atom_map_str": "acon_C_c/abcdef <=> icit_c/abcdef"
                },
            },
        },
        "v23": {
            "fluxml_id": "TCA4_v23",
            "anton_str": "ICit (abcdef) <=> AKG (abcde) + CO2 (f) + NADPH",
            "bigg_kwargs": {
                "ICDHyr": {
                    "atom_map_str": "icit_c/abcdef <=> akg_c/abcde + co2_c/f"
                },
            },
        },
        "v24": {
            "fluxml_id": "TCA5_v24___1",
            "anton_str": "AKG (abcde) --> SucCoA (bcde) + CO2 (a) + NADH",
            "bigg_kwargs": {
                "AKGDH": {
                    "atom_map_str": "akg_c/abcde --> succoa_c/bcde + co2_c/a"
                },
            },
        },
        "v25": {
            "fluxml_id": "TCA6_v25___1",
            "anton_str": "SucCoA (abcd) <=> Suc (0.5 abcd + 0.5 dcba) + ATP",
            "bigg_kwargs": {
                "SUCOAS": {
                    "coeff": -1,
                    "atom_map_str": "succ_c/abcd <=> succoa_c/abcd"
                },
            },
        },
        "v26": {
            "fluxml_id": "TCA7_v26___1|TCA7_v26___2",
            "anton_str": "Suc (0.5 abcd + 0.5 dcba) <=> Fum (0.5 abcd + 0.5 dcba) + FADH2",
            "bigg_kwargs": {
                "SUCDi": {
                    "lower_bound": -1000.0,
                    "upper_bound": 1000.0,
                    "atom_map_str": "succ_c/abcd <=> fum_c/abcd"
                },
            },
        },
        "v27": {
            "fluxml_id": "TCA8_v27___1|TCA8_v27___2",
            "anton_str": "Fum (0.5 abcd + 0.5 dcba) <=> Mal (abcd)",
            "bigg_kwargs": {
                "FUM": {
                    "atom_map_str": "fum_c/abcd <=> mal__L_c/abcd"
                },
            },
        },
        "v28": {
            "fluxml_id": "TCA9_v28___1",
            "anton_str": "Mal (abcd) <=> OAC (abcd) + NADH",
            "bigg_kwargs": {
                "MDH": {
                    "lower_bound": 0.0, 'rho_min': 0.1, # 'rho_max':0.3, inconsistent with ANTON
                    "atom_map_str": "mal__L_c/abcd ==> oaa_c/abcd"
                },
            },
        }
    },
    "Glyoxylate Shunt": {
        "v29": {
            "fluxml_id": "GOX1_v29___1|GOX1_v29___2",
            "anton_str": "ICit (abcdef) <=> Glyox (ab) + Suc (0.5 edcf + 0.5 fcde)",
            "bigg_kwargs": {
                "ICL": {
                    "atom_map_str": "icit_c/abcdef --> glx_c/ab + succ_c/edcf"
                },
            },
        },
        "v30": {
            "fluxml_id": "GOX2_v30___1",
            "anton_str": "Glyox (ab) + AcCoA (cd) --> Mal (abdc)",
            "bigg_kwargs": {
                "MALS": {
                    "atom_map_str": "accoa_c/ab + glx_c/cd --> mal__L_c/abdc"
                },
            },
        }
    },
    "Amphibolic Reactions": {
        "v31": {
            "fluxml_id": "ANA1_v31___1",
            "anton_str": "Mal (abcd) --> Pyr (abc) + CO2 (d) + NADPH",
            "bigg_kwargs": {
                "ME1": {
                    "atom_map_str": "mal__L_c/abcd --> pyr_c/abc + co2_c/d"
                },
            },
        },
        "v32": {
            "fluxml_id": "ANA2_v32___1",
            "anton_str": "Mal (abcd) --> Pyr (abc) + CO2 (d) + NADH",
            "bigg_kwargs": {
                "ME2": {
                    "atom_map_str": "mal__L_c/abcd --> pyr_c/abc + co2_c/d"
                },
            },
        },
        "v33": {
            "fluxml_id": "ANA3_v33___1",
            "anton_str": "PEP (abc) + CO2 (d) --> OAC (abcd)",
            "bigg_kwargs": {
                "PPC": {
                    "atom_map_str": "pep_c/abc + co2_c/d --> oaa_c/abcd"
                },
            },
        },
        "v34": {
            "fluxml_id": "ANA4_v34___1",
            "anton_str": "OAC (abcd) + ATP --> PEP (abc) + CO2 (d)",
            "bigg_kwargs": {
                "PPCK": {
                    "atom_map_str": "oaa_c/abcd --> pep_c/abc + co2_c/d"
                },
            },
        }
    },
    "Acetic Acid Formation": {
        "v35": {
            "fluxml_id": "AC1_v35___1",
            "anton_str": "AcCoA (ab) <=> Ac (ab) + ATP",
            "bigg_kwargs": {
                "PTAr": {
                    "atom_map_str": "accoa_c/ab <=> actp_c/ab"
                },
                "ACKr": {
                    "coeff": -1,
                    "atom_map_str": "ac_c/ab <=> actp_c/ab"
                },
            }
        }
    },
    "Amino Acid Biosynthesis": {
        "v36": {
            "fluxml_id": "AA1_v36___1",
            "anton_str": "AKG (abcde) + NADPH + NH3 --> Glu (abcde)",
            "bigg_kwargs": {
                "GLUDy": {
                    'coeff': -1,
                    "lower_bound": -1000.0,
                    "upper_bound": 0.0,
                    "atom_map_str": "glu__L_c/abcde <-- akg_c/abcde"
                },
            },
        },
        "v37": {
            "fluxml_id": "AA2_v37___1",
            "anton_str": "Glu (abcde) + ATP + NH3 --> Gln (abcde)",
            "bigg_kwargs": {
                "GLNS": {
                    "atom_map_str": "glu__L_c/abcde --> gln__L_c/abcde"
                },
            },
        },
        "v38": {
            "fluxml_id": "AA3_v38___1",
            "anton_str": "Glu (abcde) + ATP + 2 NADPH --> Pro (abcde)",
            "bigg_kwargs": {
                "pro_syn": {
                    "to_combine":{
                        "GLU5K": {},
                        "G5SD": {},
                        "G5SADs": {},
                        "P5CR": {},
                    },
                    "atom_map_str": "glu__L_c/abcde --> pro__L_c/abcde"
                },
            },
        },
        "v39": {
            "fluxml_id": "AA4_v39___1",
            "anton_str": "Glu (abcde) + CO2 (f) + Gln (ghijk) + Asp (lmno) + AcCoA (pq) + 5 ATP + NADPH --> Arg (abcdef) + AKG (ghijk) + Fum (lmno) + Ac (pq)",
            "bigg_kwargs": {
                "arg_syn": {
                    "to_combine": {
                        "ACGS": {},
                        "ACGK": {},
                        "AGPR": {
                            "coeff": -1,
                        },
                        "ACOTA": {
                            "coeff": -1,
                        },
                        "ACODA": {},
                        "CBMKr": {
                            "coeff": 0,
                        },
                        "CBPS": {},
                        "OCBT": {},
                        "ARGSS": {},
                        "ARGSL": {},
                    },
                    "atom_map_str": "glu__L_c/abcde + co2_c/f + gln__L_c/ghijk + asp__L_c/lmno + accoa_c/pq --> arg__L_c/abcdef + akg_c/ghijk + fum_c/lmno + ac_c/pq"
                },
            },
        },
        "v40": {
            "fluxml_id": "AA5_v40___1",
            "anton_str": "OAC (abcd) + Glu (efghi) --> Asp (abcd) + AKG (efghi)",
            "bigg_kwargs": {
                "ASPTA": {
                    "coeff": -1,
                    "lower_bound": -1000.0,
                    "upper_bound": 0.0,
                    "atom_map_str": "asp__L_c/abcd + akg_c/efghi <-- oaa_c/abcd + glu__L_c/efghi"
                },
            },
        },
        "v41": {
            "fluxml_id": "AA6_v41___1",
            "anton_str": "Asp (abcd) + 2 ATP + NH3 --> Asn (abcd)",
            "bigg_kwargs": {
                "ASNS2": {
                    "atom_map_str": "asp__L_c/abcd --> asn__L_c/abcd"
                },
            },
        },
        "v42": {
            "fluxml_id": "AA23_v42___1",
            "anton_str": "Pyr (abc) + Glu (defgh) --> Ala (abc) + AKG (defgh)",
            "bigg_kwargs": {
                "ALATA_L": {
                    "coeff": -1,
                    "lower_bound": -1000.0,
                    "upper_bound": 0.0,
                    "atom_map_str": "ala__L_c/abc + akg_c/defgh <-- pyr_c/abc + glu__L_c/defgh"
                },
            },
        },
        "v43": {
            "fluxml_id": "AA7_v43___1",
            "anton_str": "3PG (abc) + Glu (defgh) --> Ser (abc) + AKG (defgh) + NADH",
            "bigg_kwargs": {
                "ser_syn": {
                    "to_combine": {
                        "PGCD": {},
                        "PSERT": {},
                        "PSP_L": {},
                    },
                    "atom_map_str": "3pg_c/abc --> ser__L_c/abc",
                },
            },
        },
        "v44": {
            "fluxml_id": "AA8_v44___1",
            "anton_str": "Ser (abc) <=> Gly (ab) + MEETHF (c)",
            "bigg_kwargs": {
                "GHMT2r": {
                    "atom_map_str": "ser__L_c/abc <=> gly_c/ab + mlthf_c/c"
                },
            },
        },
        "v45": {
            "fluxml_id": "AA9_v45___1",
            "anton_str": "Gly (ab) <=> CO2 (a) + MEETHF (b) + NADH + NH3",
            "bigg_kwargs": {
                "GLYCL": {
                    "lower_bound": -1000.0,
                    "upper_bound": 1000.0,
                    "atom_map_str": "gly_c/ab <=> co2_c/a + mlthf_c/b"
                },
            },
        },
        "v46": {
            "fluxml_id": "AA10_v46___1",
            "anton_str": "Thr (abcd) --> Gly (ab) + AcCoA (cd) + NADH",
            "bigg_kwargs": {
                "THRA": {
                    "atom_map_str": "thr__L_c/abcd --> gly_c/ab + acald_c/cd"
                },
                "ACALD": {
                    "lower_bound": 0.0,
                    "atom_map_str": "acald_c/ab --> accoa_c/ab"
                },
            },
        },
        "v47": {
            "fluxml_id": "AA11_v47___1",
            "anton_str": "Ser (abc) + AcCoA (de) + 3 ATP + 4 NADPH + SO4 --> Cys (abc) + Ac (de)",
            "bigg_kwargs": {
                "cys_syn": {
                    "to_combine": {
                        "SADT2": {},
                        "ADSK": {},
                        "PAPSR": {},
                        "TRDR": {},
                        "BPNT": {},
                        "SULR": {},
                        "SERAT": {},
                        "CYSS": {},
                    },
                    "atom_map_str": "ser__L_c/abc + accoa_c/de --> cys__L_c/abc + ac_c/de"
                },
            },
        },
        "v48": {
            "fluxml_id": "AA12_v48___1|AA12_v48___2|AA12_v48___3|AA12_v48___4",
            "anton_str": "Asp (abcd) + Pyr (efg) + Glu (hijkl) + SucCoA (mnop) + ATP + 2 NADPH --> LL-DAP (0.5 abcdgfe + 0.5 efgdcba) + AKG (hijkl) + Suc (0.5 mnop + 0.5 ponm)",
            "bigg_kwargs": {
                "26dap_LL_syn": {
                    "to_combine": {
                        "ASPK": {},
                        "ASAD": {
                            "coeff": -1,
                        },
                        "DHDPS": {},
                        "DHDPRy": {},
                        "THDPS": {},
                        "SDPTA": {
                            "coeff": -1,
                        },
                        "SDPDS": {},
                    },
                    "atom_map_str": "asp__L_c/abcd + pyr_c/efg + glu__L_c/hijkl + succoa_c/mnop --> 26dap_LL_c/abcdgfe + akg_c/hijkl + succ_c/mnop"
                },
            },
        },
        "v49": {
            "fluxml_id": "AA13_v49___1|AA13_v49___2",
            "anton_str": "LL-DAP (0.5 abcdefg + 0.5 gfedcba) --> Lys (abcdef) + CO2 (g)",
            "bigg_kwargs": {
                "lys_syn": {
                    "to_combine": {
                        "DAPE": {},
                        "DAPDC":{},
                    },
                    "atom_map_str": "26dap_LL_c/abcdefg --> lys__L_c/abcdef + co2_c/g"
                },
            },
        },
        "v50": {
            "fluxml_id": "AA14_v50___1",
            "anton_str": "Asp (abcd) + 2 ATP + 2 NADPH --> Thr (abcd)",
            "bigg_kwargs": {
                "thr_syn": {
                    "to_combine": {
                        "ASPK": {},
                        "ASAD": {
                            "coeff": -1,
                        },
                        "HSDy": {
                            "coeff": -1,
                        },
                        "HSK": {},
                        "THRS": {},
                    },
                    "atom_map_str": "asp__L_c/abcd --> thr__L_c/abcd"
                }
            },
        },
        "v51": {
            "fluxml_id": "AA15_v51___1|AA15_v51___2",
            "anton_str": "Asp (abcd) + METHF (e) + Cys (fgh) + SucCoA (ijkl) + ATP + 2 NADPH --> Met (abcde) + Pyr (fgh) + Suc (0.5 ijkl + 0.5 lkji) + NH3",
            "bigg_kwargs": {
                "met_syn": {
                    "to_combine": {
                        "ASPK": {},
                        "ASAD": {
                            "coeff": -1,
                        },
                        "HSDy": {
                            "coeff": -1,
                        },
                        "HSST": {},
                        "SHSL1": {},
                        "CYSTL": {},
                        "METS": {},
                        "METAT": {
                            "coeff": 0,
                        },
                        "HCYSMT": {
                            "coeff": 0,
                        },
                        "HCYSMT2": {
                            "coeff": 0,
                        },
                    },
                    "atom_map_str": "asp__L_c/abcd + 5mthf_c/e + cys__L_c/fgh + succoa_c/ijkl --> met__L_c/abcde + pyr_c/fgh + succ_c/ijkl"
                },
            },
        },
        "v52": {
            "fluxml_id": "AA16_v52___1",
            "anton_str": "Pyr (abc) + Pyr (def) + Glu (ghijk) + NADPH --> Val (abcef) + CO2 (d) + AKG (ghijk)",
            "bigg_kwargs": {
                "val_syn": {
                    "to_combine": {
                        "ACLS": {},
                        "KARA1": {
                            "coeff": -1,
                        },
                        "DHAD1": {},
                        "VALTA": {
                            "coeff": -1,
                        },
                        "VPAMTr": {
                            "coeff": 0,
                        },
                    },
                    "atom_map_str": "pyr_c/abc + pyr_c/def + glu__L_c/ghijk --> val__L_c/abcef + co2_c/d + akg_c/ghijk"
                }
            },
        },
        "v53": {
            "fluxml_id": "AA17_v53___1",
            "anton_str": "AcCoA (ab) + Pyr (cde) + Pyr (fgh) + Glu (ijklm) + NADPH --> Leu (abdghe) + CO2 (c) + CO2 (f) + AKG (ijklm) + NADH",
            "bigg_kwargs": {
                "leu_syn": {
                    "to_combine": {
                        "ACLS": {},
                        "KARA1": {
                            "coeff": -1,
                        },
                        "DHAD1": {},
                        "IPPS": {},
                        "IPPMIb": {
                            "coeff": -1,
                        },
                        "IPPMIa": {
                            "coeff": -1,
                        },
                        "IPMD": {},
                        "OMCDC": {},
                        "LEUTAi": {},
                   },
                    "atom_map_str": "accoa_c/ab + pyr_c/cde + pyr_c/fgh + glu__L_c/ijklm --> leu__L_c/abdghe + co2_c/c + co2_c/f + akg_c/ijklm"
                },
            },
        },
        "v54": {
            "fluxml_id": "AA18_v54___1",
            "anton_str": "Thr (abcd) + Pyr (efg) + Glu (hijkl) + NADPH --> Ile (abfcdg) + CO2 (e) + AKG (hijkl) + NH3",
            "bigg_kwargs": {
                "ile_syn": {
                    "to_combine": {
                        "THRD_L": {},
                        "ACHBS": {},
                        "KARA2": {},
                        "DHAD2": {},
                        "ILETA": {
                            "coeff": -1,
                        },
                    },
                    "atom_map_str": "thr__L_c/abcd + pyr_c/efg + glu__L_c/hijkl --> ile__L_c/abfcdg + co2_c/e + akg_c/hijkl"
                }
            },
        },
        "v55": {
            "fluxml_id": "AA19_v55___1",
            "anton_str": "PEP (abc) + PEP (def) + E4P (ghij) + Glu (klmno) + ATP + NADPH --> Phe (abcefghij) + CO2 (d) + AKG (klmno)",
            "bigg_kwargs": {
                "phe_syn": {
                    "to_combine": {
                        "DDPA": {},
                        "DHQS": {},
                        "DHQTi": {},
                        "SHK3Dr": {},
                        "SHKK": {},
                        "PSCVT": {},
                        "CHORS": {},
                        "CHORM": {},
                        "PPNDH": {},
                        "PHETA1": {
                            "coeff": -1,
                        },
                    },
                    "atom_map_str": "pep_c/abc + pep_c/def + e4p_c/ghij + glu__L_c/klmno --> phe__L_c/abcefghij + co2_c/d + akg_c/klmno"
                },
            },
        },
        "v56": {
            "fluxml_id": "AA20_v56___1",
            "anton_str": "PEP (abc) + PEP (def) + E4P (ghij) + Glu (klmno) + ATP + NADPH --> Tyr (abcefghij) + CO2 (d) + AKG (klmno) + NADH",
            "bigg_kwargs": {
                "tyr_syn": {
                    "to_combine": {
                        "DDPA": {},
                        "DHQS": {},
                        "DHQTi": {},
                        "SHK3Dr": {},
                        "SHKK": {},
                        "PSCVT": {},
                        "CHORS": {},
                        "CHORM": {},
                        "PPND": {},
                        "TYRTA": {
                            "coeff": -1,
                        },
                    },
                    "atom_map_str": "pep_c/abc + pep_c/def + e4p_c/ghij + glu__L_c/klmno --> tyr__L_c/abcefghij + co2_c/d + akg_c/klmno"
                }
            },
        },
        "v57": {
            "fluxml_id": "AA21_v57___1",
            "anton_str": "Ser (abc) + R5P (defgh) + PEP (ijk) + E4P (lmno) + PEP (pqr) + Gln (stuvw) + 3 ATP + NADPH --> Trp (abcedklmnoj) + CO2 (i) + GAP (fgh) + Pyr (pqr) + Glu (stuvw)",
            "bigg_kwargs": {
                "trp_syn": {
                    "to_combine": {
                        "DDPA": {},
                        "DHQS": {},
                        "DHQTi": {},
                        "SHK3Dr": {},
                        "SHKK": {},
                        "PSCVT": {},
                        "CHORS": {},
                        "PRPPS": {},
                        "ANS": {},
                        "ANPRT": {},
                        "PRAIi": {},
                        "IGPS": {},
                        "TRPS1": {},
                    },
                    "atom_map_str": "ser__L_c/abc + r5p_c/defgh + pep_c/ijk + e4p_c/lmno + pep_c/pqr + gln__L_c/stuvw --> trp__L_c/abcedklmnoj + co2_c/i + g3p_c/fgh + pyr_c/pqr + glu__L_c/stuvw"
                }
            },
        },
        "v58": {
            "fluxml_id": "AA22_v58___1",
            "anton_str": "R5P (abcde) + FTHF (f) + Gln (ghijk) + Asp (lmno) + 5 ATP --> His (edcbaf) + AKG (ghijk) + Fum (lmno) + 2 NADH",
            "bigg_kwargs": {
                "his_syn": {
                    "to_combine": {
                        "PRPPS": {},
                        "ATPPRT": {},
                        "PRATPP": {},
                        "PRAMPC": {},
                        "PRMICI": {},
                        "IG3PS": {},
                        "AICART": {},
                        "MTHFC": {},
                        "ADSS": {},
                        "ADSL1r": {},
                        "IMPC": {
                            "coeff": -1,
                        },
                        "IGPDH": {},
                        "HSTPT": {},
                        "HISTP": {},
                        "HISTD": {},
                        "R1PK": {
                            "coeff": 0,
                        },
                        "R15BPK": {
                            "coeff": 0,
                        },
                        "ORPT": {
                            "coeff": 0,
                        },
                        "GMPR": {
                            "coeff": 0,
                        },
                        "ADSL2r": {
                            "coeff": 0,
                        },
                    },
                    "atom_map_str": "r5p_c/abcde + methf_c/f + gln__L_c/ghijk + asp__L_c/lmno --> his__L_c/edcbaf + akg_c/ghijk + fum_c/lmno"
                }
            },
        }
    },
    "One-Carbon Metabolism": {
        "v59": {
            "fluxml_id": "C1_v59___1",
            "anton_str": "MEETHF (a) + NADH --> METHF (a)",
            "bigg_kwargs": {
                "MTHFR2": {
                    "atom_map_str": "mlthf_c/a --> 5mthf_c/a"
                },
            },
        },
        "v60": {
            "fluxml_id": "C2_v60___1",
            "anton_str": "MEETHF (a) --> FTHF (a) + NADPH",
            "bigg_kwargs": {
                "MTHFD": {
                    "lower_bound": 0.0,
                    "atom_map_str": "mlthf_c/a --> methf_c/a"
                },
            },
        }
    },
    "Oxidative Phosphorylation": {
        "v61": {
            "fluxml_id": "OXP1_v61___1",
            "anton_str": "NADH + 0.5 O2 --> 2 ATP",
            "bigg_kwargs": {
                "NADH16": {
                },
                "CYTBD": {
                },
                "ATPS4r": {
                },
            }
        },
        "v62": {
            "fluxml_id": "OXP2_v62___1",
            "anton_str": "FADH2 + 0.5 O2 --> 1 ATP",
            "bigg_kwargs": {}
        }
    },
    "Transhydrogenation": {
        "v63": {
            "fluxml_id": "TH1_v63___1",
            "anton_str": "NADH <=> NADPH",
            "bigg_kwargs": {
                # "NADTRHD": {"coeff": -1},
                "THD2": {"coeff": 1},
            }
        }
    },
    "ATP Hydrolysis": {
        "v64": {
            "fluxml_id": "TH1_v63___1",
            "anton_str": "ATP --> ATP:ext",
            "bigg_kwargs": {
                "ATPM": {},
                "ADK1": {},
            }
        }
    },
    "Transport": {
        "v65": {
            "fluxml_id": "TP1_v65___1",
            "anton_str": "Ac (ab) --> Ac.ext (ab)",
            "bigg_kwargs": {
                "ACt2r": {
                    "coeff": -1, 'upper_bound': 0.0,
                    "atom_map_str": "ac_e/ab <-- ac_c/ab",
                },
                "EX_ac_e": {
                    "atom_map_str": "ac_e/ab --> ∅"
                },
            },
        },
        "v66": {
            "fluxml_id": "TP2_v66___1",
            "anton_str": "CO2 (a) --> CO2.ext (a)",
            "bigg_kwargs": {
                "OUT_CO2t": {
                    "metabolite_dct": {
                        'co2_c': -1
                    },
                    'coeff': -1,
                    "upper_bound": 1500.0,
                    "atom_map_str": "co2_c/a --> ∅"
                },
            },
        },
        "v67": {
            "fluxml_id": "TP3_v67___1",
            "anton_str": "O2.ext --> O2",
            "bigg_kwargs": {
                "O2t": {},
                "EX_o2_e": {'coeff': -1},
            },
        },
        "v68": {
            "fluxml_id": "TP4_v68___1",
            "anton_str": "NH3.ext --> NH3",
            "bigg_kwargs": {
                "NH4t": {},
                "EX_nh4_e": {'coeff': -1},
            },
        },
        "v69": {
            "fluxml_id": "TP5_v69___1",
            "anton_str": "SO4.ext --> SO4",
            "bigg_kwargs": {
                "SULabcpp": {  # NOTE THIS FUCKS UP THE ATP BALANCE
                    "metabolite_dct": {
                        "so4_p": 1,
                        "so4_e": -1,
                    },
                },
                "EX_so4_e": {'coeff': -1},
            },
        },
    },
    "Biomass Formation": {
        "v70": {
            "fluxml_id": "mu2_v70___1",
            "anton_str": "0.488 Ala + 0.281 Arg + 0.229 Asn + 0.229 Asp + 0.087 Cys + 0.250 Glu + 0.250 Gln + 0.582 Gly + 0.090 His + 0.276 Ile + 0.428 Leu + 0.326 Lys + 0.146 Met + 0.176 Phe + 0.210 Pro + 0.205 Ser + 0.241 Thr + 0.054 Trp + 0.131 Tyr + 0.402 Val + 0.205 G6P + 0.071 F6P + 0.754 R5P + 0.129 GAP + 0.619 3PG + 0.051 PEP + 0.083 Pyr + 2.510 AcCoA + 0.087 AKG + 0.340 OAC + 0.443 MEETHF + 33.247 ATP + 5.363 NADPH --> 39.68 Biomass + 1.455 NADH",
            "bigg_kwargs": {
                _bmid_ANTON: {
                    "metabolite_dct": _anton_biomass_dct,
                    "lower_bound": 0.05,
                    "atom_map_str": "biomass --> ∅"
                },
            },
        },
    },
    "CO2 Exchange": {
        "v71": {
            "fluxml_id": "GAS2_v71___1",
            "anton_str": "CO2.unlabeled (a) + CO2 (b) --> CO2 (a) + CO2.out (b)",
            "bigg_kwargs": {
                "CO2t": {
                    "coeff": 1,
                    "lower_bound": 0.0,
                    "upper_bound": 1500.0,
                    "atom_map_str": "co2_e/a --> co2_c/a",
                },
                "EX_co2_e": {
                    'coeff': -1,
                    'lower_bound': -1500.0,
                    'upper_bound': 0.0,
                    "atom_map_str": "co2_e/a --> ∅"
                },
            },
        },
    },
    "EXTRA": {
        "transport": {
            "bigg_kwargs": {
                "EX_pi_e": {},
                "PIt2r": {},
                "EX_h2o_e": {},
                "H2Ot": {},
                "EX_h_e": {},
            },
        },
        "PPA": {
            "bigg_kwargs": {
                "PPA": {},
            },
        },
        # "HCO3E": {
        #     "bigg_kwargs": {
        #         "HCO3E": {
        #             "atom_map_str": "co2_c/a <=> hco3_c/a"  # this is to make arginine synthase work
        #         },
        #     },
        # },
    },
    "MODEL_SOURCE": {
        "publication": "Integrated 13C-metabolic flux analysis of 14 parallel labeling experiments in Escherichia coli",
        "DOI": "doi.org/10.1016/j.ymben.2015.01.001",
        "website": "https://www.sciencedirect.com/science/article/pii/S1096717615000038",
        "files": {
            "atom_transitions": "1-s2.0-S1096717615000038-mmc3.docx",
            "fluxes": "1-s2.0-S1096717615000038-mmc1.xlsx"
        },
        "combo_indices": [
            38,
            60
        ],
        "fluxml_file": "ecoli_model_level_3.fml"
    }
}
_anton_measurements = {
    # definitions taken from 1-s2.0-S1096717613000840-mmc1.pdf
    'Ala232': {
        'formula': 'C10H26ONSi2',
        'anton_pos': '2-3',
        'bigg_id': 'ala__L_c|[1,2]'
    },  # 2-3
    'Ala260': {
        'formula': 'C11H26O2NSi2',
        'anton_pos': '1-2-3',
        'bigg_id': 'ala__L_c',
    },
    'Gly218': {
        'formula': 'C9H24ONSi2',
        'anton_pos': '2',
        'bigg_id': 'gly_c|[1]',
    },
    'Gly246': {
        'formula': 'C10H24O2NSi2',
        'anton_pos': '1-2',
        'bigg_id': 'gly_c',
    },
    'Val260': {
        'formula': 'C12H30ONSi2',
        'anton_pos': '2-3-4-5',
        'bigg_id': 'val__L_c|[1,2,3,4]',
    },
    'Val288': {
        'formula': 'C13H30O2NSi2',
        'anton_pos': '1-2-3-4-5',
        'bigg_id': 'val__L_c',
    },
    'Leu274': {
        'formula': 'C13H32ONSi2',
        'anton_pos': '2-3-4-5-6',
        'bigg_id': 'leu__L_c|[1,2,3,4,5]',
    },
    'Ile274': {
        'formula': 'C13H32ONSi2',
        'anton_pos': '2-3-4-5-6',
        'bigg_id': 'ile__L_c|[1,2,3,4,5]',
    },
    'Ser362': {
        'formula': 'C16H40O2NSi3',
        'anton_pos': '2-3',
        'bigg_id': 'ser__L_c|[1,2]',
    },
    'Ser390': {
        'formula': 'C10H26ONSi2',
        'anton_pos': '1-2-3',
        'bigg_id': 'ser__L_c',
    },
    'Phe302': {
        'formula': 'C17H40O3NSi3',
        'anton_pos': '1-2',
        'bigg_id': 'phe__L_c|[0,1]',
    },
    'Phe308': {
        'formula': 'C16H30ONSi2',
        'anton_pos': '2-3-4-5-6-7-8-9',
        'bigg_id': 'phe__L_c|[1,2,3,4,5,6,7,8]',
    },
    'Asp302': {
        'formula': 'C14H32O2NSi2',
        'anton_pos': '1-2',
        'bigg_id': 'asp__L_c|[0,1]',
    },
    'Asp390': {
        'formula': 'C17H40O3NSi3',
        'anton_pos': '2-3-4',
        'bigg_id': 'asp__L_c|[1,2,3]',
    },
    'Asp418': {
        'formula': 'C18H40O4NSi3',
        'anton_pos': '1-2-3-4',
        'bigg_id': 'asp__L_c',
    },
    'Glu330': {
        'formula': 'C16H36O2NSi2',
        'anton_pos': '2-3-4-5',
        'bigg_id': 'glu__L_c|[1,2,3,4]',
    },
    'Glu432': {
        'formula': 'C19H42O4NSi3',
        'anton_pos': '1-2-3-4-5',
        'bigg_id': 'glu__L_c',
    },
    'Tyr302': {
        'formula': 'C14H32O2NSi2',
        'anton_pos': '1-2',
        'bigg_id': 'tyr__L_c|[0,1]',
    },
}
_anton_substrate_purity = {
    '[]Glc':  _nist_mass['C'][13][1],  # naturally labelled glucose

    # below are taken from https://doi.org/10.1016/j.ymben.2015.01.001
    '[1,2]Glc':  0.995,
    '[2,3]Glc': 0.995,
    '[4,5,6]Glc': 0.999,
    '[2,3,4,5,6]Glc': 0.985,

    #  The isotopic purity of [U-13C]glucose tracer was determined to be 98.4±0.2 https://doi.org/10.1016/j.ymben.2012.06.003
    '[U]Glc': 0.984,

    # below are taken from https://ars.els-cdn.com/content/image/1-s2.0-S1096717613000840-mmc1.pdf
    '[1]Glc': 0.996,
    '[2]Glc': 0.995,
    '[3]Glc': 0.995,
    '[4]Glc': 0.992,
    '[5]Glc': 0.990,
    '[6]Glc': 0.985,
}
_anton_labelling_map = {
    # mixtures are taken from main text: https://doi.org/10.1016/j.ymben.2015.01.001
    '[1,2]Glc': {
        '[1,2]Glc': 0.975,  # this means the rest is naturally labelled
        '[]Glc': 1.0-0.975,
    },
    '[2,3]Glc': {
        '[2,3]Glc': 0.975,
        '[]Glc': 1.0 - 0.975,
    },
    '[4,5,6]Glc': {
        '[4,5,6]Glc': 0.975,
        '[]Glc': 1.0 - 0.975,
    },
    '[2,3,4,5,6]Glc': {
        '[2,3,4,5,6]Glc': 0.975,
        '[]Glc': 1.0 - 0.975,
    },
    '[1] + [4,5,6]Glc (1:1)': {
        '[1]Glc': 0.475,
        '[4,5,6]Glc': 0.50,
    },
    '[1] + [U]Glc (1:1)': {
        '[1]Glc': 0.475,
        '[U]Glc': 0.5,
        '[]Glc': 1.0 - 0.5 - 0.475,
    },
    '[1] + [U]Glc (4:1)': {
        '[1]Glc': 0.77,
        '[U]Glc': 0.205,
        '[]Glc': 1.0 - 0.77 - 0.205,
    },
    '20% [U]Glc': {
        '[U]Glc': 0.205,
        '[]Glc': 1.0 - 0.205,
    },
    '[1]Glc': {
        '[1]Glc': 1.0,
    },
    '[2]Glc': {
        '[2]Glc': 1.0,
    },
    '[3]Glc': {
        '[3]Glc': 1.0,
    },
    '[4]Glc': {
        '[4]Glc': 1.0,
    },
    '[5]Glc': {
        '[5]Glc': 1.0,
    },
    '[6]Glc': {
        '[6]Glc': 1.0,
    },
    'COMPLETE-MFA': {
    },
}


def _parse_anton_substrates():
    result = {}
    glucose = Formula('C6H12O6')
    glc_positions = np.zeros(6, dtype=int)
    for anton_glc_id, purity in _anton_substrate_purity.items():
        glc_positions[:] = 0
        # TODO we assume that the non-labelled positions are ALL 100% 12C, Im not sure this interpretation of atom purity is correct
        labelled_isotop = glucose.copy()
        positions = anton_glc_id.rstrip('Glc')

        if positions == '[U]':
            positions = np.arange(6) + 1
        else:
            positions = eval(positions)

        glc_positions[np.array(positions, dtype=int) - 1] = 1
        main_bigg_glc_id = f"glc__D_e/{''.join((glc_positions.astype(str)))}"

        glucose_abundances = copy.deepcopy(_nist_mass)
        glucose_abundances['C'][12] = (12.0, 1.0 - purity)
        glucose_abundances['C'][13] = (13.0033548378, purity)
        if len(positions) > 0:
            labelled_isotop['C'] = len(positions)

        isotops = isotopologues(
            labelled_isotop,
            elements_with_isotopes=['C'],
            report_abundance=True,
            isotope_threshold=1e-4,
            overall_threshold=5e-4,
            abundances=glucose_abundances,
            n_mdv=None
        )
        bigg_isotops = {main_bigg_glc_id: 0.0}

        for (formula, abundance) in isotops:
            n_C13 = formula['[13]C']
            n_combos = comb(6, n_C13)
            if (n_C13 == labelled_isotop['C']) and (labelled_isotop['C'] > 0):
                bigg_isotops[main_bigg_glc_id] += abundance
                continue
            for positions in itertools.combinations(range(6), n_C13):
                glc_positions[:] = 0
                if len(positions) > 0:
                    glc_positions[np.array(positions) - 1] = 1
                bigg_glc_id = f"glc__D_e/{''.join((glc_positions.astype(str)))}"
                bigg_isotops[bigg_glc_id] = abundance / n_combos
        result[anton_glc_id] = bigg_isotops
    purity_df = pd.DataFrame(result).fillna(0.0)

    result = {}
    for labelling_id, mixture in _anton_labelling_map.items():
        if len(mixture) == 0:
            continue
        total = pd.Series(0.0, index=purity_df.index)
        for substrate_id, fraction in mixture.items():
            total += fraction * purity_df[substrate_id]
        result[labelling_id] = total
    susbtrate_df = pd.DataFrame(result)
    susbtrate_df /= susbtrate_df.sum(0)
    susbtrate_df = susbtrate_df.T
    susbtrate_df['co2_e/0'] = 0.989
    susbtrate_df['co2_e/1'] = 1.0 - susbtrate_df['co2_e/0'].values
    susbtrate_df.to_csv(os.path.join(MODEL_DIR, 'parsed_anton_substrates.csv'))


def read_anton_substrates(which_labellings=None):
    file = os.path.join(MODEL_DIR, 'parsed_anton_substrates.csv')
    if not os.path.isfile(file):
        _parse_anton_substrates()
    if which_labellings is None:
        which_labellings = slice(None)
    return pd.read_csv(file, index_col=0).loc[which_labellings]


def _parse_anton_fluxes():
    v_map = {}
    for pway, vdct in _anton_model_kwargs.items():
        if pway in ['MODEL_SOURCE', 'EXTRA']:
            continue
        for v_num, v_kwargs in vdct.items():
            bigg_kwargs = v_kwargs.get('bigg_kwargs')
            v_map[v_num] = {}
            for bigg_id, reaction_kwargs in bigg_kwargs.items():
                v_map[v_num][bigg_id] = reaction_kwargs.get('coeff', 1)

    net_fluxes = pd.read_excel(
        os.path.join(MODEL_DIR, '1-s2.0-S1096717615000038-mmc1.xlsx'), skiprows=9, nrows=80 - 9, index_col=0
    )
    xch_fluxes = pd.read_excel(
        os.path.join(MODEL_DIR, '1-s2.0-S1096717615000038-mmc1.xlsx'), skiprows=82, nrows=105 - 82, index_col=0
    )
    xch_fluxes = xch_fluxes.replace('>1000', 'Inf')
    xch_fluxes.columns = net_fluxes.columns

    others = (f'LB95.{1}', f'UB95.{1}', 'Reaction')
    selector = ['Best Fit'] + [f'Best Fit.{i}' for i in range(1, 15)]

    net_fluxes = net_fluxes.loc[:, selector].astype(float).replace([np.inf, -np.inf], 1e5)
    net_fluxes.columns = list(_anton_labelling_map.keys())
    net_fluxes.index = net_fluxes.index.astype(int)

    xch_fluxes = xch_fluxes.loc[:, selector].astype(float).replace([np.inf, -np.inf], 1e5)
    xch_fluxes.columns = list(_anton_labelling_map.keys())
    xch_fluxes.index = xch_fluxes.index.astype(int)

    v_map = pd.DataFrame(v_map).fillna(0.0)
    v_map.columns = v_map.columns.str.replace('v', '').astype(int)

    v_map_xch = v_map.loc[:, xch_fluxes.index]
    v_map_xch = v_map_xch.loc[np.linalg.norm(v_map_xch.values, axis=1) > 0.0]

    xch_fluxes = v_map_xch @ xch_fluxes
    net_fluxes = v_map.loc[:, net_fluxes.index] @ net_fluxes

    xch_fluxes = xch_fluxes / (abs(net_fluxes.loc[xch_fluxes.index]) + xch_fluxes)
    xch_fluxes = xch_fluxes.clip(lower=LabellingReaction._RHO_MIN, upper=LabellingReaction._RHO_MAX)
    xch_fluxes.index += '_xch'

    net_fluxes = (net_fluxes / -net_fluxes.loc['EX_glc__D_e']) * 10.0  # NB scales input fluxes to 10

    model, kwargs = build_e_coli_anton_glc(build_simulator=False)
    free_id = ['ME1', 'PGK', 'ICL', 'PGI', 'EDA', 'PPC', 'biomass_rxn', 'EX_glc__D_e', 'EX_ac_e']
    model.reactions.get_by_id('EX_glc__D_e').bounds = (-10.0, -10.0)
    model.build_model(free_reaction_id=free_id)
    fcm = FluxCoordinateMapper(model, kernel_id='rref')
    thermo_pol = extract_labelling_polytope(model, coordinate_id='thermo')
    net_pol = thermo_2_net_polytope(thermo_pol)
    # pickle.dump(thermo_pol, open('tp.p', 'wb'))

    # thermo_pol = pickle.load(open('tp.p', 'rb'))
    # net_pol = thermo_2_net_polytope(thermo_pol, verbose=True)
    simplified_net_pol = simplify_polytope(net_pol, settings=PolyRoundSettings(verbose=False), normalize=False)
    trans_pol, T, T_1, tau = transform_polytope_keep_transform(simplified_net_pol, kernel_id='rref')
    full_net_fluxes = T @ net_fluxes.loc[T.columns] + tau.values

    innnn = net_pol.A @ full_net_fluxes.loc[net_pol.A.columns]
    TOLERANCE = 1e-5
    in_pol = (innnn <= (net_pol.b.values[:, None] + TOLERANCE)).all(1)
    violated_constraints_and_b = pd.concat([innnn.loc[~in_pol], net_pol.b.to_frame().loc[~in_pol]], axis=1)

    # # difff = net_fluxes - full_net_fluxes.loc[net_fluxes.index]
    # # diff = np.isclose(difff, 0.0, atol=1e-3).all(1)
    # # diff = pd.Series(diff, index=net_fluxes.index)
    xch_fluxes.loc['HCO3E_xch'] = 0.3 #  only thing missing and not deducible from the input
    thermo_fluxes = pd.concat([full_net_fluxes, xch_fluxes]).T.loc[:, thermo_pol.A.columns]
    # theta = model._fcm.map_fluxes_2_theta(thermo_fluxes, is_thermo=True, pandalize=True)
    # thermo_fluxes_2 = model._fcm.map_theta_2_fluxes(theta, return_thermo=True, pandalize=True)
    # thermo_fluxes_2.index = thermo_fluxes_2.index+ '_2'
    # thermo_flux_compare = pd.concat([thermo_fluxes, thermo_fluxes_2])
    # thermo_flux_compare = thermo_flux_compare.sort_index()
    #
    # eqqi = thermo_pol.S @ thermo_fluxes.T.values
    # equality = (np.isclose((eqqi), 0.0, atol=1e-8)).all(1)
    # where_not = thermo_pol.A.loc[~in_pol]
    # where_not = where_not.loc[:, np.linalg.norm(where_not.values, axis=0) > 0.0]

    thermo_fluxes.to_csv(os.path.join(MODEL_DIR, 'parsed_anton_fluxes.csv'))
    return thermo_fluxes


def read_anton_fluxes(labelling_ids='COMPLETE-MFA'):
    file = os.path.join(MODEL_DIR, 'parsed_anton_fluxes.csv')
    if not os.path.isfile(file):
        _parse_anton_fluxes()
    if labelling_ids is None:
        labelling_ids = slice(None)
    elif isinstance(labelling_ids, str):
        labelling_ids = [labelling_ids]
    return pd.read_csv(file, index_col=0).loc[labelling_ids]


def _parse_anton_measurements(recompute=False, verbose=False, adduct_name='M-'):
    # https://pubs.acs.org/doi/full/10.1021/ac0708893  publication on correction of natural abundances cited by Antoniewicz
    measurements = pd.read_excel(
        os.path.join(MODEL_DIR, '1-s2.0-S1096717615000038-mmc5.xlsx'), skiprows=3, nrows=107 + 3 - 7, index_col=0
    ).dropna(axis=1, how='all').dropna(axis=0, how='all').dropna(axis=1, how='any')
    meas_ids = measurements.index.to_series().str.split(' ', expand=True).rename({0: 'meas_id', 1: 'M'}, axis=1)

    # meas_ids['Mint'] = meas_ids['M'].str.lstrip('(M').str.rstrip(')').astype(int)
    # mapper = {k: _anton_measurements[k]['bigg_id'] for k in _anton_measurements}
    # meas_ids['bigg_id'] = meas_ids['meas_id'].map(mapper)

    # corr_mat_1s = []
    corr_mats = []
    sum_meas_id = []
    sum_mix_id = []
    sum_bigg_id = []
    for i, (meas_id, count) in enumerate(meas_ids['meas_id'].value_counts(sort=False).iteritems()):
        meas_dct = _anton_measurements[meas_id]
        formula = Formula(meas_dct['formula'])
        if verbose:
            print(meas_id, formula.to_chnops(), formula.mass(charge=-1))
        n_MDV = len(meas_dct['anton_pos'].split('-')) + 1
        formula['C'] -= (n_MDV - 1)
        index = meas_ids.loc[meas_ids['meas_id'] == meas_id].index
        bigg_id = _anton_measurements[meas_id]['bigg_id']

        adduct_str = f'_{{{adduct_name}}}'
        if adduct_name == 'M-H':
            adduct_str = ''
        columns = [f'{bigg_id}{adduct_str}+{i}' for i in range(n_MDV)]

        if recompute:
            corr_mat = build_correction_matrix(formula, exclude_carbon=False, n_mdv=count)
            # corr_mat_1 = np.linalg.inv(corr_mat)
            # corr_mat_1 = corr_mat_1 / corr_mat_1.sum(0)[None, :]
            # corr_mat_1s.append(pd.DataFrame(corr_mat_1[:, :n_C], index=index, columns=columns))
            corr_mats.append(pd.DataFrame(corr_mat[:, :n_MDV], index=index, columns=columns))

        sum_meas_id.append(pd.DataFrame(np.ones((count, count)), index=index, columns=index))
        sum_mix_id.append(pd.DataFrame(np.ones((count, n_MDV)), index=index, columns=columns))
        sum_bigg_id.append(pd.DataFrame(np.ones((n_MDV, n_MDV)), index=columns, columns=columns))

    if recompute:
        # corr_mat_1s = pd.concat(corr_mat_1s).fillna(0.0)
        # corr_mat_1s.to_csv('corr_mat_1.csv')
        corr_mats = pd.concat(corr_mats).fillna(0.0)
        corr_mats.to_csv('corr_mat.csv')
    else:
        # corr_mat_1s = pd.read_csv('corr_mat_1.csv', index_col=0)
        corr_mats = pd.read_csv('corr_mat.csv', index_col=0)

    sum_meas_id = pd.concat(sum_meas_id).fillna(0.0)
    sum_mix_id = pd.concat(sum_mix_id).fillna(0.0)
    sum_bigg_id = pd.concat(sum_bigg_id).fillna(0.0)

    measurements = measurements / (sum_meas_id @ measurements)  # making sure measurements sum to 1

    x = cp.Variable(corr_mats.shape[1], name='corrected')
    S = cp.Constant(sum_meas_id.values)
    S2 = cp.Constant(sum_mix_id.values)
    x_meas = cp.Parameter(corr_mats.shape[0])

    A = cp.Constant(corr_mats.values)
    cost = cp.sum_squares(A @ x - x_meas)

    # A_1 = cp.Constant(corr_mat_1s.values)
    # cost = cp.sum_squares(x - A_1.T @ x_meas)

    problem = cp.Problem(objective=cp.Minimize(cost), constraints=[
        x >= 0.0,  # non-negative labelling state
        # S2 @ x == 1.0,  # labelling state sum to 1
        S @ A @ x == 1.0,  # measured mdv sum to 1
    ])

    result = {}
    for anton_labelling_id, measurement in measurements.T.iterrows():
        x_meas.value = measurement.values
        sol = problem.solve()
        if problem.status != 'optimal':
            print(problem.status)
            raise ValueError
        result[anton_labelling_id] = pd.Series(x.value, index=sum_mix_id.columns, dtype=float)
        # meass = corr_mats @ x.value  # should be close to measurement
        # mess_1 = corr_mat_1s.T @ meass  # should be close to x
    parsed_measurements = pd.DataFrame(result)
    parsed_measurements = parsed_measurements / (sum_bigg_id @ parsed_measurements)
    # TODO: in the PDF 1-s2.0-S1096717613000840-mmc1.pdf, they report corrected MDVs for the singly labelled substrates
    #   these show minor differences from what we compute; but they have negative values somehow...
    parsed_measurements.to_csv(os.path.join(MODEL_DIR, 'parsed_anton_measurements.csv'))


def read_anton_measurements(
        labelling_ids: Iterable = ['20% [U]Glc', '[1]Glc'],
        measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
        std=0.1
):
    file = os.path.join(MODEL_DIR, 'parsed_anton_measurements.csv')

    if not os.path.isfile(file):
        _parse_anton_measurements(recompute=True)

    if labelling_ids is None:
        labelling_ids = slice(None)

    measurements = pd.read_csv(file, index_col=0).loc[:, labelling_ids]

    # ['met_id', 'formula', 'adduct_name', 'nC13']
    annotation_df = measurements.index.str.split('+', expand=True).to_frame().reset_index(drop=True).rename({
        0: 'met_id', 1: 'nC13'
    }, axis=1).astype({'nC13': int})
    annotation_df['adduct_name'] = annotation_df['met_id'].str.extract('_\{(.*)\}').fillna('M-H')
    annotation_df['met_id'] = annotation_df['met_id'].str.replace('_\{(.*)\}', '', regex=True)
    mapper = {v['bigg_id']: v['formula'] for k, v in _anton_measurements.items()}
    annotation_df['formula'] = annotation_df['met_id'].map(mapper)
    annotation_df['sigma'] = std

    measurements = measurements.T.stack()
    if measured_boundary_fluxes is not None:
        # take the average exchange fluxes of the labelling_ids as measurements
        fluxes = read_anton_fluxes(labelling_ids).loc[:, measured_boundary_fluxes].mean(0)
        fluxes.index = make_multidex({'BOM': fluxes.index}, name1='data_id')
        measurements = pd.concat([measurements, fluxes])
    measurements.index.names = ['labelling_id', 'data_id']
    return measurements, annotation_df


def _read_model(name='anton'):
    file_names = {
        'core': 'e_coli_core.xml',
        'anton': 'e_coli_anton.xml',
        'ijo': 'iJO1366.xml',
        'spiro': 'spiro.xml',
        'tomek': 'e_coli_tomek.xml',
    }
    if name not in file_names:
        raise ValueError('no such xml')
    file = os.path.join(MODEL_DIR, 'sbml', file_names[name])
    model = read_sbml_model(file)
    return model


def _parse_anton_model():
    # Antoniewicz adds ATP as a reactant every time ATP is hydrolysed to AMP + Ppi to compensate for AMP+ATP -> 2ADP
    #   we dont do this, since we add the PPA reaction

    # TODO THIS SOMEHOW MODIFIES ANTON_KWARGS UNINTENTIONALLY!
    ijo = _read_model('ijo')
    ijo.add_metabolites([
        Metabolite(id='TKT_c',  formula='C2', compartment='c'),
        Metabolite(id='TALA_c', formula='C3', compartment='c'),
    ])
    core = _read_model('core')

    anton_model = Model(id_or_model='Antoniewicz')
    anton_model.annotation = core.annotation
    anton_model.notes = core.notes

    reactions_to_add = []
    for pway, vdct in _anton_model_kwargs.items():
        if pway == 'MODEL_SOURCE':
            continue
        for v_num, v_kwargs in vdct.items():
            bigg_kwargs = v_kwargs.get('bigg_kwargs')
            for bigg_id, reaction_kwargs in bigg_kwargs.items():
                metabolite_dct = reaction_kwargs.get('metabolite_dct', None)
                to_combine = reaction_kwargs.get("to_combine", None)
                atom_map_str = reaction_kwargs.get('atom_map_str', None)

                if atom_map_str is not None:
                    reaction_kwargs[bigg_id] = atom_map_str

                if metabolite_dct is not None:
                    if bigg_id in ijo.reactions:
                        reac = ijo.reactions.get_by_id(bigg_id)
                    else:
                        reac = Reaction(bigg_id)
                    reac.add_metabolites({ijo.metabolites.get_by_id(k).copy(): v for k, v in metabolite_dct.items()}, combine=True)
                elif to_combine is not None:
                    reac = Reaction(bigg_id)
                    for combo_id, combo_kwargs in to_combine.items():
                        coeff = combo_kwargs.get('coeff', 1)
                        if coeff == 0:
                            continue
                        if combo_id in core.reactions:
                            combo_reac = core.reactions.get_by_id(combo_id)
                        elif combo_id in ijo.reactions:
                            combo_reac = ijo.reactions.get_by_id(combo_id)
                        reac += combo_reac * coeff
                else:
                    if bigg_id in core.reactions:
                        reac = core.reactions.get_by_id(bigg_id).copy()
                    elif bigg_id in ijo.reactions:
                        reac = ijo.reactions.get_by_id(bigg_id).copy()
                    else:
                        raise ValueError

                if bigg_id in ['his_syn', 'cys_syn']:
                    to_correct = reac.metabolites
                    mets = DictList(to_correct)
                    gtp = mets.get_by_id('gtp_c')
                    gdp = mets.get_by_id('gdp_c')
                    atp = mets.get_by_id('atp_c')
                    adp = ijo.metabolites.get_by_id('adp_c')
                    correction = {
                        gtp: -to_correct[gtp],
                        gdp: -to_correct[gdp],
                        atp: to_correct[gtp],
                        adp: to_correct[gdp],
                    }
                    reac.add_metabolites(correction, combine=True)
                elif bigg_id == 'arg_syn':
                    to_correct = reac.metabolites
                    mets = DictList(to_correct)
                    hco_3 = mets.get_by_id('hco3_c')
                    reac.add_metabolites({
                        ijo.metabolites.get_by_id('co2_c'): to_correct[hco_3],
                        hco_3: -to_correct[hco_3]
                    }, combine=True)

                lower_bound = reaction_kwargs.get('lower_bound', None)
                upper_bound = reaction_kwargs.get('upper_bound', None)
                if lower_bound is not None:
                    reac.lower_bound = lower_bound
                if upper_bound is not None:
                    reac.upper_bound = upper_bound

                reactions_to_add.append(reac.copy())
                # if atom_map_str is not None:
                #     reac = LabellingReaction(reac)
                #     atom_map = reac.build_atom_map_from_string(atom_map_str, metabolite_kwargs=_metabolite_kwargs)
                #     reac.set_atom_map(atom_map)
                # print(kwargs.get('anton_str'))
                # print(reac)

    anton_model.add_reactions(reactions_to_add)
    anton_model.objective = {anton_model.reactions.get_by_id('biomass_rxn'): 1}
    print(anton_model.optimize())
    cobra.io.write_sbml_model(anton_model, os.path.join(MODEL_DIR, 'sbml', 'e_coli_anton.xml'))
    return anton_model


def _correct_base_bayes_lcms(basebayes, total_intensities=None, clip_min=750.0, min_intensity_factor=500.0/750.0):
    measurements = basebayes.simulate_true_data(n_obs=1, pandalize=True).iloc[[0]]
    # these are slightly clipped intensities!
    partial_mdvs = basebayes.to_partial_mdvs(measurements, normalize=False, pandalize=True).iloc[[0]]

    corrected_annot_dfs = {}
    for labelling_id, part_mdv in partial_mdvs.T.groupby(level=0):
        if labelling_id == 'BOM':
            continue
        obmod = basebayes._obmods[labelling_id]
        total_intens = basebayes._la.tonp(obmod._scaling)
        intensities = (part_mdv.T * total_intens).T
        measurable = part_mdv.loc[(intensities > min_intensity_factor * clip_min).values].droplevel(0)
        multiple_signals = measurable.index.str.rsplit('+', n=1, expand=True).to_frame(name=['met_id', 'nC13'])
        measurable = measurable.loc[multiple_signals['met_id'].duplicated(keep=False).values]
        obs_df = obmod.observation_df.loc[measurable.index]
        corrected_annot_dfs[labelling_id] = obmod._annotation_df.loc[obs_df['annot_df_idx']].reset_index(drop=True)

    obsmods = LCMS_ObservationModel.build_models(
        basebayes._model,
        corrected_annot_dfs,
        total_intensities=total_intensities if total_intensities is not None else obmod.scaling,
        clip_min=clip_min,  # now we create the real observation models!
    )
    corrected_basebayes = _BaseBayes(basebayes._model, basebayes._substrate_df, obsmods, basebayes._prior, basebayes._bom)
    corrected_basebayes.set_true_theta(basebayes.true_theta)
    return corrected_basebayes


def build_e_coli_anton_glc(
        backend='numpy',
        auto_diff=False,
        build_simulator=True,
        ratios=False,
        batch_size=1,
        which_measurements: str='tomek',
        which_labellings=['20% [U]Glc', '[1]Glc'],
        measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
        seed=1,
) -> (LabellingModel, dict):
    if (which_measurements is not None) and not build_simulator:
        raise ValueError

    if which_labellings is not None:
        substrate_df = read_anton_substrates(which_labellings=which_labellings)
    model = _read_model('anton')

    atom_mappings = {}
    for pway, vdct in _anton_model_kwargs.items():
        if pway == 'MODEL_SOURCE':
            continue
        for v_num, v_kwargs in vdct.items():
            bigg_kwargs = v_kwargs.get('bigg_kwargs')
            for bigg_id, reaction_kwargs in bigg_kwargs.items():
                atom_mappings[bigg_id] = reaction_kwargs

    measured_metabolites = None
    if which_measurements == 'anton':
        measured_metabolites = [v['bigg_id'] for k, v in _anton_measurements.items()]
    elif which_measurements == 'tomek':
        file = os.path.join(MODEL_DIR, 'LCMS_total_intensity.csv')
        total_df = pd.read_csv(file)
        total_df = total_df.loc[total_df['met_id'].isin(model.metabolites.list_attr('id'))]
        measured_metabolites = total_df['met_id'].tolist()

    model = simulator_factory(
        id_or_file_or_model=model,
        backend=backend,
        reaction_kwargs=atom_mappings,
        metabolite_kwargs=_metabolite_kwargs,
        measurements=measured_metabolites,
        input_labelling=substrate_df.iloc[0] if which_labellings is not None else None,
        ratio_repo=_gluc_ratio_repo,
        ratios=ratios,
        build_simulator=build_simulator,
        auto_diff=auto_diff,
        batch_size=batch_size,
        seed=seed,
    )
    # if build_simulator:  # TODO wy did we do this again??
    #     model.build_simulator()

    annotation_df, measurements, basebayes = None, None, None
    if which_measurements == 'anton':
        measurements, annotation_df = read_anton_measurements(which_labellings, measured_boundary_fluxes)
    elif which_measurements == 'tomek':
        mapper = {mid: model.metabolites.get_by_id(mid).formula for mid in total_df['met_id']}
        total_df['formula'] = total_df['met_id'].map(mapper)
        total_df = total_df.loc[total_df.index.repeat(total_df['nC'] + 1)]
        annotation_df = []
        for i, df in total_df.groupby('met_id'):
            df['nC13'] = np.arange(df.shape[0])
            annotation_df.append(df)
        annotation_df = pd.concat(annotation_df, axis=0)

    if annotation_df is not None:
        observation_df = MDV_ObservationModel.generate_observation_df(model, annotation_df)

    if which_measurements == 'anton':
        sigma_ii = observation_df['sigma']
        omega = None
        annotation_dfs = {labelling_id: (annotation_df, sigma_ii, omega) for labelling_id in substrate_df.index}
        obsmods = ClassicalObservationModel.build_models(model, annotation_dfs)
    elif which_measurements == 'tomek':
        annotation_dfs = {labelling_id: annotation_df for labelling_id in substrate_df.index}
        total_intensities = observation_df.drop_duplicates('ion_id').set_index('ion_id')['total_I']
        obsmods = LCMS_ObservationModel.build_models(
            model,
            annotation_dfs,
            total_intensities=total_intensities,
            clip_min=1e-12,  # we need to post-process the measurements to create appropriate annotation_df, hence the low clip_min
        )

    thermo_fluxes, theta, comparison = None, None, None
    if annotation_df is not None:
        bom = MVN_BoundaryObservationModel(model, measured_boundary_fluxes, _bmid_ANTON)
        up = UniformRoundedFleXchPrior(model)
        basebayes = _BaseBayes(model, substrate_df, obsmods, up, bom)

        thermo_fluxes = read_anton_fluxes()
        fluxes = model._fcm.map_thermo_2_fluxes(thermo_fluxes=thermo_fluxes, pandalize=True)
        theta = model._fcm.map_fluxes_2_theta(fluxes, pandalize=True)
        basebayes.set_true_theta(theta.iloc[0])
        if which_measurements == 'anton':
            measurements = measurements.loc[basebayes.data_id]
        elif which_measurements == 'tomek':
            basebayes = _correct_base_bayes_lcms(basebayes)
            measurements = basebayes.simulate_true_data(n_obs=0, pandalize=True).iloc[[0]]

            # intensities = intensities.loc[intensities > fobmod._cmin * 0.6]
        measurements.name = which_measurements

        if which_measurements == 'tomek':
            anton_meas, anton_annot = read_anton_measurements(which_labellings, measured_boundary_fluxes)
            anton_meas = anton_meas.to_frame().T

            anton_cols = pd.Index([f'{i[0]}: {i[1]}' for i in anton_meas.columns.tolist()])
            new_anton_cols = pd.Index(anton_cols).str.replace(r'\_\{.*\}', '', regex=True)
            anton_meas.columns = new_anton_cols

            part_mdv_measurements = basebayes.to_partial_mdvs(measurements, pandalize=True)
            bb = pd.Index([f'{i[0]}: {i[1]}' for i in part_mdv_measurements.columns.tolist()])
            part_mdv_measurements.columns = bb

            intersect_cols = new_anton_cols.intersection(bb)
            intersect_cols = intersect_cols.append(pd.Index([f'{i[0]}: {i[1]}' for i in bom.boundary_id.tolist()]))

            full_anton_map = dict(zip(anton_cols, new_anton_cols))
            intersect_map = {k: v for k, v in full_anton_map.items() if v in intersect_cols}
            comparison = {
                'part_mdv_measurements': part_mdv_measurements,
                'anton_meas':anton_meas,
                'intersect_cols': intersect_cols,
                'full_anton_map': full_anton_map,
                'intersect_map': intersect_map,
            }

    kwargs = {
        'annotation_df': annotation_df,
        'substrate_df': substrate_df,
        'measured_boundary_fluxes': measured_boundary_fluxes,
        'measurements': measurements,
        'basebayes': basebayes,
        'ratio_repo': _gluc_ratio_repo,
        'thermo_fluxes': thermo_fluxes,
        'theta': theta,
        'comparison': comparison,
    }

    return model, kwargs


def _parse_tomek_model():
    # TODO include all the reactions in _extra
    core = _read_model('core')
    ijo = _read_model('ijo')

    for reac in core.boundary:
        met = list(reac.metabolites.keys())[0]
        if 'C' in met.elements:
            reac.bounds = (0.0, 0.0)

    core.reactions.get_by_id('EX_glc__D_e').bounds = (-12.0, 0.0)
    core.reactions.get_by_id('EX_ac_e').bounds = (0.0, 1000.0)
    core.reactions.get_by_id('EX_co2_e').bounds = (-1000.0, 0.0)

    core.reactions.get_by_id('FORt2').bounds = (0.0, 0.0)

    bm = core.reactions.get_by_id(_bmid_GAM)
    akg = core.metabolites.get_by_id('akg_c')
    glu = core.metabolites.get_by_id('glu__L_c')  # biomass should not produce carbon containing molecules
    bm -= core.reactions.get_by_id('GLUDy') * bm.metabolites[akg]
    bm._metabolites[glu] = round(bm.metabolites[glu], 4)

    EDD = ijo.reactions.get_by_id('EDD').copy()
    EDA = ijo.reactions.get_by_id('EDA').copy()

    co2 = core.reactions.get_by_id('CO2t')
    co2.bounds = (0.0, 1000.0)
    co2_out = Reaction(id='OUT_CO2t', lower_bound=0.0, upper_bound=1000.0)
    co2_out.add_metabolites({core.metabolites.get_by_id('co2_c'): -1})
    core.add_reactions(reaction_list=[co2_out, EDD, EDA])
    print(core.optimize())
    cobra.io.write_sbml_model(core, os.path.join(MODEL_DIR, 'sbml', 'e_coli_tomek.xml'))


def build_e_coli_tomek(
        backend='numpy',
        auto_diff=False,
        build_simulator=False,
        ratios=False,
        batch_size=1,
        anton_measured=True,
        labelling_id='[1]Glc',
        for_pta=False
):
    model = _read_model('tomek')

    substrate_df = read_anton_substrates()
    measurements = [
        'gln__L_c',
        'glu__L_c',
        'lac__D_c',
        'mal__L_c',
        '13dpg_c',
        '2pg_c',
        '3pg_c',
        'oaa_c',
        'pep_c',
        '6pgc_c',
        'ac_c',
        'pyr_c',
        'acald_c',
        's7p_c',
        'succ_c',
        'acon_C_c',
        'akg_c',
        'g3p_c',
    ]
    annot_df = []
    for rid in measurements:
        metabolite = model.metabolites.get_by_id(rid)
        annot_df.extend([(rid, metabolite.formula, 'M-H', i) for i in range(metabolite.elements['C'])])

    annot_df = pd.DataFrame(annot_df, columns = ['met_id', 'formula', 'adduct_name', 'nC13'])
    kwargs = {
        'substrate_df': substrate_df,
        'annotation_df': annot_df,
        'input_labelling': substrate_df.loc[labelling_id] if labelling_id else labelling_id,
        'ratio_repo': _gluc_ratio_repo,
        'measured_boundary_fluxes': ['EX_glc__D_e', 'EX_ac_e', 'BIOMASS_Ecoli_core_w_GAM'],
    }

    atom_mappings = {}
    for pway, vdct in _anton_model_kwargs.items():
        if pway == 'MODEL_SOURCE':
            continue
        for v_num, v_kwargs in vdct.items():
            bigg_kwargs = v_kwargs.get('bigg_kwargs')
            for bigg_id, reaction_kwargs in bigg_kwargs.items():
                if bigg_id in model.reactions:
                    reaction_kwargs.pop('lower_bound', None)
                    reaction_kwargs.pop('upper_bound', None)
                    atom_mappings[bigg_id] = reaction_kwargs

    for bigg_id, reaction_kwargs in _extra_model_kwargs.items():
        if bigg_id in model.reactions:
            reaction_kwargs.pop('lower_bound', None)
            reaction_kwargs.pop('upper_bound', None)
            atom_mappings[bigg_id] = reaction_kwargs

    model = simulator_factory(
        id_or_file_or_model=model,
        backend=backend,
        reaction_kwargs=atom_mappings,
        metabolite_kwargs=_metabolite_kwargs,
        measurements=measurements,
        input_labelling=kwargs.get('input_labelling', None),
        ratio_repo=kwargs.get('ratio_repo', None),
        ratios=ratios,
        build_simulator=False,
        auto_diff=auto_diff,
        batch_size=batch_size,
    )
    if for_pta:
        model.reactions.get_by_id('BIOMASS_Ecoli_core_w_GAM').bounds = (0.0, 1000.0)
        model.reactions.get_by_id('ATPM').bounds = (0.0, 1000.0)

    model.reactions.get_by_id('ACt2r').bounds = (-1000.0, 0.0)
    model.reactions.get_by_id('FBP').bounds = (0.0, 0.0)
    model.reactions.get_by_id('SUCCt3').bounds = (0.0, 0.0)
    if build_simulator:
        model.build_model()
    return model, kwargs


def simulator_factory(
        id_or_file_or_model='dummy',
        name=None,
        backend='numpy',
        solver='lu_solve',
        batch_size=1,
        auto_diff=False,
        fkwargs=None,
        reaction_list=None,
        reaction_kwargs=None,
        metabolite_kwargs=None,
        input_labelling=None,
        ratio_repo=None,
        measurements=None,
        build_simulator=False,
        device='cpu',
        ratios=True,
        seed=None,
        free_reaction_id=None,
) -> LabellingModel:
    if id_or_file_or_model is not None:
        try:
            id_or_file_or_model = read_sbml_model(id_or_file_or_model)
        except:
            pass

    linalg = LinAlg(
        backend=backend, batch_size=batch_size, solver=solver, device=device,
        fkwargs=fkwargs, auto_diff=auto_diff, seed=seed
    )

    kwargs = {
        'id_or_model':      id_or_file_or_model,
        'name':             name,
        'linalg':           linalg,
    }

    if ratios:
        model_type = RatioEMU_Model
    else:
        model_type = EMU_Model

    model = model_type(**kwargs)
    model.add_reactions(
        reaction_list=reaction_list,
        reaction_kwargs=reaction_kwargs,
        metabolite_kwargs=metabolite_kwargs
    )

    if (ratio_repo is not None) and ratios:
        model.set_ratio_repo(ratio_repo=ratio_repo)

    if input_labelling is not None:
        model.set_substrate_labelling(substrate_labelling=input_labelling)
    if measurements is not None:
        model.set_measurements(measurement_list=measurements)
    if build_simulator:
        model.build_model(free_reaction_id=free_reaction_id)
    return model


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    np.set_printoptions(linewidth=500)

    map_back = {}
    # from optlang.gurobi_interface import

    # model, kwargs = build_e_coli_anton_glc(backend='torch', build_simulator=True, which_measurements='tomek')
    model, kwargs = build_e_coli_anton_glc(backend='torch', build_simulator=False, which_measurements=None)

    print(len(model.labelling_reactions))

