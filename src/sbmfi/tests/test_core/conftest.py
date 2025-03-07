import pytest
from sumoflux.core.model import LabellingModel, EMU_Model, CumomerModel
from collections import OrderedDict
import pandas as pd

reaction_kwargs = {
    'EX_A': {
        'lower_bound': 100.0, 'upper_bound': 100.0, 'atom_map_str': '∅ --> A_e/abc'
    },
    'EX_E': {
        'upper_bound': 100.0, 'atom_map_str': 'E_c/a --> ∅'
    },
    'EX_F': {
        'upper_bound': 100.0, 'atom_map_str': 'F_c/abc --> ∅'
    },

    'v1': {
        'upper_bound': 100.0, 'atom_map_str': 'A_e/abc --> B_c/abc'
    },
    'v2': {
        'lower_bound': -100.0, 'upper_bound': 100.0,
        'xch_ub': 0.5,
        'atom_map_str': 'B_c/abc <=> D_c/abc',
    },
    'v4': {
        'upper_bound': 100.0, 'atom_map_str': 'B_c/abc --> C_c/bc + E_c/a'
    },
    'v5': {
        'upper_bound': 100.0, 'atom_map_str': 'B_c/abc + C_c/de --> D_c/bcd + E_c/a + E_c/e'
    },
    'v6': {
        'upper_bound': 100.0, 'atom_map_str': 'D_c/abc --> F_c/abc'
    },
    'vp': { # pseudo-reaction
        'pseudo': True, 'upper_bound': 0.0,
        'atom_map_str': 'B_c/abc + D_c/def + C_c/gh + E_c/i + A_e/jkl --> L_c/abdgil'
    },

    'cobra_R': { # cobra-reaction
        'upper_bound': 10.0,
    }
}
ratio_repo = {
    'E|v2': {
        'numerator': OrderedDict({'v5': 1}),
        'denominator': OrderedDict({'v2': 1, 'v2_rev': -1, 'v5': 1})},
}
input_labelling = OrderedDict([('A_e/010', 1.0)])
measured_metabolites = ['F_c', 'B_c']#, 'L_c']
fluxes = pd.Series({ # NOTE: from the EMU paper
    'EX_A': 100,
    'EX_E': 60,
    'EX_F': 80,
    'v1': 100,
    'v2': 110,
    'v2_rev': 50,
    'v4': 20,
    'v5': 20,
    'v6': 80,
}, dtype=float, name='emu_paper')

def parametrize(model):
    model.add_reactions(reaction_kwargs=reaction_kwargs)
    model.map_fluxes(fluxes=fluxes)
    model.set_ratio_repo(ratio_repo=ratio_repo)
    model.set_substrate_labelling(substrate_labelling=input_labelling)
    model.set_measurements(measurement_list=measured_metabolites, exclude=False)
    return model

@pytest.fixture(scope="function")
def emu_SUModel():
    M = LabellingModel(
        id_or_model='test_SUModel',
        name='test_SUModel',
    )
    return parametrize(model=M)

@pytest.fixture(scope="function")
def emu_CUModel():
    M = CumomerModel(
        id_or_model='test_SUModel',
        name='test_SUModel',
    )
    return parametrize(model=M)

@pytest.fixture(scope="function")
def emu_EMUdel():
    M = EMU_Model(
        id_or_model='test_SUModel',
        name='test_SUModel',
    )
    return parametrize(model=M)

def nofix_emu_EMUdel():
    M = EMU_Model(
        id_or_model='test_SUModel',
        name='test_SUModel',
    )
    return parametrize(model=M)


if __name__ == "__main__":
    emu_SUModel()

