from cobra import Metabolite, Object, DictList
from sbmfi.lcmsanalysis.formula import Formula
import numpy as np
import re
from abc import abstractmethod
from typing import Dict, Union


# TODO: CHARGES ARE NOT REGISTERED CORRECTLY!


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
            metabolite: Metabolite,
            symmetric: bool = False,
            formula: str = '',
            total_intensity = None,  # either a number or a distribution from which total intensities are sampled
    ):
        if isinstance(metabolite, LabelledMetabolite):
            raise NotImplementedError
        elif isinstance(metabolite, Metabolite):  # only if metabolite
            self.__dict__.update(metabolite.__dict__)
            self.formula = formula if formula else self.__dict__.pop('formula')
        else:
            raise ValueError(f'need to instantiate with a cobra Metabolite object, got {type(metabolite)}')
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

    @elements.setter
    def elements(self, elements_dict: Dict[str, Union[int, float]]) -> None:
        raise NotImplementedError

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


if __name__ == "__main__":
    water = EMU_Metabolite(metabolite='water', formula='H2O', charge=1)