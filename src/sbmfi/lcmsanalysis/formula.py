from sbmfi.lcmsanalysis.nist_mass import _nist_mass
import re
from math import factorial
import numpy as np
from collections import defaultdict, Counter
from itertools import product, combinations_with_replacement
import math
# ALMOST ALL STOLEN FROM PYTEOMICS!

# unique _chnops ordering such that Formula.to_chnops() produces unique strings for every possible isotopomer
_chnops = ['C', 'H', 'N', 'O', 'P', 'S']
_chnops = _chnops + [element for element in _nist_mass.keys() if element not in _chnops]
_chnops = {k: v for k, v in zip(_chnops, range(len(_chnops)))}
_isotope_pat = r'(?:\[(\d+)\])?([A-Z][a-z]*)(\d*)'
_isotope_rex = re.compile(_isotope_pat)
_charge_pat = r'([+-])(\d*)$'
_charge_rex = re.compile(_charge_pat)
_formula_pat = fr'^({_isotope_pat})*({_charge_pat})*$'
_formula_rex = re.compile(_formula_pat.replace(r'\\', r'\''))


class FormulAlgebra(defaultdict, Counter):
    """A generic dictionary for compositions.
    Keys should be strings, values should be integers.
    Allows simple arithmetics."""

    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, int)
        Counter.__init__(self, *args, **kwargs)
        for k, v in list(self.items()):
            if not v:
                del self[k]

    def __str__(self):
        return '{}({})'.format(type(self).__name__, dict.__repr__(self))

    def __repr__(self):
        return str(self)

    def _repr_pretty_(self, p, cycle):
        if cycle:  # should never happen
            p.text('{} object with a cyclic reference'.format(type(self).__name__))
        p.text(str(self))

    def __add__(self, other):
        result = self.copy()
        for elem, cnt in other.items():
            result[elem] += cnt
        return result

    def __iadd__(self, other):
        for elem, cnt in other.items():
            self[elem] += cnt
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        result = self.copy()
        for elem, cnt in other.items():
            result[elem] -= cnt
        return result

    def __isub__(self, other):
        for elem, cnt in other.items():
            self[elem] -= cnt
        return self

    def __rsub__(self, other):
        return (self - other) * (-1)

    def __mul__(self, other):
        if not isinstance(other, int):
            raise ValueError('Cannot multiply Composition by non-integer',
                                 other)
        return type(self)({k: v * other for k, v in self.items()})

    def __imul__(self, other):
        if not isinstance(other, int):
            raise ValueError('Cannot multiply Composition by non-integer',
                                 other)
        for elem in self:
            self[elem] *= other
        return self

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        if not isinstance(other, dict):
            return False
        self_items = {i for i in self.items() if i[1]}
        other_items = {i for i in other.items() if i[1]}
        return self_items == other_items

    # override default behavior:
    # we don't want to add 0's to the dictionary
    def __missing__(self, key):
        return 0

    def __setitem__(self, key, value):
        if math.isnan(value):
            value = 0
        if isinstance(value, float):
            value = int(round(value))
        elif not isinstance(value, int):
            raise ValueError('Only integers allowed as values in '
                                 'Composition, got {}.'.format(type(value).__name__))
        if value:  # reject 0's
            super(FormulAlgebra, self).__setitem__(key, value)
        elif key in self:
            del self[key]

    def copy(self):
        return type(self)(self)

    def __copy__(self):
        return self.copy()

    def __reduce__(self):
        class_, args, state, list_iterator, dict_iterator = super(
            FormulAlgebra, self).__reduce__()
        # Override the reduce of defaultdict so we do not provide the
        # `int` type as the first argument
        # which prevents from correctly unpickling the object
        args = ()
        return class_, args, state, list_iterator, dict_iterator


class Formula(FormulAlgebra):
    """Chemical formula stored as a dictionary-like structure.
    Contains convenient functions like the calculation of mono-isotopic mass and
    natural abundance.
    """
    # NOTE: I gave formula a default empty string, and hopefully this fixes the pickling issue in multiprocessing
    def __init__(self, formula='', charge=0):
        if formula is None:
            formula = ''
        defaultdict.__init__(self, int)
        if charge:
            self['-'] = -charge
        if isinstance(formula, dict):
            # NOTE: assume that if we are copying, that the incoming Formula is already properly formatted
            for isotop, number in formula.items():
                k, elem = self._parse_isotope_string(isotop)
                if not elem in _nist_mass:
                    raise ValueError('Unknown chemical element: ' + elem)
                self[self._make_isotope_string(k, elem)] += number
        elif isinstance(formula, str):
            # NOTE: for formulas with negative numbers like neutral loss H2O,
            #  the number and charge are confused; 'H-2O-1-0' works since its converted to 'H-2O-1'
            if _formula_rex.match(formula) is None:
                raise ValueError(f'Not a properly formatted formula: {formula}')

            charge = _charge_rex.findall(formula)
            if charge:
                polarity, number = charge[0]
                z = 1
                if number:
                    z = int(number)
                if polarity == '+':
                    z = -z
                self['-'] += z

            for isotope, elem, number in _isotope_rex.findall(formula):
                if not elem in _nist_mass:
                    raise ValueError('Unknown chemical element: ' + elem)
                self[self._make_isotope_string(int(isotope) if isotope else 0, elem)] += \
                    int(number) if number else 1

    @staticmethod
    def _parse_isotope_string(isotope_string: str):
        if isotope_string == '-':
            return 0, isotope_string
        num, elem, _ = _isotope_rex.match(isotope_string).groups()
        k = int(num) if num else 0
        if (elem not in _nist_mass) or (k not in _nist_mass[elem]):
            raise ValueError(f'Not a chemical element G')
        return k, elem

    @staticmethod
    def _make_isotope_string(k: int, elem: str):
        """Form a string label for an isotope."""
        if (elem not in _nist_mass) or (k not in _nist_mass[elem]):
            raise ValueError(f'Conjo papi')

        if k == 0:
            return elem
        else:
            return f'[{k}]{elem}'

    def mass(self, charge=0, average=False, ion=True, abundances=_nist_mass):
        """ calculates the mass of the formula object
        TODO: TEST ion=False/True!!
        Parameters
        ----------
        charge : int, default 0
            Adds or subtracts `charge` electrons from the mass and divides by charge.
            Used for m/z calculations in mass-spectrometry applications.
        average : bool, default False
            When True, calculates the average mass. Masses of all isotope of the Formulas
            atoms are multiplied by their abundance
        """
        # Calculate mass
        mass = 0.0
        for isotope_string, amount in self.items():
            if isotope_string == '-':
                amount -= charge

            k, elem = self._parse_isotope_string(isotope_string)
            # Calculate average mass if required and the isotope number is
            # not specified.
            if (not k) and average:
                for isotope, data in abundances[elem].items():
                    if isotope:
                        mass += (amount * data[0] * data[1])
            else:
                mass += (amount * abundances[elem][k][0])

        # Calculate m/z if required
        nE = charge - self['-']
        if (nE != 0) and ion:
            mass = abs(mass/nE)
        elif ion:
            raise ValueError('Not a charged molecule!')
        return mass

    def to_chnops(self) -> str:
        """Unique string representation of a given formula; elements
        C,H,N,O,P and S are at the start of the string
        """
        charge = self.pop('-', 0)
        sorter = np.zeros((2, len(self)), dtype=np.int64)
        for i, (isotope_string, number) in enumerate(self.items()):
            k, elem = self._parse_isotope_string(isotope_string)
            sorter[1,i] = _chnops[elem]
            sorter[0,i] = k
        to_sort = np.array([k + (str(v) if v != 1 else '') for k, v in self.items()])
        indices = np.lexsort(keys=sorter)
        the_string = ''.join(to_sort[indices])
        if charge > 0:
            the_string += '-' if charge == 1 else f'-{charge}'
        elif charge < 0:
            the_string += '+' if charge == -1 else f'+{-charge}'
        self['-'] = charge
        return the_string

    def add_C13(self, nC13:int):
        """Exchanges C[12] for C[13] atoms in a Formula

        Parameters
        ----------
        nC13 : int
            Number of 13C atoms to exchange for 12C atoms


        Returns
        -------
        add_C13 : Formula
            Formula where nC13 with nC13 C[13] atoms and the rest C[12] atoms

        Examples
        --------
        Formula('C2H5OH').add_C13(nC13=2)
        """
        nC = self['C'] + self['[12]C']
        if not ('[13]C' not in self) and (nC13 <= nC):
            raise ValueError('brøhh...')
        new = self.copy()
        hasC = new.pop('C', 0) + new.pop('[12]C', 0)
        if hasC > 0:
            new += {'[12]C': nC - nC13, '[13]C': nC13}
        return new

    def full_isotope(self):
        """Turn non-isotopic atoms into their most probable versions.
        Watch out, for large molecules this does not mean that the most
        probable isotopic composition is returned!

        Returns
        -------
        full_isotope : Formula
            Formula where all atoms have an isotopic specification

        Examples
        --------
        Formula('C2H5OH').full_isotope()
        TODO
        """
        new = self.copy()
        for isotope_string, number in self.items():
            k, elem = self._parse_isotope_string(isotope_string)
            if k == 0:
                new.pop(isotope_string)
                k = int(round(_nist_mass[elem][0][0]))
                new[self._make_isotope_string(k, elem)] += number
        return new

    def no_isotope(self):
        """Removes all isotopic specifications from a Formula
        Returns
        -------
        full_isotope : Formula
            Formula where all atoms have an isotopic specification

        Examples
        --------
        Formula('C[13]C[12]H[2]H4OH').no_isotope()
        TODO
        """
        new = self.copy()
        for isotope_string, number in self.items():
            k, elem = self._parse_isotope_string(isotope_string)
            if k != 0:
                new.pop(isotope_string)
                new[elem] += number
        return new

    def abundance(self, abundances=_nist_mass):
        """
        TODO slow but readable...
        Calculate the relative abundance of a given isotopic composition
        of a molecule.

        Parameters
        ----------
        elements : list, optional
            Which elements to include when calculating the abundance.
            For instance when calculating a correction matrix, all elements
            except for carbon.

        Returns
        -------
        abundance : float
            The relative abundance of a given isotopic composition.
        """
        isotopic_composition = defaultdict(dict)

        for element in self:
            k, elem = self._parse_isotope_string(element)
            if ((elem in isotopic_composition) and
                    (k == 0 or 0 in isotopic_composition[elem])):
                raise ValueError(f'{elem} all or none isotope numbers')
            else:
                isotopic_composition[elem][k] = (self[element])

        num1, num2, denom = 1.0, 1.0, 1.0
        for elem, isotope_dict in isotopic_composition.items():
            if elem == '-':
                continue
            num1 *= factorial(sum(isotope_dict.values()))
            for k, isotope_content in isotope_dict.items():
                denom *= factorial(isotope_content)
                if k:
                    num2 *= (abundances[elem][k][1] ** isotope_content)

        return num2 * (num1 / denom)

    def shift(self):
        zero_shift = self.no_isotope().full_isotope()
        self_shift = self.full_isotope()
        zero_shift = sum(self._parse_isotope_string(k)[0] * v for k, v in zero_shift.items())
        self_shift = sum(self._parse_isotope_string(k)[0] * v for k, v in self_shift.items())
        return self_shift - zero_shift


def isotopologues(
        formula,
        report_abundance=False,
        elements_with_isotopes=None,
        isotope_threshold=5e-4,
        overall_threshold=0.0,
        abundances=_nist_mass,
        n_mdv = None,
    ):
    """Iterate over possible isotopic states of a molecule.
    The molecule can be defined by formula, sequence, parsed sequence, or composition.
    The space of possible isotopic compositions is restrained by parameters
    ``elements_with_isotopes``, ``isotope_threshold``, ``overall_threshold``.

    Parameters
    ----------
    formula : str
        A string with a chemical formula.
    report_abundance : bool, optional
        If :py:const:`True`, the output will contain 2-tuples: `(composition, abundance)`.
        Otherwise, only compositions are yielded. Default is :py:const:`False`.
    elements_with_isotopes : container of str, optional
        A set of elements to be considered in isotopic distribution
        (by default, every element has an isotopic distribution).
    isotope_threshold : float, optional
        The threshold abundance of a specific isotope to be considered.
        Default is :py:const:`5e-4`.
    overall_threshold : float, optional
        The threshold abundance of the calculateed isotopic composition.
        Default is :py:const:`0`.

    Returns
    -------
    isotopologues : iterator
        Iterator over possible isotopic compositions.
    """

    formula = Formula(formula=formula).no_isotope()
    dict_elem_isotopes = {}
    for elem in formula:
        if (elements_with_isotopes is None) or (elem in elements_with_isotopes):
            compare = -1 if elem == '-' else 0
            isotopes = {k: v for k, v in abundances[elem].items() if k != compare and v[1] >= isotope_threshold}
            list_isotopes = [Formula._make_isotope_string(k, elem) for k in isotopes]
            dict_elem_isotopes[elem] = list_isotopes
        else:
            dict_elem_isotopes[elem] = [elem]

    all_isotoplogues = []
    for elem, list_isotopes in dict_elem_isotopes.items():
        n = abs(formula[elem])
        list_comb_element_n = []
        for elementXn in combinations_with_replacement(list_isotopes, n):
            list_comb_element_n.append(elementXn)
        all_isotoplogues.append(list_comb_element_n)

    for isotopologue in product(*all_isotoplogues):
        ic = Formula()
        for elem in isotopologue:
            for isotope in elem:
                ic[isotope] += 1
        if n_mdv is not None:
            if ic.shift() >= n_mdv:
                continue
        if report_abundance or overall_threshold > 0.0:
            abundance = ic.abundance(abundances=abundances)
            if abundance > overall_threshold:
                if report_abundance:
                    yield (ic, abundance)
                else:
                    yield ic
        else:
            yield ic


if __name__ == '__main__':
    a = Formula({'C': 10, 'H': 2, 'O': 3, '-': -2})
    print(list(isotopologues(a, overall_threshold=0.001, n_mdv=1)))


    # malate = Formula('C4H6O5')
    # malate_1_a = Formula('C3[13]CH6O5')
    # malate_1_b = Formula('[12]C3[13]CH6O5')
    # malate_1_c = Formula('[12]C3[13]C1H6O5')
    # neg_malate = Formula('C4H5O5-')
    # pos_malate = Formula('C4H6O5+')
    # pm = Formula(neg_malate)
    # # print(pm, pm.full_isotope().abundance())
    # print(pm.add_C13(1).abundance())

