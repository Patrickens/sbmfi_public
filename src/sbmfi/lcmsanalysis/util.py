import re
from cobra import DictList, Model
import math
from typing import Iterable, Union
from sbmfi.lcmsanalysis.formula import Formula, isotopologues
from sbmfi.core.util import _strip_bigg_rex
import pandas as pd
import numpy as np
import tables as pt


# function to compute the correction matrix to correct for the natural abundance of different isotopes of atoms
def build_correction_matrix(
        # TODO incorporate a ppm argument so that we exclude correcting isotopes that are further than resolution away!
        formula, elements=None, isotope_threshold=1e-4, overall_threshold=0.0001, exclude_carbon=True, n_mdv=None
) -> np.array:
    formula = Formula(formula=formula).no_isotope()

    if exclude_carbon:
        n_C = formula.pop('C', 4) + 1 # here we exclude carbon
    else:
        n_C = formula.get('C', 4) + 1

    if n_mdv is None:
        n_mdv = n_C

    abundances = np.zeros(shape=n_mdv, dtype=np.double)
    for (formula, abundance) in isotopologues(
            formula=formula, elements_with_isotopes=elements, report_abundance=True,
            isotope_threshold=isotope_threshold, overall_threshold=overall_threshold, n_mdv=n_mdv
    ):
        shift = formula.shift()
        if shift < 0:
            raise ValueError(f'Shift under 0 {formula.to_chnops()}')
        abundances[shift] += abundance
    corr_mat = np.zeros((n_mdv, n_mdv), dtype=np.double)
    for i in range(n_mdv):
        np.fill_diagonal(corr_mat[i:], abundances[i])
    # corr_mat = corr_mat / corr_mat.sum(0)[None, :]
    return corr_mat


def _get_bigg_metabolites(model_or_metabolites, strip_compartment=True):
    annot_metabolites = DictList()
    if hasattr(model_or_metabolites, 'metabolites'):
        iterator = DictList(model_or_metabolites.metabolites)
        if hasattr(model_or_metabolites, 'pseudo_metabolites'):
            iterator += DictList([met for met in model_or_metabolites.pseudo_metabolites if met not in iterator])
    elif hasattr(model_or_metabolites, '__iter__'):
        iterator = DictList(model_or_metabolites)
    else:
        raise ValueError('chgoser')
    for met in iterator:
        biggid = met.id
        if strip_compartment:
            biggid = _strip_bigg_rex.sub('', biggid)
        if biggid not in annot_metabolites:
            met = met.copy()
            met.id = biggid
            annot_metabolites.append(met)
    return annot_metabolites


def gen_annot_df(
        input,
        annotations='kegg.compound',
        first_annot=True,
        neutralize=True,
        strip_compartment=True,
    ):
    bigg_metabolites = _get_bigg_metabolites(input, strip_compartment=strip_compartment)

    data = {
        'met_id': bigg_metabolites.list_attr('id'),
        'name': bigg_metabolites.list_attr('name'),
        'formula': bigg_metabolites.list_attr('formula'),
        'charge': bigg_metabolites.list_attr('charge'),
    }
    if annotations is not None:
        annots = bigg_metabolites.list_attr('annotation')

    if annotations == 'all':
        data['annotations'] = annots
    elif isinstance(annotations, str):
        annotations = [annotations]
    elif annotations is None:
        annotations = []

    for annotation in annotations:
        tations = []
        for dct in annots:
            tation = dct.get(annotation, '')
            if first_annot and isinstance(tation, list):
                tation = tation[0]
            tations.append(tation)
        data[annotation] = tations

    annot_df = pd.DataFrame(data)
    # remove proteins, t-RNA and cofactor type things
    annot_df = annot_df.loc[~annot_df['formula'].str.contains(pat=r'[RX]\d*',regex=True)]
    annot_df['formula'] = annot_df.loc[:, ['formula', 'charge']].apply(
        lambda row: Formula(row['formula'], charge=row['charge']).to_chnops()
    , axis=1)
    annot_df.drop('charge', axis=1, inplace=True)

    if neutralize:
        def neutralproton(f_str):
            f = Formula(f_str)
            c = f['-']
            if (f['C'] > 0) and (c != 0):
                f += {'H': c, '-': -c}
            return f.to_chnops()
        annot_df['formula'] = annot_df['formula'].apply(neutralproton)

    annot_df['mz'] = annot_df['formula'].apply(lambda f: Formula(f).mass(ion=False))
    annot_df = annot_df.sort_values(by='mz').drop('mz', axis=1).reset_index(drop=True)
    return annot_df


if __name__ == "__main__":
    pass