import numpy as np
import tables as pt
import math, re
# np.seterr(all='raise')
from cobra.io import save_json_model
from PolyRound.api import Polytope, PolyRoundApi
from sbmfi.core.metabolite import LabelledMetabolite
import pandas as pd
import functools
import traceback
import line_profiler
from typing import Dict
# poetry add git+https://github.com/GeomScale/dingo.git
# poetry install git+https://github.com/GeomScale/dingo.git
# TODO mail this dude for postdoc in Barca: https://torres-sanchez.xyz/
# buncha regexes
_bigg_compartment_ids = [
    'c', 'e', 'p', 'm', 'x', 'r', 'v', 'n',
    'g', 'u', 'l', 'h', 'f', 's', 'im',
    'cx', 'um', 'cm', 'i', 'mm', 'w', 'y',
]
_strip_bigg_rex         = re.compile(f'(_[{"|".join(_bigg_compartment_ids)}])$')
_find_biomass_rex       = re.compile('(biomass)', flags=re.IGNORECASE)
_read_atom_map_str_rex  = re.compile(r'(.*?)([-=<>]{3})(.*$)')
_net_constraint_rex     = re.compile('(?:_net)(?:|\|lb|\|ub)$')
_optlang_reverse_id_rex = re.compile('(_reverse_.{5})$')
_rev_reactions_rex      = re.compile('_rev$')
_xch_reactions_rex      = re.compile('_xch$')
_rho_constraints_rex    = re.compile('(?:_rho)(?:|_min|_max)(?:|\|lb|\|ub)$')
_get_dictlist_idxs = np.vectorize(lambda dctlst, item: dctlst.index(id=item.id), excluded=[0], otypes=[np.int64])



def make_multidex(index_dict: Dict[str, pd.Index], name0='labelling_id', name1=None):
    if name1 is None:
        index1 = list(index_dict.values())[0]
        if index1 is not None:
            name1 = index1.name
    return pd.MultiIndex.from_frame(
        pd.DataFrame.from_dict(index_dict, orient='index').stack().reset_index().iloc[:, [0, 2]]
        , names=[name0, name1])


def profile(profiler: line_profiler.LineProfiler):
    def outer(func):  # profiler.print_stats()
        def inner(*args, **kwargs):
            profiler.add_function(func)
            profiler.enable_by_count()
            return func(*args, **kwargs)
        return inner
    return outer


def stacktrace(func):
    # https://stackoverflow.com/questions/1156023/print-current-call-stack-from-a-method-in-code
    @functools.wraps(func)
    def wrapped(*args, **kwds):
        # Get all but last line returned by traceback.format_stack()
        # which is the line below.
        callstack = '\n'.join([4*' '+ line.strip() for line in traceback.format_stack()][:-1])
        print('{}() called:'.format(func.__name__))
        print(callstack)
        return func(*args, **kwds)

    return wrapped

# very general functions to get the most important dataframes for a HDF file with spectra stored
def hdf_opener_and_closer(mode='r'):
    def outer(func):
        def inner(*args, **kwargs):
            # pt.file._open_files.close_all()
            close = False
            hdf = kwargs.get('hdf', None)
            if hdf is None:
                argi = 0
                if not isinstance(args[0], str):  # checks whether a function is a bound to a class or not
                    argi = 1
                hdf = args[argi]
            if isinstance(hdf, str):
                close = True
                hdf = pt.open_file(hdf, mode=mode)
            in_kwargs = kwargs.get('hdf', None)
            if in_kwargs is None:
                args = list(args)
                args[argi] = hdf
            else:
                kwargs['hdf'] = hdf
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                hdf.close()
                raise e
            if close:
                hdf.close()
            return result
        return inner
    return outer


def _cell_color(x):
    # can also combine args into f'background-color: {hex1}; color: {hex2}'
    if x < 0:
        color = '#a6f1a6'
    elif x > 0:
        color = 'yellow'
    else:
        return 'color: white'
    return 'background-color : ' + color


def excel_polytope(pol: Polytope, file, concat=True):
    # debugging functions to manually check system
    # TODO also store the affine transformations! Only useful if we keep the index names, which PolyRound does not currently do
    ew = pd.ExcelWriter(path=file, engine='xlsxwriter')
    pol.b.name = 'b'
    if concat:
        blank = pd.Series(0, index=pol.b.index, name='blank')
        Ab = pd.concat([pol.A, blank, pol.b], axis=1)

        # blank2 = pd.Series(0, index=pol.b.index, name='blank')
        # Ttau = pd.concat([pol.transformation.T, pol.transformation.T])
        try:
            Abs = Ab.style.applymap(lambda x: _cell_color(x))
            Abs.to_excel(ew, sheet_name='Ab')
        except:
            Ab.to_excel(ew, sheet_name='Ab')
    else:
        A = pol.A.style.applymap(lambda x: _cell_color(x))
        A.to_excel(ew, sheet_name='A')
        pol.b.to_excel(ew, sheet_name='b')
    if pol.S is not None:
        pol.h.name = 'h'
        if concat:
            blank = pd.Series(0, index=pol.h.index, name='blank')
            Sh = pd.concat([pol.S, blank, pol.h], axis=1).style.applymap(lambda x: _cell_color(x))
            Sh.to_excel(ew, sheet_name='Sh')
        else:
            S = pol.S.style.applymap(lambda x: _cell_color(x))
            S.to_excel(ew, sheet_name='S')
            pol.h.to_excel(ew, sheet_name='h')
    ew.save()


def e_coli_core_escher_map(model, fluxes, filename='map.html'):
    from escher import Builder
    json_file = os.path.join(MODEL_DIR, 'escher_input', 'model', '___escher_placholder___.json')
    assert len(fluxes)%2 == 0, 'number of fluxes needs to be even, meaning fwd & rev split!'
    assert fluxes.index.str.contains('_rev$', regex=True).sum() == len(fluxes)/2, \
        'every reac needs a reverse variable'

    fwd_fluxes = fluxes.iloc[range(0, len(fluxes), 2)]
    rev_fluxes = fluxes.iloc[range(1, len(fluxes), 2)]
    net_fluxes = fwd_fluxes - rev_fluxes.values
    net_fluxes.name = fluxes.name
    save_json_model(model=model.make_sbml_writable(), filename=json_file, sort=False)
    builder = Builder(
        model_json=json_file,
        map_json=os.path.join(MODEL_DIR, 'escher_input', 'map', 'e_coli_core_tomek.json'),
        reaction_data=net_fluxes,
    )
    os.remove(json_file)
    builder.save_html(filename)


# method to get naturally labelled abundance vector for arbitrary susbtrate
def generate_nat_labelled_isotop_state(metabolite: 'LabelledMetabolite', min_abundance=1e-3):
    raise NotImplementedError

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
    import pickle, os
    from sbmfi.models.small_models import spiro
