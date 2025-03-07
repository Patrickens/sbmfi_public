import pandas as pd
from sbmfi.core.util import _optlang_reverse_id_rex, hdf_opener_and_closer
import lxml.etree as etree
import numpy as np
import re
from cobra.util import constraint_matrices
import itertools
import copy, sys
from string import printable, digits
from cobra import Metabolite
from sympy import Matrix, matrix2numpy

_long_str = printable.replace('"', '').replace('', '').replace('\n', '').replace('\t', '').replace('\n', '').replace(' ', '')

emu_pos = lambda x: [i for i, c in enumerate(x) if c == '1']

def parse_result_fml(fml):
    xmlns = '{http://www.13cflux.net/fwdsim}'
    tree = etree.parse(fml)
    results = {}

    for el in tree.getroot().iter():
        el.tag = el.tag.replace(xmlns, '')

    fluxes = {}
    for el in tree.find('stoichiometry'):
        vid = el.attrib['id']
        net, xch = list(el)
        fwd_val = float(str(net.attrib['value']))
        rev_val = float(str(xch.attrib['value']))
        if rev_val > 0.0:
            fwd_val += rev_val
            fluxes[vid+'_rev'] = rev_val
        fluxes[vid] = fwd_val
    results['fluxes'] = pd.Series(fluxes, name='fluxes')

    simulation = tree.find('simulation')
    method = simulation.attrib['method']
    str_to_float = lambda text: float(text.strip())
    el_cfg_val = lambda el: (el.attrib.get('cfg'), str_to_float(el.text))

    if method == 'cumomer':
        simulated = []
    elif method == 'emu':
        simulated = {}

    for el in simulation:
        met_id = el.attrib['id']
        if method == 'emu':
            for e in el:
                positions = emu_pos(e.attrib['cfg'])
                if positions:
                    emu_id = f'{met_id}|{positions}'
                    vals = list(map(str_to_float, e.itertext()))
                    X = simulated.setdefault(len(positions), {})
                    X[emu_id] = vals
        elif method == 'cumomer':
            cumos = pd.Series(dict(map(el_cfg_val, el)))
            cumos.index = f'{met_id}/' + cumos.index
            simulated.append(cumos)
        else:
            raise NotImplementedError
    if method == 'cumomer':
        results[method] = pd.concat(simulated)
    elif method == 'emu':
        simulated_new = {}
        for weight, emus in simulated.items():
            emudf = pd.DataFrame(emus).T
            emudf.columns = 'M+' + emudf.columns.astype(str)
            simulated_new[weight] = emudf
        results[method] = simulated_new

    measurements = []
    for el in tree.find(f'measurements'):
        if el.attrib['type'] == 'MS':
            spec = el[0]
            met_pos, pos = spec.text.split('#M')
            met_id = re.sub('(\[[\d\-\,\:]*\]$)', '', met_pos)
            ids = [f'{met_id}+{i}' for i in pos.split(',')]
            vals = el[1:]
            vals = list(map(el_cfg_val, vals))
            measurement = {}
            for (mid, val) in zip(ids, vals):
                measurement[mid] = val[1]
            measurements.append(pd.Series(measurement))
        elif el.attrib['type'] == 'GENERIC':
            measurements.append(pd.Series(float(el[1].text), index=[el.attrib['id']]))
        else:
            raise NotImplementedError

    results['measurements'] = pd.concat(measurements)
    return results


@hdf_opener_and_closer()
def parse_h5_jacobian(hdf, net=False, config=0):
    rows = []
    if config is None:
        location = hdf.root.jacobian
    elif isinstance(config, int):
        location = hdf.root.jacobian._f_get_child(f'config_{config}')
    else:
        raise NotImplementedError

    for row in location.c.read().astype(str)[0]:
        v, nx = row.split('.')
        rev = '_rev' if nx == 'x' else ''
        rows.append(f'{v}{rev}')
    col_labels = location.r.read().astype(str)[0]

    splithing = '#M'
    sep = '+'
    try:
        col_labels[0].split(splithing)
    except:
        sep = '/'
        splithing = '#'

    cols = []
    for col in col_labels:
        met_id, meas = col.split(splithing)
        cols.append(f'{met_id}{sep}{meas}')

    mat = location.matrix.read().astype(float)
    jac = pd.DataFrame(mat, index=rows, columns=cols)
    if net:
        jac.index = jac.index.str.replace('_rev$','_xch', regex=True)
        return jac
    else:
        net_jac = jac.copy()
        net_jac.index = jac.index.str.replace('_rev$','', regex=True)
        abs = net_jac.groupby(by=net_jac.index).sum()
        abs_fluxes = abs.index[abs.index.isin(jac.index)]
        jac.loc[abs_fluxes, :] = abs.loc[abs_fluxes, :]
        return jac


@hdf_opener_and_closer()
def parse_h5_stoich(hdf):
    row = hdf.root.stoichiometry.c_fluxes.read().astype(str)[0]
    col = hdf.root.stoichiometry.r_fluxes.read().astype(str)[0]
    mat = hdf.root.stoichiometry.f.read().astype(float)
    return pd.DataFrame(mat, index=row, columns=col)


@hdf_opener_and_closer()
def parse_h5_cascade(hdf):
    cascade = {}
    for level, group in hdf.root.cascade._v_groups.items():
        level = int(level.replace('level_', ''))
        if 'B' in group:
            # process emus
            labels = []
            for label in group.labels.read().astype(str)[0]:
                met_id, pos = label.split('#')
                labels.append(f'{met_id}|{emu_pos(pos)}')

            A = pd.DataFrame(group.A.read().astype(float), index=labels, columns=labels)
            B = pd.DataFrame(group.B.read().astype(float), columns=labels)
        elif 'b' in group:
            # process cumos
            labels = np.char.replace(group.labels.read().astype(str)[0], '#', '/')
            A = pd.DataFrame(group.A.read().astype(float), index=labels, columns=labels)
            B = pd.Series(group.b.read().astype(float)[0], index=labels)
        cascade[level] = {'A': A, 'B': B}
    return cascade


def pysumo_to_fml(
        model,
        configuration_vars: dict,
        simmethod='emu',
):
    # xml_string = pysumo_to_fml(model=M)
    #     filename = os.path.join(MODEL_DIR, 'fml', 'spiro_parsed.fml')
    #     with io.open(filename, 'w', encoding='utf8') as f:
    #         f.write(xml_string)

    root = etree.Element("fluxml", nsmap={
        None: "http://www.13cflux.net/fluxml",
        'xsi':   "http://www.w3.org/2001/XMLSchema-instance",
    })

    # info elements
    info = etree.SubElement(root, "info")
    etree.SubElement(info, "version").text = "1.1"
    etree.SubElement(info, "comment", name="niks").text = f"{model.name}"

    # reactionnetwork
    reactionnetwork = etree.SubElement(root, "reactionnetwork")

    # metabolitepools
    metabolitepools = etree.SubElement(reactionnetwork, "metabolitepools")
    for m in model.metabolites_in_state:
        etree.SubElement(metabolitepools, 'pool', attrib={'atoms': str(m.elements['C']), 'id': m.id})

    # reaction
    for r in model.labelling_reactions - model.input_reactions:
        # NOTE: all reactions are unidirectional, since that resemblemes my implementation
        reaction = etree.SubElement(reactionnetwork, "reaction", attrib={'bidirectional': 'false', 'id': r.id})
        i = 0
        for met, (stoich, atoms) in r._atom_map.items():
            eltag = 'reduct' if stoich < 0 else 'rproduct'
            for atom in atoms:
                if atom is None: # NOTE: happens with biomass reactions
                    met = model.metabolites.get_by_id(met.id)
                    atom = _long_str[i: i + met.elements['C']]
                    i += met.elements['C']
                etree.SubElement(reaction, eltag, attrib={'cfg': ''.join(atom), 'id': met.id})

    # constraints
    constraints = etree.SubElement(root, "constraints")
    net = etree.SubElement(constraints, "net")
    textual = etree.SubElement(net, "textual")

    reac_cons = []
    for var in model.variables:
        name = _optlang_reverse_id_rex.sub('_rev', var.name)
        if ((var.lb, var.ub) == (0.0, 0.0)) or (name in model.input_reactions) or (name not in model.labelling_reactions):
            continue
        bounds = f'{name}&gt;={round(var.lb, 3)}; {name}&lt;={round(var.ub, 3)};'
        reac_cons.append(bounds)
    constring = '\n' + '\n'.join(reac_cons) + '\n'

    for i, cons in enumerate(model.constraints):
        if (cons.ub is not None and cons.lb is not None) and (cons.ub - cons.lb) < 1e-6:
            continue
        vars = []
        coefs = cons.get_linear_coefficients(model.variables)
        for var, coef in coefs.items():
            name = _optlang_reverse_id_rex.sub('_rev', var.name)
            if (coef == 0.0) or (name not in model.labelling_reactions):
                continue
            element = f'{coef}*{name}'
            vars.append(element)
        if not vars:
            continue
        constraint = ' + '.join(vars)
        if cons.lb is not None:
            constraint_lb = f'{constraint}&gt;={round(cons.lb, 3)};'
            constring = '\n'.join((constring, constraint_lb))
        if cons.ub is not None:
            constraint_ub = f'{constraint}&lt;={round(cons.ub, 3)};'
            constring = '\n'.join((constring, constraint_ub))

    ding = '\t' * 4
    constring = constring.replace('\n', f'\n{ding}')
    constring += f'\n{ding[:-1]}'
    textual.text = constring

    for i, (config_name, (initial_free_fluxes, measurement, input_labelling, stddev)) in enumerate(configuration_vars.items()):
        configuration = etree.SubElement(root, "configuration", attrib={'stationary': 'true', 'name': 'default'})
        configuration.attrib['name'] = f'config_{i}'  # TODO chenge to config_name

        # input labelling
        if input_labelling is None:
            input_labelling = model.substrate_labelling

        for isocumo, frac in input_labelling.iteritems():
            met_id, label = isocumo.split('/')
            input = etree.SubElement(configuration, "input", attrib={
                'pool': met_id, 'type': 'isotopomer'
            })
            label = etree.SubElement(input, "label", attrib={'cfg': label})
            label.text = str(frac)

        # labelingmeasurement
        etree_measurement = etree.SubElement(configuration, "measurement")
        etree_model = etree.SubElement(etree_measurement, "model")
        labelingmeasurement = etree.SubElement(etree_model, "labelingmeasurement")
        etree_data = etree.SubElement(etree_measurement, "data")


        # TODO update for when we have pseudo_metabolites metabolite fragments (=EMUs)
        for i, met in enumerate(set(model.measurements) - set(model.pseudo_metabolites)):
            if met.id[0] in digits:
                # print(met) # NB fml is actually completely fucking retarded, cannot deal with this
                continue
            Ms = list(range(met.elements['C']+1))
            group = etree.SubElement(labelingmeasurement, "group", attrib={
                'id': met.id, 'scale': 'one'
            })
            textual = etree.SubElement(group, "textual")
            textual.text = f'{met.id}{Ms[1:]}#M{",".join(map(str, Ms))}'.replace(' ', '')

        simulation = etree.SubElement(configuration, "simulation", attrib={'method': simmethod, 'type': 'auto'})
        variables = etree.SubElement(simulation, "variables")
        for reac_id, v in initial_free_fluxes.iteritems():
            fluxvalue = etree.SubElement(variables, "fluxvalue", attrib={'type': 'net', 'flux': reac_id})
            fluxvalue.text = str(v)

        if stddev is None:
            stddev = str(0.01)
        if measurement is None:
            measurement = model.state.iloc[i]
        for mdv_id, v in measurement.iteritems():
            met_id, weight = mdv_id.split('+')
            datum = etree.SubElement(etree_data, "datum", attrib={
                'id': met_id, 'stddev': stddev, 'weight': weight
            })
            datum.text = str(v)

        root.append(configuration)

    etree.indent(root, space="\t")
    return etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        standalone=False,
        encoding='UTF-8'
    ).decode().replace('&amp;', '&')


def fml_to_sbml(fml_file):
    # TODO: this is not done yet!!
    tree = etree.parse(fml_file)
    xmlns = '{http://www.13cflux.net/fluxml}'
    for el in tree.getroot().iter():
        el.tag = el.tag.replace(xmlns, '')

    metabolites = {}
    for el in tree.find('reactionnetwork').find('metabolitepools'):
        metabolites[el.attrib['id']] = {
            'formula': el.attrib['cfg']
        }
    reactions = {}
    bool_map = {'true': '-->', 'false': '<=>'}

    for el in tree.find('reactionnetwork'):
        if el.tag != 'reaction':
            continue
        rid = el.attrib['id']
        arrow = bool_map[el.attrib['bidirectional']]
        atom_map = {}
        totom = 0
        for subel in el.iter():
            if subel.tag == 'annotation':
                pathway = subel.text
            elif subel.tag in ['reduct', 'rproduct']:
                mid = subel.attrib['id']
                atom_list = subel.attrib['cfg'].split(' ')
                atom_list = [atom for atom in atom_list if (len(atom)>0) and (atom[0] == 'C')]
                if mid not in atom_map:
                    atom_map[mid] = (0, [])
                stoich, atoms = atom_map[mid]

                if subel.tag == 'reduct':
                    stoich -= 1
                    totom += len(atom_list)
                elif subel.tag == 'rproduct':
                    stoich += 1
                atom_map[mid] = (stoich, atoms)
                atoms.append(atom_list)

# NOTE: dont do fwd and reverse fluxes but rather do xch; this is advice from Beyss
if __name__ == "__main__":
    from sbmfi.models.build_models import spiro, e_coli_glc, e_coli_succ, multi_modal
    from sbmfi.settings import MODEL_DIR
    from sbmfi.core.observation import LCMS_ObservationModel
    import io, os

    # ding = parse_result_fml('out_multi_modal.fml')

    m = multi_modal(prepend_input=True)
    # actual_fluxes = pd.Series([10.0, 1.5], index=['a_in', 'v4'])
    # m.set_fluxes(fluxes=actual_fluxes, is_free=True)
    # measurement = m.cascade()
    # lobs = LCMS_ObservationModel(
    #     model=m,
    #     annotation_df=pd.DataFrame(
    #         [
    #             ['E', 'C2H4O2', 'M-H', 0],
    #             ['E', 'C2H4O2', 'M-H', 1],
    #             ['E', 'C2H4O2', 'M-H', 2],
    #         ],
    #         columns=['met_id', 'formula', 'adduct_name', 'nC13']
    #     ),
    #     total_intensities=pd.Series([1e4], index=['E']),
    # )
    # lobs(measurement)

    x_o_1 = pd.Series([0.09, 0.82, 0.07], index=m.state_id)
    v_init = 4.0
    initial_free_fluxes = pd.Series([10.0 - v_init, v_init], index=['v1', 'v4'])
    x_o_2 = pd.Series([0.2621, 0.4703, 0.2675], index=m.state_id)

    xml_string = pysumo_to_fml(
        model=m,
        configuration_vars={ # initial_free_fluxes, measurement, input_labelling, stddev
            'x_o_1': (initial_free_fluxes, x_o_1, None, None),
            'x_o_2': (initial_free_fluxes, x_o_2, None, None),
        },
        simmethod='emu',
    )

    filename = os.path.join(MODEL_DIR, 'fml', 'multi_modal.fml')
    with io.open(filename, 'w', encoding='utf8') as f:
        f.write(xml_string)
    print(12312313)

    # M = e_coli_core_succ(simtype='emu', return_type='mdv', lin_alg_lib='numpy')
    #     # xml_string = pysumo_to_fml(model=M, simmethod='emu')
    #     # filename = os.path.join(MODEL_DIR, 'fml', 'e_coli_core_succ.fml')
    #     # with io.open(filename, 'w', encoding='utf8') as f:
    #     #     f.write(xml_string)
    #
    #     # TODO: add biomass as free flux!
    #     # TODO: deal with symmetric metabolites
    #     # TODO: flux solution found by Wiechert based on free fluxes is not the same as M.fluxes;
    #     #   temporary solution is to change M.fluxes with the parsed fluxes from the fml file; see ipynb

