import numpy as np
import pandas as pd
import multiprocessing as mp  # TODO maybe dynamically change this import to torch.multiprocessing based on linalg?
from typing import Iterable, Union, Dict, Tuple
from sbmfi.core.model import LabellingModel, RatioMixin, EMU_Model
from sbmfi.core.observation import MDV_ObservationModel, MDV_LogRatioTransform


def init_simulator(model: LabellingModel, epsilon=1e-12):
    global _MODEL, _EPSILON
    _MODEL = model
    model.build_model(free_reaction_id=model.labelling_fluxes_id)  # necessary after pickling
    _EPSILON = epsilon


def init_observer(
        model: LabellingModel,
        mdv_observation_models: Dict[str, Tuple[MDV_ObservationModel, MDV_LogRatioTransform]],
        epsilon=1e-12
):
    init_simulator(model=model, epsilon=epsilon)
    global _OBSMODS
    _OBSMODS = mdv_observation_models


def simulator_worker(task: dict, model=None) -> dict:
    start_stop, input_labelling, fluxes_chunk, type_jacobian = task.values()

    if model is not None:
        MODEL = model
    else:
        MODEL = _MODEL

    n_state = len(MODEL.state_id)
    la = MODEL._la

    if MODEL.labelling_id != input_labelling.name:
        MODEL.set_substrate_labelling(substrate_labelling=input_labelling)

    mdv_chunk = la.get_tensor(  # by making this a tensor with -np.inf values, we can filter failed simulations
        values=np.full(shape=(fluxes_chunk.shape[0], n_state), fill_value=-np.inf, dtype=np.double)
    )

    if type_jacobian is not None:
        n = len(MODEL.labelling_fluxes_id)
        if type_jacobian == 'free':
            n = len(MODEL._fcm.theta_id)
        jacobian_chunk = la.get_tensor(values=np.full(
            shape=(fluxes_chunk.shape[0], n, n_state),
            fill_value=-np.inf,
            dtype=np.double
        ))

    step = la._batch_size
    stop = max(fluxes_chunk.shape[0], step)

    for i in range(0, stop, step, ):
        j = i + step
        if j > stop:
            j = stop
            i = j - step
        fluxes_batch = fluxes_chunk[i: j]
        try:
            MODEL.set_fluxes(labelling_fluxes=fluxes_batch, trim=False)  # trim has to be False!
            mdv_chunk[i: j] = MODEL.cascade()
        except Exception as e:
            print(1, e)

        if type_jacobian is not None:
            try:
                jacobian_batch = MODEL.compute_jacobian()
            except:
                print(2, e)
            if type_jacobian == 'free':
                jacobian_batch = MODEL._fcm.rounded_jacobian(jacobian_batch, fluxes=fluxes_batch)
            jacobian_chunk[i: j] = jacobian_batch

    # NB filter failed simulations (metabolite not summing to 1 or values outside of [0, 1]
    sum_1 = la.isclose(
        MODEL._sum @ mdv_chunk.T,
        la.ones(fluxes_chunk.shape[0], dtype=mdv_chunk.dtype), atol=_EPSILON
    ).T.all(1)
    bounds = (mdv_chunk < 1.0 + _EPSILON).all(1) & (mdv_chunk > 0.0 - _EPSILON).all(1)
    # valid_idx = la.where(sum_1 & bounds)[0]
    validx_chunk = sum_1 & bounds

    result = dict([
        ('start_stop', start_stop),
        ('input_labelling', input_labelling),
        ('mdv_chunk', mdv_chunk),
        ('validx_chunk', validx_chunk),
    ])
    if type_jacobian is not None:
        result['jacobian_chunk'] = jacobian_chunk[validx_chunk]
    return result


def obervervator_worker(task: dict, model=None):
    simulator_task = dict(
        (k, task[k]) for k in ['start_stop', 'input_labelling', 'fluxes_chunk', 'type_jacobian']
    )
    result_chunk = simulator_worker(simulator_task, model=model)
    what = task['what']
    if what == 'mdv':
        return result_chunk

    if model is not None:
        MODEL = model
    else:
        MODEL = _MODEL

    mdv_chunk = result_chunk['mdv_chunk']

    n_obs = task['n_obs']
    observation_model = _OBSMODS[task['input_labelling'].name]
    n_obshape = max(1, n_obs)
    slicer = 0 if n_obs == 0 else slice(None)
    data_chunk = MODEL._la.get_tensor(shape=(mdv_chunk.shape[0], n_obshape, observation_model._nd))
    bs = MODEL._la._batch_size

    for i in range(0, mdv_chunk.shape[0], bs):
        data_slice = observation_model(mdv=mdv_chunk[i: i + bs, :], n_obs=n_obs)  # TODO sizing!
        data_chunk[i: i + bs, slicer, :] = data_slice

    result_chunk['data_chunk'] = data_chunk

    if what != 'all':
        result_chunk.pop('mdv_chunk')

    return result_chunk


def designer_worker(task):
    simulator_task = dict(
        (k, task[k]) for k in ['start_stop', 'input_labelling', 'fluxes_chunk', 'type_jacobian']
    )
    result_chunk = simulator_worker(simulator_task)
    x_coordinate_sys = task['x_coordinate_sys']

    mdv_chunk = result_chunk['mdv_chunk']
    jacobian_chunk = result_chunk['jacobian_chunk']
    observation_model = _OBSMODS[task['input_labelling'].name]

    if x_coordinate_sys in ('sigma_v', 'all'):
        n_free = len(_MODEL._fcm.theta_id)
        summary_v_chunk = _MODEL._la.get_tensor(shape=(mdv_chunk.shape[0], 3))  # TODO
        sigma_v_chunk = _MODEL._la.get_tensor(shape=(mdv_chunk.shape[0], n_free, n_free))  # TODO
    if x_coordinate_sys in ('sigma_r', 'all') and isinstance(_MODEL, RatioMixin):
        validx = result_chunk['validx_chunk']
        fluxes_chunk = simulator_task['fluxes_chunk'][validx]
        n_ratio = len(_MODEL.ratios_id)
        summary_r_chunk = _MODEL._la.get_tensor(shape=(mdv_chunk.shape[0], 2 * n_ratio))
        sigma_r_chunk = _MODEL._la.get_tensor(shape=(mdv_chunk.shape[0], n_ratio, n_ratio))

    bs = _MODEL._la._batch_size
    for i in range(0, mdv_chunk.shape[0], bs):
        sigma_v, summary_v = observation_model.sigma_v(mdv=mdv_chunk[i: i + bs], J_sv=jacobian_chunk[i: i + bs])
        if x_coordinate_sys in ('sigma_v', 'all'):
            summary_v_chunk[i: i + bs] = summary_v
            sigma_v_chunk[i: i + bs] = sigma_v

        if x_coordinate_sys in ('sigma_r', 'all') and isinstance(_MODEL, RatioMixin):
            sigma_r, summary_r = observation_model.sigma_r(fluxes=fluxes_chunk[i: i + bs], sigma_v=sigma_v)
            summary_r_chunk[i: i + bs] = summary_r
            sigma_r_chunk[i: i + bs] = sigma_r

    if x_coordinate_sys != 'all':
        result_chunk.pop('mdv_chunk')
        result_chunk.pop('jacobian_chunk')
    if x_coordinate_sys in ('sigma_v', 'all'):
        result_chunk['sigma_v_chunk'] = sigma_v_chunk
        result_chunk['summary_v_chunk'] = summary_v_chunk
    if x_coordinate_sys in ('sigma_r', 'all') and isinstance(_MODEL, RatioMixin):
        result_chunk['sigma_r_chunk'] = sigma_r_chunk
        result_chunk['summary_r_chunk'] = summary_r_chunk
    return result_chunk


def simulator_tasks(
        fluxes,
        substrate_df: pd.DataFrame,
        fluxes_per_task: int,
        type_jacobian = None,
) -> dict:
    for labelling_id, row in substrate_df.iterrows():
        input_labelling = row[row > 0.0]
        for i in range(0, fluxes.shape[0], fluxes_per_task):
            fluxes_chunk = fluxes[i: i + fluxes_per_task]
            yield dict([
                ('start_stop', (i, i + fluxes_chunk.shape[0])),
                ('input_labelling', input_labelling),
                ('fluxes_chunk', fluxes_chunk),
                ('type_jacobian', type_jacobian),
            ])


def observator_tasks(
        fluxes,
        substrate_df: pd.DataFrame,
        fluxes_per_task: int,
        n_obs = 3,
        what = 'all',
):
    for task in simulator_tasks(fluxes, substrate_df, fluxes_per_task, type_jacobian = None):
        task['n_obs'] = n_obs
        task['what'] = what
        yield task


def designer_tasks(
        fluxes,
        substrate_df: pd.DataFrame,
        fluxes_per_task: int,
        x_coordinate_sys ='all',
        type_jacobian = 'free'
):
    for task in simulator_tasks(fluxes, substrate_df, fluxes_per_task, type_jacobian):
        task['x_coordinate_sys'] = x_coordinate_sys
        yield task


if __name__ == "__main__":
    pass