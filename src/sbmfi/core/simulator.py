import tables as pt
import warnings
import psutil
import math
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Union, Dict
import tqdm
from sbmfi.core.model import LabellingModel
from sbmfi.core.simulfuncs import (
    init_observer,
    obervervator_worker,
    observator_tasks,
)
from sbmfi.core.observation import (
    BoundaryObservationModel,
    MDV_ObservationModel,
)
from sbmfi.core.coordinater import make_theta_polytope
from sbmfi.core.util import (
    hdf_opener_and_closer,
    make_multidex,
    profile
)



warnings.simplefilter('ignore', pt.NaturalNameWarning)


"""
https://andrewcharlesjones.github.io/journal/21-effective-sample-size.html
https://emcee.readthedocs.io/en/v2.2.1/user/pt/
https://people.duke.edu/~ccc14/sta-663/MCMC.html
https://python.arviz.org/en/stable/
https://pymcmc.readthedocs.io/en/latest/modelchecking.html
https://distribution-explorer.github.io/multivariate_continuous/lkj.html
https://bayesiancomputationbook.com/markdown/chp_08.html
https://michael-franke.github.io/intro-data-analysis/bayesian-p-values-model-checking.html
"""


class _BaseSimulator(object):
    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            boundary_observation_model: BoundaryObservationModel = None,
    ):
        if not model._is_built:
            raise ValueError('need to build model')
        if not substrate_df.index.unique().all():
            raise ValueError(f'non-unique identifiers for labelling! {substrate_df.index}')

        self._obmods = {}
        self._obsize = {}

        has_log_prob = []
        i, j = 0, 0
        for k, (labelling_id, obmod) in enumerate(mdv_observation_models.items()):
            model.set_substrate_labelling(substrate_df.loc[labelling_id])  # NB check whether valid susbtrate_df
            if not model.state_id.equals(obmod.state_id):
                raise ValueError
            if not model._la == obmod._la:
                raise ValueError
            self._obmods[labelling_id] = obmod
            has_log_prob.append(hasattr(obmod, 'log_lik'))
            self._obsize[labelling_id] = i, i + obmod._nd
            i += obmod._nd

        self._is_exact = all(has_log_prob) & (hasattr(boundary_observation_model, 'log_lik') if
                                              boundary_observation_model is not None else True)
        self._model = model
        self._la = model._la
        self._fcm = model._fcm
        self._substrate_df = substrate_df.loc[list(self._obmods.keys())]
        if boundary_observation_model is not None:
            bo_id = boundary_observation_model.boundary_id.get_level_values(1)
            bo_fluxes_id = bo_id.to_series().replace({v: k for k, v in model._only_rev.items()})
            if not bo_fluxes_id.isin(model.labelling_fluxes_id).all():
                raise ValueError
            if not model._la == boundary_observation_model._la:  # TODO make sure that device also matches
                raise ValueError
            # NB this means that we always append the boundary fluxes to the end of the last dimension!
            self._bo_idx = self._la.get_tensor(  # TODO maybe make this select from prior fluxes??
                values=np.array([self._model.labelling_fluxes_id.get_loc(rid) for rid in bo_fluxes_id], dtype=np.int64)
            )
        self._bomsize = 0 if boundary_observation_model is None else len(self._bo_idx)
        self._bom = boundary_observation_model
        self._did = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        model = state.get('_model')
        self._model.build_model(free_reaction_id=model._free_reaction_id)

    @property
    def data_id(self):
        if self._did is None:
            did = []
            for labelling_id, obmod in self._obmods.items():
                did.extend(obmod.data_id.tolist())
            if self._bomsize > 0:
                did.extend(self._bom.boundary_id.tolist())
            self._did = pd.MultiIndex.from_tuples(did, names=['labelling_id', 'data_id'])
        return self._did.copy()

    def _pandalize_data(self, data, index, n_obs, return_mdvs=False):
        if return_mdvs:
            columns = make_multidex({k: self._model.state_id for k in self._obmods}, 'labelling_id', 'mdv_id')
            n_f, n_mdv, n_s = data.shape
            data = self._la.tonp(data).reshape(n_f, n_mdv * n_s)
            return pd.DataFrame(data, index=index, columns=columns)
        else:
            n_f = data.shape[0]
            n_obshape = max(1, n_obs)
            data = self._la.tonp(data).transpose(1, 0, 2).reshape((n_f * n_obshape, len(self.data_id)))
            if index is None:
                index = pd.RangeIndex(n_f)
            if n_obs > 0:
                obs_index = pd.RangeIndex(n_obshape)
                index = make_multidex({k: obs_index for k in index}, 'samples_id', 'obs_i')
            return pd.DataFrame(data, index=index, columns=self.data_id)

    def simulate(
            self,
            labelling_fluxes=None,
            n_obs=3,
            return_mdvs=False, # whether to return mdvs, observation_average or noisy observations
            pandalize=False,
            mdvs=None,
    ):
        index = None
        if isinstance(labelling_fluxes, pd.DataFrame):
            index = labelling_fluxes.index
            labelling_fluxes = self._la.get_tensor(values=labelling_fluxes.loc[:, self._fcm.fluxes_id].values)

        if len(labelling_fluxes.shape) > 2:
            raise ValueError(f'only handles (n_samples x n_fluxes) data, got {labelling_fluxes.shape}')

        if mdvs is None:
            if len(labelling_fluxes.shape) > 2:
                raise ValueError('pass n_samples x n_fluxes array!')
            n_f = labelling_fluxes.shape[0]
            self._model.set_fluxes(labelling_fluxes, index, trim=True)  # this is where wrongly shaped fluxes are caught!
            labelling_fluxes = self._model._fluxes
        elif return_mdvs:
            raise ValueError('return passed MDVS?')
        else:  # this is for when we pass theta and mdvs and the model batch_size does not match the shape of fluxes
            labelling_fluxes = self._fcm.frame_fluxes(labelling_fluxes, index, trim=True)
            n_f = mdvs.shape[0]

        slicer = 0 if n_obs == 0 else slice(None)
        n_obshape = max(1, n_obs)

        if return_mdvs:
            result = self._la.get_tensor(shape=(n_f, len(self._obmods), int(self._model._s.shape[-1])))
        else:
            result = self._la.get_tensor(shape=(n_f, n_obshape, len(self.data_id)))
            if self._bomsize > 0:
                result[:, slicer, -self._bomsize:] = self._bom.sample_observation(
                    labelling_fluxes[:, self._bo_idx], n_obs=n_obs
                )

        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            j, k = self._obsize[labelling_id]
            if mdvs is not None:
                mdv = mdvs[:, i, :]
            else:
                self._model.set_substrate_labelling(substrate_labelling=self._substrate_df.loc[labelling_id])
                mdv = self._model.cascade()
            if return_mdvs:
                result[:, i, :] = mdv
            else:
                result[:, slicer, j:k] = obmod(mdv, n_obs=n_obs)

        if pandalize:
            return self._pandalize_data(result, index, n_obs, return_mdvs)
        return result

    def to_partial_mdvs(self, data, is_mdv=False, normalize=False, pandalize=True, append_bom=True):
        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = self._la.get_tensor(values=data.values)
            if is_mdv:
                data = data[:, None, :]

        processed = []
        columns = {}
        for i, (labelling_id, obmod) in enumerate(self._obmods.items()):
            if is_mdv:
                processed.append(obmod.compute_observations(data[:, i, :]))
            else:
                if obmod._transformation is None:
                    raise ValueError(
                        'obmod does not have transformation specified, perhaps data is not log-ratio transformed'
                    )
                j, k = self._obsize[labelling_id]
                part_mdvs = obmod._transformation.inv(data[..., j:k])  # = intensities
                if normalize:
                    part_mdvs = obmod.compute_observations(part_mdvs, select=False)
                processed.append(part_mdvs)
            columns[labelling_id] = obmod._observation_df.index.copy()
        if (self._bomsize > 0) and append_bom and not is_mdv:
            processed.append(data[..., -self._bomsize:])
            columns['BOM'] = self._bom.boundary_id.get_level_values(1)
        processed = self._la.cat(processed, -1)
        if pandalize:
            processed = pd.DataFrame(self._la.tonp(processed), index=index, columns=make_multidex(columns, name1='data_id'))
        return processed

    def _verify_hdf(self, hdf: pt.file):
        substrate_df = pd.read_hdf(hdf.filename, key='substrate_df', mode=hdf.mode)
        if not self._substrate_df.equals(substrate_df):
            raise ValueError('hdf has different substrate_df')
        for what, compare in {
            'fluxes': self._fcm.fluxes_id,
            'mdv': self._model.state_id,
            'data': self.data_id,
        }.items():
            if what == 'data':
                what_id = pd.MultiIndex.from_frame(pd.read_hdf(hdf.filename, key='data_id', mode=hdf.mode))
            else:
                what_id = pd.Index(hdf.root[f'{what}_id'].read().astype(str), name=f'{what}_id')
            if not compare.equals(what_id):
                raise ValueError(f'{what}_id is different between model and hdf')

    @hdf_opener_and_closer(mode='a')
    def to_hdf(
            self,
            hdf,
            result: dict,
            dataset_id: str,
            append=True,
            expectedrows_multiplier=10,
    ):
        if 'substrate_df' not in hdf.root:
            # TODO think about storing the ilr_basis, annotation_df, total intensities and TOFPArameters to file
            # this signals that the hdf has been freshly created
            self._substrate_df.to_hdf(hdf.filename, key='substrate_df', mode=hdf.mode, format='table')
            pt.Array(hdf.root, name='mdv_id', obj=self._model.state_id.values.astype(str))
            pt.Array(hdf.root, name='fluxes_id', obj=self._model._fcm.fluxes_id.values.astype(str))  # NB these are the untrimmed fluxes
            self.data_id.to_frame(index=False).to_hdf(hdf.filename, key='data_id', mode=hdf.mode, format='table')
        else:
            self._verify_hdf(hdf)

        if (dataset_id in hdf.root) and not append:
            hdf.remove_node(hdf.root, name=dataset_id, recursive=True)

        if dataset_id not in hdf.root:
            hdf.create_group(hdf.root, name=dataset_id)

        dataset_children = hdf.root[dataset_id]._v_children
        if (len(dataset_children) > 0) and (dataset_children.keys() != result.keys()):
            raise ValueError(f'result {result.keys()} has different data than dataset {dataset_children.keys()}; cannot append!')

        dataset_shapes = []
        for item, array in result.items():
            if isinstance(array, pd.DataFrame):
                array = array.values
            if not isinstance(array, np.ndarray):
                array = self._la.tonp(array)
            if item in hdf.root[dataset_id]:
                ptarray = hdf.root[dataset_id][item]
            else:
                atom = pt.Atom.from_type(str(array.dtype))
                ptarray = pt.EArray(
                    hdf.root[dataset_id], name=item, atom=atom, shape=(0, *array.shape[1:]),
                    expectedrows=array.shape[0] * expectedrows_multiplier, chunkshape=None,
                )
            ptarray.append(array)
            dataset_shapes.append(ptarray.shape[0])
        if not all(np.array(dataset_shapes) == dataset_shapes[0]):
            raise ValueError(f'unbalanced dataset: {dataset_shapes}, its on you to fix things now, have fun!')

    @hdf_opener_and_closer(mode='r')
    def read_hdf(
            self,
            hdf:   str,
            dataset_id: str,
            what:  str,
            start: int = None,
            stop:  int = None,
            step:  int = None,
            pandalize: bool = False,
    ) -> Union[np.array, pd.DataFrame]:
        if (dataset_id not in hdf.root):
            raise ValueError(f'{dataset_id} not in hdf')
        elif (what != 'substrate_df') and (what not in hdf.root[dataset_id]):
            raise ValueError(f'{what} not in {dataset_id}')

        self._verify_hdf(hdf)

        if (what == 'substrate_df') or pandalize:
            substrate_df = pd.read_hdf(hdf.filename, key='substrate_df', mode=hdf.mode)
            if what == 'substrate_df':
                return substrate_df

        xcsarr = hdf.root[dataset_id][what].read(start, stop, step) # .squeeze()  # TODO why did we squeeze before?

        if not pandalize:
            return self._la.get_tensor(values=xcsarr)

        labelling_id = substrate_df.index.rename('labelling_id')

        if start is None:
            start = 0
        if step is None:
            step = 1
        if stop is None:
            stop = xcsarr.shape[0]

        samples_id = pd.RangeIndex(start, stop, step, name='samples_id')

        if what == 'fluxes':
            xcs_id = pd.Index(hdf.root[f'{what}_id'].read().astype(str), name=f'{what}_id')
            return pd.DataFrame(xcsarr, index=samples_id, columns=xcs_id)

        elif what == 'validx':
            return pd.DataFrame(xcsarr, index=samples_id, columns=labelling_id)

        elif what == 'data':
            i_obs = (*range(xcsarr.shape[1]), )
            dataframes = [pd.DataFrame(xcsarr[:, i, :], index=samples_id, columns=self.data_id) for i in i_obs]
            dataframe = pd.concat(dataframes, keys=i_obs, names=['i_obs', 'samples_id'])
            return dataframe.swaplevel(0, 1, 0).sort_values(by=['samples_id', 'i_obs'], axis=0).loc[:, labelling_id]

        elif what == 'mdv':
            return pd.concat([
                pd.DataFrame(xcsarr[:, i], index=samples_id, columns=self._model.state_id)
                for i in range(len(labelling_id))], axis=1, keys=labelling_id
            )

    def __call__(self, labelling_fluxes, n_obs=3, pandalize=False, **kwargs):
        vape = labelling_fluxes.shape
        data = self.simulate(labelling_fluxes, n_obs, return_mdvs=False, pandalize=pandalize)
        if pandalize:
            return data
        n_obshape = max(1, n_obs)
        return self._la.view(data, shape=(*vape[:-1], n_obshape, vape[-1]))


class DataSetSim(_BaseSimulator):
    def __init__(
            self,
            model: LabellingModel,
            substrate_df: pd.DataFrame,
            mdv_observation_models: Dict[str, MDV_ObservationModel],
            boundary_observation_model: BoundaryObservationModel = None,
            num_processes=0,
    ):
        super(DataSetSim, self).__init__(model, substrate_df, mdv_observation_models, boundary_observation_model)

        if num_processes < 0:
            num_processes = psutil.cpu_count(logical=False)
        self._num_processes = num_processes

        self._mp_pool = None
        if num_processes > 0:
            self._mp_pool = mp.Pool(
                processes=self._num_processes, initializer=init_observer,
                initargs=(self._model, self._obmods)
            )

    def __getstate__(self):
        if self._mp_pool is not None:
            self._mp_pool.terminate()
            self._mp_pool.join()
        self._mp_pool = None
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._num_processes > 0:
            self._mp_pool = mp.Pool(self._num_processes)

    def _fill_results(self, result, worker_result):
        input_labelling = worker_result['input_labelling']
        labelling_id = input_labelling.name
        start, stop = worker_result['start_stop']
        start_stop_idx = self._la.arange(start, stop)
        i, j = self._obsize[labelling_id]
        i_obs = self._substrate_df.index.get_loc(labelling_id)
        result['validx'][start_stop_idx, i_obs] = worker_result['validx_chunk']

        for key in result.keys():
            if key == 'data':
                result['data'][start_stop_idx, :, i:j] = worker_result['data_chunk']
            elif (key == 'validx'):
                continue
            else:
                chunk = worker_result.get(f'{key}_chunk')
                if chunk is None:
                    continue
                result[key][start_stop_idx, i_obs] = chunk

    def simulate_set(
            self,
            labelling_fluxes,
            n_obs=3,
            fluxes_per_task=None,
            what='data',
            break_i=-1,
            show_progress=False,
    ) -> {}:

        if isinstance(labelling_fluxes, pd.DataFrame):
            labelling_fluxes = self._la.get_tensor(values=labelling_fluxes.loc[:, self._fcm.fluxes_id].values)

        if labelling_fluxes.ndim > 2:
            raise ValueError(f'only handles (n_samples x n_fluxes) data, got {labelling_fluxes.shape}')

        result = {}
        result['validx'] = self._la.get_tensor(shape=(labelling_fluxes.shape[0], len(self._obmods)), dtype=np.bool_)
        result['fluxes'] = labelling_fluxes

        labelling_fluxes = self._model._fcm.frame_fluxes(labelling_fluxes, trim=True)

        if what not in ('all', 'data', 'mdv'):
            raise ValueError('not sure what to simulate')
        if labelling_fluxes.shape[0] < self._la._batch_size:
            raise ValueError(f'n must be at least batch size: {self._la._batch_size}')

        if what != 'data':
            result['mdv'] = self._la.get_tensor(shape=(labelling_fluxes.shape[0], len(self._obmods), len(self._model.state_id)))
        if what != 'mdv':
            n_obshape = max(1, n_obs)
            result['data'] = self._la.get_tensor(shape=(labelling_fluxes.shape[0], n_obshape, len(self.data_id)))

        if (self._bomsize > 0) and (what != 'mdv'):
            slicer = 0 if n_obs == 0 else slice(None)
            bo_fluxes = labelling_fluxes[:, self._bo_idx]
            result['data'][:, slicer, -self._bomsize:] = self._bom(bo_fluxes, n_obs=n_obs)

        if fluxes_per_task is None:
            fluxes_per_task = math.ceil(labelling_fluxes.shape[0] / max(self._num_processes, 1))

        fluxes_per_task = min(labelling_fluxes.shape[0], fluxes_per_task)

        tasks = observator_tasks(
            labelling_fluxes, substrate_df=self._substrate_df, fluxes_per_task=fluxes_per_task, n_obs=n_obs, what=what
        )
        if show_progress:
            pbar = tqdm.tqdm(total=labelling_fluxes.shape[0] * self._substrate_df.shape[0], ncols=100)

        if self._num_processes == 0:
            init_observer(self._model, self._obmods)
            for i, task in enumerate(tasks):
                worker_result = obervervator_worker(task)
                self._fill_results(result, worker_result)
                if show_progress:
                    i, j = worker_result['start_stop']
                    pbar.update(n = j - i)
                # self._fill_results(result, obervervator_worker(task))
                if (break_i > -1) and (i > break_i):
                    break
        else:
            for worker_result in self._mp_pool.imap_unordered(obervervator_worker, iterable=tasks):
                self._fill_results(result, worker_result)
                if show_progress:
                    i, j = worker_result['start_stop']
                    pbar.update(n = j - i)
        if show_progress:
            pbar.close()
            result['running_time'] = pbar.format_dict['elapsed']
        return result

    def __call__(
            self, labelling_fluxes, n_obs=5, fluxes_per_task=None, close_pool=False, show_progress=False, pandalize=False,
            return_time=False, **kwargs
    ):
        index = None
        if isinstance(labelling_fluxes, pd.DataFrame):
            index = labelling_fluxes.index

        vape = labelling_fluxes.shape
        result = self.simulate_set(
            labelling_fluxes, n_obs,
            fluxes_per_task=fluxes_per_task,
            what='data',
            show_progress=show_progress,
        )

        data = result['data']
        n_obshape = max(n_obs, 1)

        if pandalize:
            return self._pandalize_data(data, index, n_obs)
        data = self._la.view(data, shape=(*vape[:-1], n_obshape, len(self._did)))
        if return_time and show_progress:
            return data, result['running_time']
        return data


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    from sbmfi.priors.uniform import UniformRoundedFleXchPrior

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    model, kwargs = spiro(
        backend='torch', v2_reversible=True, v5_reversible=False, build_simulator=True, which_measurements='com',
        which_labellings=['A', 'B']

    )
    up = UniformRoundedFleXchPrior(model.flux_coordinate_mapper)
    dss = DataSetSim(
        model=model,
        substrate_df=kwargs['substrate_df'],
        mdv_observation_models=kwargs['basebayes']._obmods,
        boundary_observation_model=kwargs['basebayes']._bom,
        num_processes=2,
    )

    samples = up.sample((1000,))
    labelling_fluxes = model.flux_coordinate_mapper.map_theta_2_fluxes(samples, rescale_val=None)

    result = dss.simulate_set(
        labelling_fluxes=labelling_fluxes,
        n_obs=2,
        show_progress=True,
    )
    print(result.keys())


    # smc_tomek = "C:\python_projects\sbmfi\SMC_e_coli_glc_tomek_obsmod_copy_NEW.nc"
    # v_rep = pd.read_excel(
    #     r"C:\python_projects\sbmfi\src\sbmfi\inference\VREP_MCMC_e_coli_glc_anton_obsmod_copy_NEWWP.xlsx",
    #     index_col=None)
    # pol = pickle.load(open(r"C:\python_projects\sbmfi\build_e_coli_anton_glc_F_round_pol.p", 'rb'))
    # psmc = SMC_PLOT(pol, inference_data=smc_tomek, v_rep=v_rep)
    # # model, kwargs = build_e_coli_anton_glc(
    # #     backend='numpy',
    # #     auto_diff=False,
    # #     build_simulator=True,
    # #     ratios=False,
    # #     batch_size=25,
    # #     which_measurements='tomek',
    # #     which_labellings=['20% [U]Glc', '[1]Glc'],
    # #     measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
    # #     seed=1,
    # # )
    # import pickle
    #
    # # pickle.dump((model, kwargs), open('m_k.p', 'wb'))
    # model, kwargs = pickle.load(open('m_k.p', 'rb'))
    #
    # bay = kwargs['basebayes']
    # up = UniNetFluxPrior(model, cache_size=2000)
    # smc = SMC(
    #     model=model,
    #     substrate_df=kwargs['substrate_df'],
    #     mdv_observation_models=bay._obmods,
    #     boundary_observation_model=bay._bom,
    #     prior=up,
    #     num_processes=0,
    # )
    #
    # smc.to_partial_mdvs(psmc._data.posterior_predictive.data.values[-1], normalize=False, pandalize=False)

    #
    # model, kwargs = spiro(backend='torch', v2_reversible=True, build_simulator=True, which_measurements='lcms',
    #                       which_labellings=list('AB'))
    # up = UniNetFluxPrior(model._fcm, cache_size=50)
    # dss = DataSetSim(
    #     model,
    #     substrate_df=kwargs['substrate_df'],
    #     mdv_observation_models=kwargs['basebayes']._obmods,
    #     boundary_observation_model=kwargs['basebayes']._bom,
    #     num_processes=0,
    # )
    # theta = up.sample((10,))
    # result = dss.simulate_set(theta, what='all', n_obs=0, show_progress=True, close_pool=False)
    # mdvs = result['mdv']
    #
    # data = dss.simulate(mdvs=mdvs, n_obs=0)
