import numpy as np
import pandas as pd
from typing import Iterable, Union, Dict, Tuple
from itertools import product, cycle

from scipy.linalg import helmert
from cobra import Metabolite
from sbmfi.core.linalg import LinAlg
from sbmfi.core.model import LabellingModel, RatioMixin
from sbmfi.core.metabolite import EMU
from sbmfi.core.coordinater import FluxCoordinateMapper
from sbmfi.core.polytopia import (
    project_polytope,
    LabellingPolytope,
    simplify_polytope
)
from sbmfi.core.util import (
    make_multidex,
    _bigg_compartment_ids
)
from sbmfi.lcmsanalysis.util import (
    build_correction_matrix,
    gen_annot_df,
    _strip_bigg_rex
)
# from sbmfi.lcmsanalysis.zemzed import add_formulas
from sbmfi.lcmsanalysis.formula import Formula
from sbmfi.lcmsanalysis.adducts import emzed_adducts


class MDV_ObservationModel(object):
    def __init__(
            self,
            model: LabellingModel,
            annotation_df: pd.DataFrame,
            labelling_id: str,
            transformation=None,
            correct_natab=False,
            clip_min=750.0,
            **kwargs,
    ):
        self._la = model._la
        self._annotation_df = annotation_df
        if labelling_id == 'BOM':
            raise ValueError('BOM is not a valid labelling_id')
        self._labelling_id = labelling_id  # TODO make the observation_id incorporate the labelling_id
        self._observation_df = self.generate_observation_df(model=model, annotation_df=annotation_df)
        self._n_o = self._observation_df.shape[0]
        self._natcorr = correct_natab
        if correct_natab:
            self._natab = self._set_natural_abundance_correction() # TODO this currently sucks
        self._state_id = model.state_id

        self._scaling = self._la.get_tensor(shape=(self._n_o, ))
        self._scaled = True
        self._mdv_scaling = self._la.get_tensor(shape=(len(model.state_id),))

        self._cmin = clip_min

        if transformation is not None:
            if (clip_min is None) or (clip_min <= 0.0):
                raise ValueError(f'For log-ratio transforms to work, set a positive clip_min, not: {clip_min}')
            ilr_basis = kwargs.pop('ilr_basis', 'helmert')
            transformation = MDV_LogRatioTransform(
                observation_df=self._observation_df,
                linalg=self._la,
                transformation=transformation,
                ilr_basis=ilr_basis
            )
        self._transformation = transformation
        self._nd = len(self.data_id)

    @property
    def labelling_id(self):
        return self._labelling_id

    @property
    def data_id(self) -> pd.Index:
        if self._transformation is not None:
            did = self._transformation.transformation_id.copy()
        else:
            did = self._observation_df.index.copy()
        did.name = 'data_id'
        return make_multidex({self._labelling_id: did}, 'labelling_id', 'data_id')

    @property
    def transformation(self):
        if self._transformation is None:
            return
        return self._transformation._transformation

    @property
    def state_id(self) -> pd.Index:
        return self._state_id.copy()

    @property
    def annotation_df(self):
        return self._annotation_df.copy()

    @property
    def observation_df(self):
        return self._observation_df.copy()

    @property
    def scaling(self):
        scaling = {}
        for ion_id, indices in self._ionindices.items():
            scaling[ion_id] = self._la.tonp(self._scaling[indices][0])
        return pd.Series(scaling, name='scaling')

    @staticmethod
    def generate_observation_df(model: LabellingModel, annotation_df: pd.DataFrame, verbose=False):
        columns = pd.Index(['met_id', 'formula', 'adduct_name', 'nC13'])
        assert columns.isin(annotation_df.columns).all()
        annotation_df.reset_index(drop=True, inplace=True)  # necessary for annot_df_idx to be set correctly

        return_ids = model.state_id
        cols = []
        for i, (met_id, formula, adduct_name, nC13) in annotation_df.loc[:, columns].iterrows():
            if met_id not in model.measurements:
                if verbose:
                    print(f'{met_id} not in model.measurements')
                continue

            if adduct_name in ['M-H', 'M+H']:
                adduct_str = ''
            else:
                adduct_str = f'_{{{adduct_name}}}'
            oid = f'{met_id}{adduct_str}'  # id of the observation
            f = Formula(formula)
            met = model.measurements.get_by_id(met_id)
            if isinstance(met, EMU):
                n_C = len(met.positions)
            elif isinstance(met, Metabolite):
                n_C = met.elements['C']
            if (n_C != f['C']) and verbose:
                print(f'model measurement {met} with {n_C} carbons is different from annotated formula {formula}')
            model_return_id = f'{met_id}+{nC13}'
            state_idx = np.where(return_ids == model_return_id)[0][0]

            ion_row = emzed_adducts.loc[adduct_name]
            f = f * int(ion_row['m_multiplier']) \
                + Formula(ion_row['adduct_add']) \
                - Formula(ion_row['adduct_sub']) \
                + {'-': int(ion_row['z']) * -int(ion_row.get('sign_z', 1))}
            f = f.add_C13(nC13)
            isotope_decomposition = f.to_chnops()
            cols.append((i, met_id, oid, formula, adduct_name, nC13, isotope_decomposition, state_idx))

        obs_df = pd.DataFrame(cols, columns=[
            'annot_df_idx', 'met_id', 'ion_id', 'formula', 'adduct_name', 'nC13', 'isotope_decomposition', 'state_idx'
        ])
        obs_df.index = obs_df['ion_id'] + '+' + obs_df['nC13'].astype(str)
        obs_df.index.name = 'observation_id'
        obs_df = obs_df.drop_duplicates()  # TODO figure out a bug that replicates rows a bunch of times
        obs_df = obs_df.sort_values(by=['met_id', 'adduct_name', 'nC13'])  # sorting is essential for block-diagonal structure!

        if 'sigma' in annotation_df.columns:
            # sigma is defined per measurement
            obs_df['sigma'] = annotation_df.loc[obs_df['annot_df_idx'].values, 'sigma'].values

        def check_ion_equality(df, column_id, tol=1e-5):
            vals = df[column_id].values
            return (abs(vals - vals[0]) < tol).all()

        if 'omega' in annotation_df.columns:
            # omega is defined per ion
            obs_df['omega'] = annotation_df.loc[obs_df['annot_df_idx'], 'omega'].values
            obs_df['omega'] = obs_df['omega'].fillna(1.0)
            all_ion = obs_df.groupby('ion_id').apply(check_ion_equality, column_id='omega')
            if not all_ion.all():
                raise ValueError(
                    f'omega incorrectly set: {all_ion}, '
                    f'they should be equal across ions (metabolite + ionization)'
                )

        if 'total_I' in annotation_df.columns:
            # total_I is defined per ion
            obs_df['total_I'] = annotation_df.loc[obs_df['annot_df_idx'], 'total_I'].values
            all_ion = obs_df.groupby('ion_id').apply(check_ion_equality, column_id='total_I')
            if not all_ion.all():
                raise ValueError(
                    f'total_I incorrectly set: {all_ion}, '
                    f'they should be equal across ions (metabolite + ionization)'
                )
        return obs_df

    def _set_scaling(self, scaling: pd.Series, ionindices, transform_scaling=True):
        if scaling.index.duplicated().any():
            raise ValueError('double ions')
        # num_C = self._observation_df['isotope_decomposition'].apply(lambda x: Formula(x).no_isotope()['C'])
        # ions = num_C.index.str.rsplit('+', n=1, expand=True).to_frame().reset_index(drop=True)[0]
        odf = self._observation_df
        for ion_id, value in scaling.items():
            indices = ionindices.get(ion_id)
            if indices is not None:
                value = self._la.get_tensor(values=np.array([value]))
                self._scaling[indices] = value
                mdv_indices = self._la.get_tensor(values=odf.loc[(odf['ion_id'] == ion_id), 'state_idx'].values)
                self._mdv_scaling[mdv_indices] = value
        if (self._scaling <= 0.0).any():
            raise ValueError(f'a total intensity is not set: {self.scaling}')
        if (self._transformation is not None) and transform_scaling:
            self._transformation.set_scaling(self._scaling)
        self._scaled = transform_scaling

    def check_x_meas(self, x_meas: pd.Series, atol=1e-3):
        # check whether the scaling makes sense and whether the clips are respected
        if isinstance(x_meas, pd.Series):
            x_meas = x_meas.to_frame().T
        if isinstance(x_meas, pd.DataFrame):
            x_meas = x_meas.values
        x_meas = self._la.atleast_2d(self._la.get_tensor(values=x_meas))
        if self._transformation is not None:
            x_meas = self._transformation.inv(x_meas)
        totals = (self._denom_sum @ x_meas.T)[self._denomi].T

        if self._scaled:
            # can unfortunately not check this for LCMS model, since we lose the total intensity information!
            correct_scaling = (abs(totals - self._scaling) <= atol).all()
            over_cmin = True if self._cmin is None else (totals >= self._cmin).all()

            if not all((correct_scaling, over_cmin)):
                raise ValueError(
                    f'the measurement cannot be produced by the observation model. '
                    f'Correct scaling: {correct_scaling}, '
                    f'over clip_min: {over_cmin}'
                )

    def _set_natural_abundance_correction(self, isotope_threshold=1e-4, correction_threshold=0.001):
        if self._observation_df.empty:
            raise ValueError('first set observations G')
        indices = []
        values = []
        tot_obs = 0

        for (Mid, isotope_decomposition), df in self._observation_df.groupby(by=['met_id', 'isotope_decomposition']):
            formula = Formula(formula=isotope_decomposition)
            mat = build_correction_matrix(
                formula=formula, isotope_threshold=isotope_threshold, overall_threshold=correction_threshold
            )
            slicer = df['nC13'].values
            normalizing_cons = mat[:, 0].sum()  # NOTE: making sure that the last row/ first col sum to 1
            mat /= normalizing_cons
            mat = mat[slicer, :][:, slicer]
            index = np.nonzero(mat)
            vals = mat[index]
            values.append(vals)
            indices.append(np.array(index, dtype=np.int64).T + tot_obs)
            tot_obs += df.shape[0]
        values = np.concatenate(values)
        indices = np.concatenate(indices)

        return self._la.get_tensor(shape=(tot_obs, tot_obs), values=values, indices=indices)

    def sample_observations(self, mdv, n_obs=3, **kwargs):
        raise NotImplementedError

    def _sum_square_diff(self, data, x_meas):
        # data = n_samples x n_obs x n_data
        if len(data.shape) < 3:
            data = self._la.unsqueeze(data, -2)  # adds the n_obs dimension for simulations with n_obs=0 in the previous step
        return self._la.sum((data - x_meas) ** 2, -1, keepdims=False) # sum over n_data

    def euclidean(self):
        raise NotImplementedError

    def rmse(self, data, x_meas):
        if len(data.shape) < 2:
            raise ValueError
        # diff_2 = n_samples x n_obs
        diff_2 = self._sum_square_diff(data, x_meas)
        # rmse = n_samples
        return self._la.sqrt(self._la.mean(diff_2, -1, keepdims=False)) # mean over n_obs

    def __call__(self, mdv, n_obs=3, pandalize=False, **kwargs):
        index = None
        if isinstance(mdv, pd.DataFrame):
            index = mdv.index
            mdv = self._la.get_tensor(values=mdv.loc[:, self.state_id].values)

        result = self.sample_observations(mdv, n_obs=n_obs, **kwargs)
        if self._transformation is not None:
            result = self._transformation(result)

        if pandalize:
            n_samples = mdv.shape[0]
            if index is None:
                index = pd.RangeIndex(n_samples)
            n_obshape = max(1, n_obs)
            obs_index = pd.RangeIndex(n_obshape)
            index = make_multidex({i: obs_index for i in index}, 'samples_id', 'obs_i')
            if len(result.shape) > 2:
                result = self._la.tonp(result).transpose(1, 0, 2)
            result = result.reshape((n_samples*n_obshape, len(self.data_id)))
            result = pd.DataFrame(result, index=index, columns=self.data_id)
        return result


class MDV_LogRatioTransform():
    # https://www.tandfonline.com/doi/full/10.1080/03610926.2021.2014890?scroll=top&needAccess=true
    # http://www.leg.ufpr.br/lib/exe/fetch.php/pessoais:abtmartins:a_concise_guide_to_compositional_data_analysis.pdf
    # TODO: instead of only ILR transform, introduce parameter alpha that controls a power-transform (Box-Cox) transform
    #   at alpha=0, we end up with the ilr and with alpha=1 we end up with the original data;
    #   note that this is DATA-DEPENDENT and thus we would need to prior sample or perhaps adjust in SNPE
    def __init__(
            self,
            observation_df: pd.DataFrame,
            linalg: LinAlg,
            transformation: str = 'ilr',
            ilr_basis: str = 'helmert',
    ):
        # TODO come up with some reasonable transform_id!
        # self._obsmod = mdv_observation_model

        self._observation_df = observation_df
        self._la = linalg

        n_o = observation_df.shape[0]
        self._n_t = n_o # number of transformed variables
        if transformation != 'clr':
            self._n_t = n_o - observation_df['ion_id'].unique().shape[0]

        if transformation == 'ilr':
            if ilr_basis == 'helmert':
                self._ilr_basis = self._la.get_tensor(shape=(n_o, self._n_t))
                self._sumatrix  = self._la.get_tensor(shape=(n_o, n_o))
                i, j = 0, 0
                for ion_id, df in observation_df.groupby('ion_id', sort=False):
                    # basis = self._gramm_schmidt_basis(df.shape[0])
                    basis = self._la.get_tensor(values=helmert(df.shape[0], full=False))
                    k, l = basis.shape
                    self._ilr_basis[j: j + l, i: i + k] = basis.T
                    self._sumatrix[j: j + l, j: j + l]  = 1.0
                    i += k
                    j += l
                self._scaled_sumatrix = self._la.vecopy(self._sumatrix)
                self._meantrix = self._sumatrix / self._la.sum(self._sumatrix, 0, keepdims=True)
            elif ilr_basis == 'random_sbp':
                # TODO make a random sequental binary partition?
                # TODO pass sequential binary partition for every ion;
                #   this would serve to see how much different bases affect convergence of learning
                #   {ion:
                #       [[  1,  1, -1],
                #        [  1, -1,  0]],
                #   }
                raise NotImplementedError
            else:
                raise ValueError
        elif transformation == 'alr':
            raise NotImplementedError('TODO, also need to implement self.transformation_id')
        elif transformation == 'clr':
            pass
        else:
            raise ValueError('not a valid log-ratio transoformation')

        self._transfunc = eval(f'self._{transformation}')
        self._inv_transfunc = eval(f'self._{transformation}_inv')
        self._transformation = transformation

    @property
    def transformation_id(self):
        if self._transformation == 'clr':
            return 'clr_' + self._observation_df.index
        odf_ion = self._observation_df['ion_id']
        counts = (odf_ion.value_counts() - 1)[odf_ion.unique()] # TODO this might not preserve ordering...
        return pd.Index(
            [f'{self._transformation}_{k}_{v}' for k in counts.index for v in range(counts.loc[k])],
            name='transformation_id'
        )

    def _closure(self, mat, sumatrix=None):
        if sumatrix is None:
            sum = self._la.sum(mat, dim=-1, keepdims=True)
            # sum = self._la.sum(mat, -1, True)
        else:
            sum = mat @ sumatrix
        return mat / sum

    def _clr_inv(self, clrs, sumatrix=None):
        expclrs = self._la.exp(clrs)
        return self._closure(expclrs, sumatrix)

    def _clr(self, observations, meantrix=None):
        logobs = self._la.log(observations)
        if meantrix is None:
            mean = self._la.mean(logobs, dim=-1, keepdim=True)
        else:
            mean = logobs @ meantrix
        return logobs - mean

    def _ilr_inv(self, ilrs):
        return self._clr_inv(ilrs @ self._ilr_basis.T, sumatrix=self._scaled_sumatrix)

    def _ilr(self, observations):
        return self._clr(observations, meantrix=self._meantrix) @ self._ilr_basis

    def _alr_inv(self, alrs):
        raise NotImplementedError

    def _alr(self, observations):
        raise NotImplementedError

    def _gramm_schmidt_basis(self, n):
        if n == 1:
            return self._la.get_tensor(shape=(1, 1))
        basis = self._la.get_tensor(shape=(n, n - 1))
        for j in range(n - 1):
            i = j + 1
            e = self._la.get_tensor(
                values=np.array([(1 / i)] * i + [-1] + [0] * (n - i - 1), dtype=np.double)
            ) * np.sqrt(i / (i + 1))
            basis[:, j] = e
        return basis.T

    def _sbp_basis(self, sbp):
        n_pos = (sbp == 1).sum(axis=1)
        n_neg = (sbp == -1).sum(axis=1)
        psi = np.zeros(sbp.shape)
        for i in range(0, sbp.shape[0]):
            psi[i, :] = sbp[i, :] * np.sqrt((n_neg[i] / n_pos[i]) ** sbp[i, :] / np.sum(np.abs(sbp[i, :])))
        return self._clr_inv(psi)

    def set_scaling(self, scaling):
        scaling = self._la.atleast_2d(scaling)
        self._scaled_sumatrix = self._sumatrix * (1.0 / scaling)

    def inv(self, transform):
        return self._inv_transfunc(transform)

    def __call__(self, observations):
        return self._transfunc(observations)


class _BlockDiagGaussian(object):
    """convenience class to set a bunch of indices and compute observation"""
    def __init__(self, linalg: LinAlg, observation_df: pd.DataFrame):
        self._la = linalg
        self._observation_df = observation_df
        self._no = observation_df.shape[0]

        sigma_indices = []
        tot_features = 0

        self._ionindices = {}  # for setting total intensity
        for denomi, ((model_id, ion_id, ion), df) in enumerate(
                self._observation_df.groupby(['met_id', 'ion_id', 'adduct_name'], sort=False)
        ):
            n_idion = df.shape[0]
            indices_feature = np.arange(tot_features, n_idion + tot_features, dtype=np.int64)
            self._ionindices[ion_id] = self._la.get_tensor(values=indices_feature)
            indices_block = list(product(indices_feature, indices_feature, [denomi]))
            sigma_indices += indices_block
            tot_features += n_idion

        _indices_columns = ['Σ_row_idx', 'Σ_col_idx', 'denomi', 'mdv_idx']
        sigma_indices = np.array(sigma_indices)[:, [1, 0, 2]]
        map_feat_to_mdv = dict(zip(range(self._observation_df.shape[0]), self._observation_df['state_idx'].values))
        sigma_indices = np.concatenate(
            [sigma_indices, np.vectorize(map_feat_to_mdv.get)(sigma_indices[:, 0])[:, None]], axis=1
        ).astype(np.int64)

        # these indices are used to distribute values into a sparse block-diagonal matrix
        self._indices = self._la.get_tensor(values=sigma_indices)
        self._row = self._la.get_tensor(values=sigma_indices[:, 0])
        self._col = self._la.get_tensor(values=sigma_indices[:, 1])
        self._denom = self._la.get_tensor(values=sigma_indices[:, 2])
        self._mdv = self._la.get_tensor(values=sigma_indices[:, 3])

        # these are booleans to indicate diagonal and upper triangular indices from self._indices above
        offdiag_uptri = np.array([True if (j > denomi) else False for (denomi, j, k, l) in sigma_indices])
        diagionals = sigma_indices[:, 0] == sigma_indices[:, 1]
        self._diag = self._la.get_tensor(values=diagionals)
        self._uptri = self._la.get_tensor(values=offdiag_uptri)

        # these are used to distribute values into a vector of features
        self._numi = self._la.get_tensor(values=observation_df['state_idx'].values)
        self._denomi = self._denom[self._diag]

        denom_sum_indices = np.unique(sigma_indices[:, [2, 1]], axis=0)
        self._denom_sum = self._la.get_tensor(
            shape=(sigma_indices[:, 2].max() + 1, self._observation_df.shape[0]),
            indices=denom_sum_indices, values=np.ones(denom_sum_indices.shape[0], dtype=np.double)
        )  # needs to be distributed with either self._denomi or self._denom!

        self._sigma = self._la.get_tensor(shape=(self._la._batch_size, self._no, self._no))
        self._sigma_1 = self._la.get_tensor(shape=(self._la._batch_size, self._no, self._no))
        self._chol = None
        self._bias = self._la.get_tensor(shape=(self._la._batch_size, self._no,))

    @property
    def sigma_1(self):
        if self._sigma_1 is None:
            self._sigma_1 = self._la.pinv(self._sigma, rcond= 1e-12, hermitian=False)
        return self._sigma_1

    @staticmethod
    def construct_sigma_x(observation_df: pd.DataFrame, diagonal_std: pd.Series = 0.1, corr=0.0):
        la = LinAlg(backend='numpy')
        if isinstance(diagonal_std, float):
            diagonal_std = pd.Series(diagonal_std, index=observation_df.index)
        elif isinstance(diagonal_std, pd.Series) and (len(diagonal_std) != observation_df.shape[0]):
            try:
                diagonal_std = diagonal_std.loc[observation_df.index]
            except:
                raise ValueError('wrong shape')

        diagonal_std = diagonal_std.loc[observation_df.index]
        idx = _BlockDiagGaussian(linalg=la, observation_df=observation_df)
        nf = len(diagonal_std)
        sigma = np.zeros((nf, nf))
        diagi = np.diag_indices(n=nf)
        variance = diagonal_std.values ** 2
        std = np.sqrt(variance)
        sigma[diagi] = variance / 2
        if corr > 0.0:
            sigma[idx._indices[idx._uptri, 0], idx._indices[idx._uptri, 1]] = \
                np.prod(std[idx._indices[idx._uptri, :2]], axis=1) * corr
        sigma += sigma.T
        return pd.DataFrame(sigma, index=observation_df.index, columns=observation_df.index)

    def set_sigma(self, sigma, verify=True):
        # this is mainly to set constant sigma

        if isinstance(sigma, pd.Series):
            sigma = self.construct_sigma_x(self._observation_df, diagonal_std=sigma)

        if isinstance(sigma, pd.DataFrame):
            sigma = self._la.get_tensor(
                values=sigma.loc[:, self._observation_df.index].loc[self._observation_df.index, :].values[None, :, :],
                squeeze=False
            )  # this throws error if wrong shape or features not represented

        if verify:
            sigma = self._la.get_tensor(values=sigma)  # making sure we have the correct type
            if not sigma.shape[-1] == sigma.shape[-2] == self._no:
                raise ValueError
            for i in range(sigma.shape[0]):

                variance = sigma[i, self._row[self._diag], self._col[self._diag]]
                std = self._la.sqrt(variance)
                offtri_cov = sigma[i, self._row[self._uptri], self._col[self._uptri]]
                corr_1 = std[self._row[self._uptri]] * std[self._col[self._uptri]]

                positive_var = all(variance >= 0.0)  # variance must be positive
                valid_cov = all(abs(offtri_cov) < corr_1)  # abs(correlation) <= 1
                is_diagonal = self._la.allclose(sigma, self._la.transax(sigma), rtol=1e-10)  # sigma must be diagonal
                if not (positive_var and valid_cov and is_diagonal):
                    raise ValueError(f'positive variance: {positive_var}, diagional: {is_diagonal}, valid_cov: {valid_cov}')

        self._sigma = sigma
        self._chol = self._la.cholesky(self._sigma)  # NB fails if not invertible!
        self._sigma_1 = None

    def compute_observations(self, s: pd.DataFrame, select=True, pandalize=False):
        # can take both simulations (full MDVs) or observation that need to be renormalized
        # s.shape = (n_simulations, n_mdv | n_observation)
        # select = True is used when passing MDVs, select = False is used when passing intensities
        index = None
        if isinstance(s, pd.DataFrame):
            index = s.index
            s = s.values
        s = self._la.get_tensor(values=s)
        observations_num = s
        if select:
            observations_num = s[..., self._numi]
        observations_denom = self._la.tensormul_T(self._denom_sum, observations_num)
        observations_denom[observations_denom == 0.0] = 1.0
        observations = observations_num / observations_denom[..., self._denomi]
        if pandalize:
            observations = pd.DataFrame(self._la.tonp(observations), index=index, columns=self._observation_df.index)
        return observations

    def sample_sigma(self, shape=(1, )):
        noise = self._la.randn((*shape, self._no, 1))
        res =  (self._la.unsqueeze(self._chol, 1) @ noise).squeeze(-1)
        return res


class ClassicalObservationModel(MDV_ObservationModel, _BlockDiagGaussian):
    # TODO incorporate natural abundance!

    def __init__(
            self,
            model: Union[LabellingModel, RatioMixin],
            annotation_df: pd.DataFrame,
            labelling_id: str,
            sigma_df: pd.DataFrame = None,
            omega: pd.Series = None,
            transformation = None,
            correct_natab=False,
            clip_min=0.0,
            normalize=True,
            **kwargs,
    ):
        if clip_min > 1.0:
            raise ValueError('not a valid clip_min for the the classical observation model')
        MDV_ObservationModel.__init__(
            self, model, annotation_df, labelling_id, transformation, correct_natab, clip_min, **kwargs
        )
        _BlockDiagGaussian.__init__(self, linalg=self._la, observation_df=self._observation_df)
        # TODO introduce scaling factor w as in Wiechert publications
        self._normalize = normalize
        self._initialize_J_xs()

        # variables needed for dealing with a singular FIM
        self._permutations = {}
        self._selector = None

        if sigma_df is not None:
            self.set_sigma(sigma_df, verify=True)

        # TODO complete omega (scaling factors) check whether in _x_meas the things sum to
        if omega is None:
            ion_ids = self._observation_df['ion_id'].unique()
            omega = pd.Series(1.0, index=ion_ids)
        omega = omega.fillna(1.0)
        self._set_scaling(omega, self._ionindices)

    @staticmethod
    def build_models(
            model,
            annotation_dfs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]],
            normalize=True,
            transformation=None,
            clip_min=0.0,
    ) -> dict:
        obsims = {}
        for labelling_id, (annotation_df, sigma_df, omega) in annotation_dfs.items():
            obsim = None
            if annotation_df is not None:
                obsim = ClassicalObservationModel(
                    model,
                    annotation_df=annotation_df,
                    labelling_id=labelling_id,
                    sigma_df=sigma_df,
                    omega=omega,
                    transformation=transformation,
                    clip_min=clip_min,
                    normalize=normalize,
                )
                if sigma_df is None:
                    sigma = _BlockDiagGaussian.construct_sigma_x(obsim.observation_df)
                    obsim.set_sigma(sigma, verify=True)
            obsims[labelling_id] = obsim
        return obsims

    def set_sigma_x(self, sigma_x: pd.DataFrame):
        self.set_sigma(sigma=sigma_x, verify=True)

    def _initialize_J_xs(self):
        num_sum_indices = []
        num_sum_values = []
        for i, (rowi, coli) in enumerate(zip(self._row, self._col)):
            all_denom = self._row[self._col == rowi]
            if rowi == coli:
                indices = all_denom[all_denom != rowi]
                if indices.size:
                    num_sum_indices.append([*zip(cycle([i]), self._la.tonp(indices))])
                    num_sum_values.extend([1] * indices.shape[0])
            else:
                num_sum_indices.append([[i, self._la.tonp(coli)]])
                num_sum_values.append(-1)
        num_sum_indices = np.vstack(num_sum_indices)
        self._num_sum = self._la.get_tensor(
            shape=(self._row.shape[0], self._observation_df.shape[0]),
            indices=num_sum_indices,
            values=np.array(num_sum_values, dtype=np.double)
        )  # does not need to be distributed
        self._J_xs = self._la.get_tensor(shape=(self._la._batch_size, len(self._state_id), self._n_o))

    def J_xs(self, mdv):
        mdv = self._la.atleast_2d(mdv)
        num = mdv[..., self._numi]
        denom = self._denom_sum @ num.T
        jac_num = self._num_sum @ num.T
        self._J_xs[:, self._mdv, self._col] = (jac_num / (denom ** 2)[self._denom]).T
        return self._J_xs

    def J_xv(self, mdv, J_sv=None):
        J_xs = self.J_xs(mdv=mdv)
        return J_sv @ J_xs

    def _random_selector_permutation(self, fullrank, rank):
        while True:
            permutation = self._la.randperm(n=fullrank)[:rank]
            permtup = tuple(permutation)
            if permtup not in self._permutations:
                self._permutations[permtup] = 0.0  # store determinants for every permutation
                selector = self._la.vecopy(self._selector)
                selector[permutation] = True
                yield selector

    def sigma_v(self, mdv, J_sv, rtol=1e-10, n_tries=500):
        raise NotImplementedError('needs to be able to deal with jacobians towards different flux-bases!')
        if self._selector is None:
            self._selector = self._la.get_tensor(values=np.zeros(len(model._fcm.theta_id), dtype=np.bool_))
        J_xv = self.J_xv(mdv=mdv, J_sv=J_sv)
        FIM = (J_xv @ self._la.unsqueeze(self.sigma_1, 0) @ self._la.transax(J_xv)).squeeze(0)  # Fisher Information Matrix
        sigma_v = self._la.get_tensor(shape=FIM.shape)
        summary_v = self._la.get_tensor(shape=(mdv.shape[0], 3))
        for i in range(mdv.shape[0]):
            # TODO make this actually batched computation (difficult for different ranks in same batch...)
            invertible = False
            U, S, V = self._la.svd(A=FIM[i], full_matrices=True)
            fullrank = S.shape[0]
            # numerical rank TODO maybe refine rank determination by looking at relative eigvals, filtering small/largest
            rank_FIM = sum(S > max(S) * fullrank * rtol)

            j = 0

            if rank_FIM == fullrank:
                rank = rank_FIM
                selector = self._la.vecopy(self._selector)
                selector[:] = True
                invertible = True
            else:
                self._permutations = {}
                selector_generator = self._random_selector_permutation(fullrank=fullrank, rank=rank_FIM)

            while not invertible and not (j > n_tries):
                selector = next(selector_generator)
                FIM_act = FIM[i, selector, :][:, selector]
                U, S, V = self._la.svd(A=FIM_act, full_matrices=True)
                rank_act = S.shape[0]
                rank = sum(S > max(S) * rank_act * rtol)
                if rank_act == rank:
                    # TODO we now break as soon as we find a valid invertible matrix
                    #  should make an effort to look for a minimum determinant combination??
                    invertible = True
                j += 1
            if invertible:
                mask = selector[:, None] & selector[None, :]
                sigma_v[i, mask] = (V.T @ self._la.diag(1.0 / S) @ U.T).flatten()
                summary_v[i, 0] = rank / fullrank
                summary_v[i, 1] = self._la.trace(sigma_v[i])
                summary_v[i, 2] = 1.0 / self._la.prod(S)  # |sigma_v| = 1.0 / |FIM|
        return sigma_v, summary_v

    def sample_observations(self, mdv, n_obs=3, **kwargs):
        # truncate will restrict out to the domain [0, ∞)
        # normalize will make sure that the Σ out = 1
        if self._chol is None:
            raise ValueError('set sigma_x')
        mdv = self._la.atleast_2d(mdv)  # shape = batch x n_mdv
        observations = self.compute_observations(s=mdv, select=True)  # batch x n_observables

        if self._natcorr:
            observations = self._natab @ observations

        observations *= self._scaling
        if n_obs == 0:  # this means we return the 'mean'
            return observations

        noise = self.sample_sigma(shape=(mdv.shape[0], n_obs))
        noisy_observations = observations[:, None, ...] + noise

        if (self._cmin is not None):
            noisy_observations = self._la.clip(noisy_observations, self._cmin, None)

        if self._normalize:
            if (self._cmin is None) or (self._cmin < 0.0):
                raise ValueError('cannot normalize if we have negative values')
            noisy_observations = self.compute_observations(noisy_observations, select=False)  # n_obs x batch x features
        return noisy_observations

    def log_lik(self, x_meas, mu_o):
        if self._cmin is not None:
            raise ValueError('cannot compute log_lik for observation models with a clip_min, '
                             'this would be a clipped Gaussian, which is not implemented here')
        x_meas = self._la.atleast_2d(x_meas)  # shape = n_meas x n_mdv
        mu_o = self._la.atleast_2d(mu_o)  # shape = batch x n_d
        diff = mu_o[:, None, :] - x_meas[:, None, :]  # shape = n_obs x batch x n_d
        log_lik = -0.5 * ((diff @ self.sigma_1) * diff).sum(-1)
        return log_lik


class TOF6546Alaa5minParameters(object):
    def __init__(self, is_diagonal=False, has_bias=False):
        self._la = None
        self._is_diagonal = is_diagonal
        self._has_bias = has_bias

        self._popt_cv = {
            'cv_0': 16.719278619482555,
            'lambada': -2.0877441212691403,
            'cv_o': 0.0031074197067964154,  # offset
        }
        self._popt_corr = {
            'a': -0.14393031668998882,
            'b': -0.14879596171381898,
            'c': 0.24163561630186947,
            'd': 0.43951441482203674,
            'e': 0.199848985709391,
            'f': -1.5323041071704333,
        }

    def _exp_decay(self, I, lambada, cv_0, cv_o):
        return self._la.exp(I * lambada) * cv_0 + cv_o

    def _quadratic_surface(self, I_arr, a, b, c, d, e, f):
        x = I_arr[..., 0]
        y = I_arr[..., 1]
        return  a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f

    def CV(self, I):
        return self._la.clip(  # CLIP TO ONE ORDER OF MAGNITUDE!!
            self._exp_decay(I, **self._popt_cv), None, 1.0
        )

    def std(self, I):
        CV = self.CV(I=I)
        return I * CV

    def var(self, I):
        std = self.std(I=I)
        return std**2

    def corr(self, I_arr):
        # make sure that x >= y, as assumed in the correlation surface
        switch = I_arr[..., 0] < I_arr[..., 1]
        memberberry = I_arr[..., 0][switch]
        I_arr[..., 0][switch] = I_arr[..., 1][switch]
        I_arr[..., 1][switch] = memberberry

        corr = self._quadratic_surface(I_arr, **self._popt_corr)
        corr[corr < 0.0] = 0.0
        return corr

    def cov(self, I_arr, std):
        corr = self.corr(I_arr=I_arr)
        return std[..., 0] * std[..., 1] * corr

    def bias(self, I):
        raise NotImplementedError


class LCMS_ObservationModel(MDV_ObservationModel, _BlockDiagGaussian):
    def __init__(
            self,
            model: Union[LabellingModel, RatioMixin],
            annotation_df: pd.DataFrame,
            total_intensities: pd.Series,
            labelling_id: str,
            parameters = TOF6546Alaa5minParameters(),
            transformation = None,
            correct_natab=False,
            clip_min=750.0,
            **kwargs
    ):
        MDV_ObservationModel.__init__(
            self, model, annotation_df, labelling_id, transformation, correct_natab, clip_min, **kwargs
        )
        _BlockDiagGaussian.__init__(self, linalg=self._la, observation_df=self._observation_df)
        self._total_intensities = total_intensities
        self._p = parameters
        self._p._la = self._la

        # transform_scaling=True would be incorrect since we lose information about intensities when mulitplying MDVs
        self._set_scaling(total_intensities, self._ionindices, transform_scaling=False)

    @staticmethod
    def build_models(
            model,
            annotation_dfs: Dict[str, pd.DataFrame],
            total_intensities: pd.Series = None,
            parameters: TOF6546Alaa5minParameters = TOF6546Alaa5minParameters(),
            clip_min=750.0,
            transformation='ilr',
    ) -> dict:
        obsims = {}
        for labelling_id, annotation_df in annotation_dfs.items():
            obsim = None
            if annotation_df is not None:
                obsim = LCMS_ObservationModel(
                    model,
                    annotation_df=annotation_df,
                    labelling_id=labelling_id,
                    total_intensities=total_intensities,
                    parameters=parameters,
                    clip_min=clip_min,
                    transformation=transformation
                )
            obsims[labelling_id] = obsim
        return obsims

    def construct_sigma(self, logI):
        if self._p is None:
            raise ValueError('set parameters to compute sigma elements')

        sigma = self._la.get_tensor(shape=(logI.shape[0], self._n_o, self._n_o), squeeze=False)

        std = self._p.std(I=logI) / 2
        sigma[:, self._indices[self._diag, 0], self._indices[self._diag, 1]] = std**2 / 2

        if not self._p._is_diagonal:
            cov = self._p.cov(I_arr=logI[:, self._indices[self._uptri, :2]], std=std[:, self._indices[self._uptri, :2]])
            sigma[:, self._indices[self._uptri, 0], self._indices[self._uptri, 1]] = cov

        sigma += self._la.vecopy(self._la.transax(sigma))  # easiest way to make it diagonal
        return sigma

    def sample_observations(self, mdv, n_obs=3, atol=1e-10, **kwargs):
        if self._cmin == 0.0:
            raise ValueError('need to clip brohh')
        if self._total_intensities is None:
            raise ValueError(f'set total intensities')

        mdv = self._la.atleast_2d(mdv)  # shape = batch x n_mdv
        intensities = mdv * self._mdv_scaling  #

        observations_num = intensities[..., self._numi]
        if self._natcorr:
            observations_num = self._la.tensormul_T(self._natcorr, observations_num)

        if n_obs == 0:
            clip_observations_num = self._la.clip(observations_num, self._cmin, None)
            return self.compute_observations(clip_observations_num, select=False)

        log10_observations_num = self._la.log10(observations_num + atol)
        noisy_observations = self._la.tile(log10_observations_num[:, None, :], (1, n_obs, 1))
        sigma_x = self.construct_sigma(logI=log10_observations_num)
        self.set_sigma(sigma=sigma_x, verify=False)
        noisy_observations += self.sample_sigma(shape=(n_obs,))

        if self._p._has_bias:
            bias = self._p.bias(I=log10_observations_num)
            noisy_observations += bias

        noisy_observations = 10 ** noisy_observations
        # THESE ARE THE OBSERVED INTENSITIES RESPECTING THE CLIP!
        noisy_observations = self._la.clip(noisy_observations, self._cmin, None)
        return self.compute_observations(noisy_observations, select=False)  # batch x n_obs x features


class BoundaryObservationModel(object):
    def __init__(
            self,
            model: LabellingModel,
            measured_boundary_fluxes: Iterable,
            biomass_id: str = None,  # 'bm', 'BIOMASS_Ecoli_core_w_GAM'
            check_noise_support: bool = False,
    ):
        self._la = model._la
        self._call_kwargs = {}
        self._fcm = model._fcm
        boundary_rxns = (self._fcm._Fn.S >= 0.0).all(0) | (self._fcm._Fn.S <= 0.0).all(0)
        self._bound_id = pd.Index(measured_boundary_fluxes)

        boundary_ids = self._fcm._Fn.S.columns[boundary_rxns]
        if biomass_id is not None:
            if not (biomass_id in self._fcm.fluxes_id):
                raise ValueError
            boundary_ids = boundary_ids.union([biomass_id])
        if not self._bound_id.isin(boundary_ids).all():
            raise ValueError('can only handle boundary fluxes and biomass for this observation model')

        n = len(self._bound_id)
        self._check = check_noise_support
        self._boundary_pol = None
        if check_noise_support:
            pol = self._fcm._Fn
            settings = self._fcm._sampler._pr_settings
            pol = simplify_polytope(pol, settings=settings, normalize=False)
            P = pd.DataFrame(0.0, index=self._bound_id, columns=pol.A.columns)
            P.loc[self._bound_id, self._bound_id] = np.eye(n)
            self._boundary_pol = project_polytope(pol, P)
            self._A = self._la.get_tensor(values=self._boundary_pol.A.values)
            self._b = self._la.get_tensor(values=self._boundary_pol.b.values)[:, None]

    @property
    def boundary_id(self):
        # return self._bound_id.copy()
        return make_multidex({'BOM': self._bound_id}, 'BOM', 'boundary_id')

    def sample_observation(self, mu_bo, n_obs=1, **kwargs):
        raise NotImplementedError

    def log_lik(self, bo_meas, mu_bo):
        raise NotImplementedError

    def __call__(self, mu_bo, n_obs=1):
        # vape = boundary_fluxes.shape
        # flat = boundary_fluxes.view(vape[:-1].numel(), vape[-1])
        return self.sample_observation(mu_bo, n_obs, **self._call_kwargs)


class MVN_BoundaryObservationModel(BoundaryObservationModel):
    def __init__(
            self,
            model: LabellingModel,
            measured_boundary_fluxes: Iterable,
            biomass_id: str = None,  # 'bm', 'BIOMASS_Ecoli_core_w_GAM'
            check_noise_support: bool = False,
            sigma_o=None,
            biomass_std=0.1,
            boundary_std=0.3,
    ):
        super(MVN_BoundaryObservationModel, self).__init__(
            model, measured_boundary_fluxes, biomass_id, check_noise_support
        )
        n = len(self._bound_id)
        if sigma_o is None:
            sigma_o = np.eye(n) * boundary_std ** 2
            if biomass_id is not None:
                bm_idx = self._bound_id.get_loc(biomass_id)
                sigma_o[bm_idx, bm_idx] = biomass_std ** 2
        self._sigma_o = self._la.get_tensor(values=sigma_o)
        self._sigma_o_1 = self._la.pinv(self._sigma_o, rcond= 1e-12, hermitian=False)

    def sample_observation(self, mu_bo, n_obs=1, **kwargs):
        if n_obs == 0:  # consistent with the MDV observation models!
            return mu_bo
        n, n_b = mu_bo.shape
        mu_bo = mu_bo[:, None, :]
        if not self._check:
            noise = self._la.randn(shape=(n, n_obs, len(self._bound_id))) @ self._sigma_o
            return abs(mu_bo + noise)  # .squeeze(0)

        output = self._la.get_tensor(shape=(n, n_obs, n_b))
        for i in range(n):
            mean = mu_bo[i, 0, :]
            j = 0
            rounds = 0
            while j < n_obs:
                noise = self._la.randn(shape=(n_obs * 5, len(self._bound_id))) @ self._sigma_o
                samples = mean + noise

                valid = self._la.transax((self._A @ self._la.transax(samples) <= self._b)).all(-1)
                k = min((j + min(valid.sum(), n_obs)), n_obs)
                if k > j:
                    output[i, j:k, :] = samples[valid][: (k - j), :]
                    j = k
                rounds += 1
                if rounds > 20:
                    raise ValueError('distribution samples outside of support')
        return output

    def log_lik(self, bo_meas, mu_bo):
        mu_bo = self._la.atleast_2d(mu_bo)  # shape = batch x n_bo
        bo_meas = self._la.atleast_2d(bo_meas)  # shape = n_obs x n_bo
        diff = mu_bo[:, None, :] - bo_meas[:, None, :]  # shape = batch x n_obs x n_bo
        return - 0.5 * ((diff @ self._sigma_o_1) * diff).sum(-1)


def _process_flat_frame(mdvdff, total_intensities=None, min_signal=0.05, min_frac=0.33):
    metabolites = mdvdff.columns.str.replace('\+\d+$', '', regex=True)

    if total_intensities is None:
        # NB now we work with probability vectors
        total_intensities = pd.Series(1.0, index=metabolites.unique())

    if not metabolites.unique().isin(total_intensities.index).all():
        raise ValueError('mdvdf metabolites do not have a total signal assigned')
    totI_vals = total_intensities.loc[metabolites].values

    filter_df = mdvdff * totI_vals[None, :]
    frac_ser = (filter_df > min_signal).sum(0) / mdvdff.shape[0]
    frac_ser = frac_ser.loc[frac_ser >= min_frac]

    # make sure that we do not count mdvs with only 1 signal
    n_mdv = frac_ser.index.str.rsplit('+', expand=True).to_frame(name=['met_id', 'nC13']).set_index('met_id')
    counts = n_mdv.index.value_counts()
    keep = n_mdv.index.isin(counts[counts > 1].index)
    frac_ser = frac_ser.loc[keep]
    # indices = np.where(frac_ser.index.values[:, None] == mdvdf.columns.values[None, :])[1]
    return frac_ser


def exclude_low_massiso(
        mdvdf,
        total_intensities: pd.Series = None,
        min_frac=0.33,
        min_signal=0.05,
):
    if isinstance(mdvdf.columns, pd.MultiIndex):
        res = []
        if not mdvdf.columns.names[0] == 'labelling_id':
            raise ValueError('niks')
        for i, df in mdvdf.groupby(level=0, axis=1):
            res.append(_process_flat_frame(df.droplevel(0, axis=1), total_intensities, min_signal, min_frac))
        return pd.concat(res, keys=mdvdf.columns.get_level_values(0).unique())
    else:
        return _process_flat_frame(mdvdf, total_intensities, min_signal, min_frac)


if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    # from sbmfi.core.polytopia import coordinate_hit_and_run_cpp

    import pandas as pd

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


    model, kwargs = spiro(which_measurements='lcms', build_simulator=True, L_12_omega=1.0)
    annotation_df = kwargs['annotation_df']
    fluxes = kwargs['fluxes']
    substrate_df = kwargs['substrate_df']
    model.set_substrate_labelling(substrate_labelling=substrate_df.iloc[1])

    # observation_df = LCMS_ObservationModel.generate_observation_df(model, annotation_df)
    # com = ClassicalObservationModel(model, kwargs['annotation_df'])

    # observation_df = LCMS_ObservationModel.generate_observation_df(model, annotation_df)
    # total_intensities = observation_df.drop_duplicates('ion_id').set_index('ion_id')['total_I']
    # lcms = LCMS_ObservationModel(model, annotation_df, total_intensities, 'niks', transformation='ilr')
    sim = kwargs['basebayes']
    print(sim.data_id)

    # model.set_fluxes(fluxes)
    # mdv = model.cascade()
    # obs = lcms(mdv, pandalize=True)
    # aa = pd.DataFrame(lcms._transformation.inv(obs.values), columns=lcms._observation_df.index)
    # # print(aa)
    # obs = lcms(mdv, n_obs=0, pandalize=True)
    # aa = pd.DataFrame(lcms._transformation.inv(obs.values), columns=lcms._observation_df.index)
    # # print(aa)




    # total_intensities = {}
    # unique_ion_ids = observation_df.drop_duplicates(subset=['ion_id'])
    # for _, row in unique_ion_ids.iterrows():
    #     total_intensities[row['ion_id']] = annotation_df.loc[
    #         (annotation_df['met_id'] == row['met_id']) & (annotation_df['adduct_name'] == row['adduct_name']),
    #         'total_I'
    #     ].values[0]
    # total_intensities = pd.Series(total_intensities)
    #
    #
    # obsmod_a = LCMS_ObservationModel(model, annotation_df, total_intensities)
    #
    # obsmod_b = ClassicalObservationModel(model, annotation_df)
    # sigma_x = obsmod_b.construct_sigma_x(observation_df)
    # obsmod_b.set_sigma(sigma_x)
    #
    # model.set_fluxes(fluxes)
    # mdv = model.cascade()
    #
    # mdv = model._la.tile(mdv.T, (4, )).T
    # x_meas = obsmod_a.sample_observations(mdv, n_obs=3)
    #
    # # obsmod_a.distance()
    #
    # # obsmod_b.sample_observations(mdv, n_obs=3)


    # model, kwargs = spiro(backend='numpy', build_simulator=True, batch_size=bs)
    # model, kwargs = build_e_coli_anton_glc(backend='numpy', build_simulator=False, batch_size=bs)
    # model.set_measurements(model.measurements.list_attr('id') + ['glc__D_e'])
    # adf = kwargs['anton']['annot_df']
    # adf.loc[65] = ['glc__D_e', 0, 'C6H12O6', 'M+H']
    # com = ClassicalObservationModel(model, adf)

    # model.build_simulator(free_reaction_id=['d_out', 'h_out', 'bm'])
    # bom = MVN_BoundaryObservationModel(model, measured_boundary_fluxes=['d_out', 'h_out', 'bm'], biomass_id='bm',
    #                                    check_noise_support=True)
    # sdf = kwargs['substrate_df']
    # prior = UniFluxPrior(model._fcm)
    # t, f = prior.sample_dataframes(bs)

    # obsim = ClassicalObservationModel(model, kwargs['annotation_df'])
    # sigma = _BlockDiagGaussian.construct_sigma_x(obsim.observation_df)
    # obsim.set_sigma(sigma)
    # model.set_fluxes(f)
    # mdv = model.cascade(pandalize=True)
    # o = obsim(mdv, n_obs=2, pandalize=True)


    # model, kwargs = spiro()
    # fcm = FluxCoordinateMapper(model, free_reaction_id=['d_out', 'h_out', 'bm'])
    # pickle.dump(fcm, open('spiro_fcm.p', 'wb'))
    # fcm = pickle.load(open('spiro_fcm.p', 'rb'))
    # bom = MVN_BoundaryObservationModel(fcm, biomass_id='bm', measured_boundary_fluxes=['d_out', 'h_out', 'bm'], check_noise_support=True)
    # samples = coordinate_hit_and_run(bom._boundary_pol, n=20)['fluxes']
    # aa = bom(samples.values, n_obs=5)
