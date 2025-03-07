import math

from sbmfi.core.polytopia import (
    PolytopeSamplingModel,
    V_representation,
    fast_FVA,
    LabellingPolytope,
    thermo_2_net_polytope
)
from sbmfi.core.coordinater import FluxCoordinateMapper
from sbmfi.inference.arviz_monkey import *
import pandas as pd
import holoviews as hv
from scipy.spatial import ConvexHull
import numpy as np
from typing import Iterable, Union, Dict, Tuple
from bokeh.plotting import show
from bokeh.io import output_file
import matplotlib.pyplot as plt
import colorcet


class PlotMonster(object):
    _ALLFONTSIZES = {
        'xlabel': 12,
        'ylabel': 12,
        'zlabel': 12,
        'labels': 12,
        'xticks': 10,
        'yticks': 10,
        'zticks': 10,
        'ticks': 10,
        'minor_xticks': 8,
        'minor_yticks': 8,
        'minor_ticks': 8,
        'title': 14,
        'legend': 12,
        'legend_title': 12,
    }
    _FONTSIZES = {
        'labels': 14,
        'ticks': 12,
        'minor_ticks': 8,
        'title': 16,
        'legend': 12,
        'legend_title': 14,
    }
    def __init__(
            self,
            fcm: FluxCoordinateMapper,  # this should be in the sampled basis!
            inference_data: az.InferenceData,
            v_rep: pd.DataFrame = None,
            hv_backend='matplotlib',
    ):
        # TODO make sure we can plot exchange fluxes!
        hv.extension(hv_backend)
        self._hvb = hv_backend == 'matplotlib'
        if isinstance(inference_data, str):
            inference_data = az.from_netcdf(inference_data)
        self._data = inference_data

        prior_color = '#2855de'
        post_color = '#e02450'

        self._colors = {
            'polytope': '#ffec58',
            'true': '#ff0000',
            'map': '#13f269',
            'prior': prior_color,
            'prior_predictive': prior_color,
            'posterior': post_color,
            'posterior_predictive': post_color,
        }
        self._group_var_map = {
            'posterior': 'theta',
            'prior': 'theta',
            'posterior_predictive': 'data',
            'prior_predictive': 'data',
        }
        self._var_id_map = {
            'data': 'data_id',
            'theta': 'theta_id',
        }
        self._px = 1/plt.rcParams['figure.dpi']

        self._ingroup = 'posterior'
        if ('posterior' not in inference_data) and ('prior' in inference_data):
            self._ingroup = 'prior'

        if not all(fcm.theta_id().isin(inference_data[self._ingroup].theta_id.values)):
            print(self._ingroup, )
            print(fcm.theta_id())
            print(inference_data[self._ingroup].theta_id.values)
            raise ValueError

        if v_rep is None:
            v_rep = V_representation(fcm._sampler._F_round)

        self._v_rep = v_rep
        self._fva = fast_FVA(fcm._sampler._F_round)
        if 'observed_data' in self._data:
            self._odf = self._load_observed_data()
        self._ttdf = self._load_true_theta()
        # self._max_prob_point = self._load_max_prob_point()

    @property
    def obsdat_df(self):
        return self._odf

    @property
    def theta_id(self):
        if 'prior' in self._data:
            return self._data.prior.theta_id.values
        elif 'posterior' in self._data:
            return self._data.posterior.theta_id.values
        else:
            raise ValueError

    @property
    def data_id(self):
        if 'posterior_predictive' in self._data:
            return self._data.posterior_predictive.data_id.values
        elif 'prior_predictive' in self._data:
            return self._data.prior_predictive.data_id.values

    def _axes_range(
            self,
            var_id,
            return_dimension=True,
            label=None,
            tol=12,
    ):
        if '_xch' in var_id:
            range = (-0.05, 1.05)
        else:
            fva_min = self._fva.loc[var_id, 'min']
            fva_max = self._fva.loc[var_id, 'max']

            if tol > 0:
                tol = abs(fva_min - fva_max) / 10
            range = (fva_min - tol, fva_max + tol)

        if return_dimension:
            kwargs = dict(spec=var_id, range=range)
            if label is not None:
                kwargs['label'] = label
            return hv.Dimension(**kwargs)
        return range

    def _process_points(self, points: np.ndarray):
        hull = ConvexHull(points)
        verts = hull.vertices.copy()
        verts = np.concatenate([verts, [verts[0]]])
        return hull.points[verts]

    def _get_samples(self, *args, obs_idx=None, group='posterior', num_samples=30000):
        slicer = list(args)
        first_item = slicer[0]
        if len(slicer) == 1 and isinstance(first_item, Iterable) and not isinstance(first_item, str):
            slicer = slicer[0]

        if isinstance(obs_idx, int):
            obs_idx = [obs_idx]
        elif isinstance(obs_idx, Iterable):
            obs_idx = list(obs_idx)

        if ('epsilons' in self._data.attrs) and ('posterior' in group):  # this signals that we are using an SMC dataset
            last_chain = self._data[group]['chain'].values[-1]
            data = self._data.sel(chain=[last_chain])
        else:
            data = self._data

        var_names = self._group_var_map[group]

        data = az.extract(
            data,
            group=group,
            var_names=var_names,
            combined=True,
            num_samples=num_samples,
            rng=True,
        )  # <class 'xarray.core.dataarray.DataArray'> FUCK I HATE XARRAY IT IS SO FUCKING COMPLICATED

        coord_id = self._var_id_map[var_names]
        result = data.sel({coord_id: slicer}).values
        if 'obs_idx' in data.coords:
            if obs_idx is None:
                obs_idx = data.coords['obs_idx'].values
            result = result[obs_idx].swapaxes(1, 2)
            result = result.reshape(math.prod(result.shape[:-1]), result.shape[-1]).T
        return result.T

    def _plot_distribution(
            self,
            to_plot: pd.DataFrame,
            var_id,
            label=None,
            show_grid=True,
            bandwidth=None,
            color=None,
            muted=False,
            alpha=0.3,
            muted_alpha=0.07,
            show_legend=True,
            **opts
    ):
        kwargs = locals()
        kwargs.pop('opts')
        kwargs = {k: v for i, (k, v) in enumerate(kwargs.items()) if i > 3}
        kwargs = {**kwargs, **self._size_opts(), **opts}
        if self._hvb:
            kwargs.pop('muted', None)
            kwargs.pop('muted_alpha', None)
            if 'color' in kwargs:
                kwargs['facecolor'] = kwargs.pop('color')
        return hv.Distribution(to_plot, kdims=[var_id], label=label).opts(**kwargs)

    def _get_chain(self, *args, chain_idx=-1, group='posterior'):
        var = self._group_var_map[group]
        return self._data[group][var].sel({f'{var}_id': list(args)}).values[chain_idx]

    def _size_opts(self, width=500, height=400):
        kwargs = dict(height=height, width=width,)
        if self._hvb:
            kwargs = dict(fig_inches=(height * self._px, width * self._px),)
        return kwargs

    def density_plot(
            self,
            var_id,
            num_samples=30000,
            group='posterior',
            include_fva = False,
            tol = 12,
            color=None,
            label=None,
    ):
        sampled_points = self._get_samples(var_id, group=group, num_samples=num_samples)
        if group in ['posterior', 'prior'] and var_id in self._fva.index:
            xax = self._axes_range(var_id, tol=tol)
        else:
            maxx = sampled_points.max()
            minn = sampled_points.min()
            if tol > 0:
                tol = abs(minn - maxx) / 10
            range = (minn - tol, maxx + tol)
            xax = hv.Dimension(var_id, range=range)

        if color is None:
            color = self._colors[group]
        if label is None:
            label = group

        density = self._plot_distribution(sampled_points, xax, label, color=color)
        if include_fva and (group in ['posterior', 'prior']):
            plots = [density]
            fva_min, fva_max = self._axes_range(var_id, return_dimension=False, tol=0)
            opts = dict(color='#000000', line_dash='dashed')
            if self._hvb:
                opts['linestyle'] = opts.pop('line_dash')
            plots.extend([
                hv.VLine(fva_min).opts(**opts), hv.VLine(fva_max).opts(**opts),
            ])

            return hv.Overlay(plots).opts(
                xrotation=90, show_grid=True, fontsize=self._FONTSIZES,
                show_legend=True, legend_position='right', **self._size_opts()
            )
        return density.opts(
                xrotation=90, show_grid=True, fontsize=self._FONTSIZES,
                show_legend=True, **self._size_opts()
            )

    def _plot_area(self, vertices: np.ndarray, var1_id, var2_id, label=None, color='#ebb821'):
        xax = self._axes_range(var1_id)
        yax = self._axes_range(var2_id)

        opts = dict(color=color)
        if self._hvb:
            opts['facecolor'] = opts.pop('color')

        plots = [
            hv.Area(vertices, kdims=[xax], vdims=[yax], label=label).opts(
                alpha=0.2, show_grid=True, **{**opts, **self._size_opts()}
            ),
            hv.Curve(vertices, kdims=[xax], vdims=[yax], label=label).opts(color=color)
        ]
        return hv.Overlay(plots).opts(fontsize=self._FONTSIZES)

    def _plot_polytope_area(self, var1_id, var2_id):
        pol_verts = self._v_rep.loc[:, [var1_id, var2_id]].drop_duplicates()
        vertices = self._process_points(pol_verts.values)
        return self._plot_area(vertices, var1_id, var2_id, label='polytope', color=self._colors['polytope'])

    def _data_hull(
            self,
            var1_id,
            var2_id,
            group='posterior',
            num_samples=30000
    ):
        sampled_points = self._get_samples(var1_id, var2_id, group=group, num_samples=num_samples)
        vertices = self._process_points(sampled_points)
        return self._plot_area(vertices, var1_id, var2_id, label=f'{group} sampled support', color=self._colors[group])

    def _bivariate_plot(
            self,
            var1_id,
            var2_id,
            group='posterior',
            num_samples=30000,
            bandwidth=None,
    ):
        sampled_points = self._get_samples(var1_id, var2_id, group=group, num_samples=num_samples)
        xax = self._axes_range(var1_id)
        yax = self._axes_range(var2_id)
        return hv.Bivariate(sampled_points, kdims=[xax, yax], label='density').opts(
            bandwidth=bandwidth, filled=True, alpha=1.0, cmap='Blues', fontsize=self._FONTSIZES
        )

    def _load_observed_data(self):
        measurement_id = self._data.observed_data['measurement_id']
        data_id = self._data.observed_data['data_id'].values
        return pd.DataFrame(
            self._data.observed_data['observed_data'].values, index=measurement_id, columns=data_id
        )

    def _load_true_theta(self):
        theta_id = self._data[self._ingroup]['theta_id'].values
        true_theta = self._data.attrs.get('true_theta')
        if true_theta is None:
            return
        if true_theta.ndim == 1:
            true_theta = true_theta[None, :]
        return pd.DataFrame(true_theta, columns=theta_id)

    def point_plot(self, var1_id, var2_id=None, what_var='theta', what_point='true', label=None, color=None):
        if what_var == 'theta':
            xax = self._axes_range(var1_id)
            if var2_id is not None:
                yax = self._axes_range(var2_id)
        elif what_var == 'data':
            xax = hv.Dimension(var1_id)
            yax = hv.Dimension(var1_id)
        else:
            raise ValueError

        if what_point == 'max_prob':
            if what_var not in self._max_prob_point:
                raise ValueError(f'{what_var} not in InferenceData')
            to_plot = self._max_prob_point[what_var]
            linstyle = 'dashed'
        elif what_point == 'true':
            if self._ttdf is None:
                raise ValueError('no true theta in this InferenceData')
            if what_var == 'theta':
                to_plot = self._ttdf
            else:
                to_plot = self._odf
            linstyle = 'dotted'
        print(to_plot)

        if label is None:
            label = what_point
        if color is None:
            color = self._colors[what_point]
        if var2_id is None:
            data = to_plot.loc[:, var1_id].values
            opts = dict(color=color, line_dash=linstyle, xrotation=90, line_width=2.5, show_legend=True)
            if self._hvb:
                opts['linewidth'] = opts.pop('line_width')
                opts['linestyle'] = opts.pop('line_dash')
            return hv.VLine(data).opts(**opts) * hv.Spikes(data, label=label).opts(**opts)
        opts = dict(size=4)
        if self._hvb:
            opts['s'] = opts.pop('size')
        return hv.Points(to_plot.loc[:, [var1_id, var2_id]], kdims=[xax, yax], label=what_point).opts(
            color=color, fontsize=self._FONTSIZES, **opts
        )

    # def observed_data_plot(self, var1_id, var2_id=None, what='max_prob'):
    #     if var2_id is None:
    #         return hv.VLine(self.obsdat_df.loc[:, var1_id].values).opts(
    #             color=self._colors['true_theta'], line_dash='dashed', xrotation=90
    #         )
    #     return hv.Points(self.obsdat_df.loc[:, [var1_id, var2_id]], kdims=[var1_id, var2_id]).opts(
    #         color=self._colors['true_theta'], size=7, fontsize=self._FONTSIZES
    #     )

    def _load_max_prob_point(self):
        if 'sample_stats' not in self._data:
            return
        lp = self._data.sample_stats.lp.values
        chain_idx, draw_idx = np.argwhere(lp == lp.max()).T
        row, col = chain_idx[0], draw_idx[0]
        max_lp = lp[row, col]

        theta_id = self._data['posterior']['theta_id'].values
        theta = pd.DataFrame(
            self._data['posterior']['theta'].values[row, col, :], index=theta_id
        ).T

        result = {'lp': max_lp, 'theta': theta}

        if 'posterior_predictive' in self._data:
            data_id = self._data['posterior_predictive']['data_id'].values
            data = pd.DataFrame(
                self._data['posterior_predictive']['data'].values[row, col, :], index=data_id
            ).T
            result['data']=data
        return result

    def grand_data_plot(self, var_names: Iterable, plot_map=True):
        plots = []
        cols = 3
        for i, var_id in enumerate(var_names):
            show_legend = True if i == cols - 1 else False
            postpred = self.density_plot(var_id, group='posterior_predictive')
            priopred = self.density_plot(var_id, group='prior_predictive')
            true = self.point_plot(var_id, what_var='data', what_point='true')
            width = 600 if i % cols == cols - 1 else 400
            size_opts = self._size_opts(width=width)
            layers = [postpred * priopred * true]
            if plot_map:
                map = self.point_plot(var_id, what_var='data', what_point='map')
                layers = layers + [map]
            panel = hv.Overlay(layers).opts(
                legend_position='right', show_legend=show_legend, show_grid=True, fontsize=self._FONTSIZES,
                ylabel='', **size_opts
            )
            plots.append(panel)

        return hv.Layout(plots).cols(cols).opts(shared_axes=False)

    def grand_theta_plot(self, var1_id, var2_id, group='posterior', bandwidth=None):
        plots = [
            self._plot_polytope_area(var1_id, var2_id),
            self._data_hull(var1_id=var1_id, var2_id=var2_id, group=group),
            self._bivariate_plot(var1_id=var1_id, var2_id=var2_id, group=group, bandwidth=bandwidth),
        ]
        if group == 'posterior':
            if self._ttdf is not None:
                plots.append(self.point_plot(var1_id=var1_id, var2_id=var2_id, what_point='true'))
            if hasattr(self, '_map'):
                plots.append(self.point_plot(var1_id=var1_id, var2_id=var2_id, what_point='map'))
        return hv.Overlay(plots).opts(legend_position='right', show_legend=True, fontsize=self._FONTSIZES)


class MCMC_PLOT(PlotMonster):
    def __init__(
            self,
            fcm: FluxCoordinateMapper,  # this should be in the sampled basis!
            inference_data: az.InferenceData,
            v_rep: pd.DataFrame = None,
            hv_backend='bokeh',
    ):
        self._max_prob_measure = 'lp'
        super().__init__(fcm, inference_data, v_rep, hv_backend)


class SMC_PLOT(PlotMonster):

    def __init__(
            self,
            fcm: FluxCoordinateMapper,  # this should be in the sampled basis!
            inference_data: az.InferenceData,
            v_rep: pd.DataFrame = None,
            hv_backend='bokeh',
    ):
        self._max_prob_measure = 'dist'
        super().__init__(fcm, inference_data, v_rep, hv_backend)

    def plot_evolution(self, var1_id, var2_id=None, include_prior=True, include_fva=True):
        plots = []

        if var2_id is None:
            xax = self._axes_range(var1_id)
            num_chains = len(self._data['posterior']['chain'].values)
            for i in range(num_chains):
                color = colorcet.glasbey[i]
                if i == 0:
                    color = self._colors['prior']
                    to_plot = self._get_samples(var1_id, group='prior')
                    label = 'prior'
                    muted = False
                else:
                    to_plot = self._get_chain(var1_id, chain_idx=i, group='posterior')
                    label = f'pop_{i}'
                    muted = True
                    if i == num_chains-1:
                        muted = False
                        color = self._colors['posterior']
                        label = 'posterior'
                plots.append(
                    self._plot_distribution(to_plot, var_id=xax, label=label, color=color, muted=muted)
                )
            if include_fva and (var2_id is None):
                fva_min, fva_max = self._axes_range(var1_id, return_dimension=False, tol=0)
                opts = dict(color='#000000', line_dash='dashed')
                if self._hvb:
                    opts['linestyle'] = opts.pop('line_dash')
                plots.extend([
                    hv.VLine(fva_min).opts(**opts), hv.VLine(fva_max).opts(**opts),
                ])

        plots.append(self.point_plot(var1_id))
        return hv.Overlay(plots).opts(legend_position='right', fontsize=self._FONTSIZES, **self._size_opts(width=600), )

    # def _get_samples(self, *args, chain=None, obs_idx=0, group='posterior', num_samples=30000):
    #     if (chain is None) and ('posterior' in group):
    #         chain = -1
    #     print(args)
    #     aa = PlotMonster._get_samples(self, *args, chain=chain, obs_idx=obs_idx, group=group, num_samples=num_samples)
    #     bb = self._get_chain(*args, chain_idx=-1, group=group)
    #     print(bb.shape, aa.shape)


def speed_plot():
    pickle.dump(model._fcm._sampler.basis_polytope, open('pol.p', 'wb'))
    pol = pickle.load(open('pol.p', 'rb'))
    # nc_file = "C:\python_projects\sbmfi\src\sbmfi\inference\e_coli_anton_glc7_prior.nc"
    nc_file = "C:\python_projects\sbmfi\spiro_cdf.nc"
    post = az.from_netcdf(nc_file)

    v_rep = None
    # v_rep = pd.read_excel('v_rep.xlsx', index_col=0)
    pm = PlotMonster(pol, post, v_rep=v_rep)
    pm._v_rep.to_excel('v_rep.xlsx')

    var1_id = 'B_svd2'
    var2_id = 'B_svd3'
    group = 'posterior'

    map = pm._load_max_prob_point()
    measurements = pm._load_observed_data()
    boli = measurements.columns.str.contains('[CD]\+', regex=True)
    plot = pm.grand_data_plot(measurements.columns[boli])
    # hv.save(plot, 'pltts.png')

    # plot = pm.grand_theta_plot(var1_id, var2_id, group='prior')

    # aa = pm.plot_density('D: C+0', group='posterior_predictive', var_names='simulated_data')
    #
    # a = pm._plot_polytope_area(var1_id, var2_id)
    # b = pm._data_hull(var1_id=var1_id, var2_id=var2_id, group=group)
    # c = pm._bivariate_plot(var1_id=var1_id, var2_id=var2_id, group=group)
    # plot = a * b * c
    # if group == 'posterior':
    #     d = pm.point_plot(var1_id=var1_id, var2_id=var2_id, what='map')
    #     e = pm.point_plot(var1_id=var1_id, var2_id=var2_id, what='true_theta')
    #     plot = plot * d * e
    # plot = plot.opts(legend_position='right', show_legend=True)
    # d = pm.density_plot(var1_id)
    # e = pm.density_plot(var1_id, group=group)
    output_file('test.html')
    show(hv.render(plot))
    # show(hv.render(d))


def _load_data(hv_backend = 'matplotlib'):
    v_rep = pd.read_excel(r"C:\python_projects\sbmfi\v_rep.xlsx", index_col=0)
    pol = pickle.load(open(r"C:\python_projects\sbmfi\pol.p", 'rb'))

    mcmc_anton = az.from_netcdf(r"C:\python_projects\sbmfi\MCMC_e_coli_glc_anton_obsmod_wprior.nc")
    pmcmc_anton = MCMC_PLOT(pol, mcmc_anton, v_rep, hv_backend)

    mcmc_tomek = az.from_netcdf(r"C:\python_projects\sbmfi\MCMC_e_coli_glc_tomek_obsmod_CORR.nc")
    pmcmc_tomek = MCMC_PLOT(pol, mcmc_tomek, v_rep, hv_backend)

    smc_tomek = az.from_netcdf(r"C:\python_projects\sbmfi\SMC_e_coli_glc_tomek_obsmod_CORR_winv.nc")
    psmc_tomek = SMC_PLOT(pol, smc_tomek, v_rep, hv_backend)
    return pmcmc_anton, pmcmc_tomek, psmc_tomek

def MARGINAL_PLOT():
    variables = ['B_svd0', 'B_svd1', 'B_svd2', 'B_svd3', 'B_svd4', 'B_svd5', 'B_svd6',
                 'B_svd7', 'B_svd8', 'B_svd9', 'PGI_xch', 'FBA_xch', 'TPI_xch',
                 'GAPD_xch', 'PGK_xch', 'PGM_xch', 'ENO_xch', 'RPE_xch', 'RPI_xch',
                 'TKTa_xch', 'TKTb_xch', 'TKTc_xch', 'TALAa_xch', 'TALAb_xch',
                 'ACONTa_xch', 'ACONTb_xch', 'ICDHyr_xch', 'SUCOAS_xch', 'SUCDi_xch',
                 'FUM_xch', 'MDH_xch', 'PTAr_xch', 'ACKr_xch', 'GHMT2r_xch', 'GLYCL_xch']

    pmcmc_anton, pmcmc_tomek, psmc_tomek = _load_data()

    maxI = 10
    layout = []
    for i in range(maxI):
        varid = variables[i]

        dens_anton = pmcmc_anton.density_plot(var_id=varid)
        dens = dens_anton.data[('Distribution', 'Posterior')]
        dens.opts(color='#ffa500')
        dens_anton.data[('Distribution', 'Posterior')] = dens.relabel('Antoniewicz MCMC')

        dens_mcmc_tomek = pmcmc_tomek.density_plot(var_id=varid)
        dens = dens_mcmc_tomek.data[('Distribution', 'Posterior')]
        dens.opts(color='#b55ef1')
        dens_mcmc_tomek.data[('Distribution', 'Posterior')] = dens.relabel('LCMS MCMC')

        to_plot = psmc_tomek._get_chain(varid, chain_idx=-1)
        dens_smc_tomek = psmc_tomek._plot_distribution(to_plot, var_id=varid, label='LCMS SMC',
                                                       color=psmc_tomek._colors['posterior'])

        prior_dens = psmc_tomek.density_plot(var_id=varid, group='prior')

        true_plot = pmcmc_anton.point_plot(varid, label='Antoniewicz MLE')

        plot = (prior_dens * dens_anton * dens_mcmc_tomek * dens_smc_tomek * true_plot).opts(
            legend_position='right', **psmc_tomek._size_opts(width=600), fontsize=PlotMonster._FONTSIZES
        )
        if i != maxI - 1:
            plot.opts(show_legend=False)
        layout.append(plot)

    plot = hv.Layout(layout).cols(2).opts(hspace=0.1, vspace=0.2, fig_size=150)
    hv.save(plot, 'naks.png', fmt='png', dpi=400)


def PPC_PLOT():

    pmcmc_anton, pmcmc_tomek, psmc_tomek = _load_data()

    anton_ids = pd.Series(pmcmc_anton._data.posterior_predictive.data_id.values)
    tomek_ids = pd.Series(pmcmc_tomek._data.posterior_predictive.data_id.values)
    anton_ids = anton_ids.str.replace('_{M-}', '', regex=False)
    shared = anton_ids.loc[anton_ids.isin(tomek_ids)].values
    varid = shared[1]

    plot = pmcmc_anton.point_plot(varid, what_var='data')

    hv.save(plot, 'noks.png', fmt='png', dpi=400)



if __name__ == "__main__":
    from sbmfi.models.small_models import spiro
    from sbmfi.models.build_models import build_e_coli_anton_glc, _bmid_ANTON
    from bokeh.plotting import figure, output_file, save
    import matplotlib
    import pickle

    file = "C:\python_projects\sbmfi\mog_polytope_50k_samples.nc"
    data = az.from_netcdf(file)
    fcm = pickle.load(open(r"C:\python_projects\sbmfi\fcm.p", 'rb'))

    plotter = MCMC_PLOT(fcm, data)
    plott = plotter.grand_theta_plot('R_h_out', 'R_bm')

    output_file(filename="custom_filename1.html", title="plot1")
    aaa = show(hv.render(plott), new='tab')

    # model, kwargs = spiro(
    #     backend='torch',
    #     auto_diff=False,
    #     batch_size=1,
    #     add_biomass=True,
    #     v2_reversible=True,
    #     ratios=True,
    #     build_simulator=True,
    #     add_cofactors=True,
    #     which_measurements=None,
    #     seed=2,
    #     measured_boundary_fluxes=('h_out',),
    #     which_labellings=['A', 'B'],
    #     include_bom=True,
    #     v5_reversible=False,
    #     n_obs=0,
    #     kernel_id='svd',
    #     L_12_omega=1.0,
    #     clip_min=None,
    #     transformation='ilr',
    # )
    # data = az.InferenceData.from_netcdf(file)
    # fcm = model.flux_coordinate_mapper
    # smc_plot = SMC_PLOT(
    #     fcm=fcm,  # this should be in the sampled basis!
    #     inference_data=data,
    #     v_rep=None,
    #     hv_backend='bokeh',
    # )


    # plott = smc_plot.plot_evolution('R_svd0',)
    # plott = smc_plot.grand_data_plot(['A: ilr_C_0', 'A: ilr_C_1', 'A: ilr_D_0', 'A: ilr_D_1', 'A: ilr_H_0',
    #    'A: ilr_L_0', 'A: ilr_L_1', 'A: ilr_L_2', 'A: ilr_L|[1,2]_0',
    #    'B: ilr_C_0', 'B: ilr_D_0', 'B: ilr_H_{M+Cl}_0', 'B: ilr_H_0',
    #    'B: ilr_L|[1,2]_0', 'BOM: h_out', 'BOM: bm'], plot_map=False)


    # data_id = ['A: ilr_C_0', 'A: ilr_C_1', 'A: ilr_D_0', 'A: ilr_D_1', 'A: ilr_H_0',
    #    'A: ilr_L_0', 'A: ilr_L_1', 'A: ilr_L_2', 'A: ilr_L|[1,2]_0',
    #    'B: ilr_C_0', 'B: ilr_D_0', 'B: ilr_H_{M+Cl}_0', 'B: ilr_H_0',
    #    'B: ilr_L|[1,2]_0', 'BOM: h_out', 'BOM: bm']
    #
    # ding = smc_plot.grand_data_plot(data_id, plot_map=True)
    #
    # # ding = smc_plot.plot_evolution('R_svd0', 'R_svd1')
    # output_file(filename="custom_filename.html", title="plot1")
    # show(hv.render(ding))

    # pol = pickle.load(open(r"C:\python_projects\sbmfi\pol.p", 'rb'))
    # v_rep = pd.read_excel(
    #     'C:\python_projects\sbmfi\src\sbmfi\inference\VREP_MCMC_e_coli_glc_anton_obsmod_copy_NEWWP.xlsx')
    # smc_res = az.from_netcdf("C:/python_projects/sbmfi/SMC_e_coli_glc_tomek_obsmod_copy_NEW.nc")
    # smc_res = smc_res.sel(draw=slice(5000, None))
    #
    # smp = SMC_PLOT(pol, smc_res, v_rep=v_rep, hv_backend='bokeh')
    #
    # ding = smp.plot_evolution('R_svd0')
    # output_file(filename="custom_filename.html", title="plot1")
    # show(hv.render(ding))

    # var1 = ['[1]Glc: ilr_val__L_c_0', '[1]Glc: ilr_val__L_c_1', '20% [U]Glc: ilr_2pg_c_0']
    #
    # data = smp._get_samples(var1, group='posterior_predictive')
    # theta = smp._get_samples('R_svd0', group='posterior')
    # smp.density_plot(varr, group='posterior_predictive', tol=20, label='konker')


    # model, kwargs = build_e_coli_anton_glc(
    #     backend='torch',
    #     auto_diff=False,
    #     build_simulator=True,
    #     ratios=False,
    #     batch_size=25,
    #     which_measurements=None,
    #     which_labellings=['20% [U]Glc', '[1]Glc'],
    #     measured_boundary_fluxes=[_bmid_ANTON, 'EX_glc__D_e', 'EX_ac_e'],
    #     seed=1,
    # )
    # print(model._fcm.theta_id)

    # var = 'FUM_xch'
    # v_rep = pd.read_excel('VREP_MCMC_e_coli_glc_anton_obsmod_copy_NEWWP.xlsx')
    # print(v_rep.shape)
    # spres = az.from_netcdf("C:\python_projects\sbmfi\MCMC_e_coli_glc_anton_obsmod_copy_NEWWP.nc")
    # mc = MCMC_PLOT(model._fcm._sampler._F_round, inference_data=spres, v_rep=v_rep)
    # aa = mc.density_plot(var, group='posterior')
    # bb = mc.density_plot(var, group='prior')
    # output_file(filename="custom_filename.html", title="plot1")
    # show(hv.render(aa * bb))
    #
    # spres = az.from_netcdf("C:\python_projects\sbmfi\SMC_e_coli_glc_tomek_obsmod_copy_NEW.nc")
    # mc = SMC_PLOT(model._fcm._sampler._F_round, inference_data=spres, v_rep=v_rep)
    # # mc._v_rep.to_excel('VREP_MCMC_e_coli_glc_anton_obsmod_copy_NEWWP.xlsx', index=False)
    # aa = mc.density_plot(var, group='posterior')
    # bb = mc.density_plot(var, group='prior')
    # output_file(filename="custom_filename.html", title="plot2")
    # show(hv.render(aa * bb))

    # model, kwargs = spiro(
    #     seed=None,
    #     batch_size=100,
    #     backend='torch', v2_reversible=True, ratios=False, build_simulator=True,
    #     which_measurements='com', which_labellings=['A'], v5_reversible=True
    # )
    # model, kwargs = spiro(
    #     seed=None,
    #     batch_size=10,
    #     backend='torch', v2_reversible=True, ratios=False, build_simulator=True, which_labellings=['A', 'B'],
    #     v5_reversible=True
    # )
    # v_rep = pd.read_excel('vrep.xlsx')
    # var = 'R_svd3'
    #
    # spres = az.from_netcdf("C:\python_projects\sbmfi\src\sbmfi\inference\spiro_TEST_SMC.nc")
    # mc = SMC_PLOT(model._fcm.make_theta_polytope(), inference_data=spres, v_rep=v_rep)
    # # R_svd0    R_svd1    R_svd2    R_svd3  v2_xch  v5_xch
    # aa = mc.plot_evolution(var)
    # output_file(filename="custom_filename.html", title="plot2")
    # show(hv.render(aa))
    # # mc._v_rep.to_excel('vrep.xlsx', index=False)
    # #
    # spres = az.from_netcdf("C:\python_projects\sbmfi\src\sbmfi\inference\spiro_TEST_MCMC.nc")
    # spres = spres.sel(draw=slice(1000, None))
    # mc = MCMC_PLOT(model._fcm.make_theta_polytope(), inference_data=spres, v_rep=v_rep)
    # aa = mc.density_plot(var)
    # output_file(filename="custom_filename.html", title="plot1")
    # show(hv.render(aa))

