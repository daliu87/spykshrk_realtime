
import matplotlib.pyplot as plt
from matplotlib import patches
import functools
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews.operation.datashader import datashade, aggregate, shade, regrid, dynspread
from holoviews.operation import decimate
import datashader as ds

import functools

idx = pd.IndexSlice

class MissingDataError(RuntimeError):
    pass


from spykshrk.franklab.pp_decoder.data_containers import pos_col_format, Posteriors, LinearPosition, \
    EncodeSettings, StimLockout


class DecodeVisualizer:

    def __init__(self, posteriors: Posteriors, enc_settings: EncodeSettings,
                 linpos: LinearPosition=None, relative=False):
        self.posteriors = posteriors
        self.linpos = linpos
        self.linflat = linpos.get_mapped_single_axis()
        self.enc_settings = enc_settings
        self.relative = relative

    def plot_decode_image(self, time, x_range=None, y_range=None, plt_range=10):

        sel_range = [time, time+plt_range]

        img_sel = self.posteriors.get_distribution_view().values.T
        img_sel = np.flip(img_sel, axis=0)
        img = hv.Image(img_sel, bounds=(self.posteriors.index.get_level_values('time')[0], 0,
                                        self.posteriors.index.get_level_values('time')[-1],
                                        self.enc_settings.pos_bins[-1]),
                       kdims=['time (sec)', 'linpos (cm)'], vdims=['probability'],)

        img = img.redim(probability={'range': (0, 0.3)})
        img.extents = (sel_range[0], 0, sel_range[1], self.enc_settings.pos_bins[-1])

        if (x_range is None):
            x_range = [time, time+plt_range]

        if (y_range is None):
            y_range = [0, self.enc_settings.pos_bins[-1]]

        rgb = shade(regrid(img, aggregator='mean', dynamic=False,
                           x_range=x_range, y_range=y_range),
                    cmap=plt.get_cmap('hot'), normalization='linear', dynamic=False)

        rgb.extents = (sel_range[0], 0, sel_range[1], self.enc_settings.pos_bins[-1])
        return rgb

    def plot_arm_boundaries(self, plt_range):
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[0][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[0][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[1][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[1][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[2][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[2][1]]*2, '--', color='0.4', zorder=1)

        plt.ylim([self.enc_settings.arm_coordinates[0][0] - 5, self.enc_settings.arm_coordinates[2][1] + 15])

    def plot_linear_pos(self, time, x_range=None, y_range=None, plt_range=10):
        linflat_sel_data = self.linflat['linpos_flat'].values
        linflat_sel_time = self.linflat.index.get_level_values('time')
        pos = hv.Points((linflat_sel_time, linflat_sel_data), kdims=['sec', 'linpos'],
                        extents=(time, None, time + plt_range, None))

        return pos

    def plot_shade_linear_pos(self, time, x_range=None, y_range=None, plt_range=10):
        linflat_sel_data = self.linflat['linpos_flat'].values
        linflat_sel_time = self.linflat.index.get_level_values('time')
        pos = hv.Points((linflat_sel_time, linflat_sel_data), kdims=['sec', 'linpos'],
                        extents=(time, None, time + plt_range, None))

        if x_range is None:
            x_range = (time, time+plt_range)
        if y_range is None:
            y_range = (0, self.enc_settings.pos_bins[-1])
        rgb = dynspread(datashade(pos, dynamic=False, x_range=x_range, y_range=y_range,
                                  agg=ds.reductions.mean, cmap=['#8888FF']), max_px=1, threshold=0.3)
        rgb.extents = (time, 0, time+plt_range, self.enc_settings.pos_bins[-1])
        return rgb

    def plot_all(self, time=None, x_range=None, y_range=None, plt_range=10):

        img = self.plot_decode_image(time, x_range, y_range, plt_range)
        pos = self.plot_linear_pos(time, x_range, y_range, plt_range)

        return img * pos

    def plot_all_dynamic(self, stream, slide=10, plt_range=10):
        dmap = hv.DynamicMap(functools.partial(self.plot_all, plt_range=plt_range),
                             kdims=hv.Dimension('time',
                                                values=np.arange(self.posteriors.index.get_level_values('time')[0],
                                                                 self.posteriors.index.get_level_values('time')[-1],
                                                                 slide)),
                             streams=[stream])
        return dmap

    @staticmethod
    def plot_stim_lockout(ax, stim_lock: StimLockout, plt_range, plt_height):

        stim_lock_sel = stim_lock.get_range_sec(*plt_range)

        for stim_item in stim_lock_sel['time'].itertuples():
            pat = patches.Rectangle(xy=(stim_item.on, 0), width=(stim_item.off - stim_item.on), height=plt_height,
                                    fill=True, alpha=0.5, color='gray', zorder=3)

            ax.add_patch(pat)

        return ax


class DecodeErrorVisualizer:

    arms = ['center', 'left', 'right']

    def __init__(self, error_table):
        self.error_table = error_table

        self.arm_error_tables = {}

        maxlist = []
        minlist = []

        for arm in self.arms:

            arm_table = error_table.loc[:, idx[arm, ['real_pos', 'plt_error_up', 'plt_error_down']]]
            arm_table.columns = arm_table.columns.droplevel(0)
            arm_table = arm_table.reset_index(level=['time']).reindex(columns=['time','real_pos', 'plt_error_up',
                                                                               'plt_error_down'])

            self.arm_error_tables[arm] = hv.Table(arm_table.dropna(), kdims='time',
                                                  vdims=['real_pos', 'plt_error_down', 'plt_error_up'])

            maxlist.append(max(self.arm_error_tables[arm]['real_pos'] + self.arm_error_tables[arm]['plt_error_up']))
            minlist.append(min(self.arm_error_tables[arm]['real_pos'] + self.arm_error_tables[arm]['plt_error_down']))

        self.min_pos = min(minlist)
        self.max_pos = max(maxlist)

    def plot_arms_error(self, start_time=None, interval=None, lookahead=0, lookbehind=0):

        error_plots = {}
        real_pos_plots = {}
        joint_pos_plots = {}

        for arm, arm_table in self.arm_error_tables.items():

            if not (start_time is None or interval is None):
                #arm_table = arm_table[start_time: start_time+interval]
                pass

            if len(arm_table) == 0:
                # dummy table
                arm_table = hv.Table([[start_time,self.max_pos,0,0]], kdims=['time'],
                                     vdims=['real_pos','plt_error_down','plt_error_up'])
            error_plots[arm] = hv.ErrorBars(arm_table, kdims='time',
                                            vdims=['real_pos', 'plt_error_down', 'plt_error_up'],
                                            extents=(start_time, self.min_pos, start_time+interval, self.max_pos))
            error_plots[arm].redim(time={'range': (start_time, start_time+interval)})
            real_pos_plots[arm] = arm_table.to.points(kdims=['time', 'real_pos'], vdims=['real_pos'],
                                                      extents=(start_time, self.min_pos,
                                                               start_time+interval, self.max_pos))
            #joint_pos_plots[arm] = real_pos_plots[arm] * error_plots[arm]

        errorbar_overlay = hv.NdOverlay(error_plots, kdims='arm')
        errorbars = hv.NdOverlay(error_plots, kdims='arm')
        points = hv.NdOverlay(real_pos_plots, kdims='arm')

        return errorbars * points

    def plot_arms_error_dmap(self, slide_interval, plot_interval=None):

        if plot_interval is None:
            plot_interval = slide_interval

        dmap = hv.DynamicMap(functools.partial(self.plot_arms_error, interval=plot_interval),
                             kdims=[hv.Dimension('start_time',
                                                 values=np.arange(self.error_table.index.get_level_values('time')[0],
                                                                  self.error_table.index.get_level_values('time')[-1],
                                                                  slide_interval))])

        return dmap
