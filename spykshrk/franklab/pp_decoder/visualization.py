
import matplotlib.pyplot as plt
from matplotlib import patches
import functools
import pandas as pd
import numpy as np
import holoviews as hv
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

    def plot_decode_image(self, time, plt_range=10, lookahead=5, lookbehind=5):

        behind_time = max(time-lookbehind, self.posteriors.index.get_level_values('time')[0])
        img_sel = self.posteriors.get_distribution_view().query('time > {} and time < {}'.
                                                                format(behind_time,
                                                                       time+plt_range+lookahead)).values.T
        img_sel = np.flip(img_sel, axis=0)
        img = hv.Image(img_sel, bounds=(behind_time, 0, time+plt_range+lookahead, self.enc_settings.pos_bins[-1]),
                       kdims=['time (sec)', 'linpos (cm)'], vdims=['probability'],
                       extents=(time, None, time + plt_range, None))

        img = img.redim(probability={'range': (0, 0.5)})

        return img

    def plot_arm_boundaries(self, plt_range):
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[0][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[0][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[1][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[1][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[2][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[2][1]]*2, '--', color='0.4', zorder=1)

        plt.ylim([self.enc_settings.arm_coordinates[0][0] - 5, self.enc_settings.arm_coordinates[2][1] + 15])

    def plot_linear_pos(self, time, plt_range=10, lookahead=5, lookbehind=5):
        behind_time = max(time-lookbehind, self.posteriors.index.get_level_values('time')[0])
        linflat_sel = self.linflat.query('time > {} and time < {}'.
                                         format(behind_time, time+plt_range+lookahead))['linpos_flat']
        linflat_sel_data = linflat_sel.values
        linflat_sel_time = linflat_sel.index.get_level_values('time')
        pos = hv.Points((linflat_sel_time, linflat_sel_data), kdims=['sec', 'linpos'],
                        extents=(time, None, time + plt_range, None))
        return pos

    def plot_all(self, time, plt_range=10, lookahead=5, lookbehind=5):

        img = self.plot_decode_image(time, plt_range, lookahead, lookbehind)
        pos = self.plot_linear_pos(time, plt_range, lookahead, lookbehind)

        return img * pos

    def plot_all_dynamic(self, slide=10, plt_range=10, lookahead=5, lookbehind=5):
        dmap = hv.DynamicMap(functools.partial(self.plot_all, plt_range=plt_range, lookahead=lookahead,
                                               lookbehind=lookbehind),
                             kdims=hv.Dimension('time',
                                                values=np.arange(self.posteriors.index.get_level_values('time')[0],
                                                                 self.posteriors.index.get_level_values('time')[-1],
                                                                 slide)))
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
                arm_table = arm_table[start_time: start_time+interval]

            if len(arm_table) == 0:
                # dummy table
                arm_table = hv.Table([[start_time,self.max_pos,0,0]], kdims=['time'],
                                     vdims=['real_pos','plt_error_down','plt_error_up'])
            error_plots[arm] = hv.ErrorBars(arm_table, kdims='time',
                                            vdims=['real_pos', 'plt_error_down', 'plt_error_up'],
                                            extents=(start_time, self.min_pos, start_time+interval, self.max_pos))
            #error_plots[arm].redim(time={'range': (start_time, start_time+interval)})
            #real_pos_plots[arm] = arm_table.to.points(kdims=['time', 'real_pos'], vdims=['real_pos'],
            #                                          extents=(start_time, self.min_pos,
            #                                                   start_time+interval, self.max_pos))
            #joint_pos_plots[arm] = real_pos_plots[arm] * error_plots[arm]


        errorbar_overlay = hv.NdOverlay(error_plots, kdims=['arm'])



        return errorbar_overlay
        #return overlay
        #return errorbar_overlay

    def plot_arms_error_dmap(self, slide_interval, plot_interval=None):

        if plot_interval is None:
            plot_interval = slide_interval

        dmap = hv.DynamicMap(functools.partial(self.plot_arms_error, interval=plot_interval),
                             kdims=[hv.Dimension('start_time',
                                                 values=np.arange(self.error_table.index.get_level_values('time')[0],
                                                                  self.error_table.index.get_level_values('time')[-1],
                                                                  slide_interval))])

        return dmap
