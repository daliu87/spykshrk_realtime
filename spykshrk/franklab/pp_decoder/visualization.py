
import matplotlib.pyplot as plt
from matplotlib import patches
import functools
import pandas as pd
import numpy as np
import holoviews as hv

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
        self.enc_settings = enc_settings
        self.relative = relative

    def plot_decode_image(self, plt_range=None, x_tick=1.0):

        if plt_range is None:
            plt_range = self.posteriors.get_time_range()
        else:
            if plt_range[0] == None:
                plt_range[0] = self.posteriors.get_time_start()
            if plt_range[1] == None:
                plt_range[1] = self.posteriors.get_time_end()

        post_plot = self.posteriors.query('time > {} and time < {}'.
                                          format(*plt_range)).loc[:, pos_col_format(0, self.enc_settings.pos_num_bins):
                                                                  pos_col_format(self.enc_settings.pos_num_bins-1,
                                                                  self.enc_settings.pos_num_bins)]

        ax = plt.imshow(post_plot.values.T, extent=[plt_range[0], plt_range[1], 0, self.enc_settings.pos_bins[-1]],
                        origin='lower', aspect='auto', cmap='hot', zorder=0, vmax=0.3)

        plt.xticks(np.arange(plt_range[0], plt_range[1], x_tick))

        plt.colorbar()
        ax.axes.set_facecolor('black')

        plt.xlabel('seconds')
        plt.ylabel('1D pos map')

        return ax

    def plot_arm_boundaries(self, plt_range):
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[0][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[0][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[1][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[1][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[2][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [self.enc_settings.arm_coordinates[2][1]]*2, '--', color='0.4', zorder=1)

        plt.ylim([self.enc_settings.arm_coordinates[0][0] - 5, self.enc_settings.arm_coordinates[2][1] + 15])

    def plot_linear_pos(self, plt_range=None, alpha=0.5):

        if self.linpos is None:
            raise MissingDataError('Missing linpos data, never set.')
        if plt_range is None:
            plt_range = self.posteriors.get_time_range()

        linpos_sing = self.linpos.get_mapped_single_axis()

        self.linpos_sel = linpos_sing.loc[:, ['linpos_flat']].query('time > {} and time < {}'.format(*plt_range))
        ax = plt.plot(self.linpos_sel.index.get_level_values('time'), self.linpos_sel.values, 'co', zorder=2, markersize=6,
                      alpha=alpha)

        return ax

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
        for arm in self.arms:

            arm_table = error_table.loc[:, idx[arm, ['real_pos', 'plt_error_up', 'plt_error_down']]]
            arm_table.columns = arm_table.columns.droplevel(0)
            arm_table = arm_table.reset_index(level=['time']).reindex(columns=['time','real_pos', 'plt_error_up',
                                                                               'plt_error_down'])

            self.arm_error_tables[arm] = hv.Table(arm_table.dropna(), kdims='time',
                                                  vdims=['real_pos', 'plt_error_up', 'plt_error_down'])

    def plot_arms_error(self, start_time=None, interval=None):

        error_plots = {}
        real_pos_plots = {}

        for arm, arm_table in self.arm_error_tables.items():

            if not (start_time is None or interval is None):
                arm_table = arm_table[start_time: start_time+interval]

            error_plots[arm] = hv.ErrorBars(arm_table, kdims='time',
                                            vdims=['real_pos', 'plt_error_up', 'plt_error_down'])
            real_pos_plots[arm] = arm_table.to.points(kdims=['time', 'real_pos'], vdims=['real_pos'])

            error_plots[arm].opts(plot={'apply_extent': (start_time, None, start_time+interval, None)})
            real_pos_plots[arm].opts(plot={'apply_extent': (start_time, None, start_time+interval, None)})

        errorbar_overlay = hv.NdOverlay(error_plots)
        errorbar_overlay.opts(plot={'apply_extent': (start_time, None, start_time+interval, None)})

        overlay = hv.NdOverlay(real_pos_plots, kdims=['arm'])

        overlay.opts(plot={'apply_extent': (start_time, None, start_time+interval, None)})
        plot = overlay * errorbar_overlay
        plot.opts(plot={'apply_extent': (start_time, None, start_time+interval, None)})
        return plot

    def plot_arms_error_dmap(self, slide_interval, plot_interval=None):

        if plot_interval is None:
            plot_interval = slide_interval

        dmap = hv.DynamicMap(functools.partial(self.plot_arms_error, interval=plot_interval),
                             kdims=[hv.Dimension('start_time',
                                                 values=np.arange(self.error_table.index.get_level_values('time')[0],
                                                                  self.error_table.index.get_level_values('time')[-1],
                                                                  slide_interval))])

        return dmap
