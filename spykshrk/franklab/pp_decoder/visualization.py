
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
import numpy as np

from spykshrk.franklab.pp_decoder.data_containers import pos_col_format, Posteriors, LinearPosition, \
    EncodeSettings, StimLockout


class DecodeVisualizer:

    @staticmethod
    def plot_decode_image(posteriors: Posteriors, plt_range, enc_settings: EncodeSettings, x_tick=1.0):

        post_plot = posteriors.query('time > {} and time < {}'.
                                     format(*plt_range)).loc[:, pos_col_format(0, enc_settings.pos_num_bins):
                                                             pos_col_format(enc_settings.pos_num_bins-1,
                                                             enc_settings.pos_num_bins)]

        ax = plt.imshow(post_plot.values.T, extent=[plt_range[0], plt_range[1], 0, enc_settings.pos_num_bins],
                        origin='lower', aspect='auto', cmap='hot', zorder=0, vmax=0.3)

        plt.xticks(np.arange(plt_range[0], plt_range[1], x_tick))

        plt.colorbar()
        ax.axes.set_facecolor('black')

        plt.xlabel('seconds')
        plt.ylabel('1D pos map')

        return ax

    @staticmethod
    def plot_arm_boundaries(plt_range, enc_settings: EncodeSettings):
        plt.plot(plt_range, [enc_settings.arm_coordinates[0][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[0][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[1][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[1][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[2][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[2][1]]*2, '--', color='0.4', zorder=1)

        plt.ylim([enc_settings.arm_coordinates[0][0] - 5, enc_settings.arm_coordinates[2][1] + 15])

    @staticmethod
    def plot_linear_pos(linpos: LinearPosition, plt_range):
        linpos_sing = linpos.get_mapped_single_axis()

        linpos_sel = linpos_sing.loc[:, ['linpos_flat']].query('time > {} and time < {}'.format(*plt_range))
        ax = plt.plot(linpos_sel.index.get_level_values('time'), linpos_sel.values, 'co', zorder=2, markersize=6)

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

    @staticmethod
    def plot_arms_error(dec_error, plt_range=None):

        plt.xlabel('seconds')
        plt.ylabel("distance from arm's well")
        plt.legend(['center arm', 'left arm', 'right arm'])
