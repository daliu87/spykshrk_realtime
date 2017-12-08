
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from spykshrk.franklab.pp_decoder.data_containers import pos_col_format, Posteriors, LinearPositionContainer, \
    EncodeSettings, StimLockout


class DecodeVisualizer:

    @staticmethod
    def plot_decode_image(posteriors: Posteriors, plt_range, enc_settings: EncodeSettings, x_tick=1.0):

        post_plot = posteriors.data.query('time > {} and time < {}'.
                                          format(*plt_range)).loc[:, pos_col_format(0, enc_settings.pos_num_bins):
                                                                  pos_col_format(enc_settings.pos_num_bins-1,
                                                                                 enc_settings.pos_num_bins)]

        ax = plt.imshow(post_plot.T, extent=[plt_range[0], plt_range[1], 0, enc_settings.pos_num_bins],
                        origin='lower', aspect='auto', cmap='hot', zorder=0)

        plt.plot(plt_range, [enc_settings.arm_coordinates[0][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[0][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[1][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[1][1]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[2][0]]*2, '--', color='0.4', zorder=1)
        plt.plot(plt_range, [enc_settings.arm_coordinates[2][1]]*2, '--', color='0.4', zorder=1)

        plt.xticks(np.arange(plt_range[0], plt_range[1], x_tick))

        plt.ylim([enc_settings.arm_coordinates[0][0] - 5, enc_settings.arm_coordinates[2][1] + 15])

        ax.axes.set_facecolor('black')

        return ax

    @staticmethod
    def plot_linear_pos(linpos: LinearPositionContainer, plt_range):
        linpos_sing = linpos.get_mapped_single_axis()
        linpos_sel = linpos_sing[(linpos_sing.index.get_level_values('time') > plt_range[0]) &
                                 (linpos_sing.index.get_level_values('time') < plt_range[1])]
        ax = plt.plot(linpos_sel.index.get_level_values('time'), linpos_sel.values, 'co', zorder=2, markersize=4)

        return ax


    @staticmethod
    def plot_stim_lockout(stim_lock: StimLockout, plt_range, plt_location):

        stim_lock_sel = stim_lock.get_range_sec(*plt_range)

        ax = plt.plot(stim_lock_sel['time'].values.transpose(), np.tile([[plt_location], [plt_location]],
                                                                        [1, len(stim_lock_sel)]), 'c-*',
                      linewidth=3, markersize=10)

        return ax


