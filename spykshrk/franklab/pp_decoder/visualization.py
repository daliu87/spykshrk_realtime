
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from spykshrk.franklab.pp_decoder.data_containers import pos_col_format


class DecodeVisualizer:

    @staticmethod
    def plot_decode_2d(posteriors, plt_range, num_pos_bin, x_tick=1.0):
        post_plot = posteriors.data.query('timestamp > {} and timestamp < {}'.
                                          format(*plt_range)).loc[:, pos_col_format(0, num_pos_bin):
                                                                  pos_col_format(num_pos_bin-1, num_pos_bin)]

        plt.imshow(post_plot)



    def plot_decode(posteriors , stim_lockout_ranges, linpos_flat, plt_range, x_tick=1.0):
        stim_lockout_ranges_sec = stim_lockout_ranges/30000
        stim_lockout_range_sec_sub = stim_lockout_ranges_sec[(stim_lockout_ranges_sec[1] > plt_range[0]) &
                                                             (stim_lockout_ranges_sec[0] < plt_range[1])]

        plt.imshow(dec_est[(dec_bin_times > plt_range[0]*30000) & (dec_bin_times < plt_range[1]*30000)].transpose(),
                   extent=[plt_range[0], plt_range[1], 0, 450], origin='lower', aspect='auto', cmap='hot', zorder=0)

        plt.colorbar()

        # Plot linear position
        if isinstance(linpos_flat.index, pd.MultiIndex):
            linpos_index_s = linpos_flat.index.get_level_values('timestamp') / 30000
        else:
            linpos_index_s = linpos_flat.index / 30000
        index_mask = (linpos_index_s > plt_range[0]) & (linpos_index_s < plt_range[1])

        plt.plot(linpos_index_s[index_mask],
                 linpos_flat.values[index_mask], 'c.', zorder=1, markersize=5)

        plt.plot(stim_lockout_range_sec_sub.values.transpose(), np.tile([[440], [440]], [1, len(stim_lockout_range_sec_sub)]), 'c-*' )

        for stim_lockout in stim_lockout_range_sec_sub.values:
            plt.axvspan(stim_lockout[0], stim_lockout[1], facecolor='#AAAAAA', alpha=0.3)

        plt.plot(plt_range, [74, 74], '--', color='gray')
        plt.plot(plt_range, [148, 148], '--', color='gray')
        plt.plot(plt_range, [256, 256], '--', color='gray')
        plt.plot(plt_range, [298, 298], '--', color='gray')
        plt.plot(plt_range, [407, 407], '--', color='gray')

        plt.xticks(np.arange(plt_range[0], plt_range[1], x_tick))

