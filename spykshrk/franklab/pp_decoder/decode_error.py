import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from functools import partial


def conv_center_pos(pos, arm_coord):
    """
    Compute distance from center arm
    """
    if pos < arm_coord[0][1]:
        return pos
    elif (pos >= arm_coord[1][0]) and (pos < arm_coord[1][1]):
        return arm_coord[1][1]-pos+arm_coord[0][1]
    elif (pos >= arm_coord[2][0]) and (pos < arm_coord[2][1]):
        return arm_coord[2][1]-pos+arm_coord[0][1]
    else:
        raise ValueError("Position {} not valid for arm coordinate.".format(pos))


def conv_left_pos(pos, arm_coord):
    if pos < arm_coord[0][1]:
        return arm_coord[0][1]-pos+arm_coord[1][1]-arm_coord[1][0]
    elif (pos >= arm_coord[1][0]) and (pos < arm_coord[1][1]):
        return pos-arm_coord[1][0]
    elif (pos >= arm_coord[2][0]) and (pos < arm_coord[2][1]):
        return arm_coord[2][1]-pos+arm_coord[1][1]-arm_coord[1][0]
    else:
        raise ValueError("Position {} not valid for arm coordinate.".format(pos))


def conv_right_pos(pos, arm_coord):
    if pos < arm_coord[0][1]:
        return arm_coord[0][1]-pos+arm_coord[2][1]-arm_coord[2][0]
    elif (pos >= arm_coord[1][0]) and (pos < arm_coord[1][1]):
        return arm_coord[1][1]-pos+arm_coord[2][1]-arm_coord[2][0]
    elif (pos >= arm_coord[2][0]) and (pos < arm_coord[2][1]):
        return pos-arm_coord[2][0]
    else:
        raise ValueError("Position {} not valid for arm coordinate.".format(pos))


def bin_pos_data(pos_data, bin_size):
    pos_bin_ids = np.floor((pos_data.index - pos_data.index[0])/bin_size).astype('int')
    pos_data['bin'] = pos_bin_ids
    pos_bin_ids_unique = np.unique(pos_bin_ids)

    start_bin_time = np.floor(pos_data.index[0] / bin_size) * bin_size

    pos_bin_times = (pos_bin_ids_unique * bin_size + start_bin_time)

    pos_data_bins = pd.DataFrame()

    for ind, bin_id in enumerate(pos_bin_ids_unique):
        pos_in_bin = pos_data[pos_data['bin'] == bin_id]
        pos_bin_mean = pos_in_bin.mean()
        pos_bin_mean.name = pos_bin_times[ind]

        pos_data_bins = pos_data_bins.append(pos_bin_mean)

    return pos_data_bins


def conv_arm_pos(arm_dec_est, arm_coordinates, conv_func):
    new_arm_dec_est = pd.DataFrame()
    new_arm_dec_est.loc[:, 'est_pos'] = arm_dec_est['est_pos'].map(partial(conv_func,
                                                                           arm_coord=arm_coordinates))
    new_arm_dec_est.loc[:, 'real_pos'] = arm_dec_est['well_center']

    return new_arm_dec_est


def calc_error_for_plot(dec_est):
    dec_est.loc[:, 'error'] = dec_est['real_pos'] - dec_est['est_pos']
    dec_est.loc[:, 'abs_error'] = np.abs(dec_est['error'])
    dec_est.loc[:, 'plt_error_up'] = dec_est['error']
    dec_est.loc[dec_est['error'] < 0, 'plt_error_up'] = 0
    dec_est.loc[:, 'plt_error_down'] = dec_est['error']
    dec_est.loc[dec_est['error'] > 0, 'plt_error_down'] = 0
    dec_est.loc[:, 'plt_error_down'] = np.abs(dec_est['plt_error_down'])

    return dec_est


def calc_error_table(dec_est_and_linpos, arm_coordinates, vel_thresh):

    # Select rows only when velocity meets criterion
    dec_est_and_linpos = dec_est_and_linpos[np.abs(dec_est_and_linpos['lin_vel_center']) >= vel_thresh]

    # Separate out each arm's position
    center_dec_est_merged = dec_est_and_linpos[dec_est_and_linpos['seg_idx'] == 1]
    left_dec_est_merged = dec_est_and_linpos[(dec_est_and_linpos['seg_idx'] == 2) |
                                             (dec_est_and_linpos['seg_idx'] == 3)]
    right_dec_est_merged = dec_est_and_linpos[(dec_est_and_linpos['seg_idx'] == 4) |
                                              (dec_est_and_linpos['seg_idx'] == 5)]

    # Apply "closest well centric" tranform to each arm's data
    center_dec_est = conv_arm_pos(center_dec_est_merged, arm_coordinates, conv_center_pos)

    left_dec_est = conv_arm_pos(left_dec_est_merged, arm_coordinates, conv_left_pos)

    right_dec_est = conv_arm_pos(right_dec_est_merged, arm_coordinates, conv_right_pos)

    center_dec_est = calc_error_for_plot(center_dec_est)

    left_dec_est = calc_error_for_plot(left_dec_est)

    right_dec_est = calc_error_for_plot(right_dec_est)

    return center_dec_est, left_dec_est, right_dec_est


