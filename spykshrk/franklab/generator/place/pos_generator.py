import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

from spykshrk.franklab.data_containers import FlatLinearPosition
from spykshrk.franklab.wtrack import WtrackArm, Direction


class WtrackPosConstSimulator:

    def __init__(self, trial_time, num_series, encode_settings):
        self.trial_time = trial_time
        self.num_series = num_series

        pos_series, pos_series_enum, pos_series_dir_enum = \
            WtrackPosConstSimulator.simulate_series_const_speed(encode_settings.wtrack_arm_coordinates,
                                                                trial_time, encode_settings.pos_sampling_rate)

        pos_series_timestamp = np.linspace(0, 4 * trial_time * encode_settings.sampling_rate,
                                           4 * trial_time * encode_settings.pos_sampling_rate,
                                           endpoint=False).astype('int')

        pos_series_time = np.arange(0, pos_series.size / encode_settings.sampling_rate,
                                    encode_settings.pos_sampling_rate / encode_settings.sampling_rate)

        pos_series_vel = np.ones(len(pos_series)) * ((pos_series[1] - pos_series[0]) /
                                                     (pos_series_time[1] - pos_series_time[0]))

        pos_epoch = np.array([])
        pos_timestamp = np.array([])
        pos_time = np.array([])
        pos_vel = np.array([])
        pos_enum = np.array([])
        pos_dir_enum = np.array([])

        # Duplicate
        num_dup = num_series - 1
        for ii in range(num_series):
            pos_epoch = np.append(pos_epoch, pos_series)
            pos_timestamp = np.append(pos_timestamp, pos_series_timestamp + ii * 4 * trial_time *
                                      encode_settings.sampling_rate)
            #pos_time = np.append(pos_time, pos_series_time + ii * trial_time)
            pos_vel = np.append(pos_vel, pos_series_vel)
            pos_enum = np.append(pos_enum, pos_series_enum)
            pos_dir_enum = np.append(pos_dir_enum, pos_series_dir_enum)

        self.linpos_flat = FlatLinearPosition.from_numpy_single_epoch(1, 1, pos_timestamp, pos_epoch, pos_vel,
                                                                      encode_settings.sampling_rate,
                                                                      encode_settings.wtrack_arm_coordinates)

        self.linpos_flat['arm'] = pd.Series(index=self.linpos_flat.index, data=pos_enum, dtype='category')
        self.linpos_flat['direction'] = pd.Series(index=self.linpos_flat.index,
                                                  data=pos_dir_enum, dtype='category')

    @staticmethod
    def simulate_arm_const_speed(arm_range, trial_time, trial_len, sampling_rate):
        pos_arm = np.arange(arm_range.x1, arm_range.x2,
                            arm_range.len/sampling_rate/(trial_time*arm_range.len/trial_len))
        return pos_arm

    @staticmethod
    def simulate_trial_const_speed(arm1_range, arm1_enum, arm2_range, arm2_enum, trial_time, sampling_rate):
        pos_trial = np.array([])
        pos_trial_enum = np.array([])

        pos_arm = WtrackPosConstSimulator.simulate_arm_const_speed(arm1_range, trial_time,
                                                                   arm1_range.len + arm2_range.len,
                                                                   sampling_rate)
        pos_trial = np.append(pos_trial, pos_arm)
        pos_trial_enum = np.append(pos_trial_enum, np.tile(arm1_enum, len(pos_arm)))

        pos_arm = WtrackPosConstSimulator.simulate_arm_const_speed(arm2_range, trial_time,
                                                                   arm1_range.len + arm2_range.len,
                                                                   sampling_rate)
        pos_trial = np.append(pos_trial, pos_arm)
        pos_trial_enum = np.append(pos_trial_enum, np.tile(arm2_enum, len(pos_arm)))
        return pos_trial, pos_trial_enum

    @staticmethod
    def simulate_run_const_speed(arm1_dir, arm1_enum, arm2_dir, arm2_enum, trial_time, sampling_rate):
        pos_run = np.array([])
        pos_run_enum = np.array([])
        pos_run_dir_enum = np.array([])

        pos_trial, pos_trial_enum = \
                WtrackPosConstSimulator.simulate_trial_const_speed(arm1_dir.outbound, arm1_enum,
                                                                   arm2_dir.outbound, arm2_enum,
                                                                   trial_time, sampling_rate)
        pos_run = np.append(pos_run, pos_trial)
        pos_run_enum = np.append(pos_run_enum, pos_trial_enum)
        pos_run_dir_enum = np.append(pos_run_dir_enum, np.tile(Direction.outbound, len(pos_trial)))

        pos_trial, pos_trial_enum = \
                WtrackPosConstSimulator.simulate_trial_const_speed(arm2_dir.inbound, arm2_enum,
                                                                   arm1_dir.inbound, arm1_enum,
                                                                   trial_time, sampling_rate)
        pos_run = np.append(pos_run, pos_trial)
        pos_run_enum = np.append(pos_run_enum, pos_trial_enum)
        pos_run_dir_enum = np.append(pos_run_dir_enum, np.tile(Direction.inbound, len(pos_trial)))

        return pos_run, pos_run_enum, pos_run_dir_enum

    @staticmethod
    def simulate_series_const_speed(w_coor, trial_time, sampling_rate):
        pos_series = np.array([])
        pos_series_enum = np.array([])
        pos_series_dir_enum = np.array([])

        pos_run, pos_run_enum, pos_run_dir_enum = \
                WtrackPosConstSimulator.simulate_run_const_speed(w_coor.center, WtrackArm.center,
                                                                 w_coor.left, WtrackArm.left,
                                                                 trial_time, sampling_rate)
        pos_series = np.append(pos_series, pos_run)
        pos_series_enum = np.append(pos_series_enum, pos_run_enum)
        pos_series_dir_enum = np.append(pos_series_dir_enum, pos_run_dir_enum)

        pos_run, pos_run_enum, pos_run_dir_enum = \
                WtrackPosConstSimulator.simulate_run_const_speed(w_coor.center, WtrackArm.center,
                                                                 w_coor.right, WtrackArm.right,
                                                                 trial_time, sampling_rate)
        pos_series = np.append(pos_series, pos_run)
        pos_series_enum = np.append(pos_series_enum, pos_run_enum)
        pos_series_dir_enum = np.append(pos_series_dir_enum, pos_run_dir_enum)

        return pos_series, pos_series_enum, pos_series_dir_enum
