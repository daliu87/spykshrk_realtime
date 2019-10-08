import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd

from spykshrk.franklab.data_containers import SpikeFeatures, FlatLinearPosition
from spykshrk.util import AttrDict


class UnitNormalGenerator:
    """
    Neural activity simulator for tetrodes.  For each unit found on a single tetrode, models the unit as
    a multivariate gaussians in 4-D amplitude (mark) space and firing rate in 1-D position space.
    Simulates spike trains as a poisson process with input of time and position.
     Returns a table of spike times and their matching mark features.
    """
    def __init__(self, elec_grp_id, mark_mean, mark_cov, pos_mean, pos_var, peak_fr, sampling_rate):
        """
        Constructor to setup 4-D gaussian model of amplitude (mark) space and 1-D position.  The model is a
        mark poisson distribution of when and what mark features spike events would have given an input of
        position.
        :param elec_grp_id: scalar ID for unique tetrodes
        :param mark_mean: list specifying the mean of each unit's mark gaussian distribution (4-D)
        :param mark_cov:  list specifying the covariance of each unit's mark gaussian distribution (4-D)
        :param pos_mean: scalar specifying the mean of each unit's position gaussian distribution (1-D)
        :param pos_var: scalar specifying the variance of each unit's position gaussian distribution (1-D)
        :param peak_fr: scalar specifying the peak firing rate of each unit.
        :param sampling_rate: uniform sampling rate to expect for simulation input
        """
        self.elec_grp_id = elec_grp_id
        self.mark_mean = mark_mean
        self.mark_cov = mark_cov
        self.pos_mean = pos_mean
        self.pos_var = pos_var
        self.rv_marks = sp.stats.multivariate_normal(mean=mark_mean, cov=np.diag(mark_cov))
        self.rv_pos = sp.stats.norm(loc=pos_mean, scale=pos_var)
        self.peak_fr = peak_fr
        self.sampling_rate = sampling_rate

    def simulate_spikes_over_pos(self, linpos_flat: FlatLinearPosition):
        """Simulate spikes given a list of uniformly sampled position data.

        :param linpos_flat: a Pandas Dataframe of uniformly sampled position data. Index should be time
                            and 'linpos_flat' should be the column name of 1-D positions
        :return: a SpikeFeatures dataframe of simulated spike times and corresponding amplitude (mark)
        features
        """

        # Generate the probability of a spike occurring at each position depending on the
        # firing rate - position map of the unit.
        prob_field = self.rv_pos.pdf(linpos_flat['linpos_flat'].values)/self.rv_pos.pdf(self.pos_mean)

        # Simulates spike train by treating each time point as a bernoulli trial
        spike_train = sp.stats.bernoulli(p=self.peak_fr/self.sampling_rate * prob_field).rvs()

        # Generate the mark features based on mark kernel.  Assumes mark probability is uniformly
        # distributed over position.
        marks = np.atleast_2d(self.rv_marks.rvs(sum(spike_train))).astype('i4')

        # list of spike indexes
        sample_num = np.nonzero(spike_train)[0]

        # reorganizing linpos data into a list of spike times
        time_ind = linpos_flat.index[sample_num]
        ind_levels = time_ind.levels.copy()
        ind_levels.append([self.elec_grp_id])
        ind_labels = time_ind.codes.copy()
        ind_labels.append([0]*len(time_ind))
        ind_names = time_ind.names.copy()
        ind_names.append('elec_grp_id')

        # Organizes returning DataFrame
        new_ind = pd.MultiIndex(levels=ind_levels, codes=ind_labels, names=ind_names)
        new_ind = new_ind.reorder_levels(['day', 'epoch', 'elec_grp_id', 'timestamp', 'time'])
        #new_ind = new_ind.sortlevel(['day', 'epoch', 'elec_grp', 'timestamp', 'time'])

        # Packages Pandas data into a SpikeFeatures dataframe
        spk_amp = SpikeFeatures(marks, columns=['c00', 'c01', 'c02', 'c03'],
                                index=new_ind)
        mark_linpos = linpos_flat.iloc[sample_num].copy()
        mark_linpos['elec_grp_id'] = self.elec_grp_id
        mark_linpos.set_index('elec_grp_id', append=True, inplace=True)
        mark_linpos = mark_linpos.reorder_levels(['day','epoch','elec_grp_id','timestamp','time'])

        return spk_amp, mark_linpos, prob_field


class TetrodeUniformUnitNormalGenerator:

    def __init__(self, sampling_rate=1000, num_marks=4, num_units=1, mark_mean_range=(40, 100),
                 mark_cov_range=(10, 20), firing_rate_range=(5, 20), pos_field_range=(0, 100),
                 pos_field_var_range=(5, 10)):

        self.sampling_rate = sampling_rate
        self.num_units = num_units
        self.num_marks = num_marks
        self.mark_mean_range = mark_mean_range
        self.mark_cov_range = mark_cov_range
        self.firing_rate_range = firing_rate_range
        self.pos_field_range = pos_field_range
        self.pos_field_var_range = pos_field_var_range

        self.unit_mean = np.random.randint(*self.mark_mean_range, size=[self.num_units, self.num_marks])
        self.unit_cov = np.random.randint(*self.mark_cov_range, size=[self.num_units, self.num_marks])
        self.unit_pos_mean = np.random.uniform(*self.pos_field_range, size=self.num_units)
        self.unit_pos_var = np.random.uniform(*self.pos_field_var_range, size=self.num_units)
        self.unit_fr = np.random.randint(*self.firing_rate_range, size=self.num_units)

        self.units = {}
        for unit_ii in range(self.num_units):
            self.units[unit_ii] = UnitNormalGenerator(elec_grp_id=1,
                                                      mark_mean=self.unit_mean[unit_ii, :],
                                                      mark_cov=self.unit_cov[unit_ii, :],
                                                      pos_mean=self.unit_pos_mean[unit_ii],
                                                      pos_var=self.unit_pos_var[unit_ii],
                                                      peak_fr=self.unit_fr[unit_ii],
                                                      sampling_rate=self.sampling_rate)

    def simulate_tetrode_over_pos(self, linpos_flat: FlatLinearPosition):
        unit_spks = {}
        spk_amps = pd.DataFrame()

        for unit_ii in range(self.num_units):
            unit_marks, mark_pos, field = self.units[unit_ii].simulate_spikes_over_pos(linpos_flat)
            unit_spks[unit_ii] = unit_marks.merge(mark_pos, how='outer', left_index=True, right_index=True)

            spk_amps = spk_amps.append(unit_marks)

        spk_amps.sort_index(level='timestamp', inplace=True)

        spk_amps = spk_amps[~spk_amps.index.duplicated(keep='first')]

        return spk_amps, unit_spks
