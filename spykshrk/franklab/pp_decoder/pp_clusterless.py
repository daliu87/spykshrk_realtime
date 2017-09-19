import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from spykshrk.franklab.pp_decoder.util import gaussian, normal2D, apply_no_anim_boundary


# Calculate State Transition Matrix
def calc_learned_state_trans_mat(linpos_flat, x_bins, x_no_anim_bounds):
    pos_num_bins = len(x_bins)

    # Smoothing kernel for learned pos transition matrix
    xv, yv = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21))
    kernel = normal2D(xv, yv, 1)
    kernel /= kernel.sum()

    linpos_state = linpos_flat
    linpos_ind = np.searchsorted(x_bins, linpos_state, side='right') - 1

    # Create learned pos transition matrix
    learned_trans_mat = np.zeros([pos_num_bins, pos_num_bins])
    for first_pos_ind, second_pos_ind in zip(linpos_ind[:-1], linpos_ind[1:]):
        learned_trans_mat[first_pos_ind, second_pos_ind] += 1

    # normalize
    learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
    learned_trans_mat[np.isnan(learned_trans_mat)] = 0

    # smooth
    learned_trans_mat = sp.signal.convolve2d(learned_trans_mat, kernel, mode='same')
    learned_trans_mat = apply_no_anim_boundary(x_bins, x_no_anim_bounds, learned_trans_mat)

    # uniform offset
    uniform_gain = 0.01
    uniform_dist = np.ones(learned_trans_mat.shape)

    # no-animal boundary
    uniform_dist = apply_no_anim_boundary(x_bins, x_no_anim_bounds, uniform_dist)

    # normalize uniform offset
    uniform_dist = uniform_dist / (uniform_dist.sum(axis=0)[None, :])
    uniform_dist[np.isnan(uniform_dist)] = 0

    # apply uniform offset
    learned_trans_mat = learned_trans_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

    # renormalize
    learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
    learned_trans_mat[np.isnan(learned_trans_mat)] = 0

    return learned_trans_mat


# Compute artificial gaussian state transition matrix
def calc_simple_trans_mat(x_bins):
    pos_num_bins = len(x_bins)

    # Setup transition matrix
    transition_mat = np.ones([pos_num_bins, pos_num_bins])
    for bin_ii in range(pos_num_bins):
        transition_mat[bin_ii, :] = gaussian(x_bins, x_bins[bin_ii], 3)

    # uniform offset
    uniform_gain = 0.01
    uniform_dist = np.ones(transition_mat.shape)

    # normalize transition matrix
    transition_mat = transition_mat/( transition_mat.sum(axis=0)[None,:])

    # normalize uniform offset
    uniform_dist = uniform_dist/( uniform_dist.sum(axis=0)[None,:])

    # apply uniform offset
    transition_mat = transition_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

    return transition_mat


# Loop through each bin and generate the observation distribution from spikes in bin
def calc_observation_intensity(spike_decode, dec_bins, x_bins, pos_kernel, x_no_anim_bounds):
    pos_num_bins = len(x_bins)
    pos_bin_delta = x_bins[1] - x_bins[0]

    dec_bin_ids = np.unique(dec_bins)
    dec_est = np.zeros([dec_bin_ids[-1] + 1, pos_num_bins])

    # initialize conditional intensity function
    firing_rate = {ntrode_id: np.zeros(pos_num_bins) for ntrode_id in spike_decode['ntrode_id'].unique()}

    groups = spike_decode.groupby('dec_bin')
    bin_num_spikes = [0] * len(dec_est)

    for bin_id, spikes_in_bin in groups:
        dec_in_bin = np.ones(pos_num_bins)

        bin_num_spikes[bin_id] = len(spikes_in_bin)

        # Count spikes for occupancy firing rate (conditional intensity function)
        for ntrode_id, pos in spikes_in_bin.loc[:, ('ntrode_id', 'position')].values:
            firing_rate[ntrode_id][np.searchsorted(x_bins, pos, side='right') - 1] += 1

        for dec_ii, dec in enumerate(spikes_in_bin.loc[:, 'x0':'x{:d}'.format(pos_num_bins - 1)].values):
            smooth_dec = np.convolve(dec, pos_kernel, mode='same')
            dec_in_bin = dec_in_bin * smooth_dec
            dec_in_bin = dec_in_bin / (np.sum(dec_in_bin) * pos_bin_delta)

        dec_est[bin_id, :] = dec_in_bin

    # Smooth and normalize firing rate (conditional intensity function)
    for fr_key in firing_rate.keys():
        firing_rate[fr_key] = np.convolve(firing_rate[fr_key], pos_kernel, mode='same')

        firing_rate[fr_key] = apply_no_anim_boundary(x_bins, x_no_anim_bounds, firing_rate[fr_key])

        firing_rate[fr_key] = firing_rate[fr_key] / (firing_rate[fr_key].sum() * pos_bin_delta)

    return dec_est, bin_num_spikes, firing_rate


# Compute the likelihood of each bin
def calc_likelihood(dec_est, bin_num_spikes, prob_no_spike, pos_bin_delta):
    likelihoods = np.ones(dec_est.shape)

    for num_spikes, (dec_ind, dec_est_bin) in zip(bin_num_spikes, enumerate(dec_est)):
        if num_spikes > 0:
            likelihoods[dec_ind, :] = dec_est_bin

            for prob_no in prob_no_spike.values():
                likelihoods[dec_ind, :] *= prob_no
        else:

            for prob_no in prob_no_spike.values():
                likelihoods[dec_ind, :] *= prob_no

        # Normalize
        likelihoods[dec_ind, :] = likelihoods[dec_ind, :] / (likelihoods[dec_ind, :].sum() * pos_bin_delta)
    return likelihoods


def calc_posterior(likelihoods, transition_mat, pos_num_bins, pos_bin_delta):
    last_posterior = np.ones(pos_num_bins)

    posteriors = np.zeros(likelihoods.shape)

    for like_ii, like in enumerate(likelihoods):
        posteriors[like_ii, :] = like * (transition_mat * last_posterior).sum(axis=1)
        posteriors[like_ii, :] = posteriors[like_ii, :] / (posteriors[like_ii, :].sum() * pos_bin_delta)
        last_posterior = posteriors[like_ii, :]

    return posteriors


def plot_decode_2d(dec_est, dec_bin_times, stim_lockout_ranges, linpos_flat, plt_range):
    stim_lockout_ranges_sec = stim_lockout_ranges/30000
    stim_lockout_range_sec_sub = stim_lockout_ranges_sec[(stim_lockout_ranges_sec[1] > plt_range[0]) & (stim_lockout_ranges_sec[0] < plt_range[1])]

    plt.imshow(dec_est[(dec_bin_times > plt_range[0]*30000) & (dec_bin_times < plt_range[1]*30000)].transpose(),
               extent=[plt_range[0], plt_range[1], 0, 450], origin='lower', aspect='auto', cmap='hot', zorder=0)

    plt.colorbar()

    # Plot linear position
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
