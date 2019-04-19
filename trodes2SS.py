"""
Parse trodes data into SS data containers
# Written by AKG
# Edited 3-8-19 by MEC to accomodate tetrodes and tritrodes (line 94)
# Edited 3-22-19 by MEC to make a filter for marks with large negative channels because
#                       this crashes the decoder by going outside the bounds of the
#                       normal_pdf_int_lookup function

"""

import numpy as np
import scipy as sp
import scipy.stats
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import holoviews as hv
import xarray as xr

import json
import functools

import dask
import dask.dataframe as dd
import dask.array as da

from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, \
        EncodeSettings, pos_col_format, SpikeObservation, RippleTimes

def get_all_below_threshold(self, threshold):
    ind = np.nonzero(np.all(self.values < threshold, axis=1))
    return self.iloc[ind]

def get_any_above_threshold(self, threshold):
    ind = np.nonzero(np.any(self.values >= threshold, axis=1))
    return self.iloc[ind]

def get_all_above_threshold(self, threshold):
    ind = np.nonzero(np.all(self.values > threshold, axis=1))
    return self.iloc[ind]

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

def threshold_marks_negative(marks, negthresh=-999):
		pre_length = marks.shape
		marks = get_all_above_threshold(marks, negthresh)
		print(str(pre_length[0]-marks.shape[0])+' below '+str(negthresh)+'uV events removed')

		return marks

def threshold_marks(marks, maxthresh=2000, minthresh=0):
		pre_length = marks.shape
		marks = get_all_below_threshold(marks, maxthresh)
		print(str(pre_length[0]-marks.shape[0])+' above '+str(maxthresh)+'uV events removed')
		pre_length = marks.shape
		marks = get_any_above_threshold(marks, minthresh)
		print(str(pre_length[0]-marks.shape[0])+' below '+str(minthresh)+'uV events removed')

		return marks

class TrodesImport:
	""" animalinfo - takes in animal d/e/t information; parses FFmat files into dataframes of each datatype 
	"""
	def __init__(self, ff_dir, name, days, epochs, tetrodes, Fs=3e4):
	    """ init function
	    
	    Args:
	        base_dir: root directory of animal data
	        name: name of animal
	        days: array of which days of data to process
	        tetrodes: array of which tetrodes to process
	        epochs: list of epochs for encoding
	    
	    """
	    self.ff_dir = ff_dir
	    self.name = name
	    self.days = days
	    self.epochs = epochs
	    self.tetrodes = tetrodes
	    self.Fs = Fs


	def import_marks(self):

		spk_amps = pd.DataFrame()
		

		for day in self.days:
			markname = self.ff_dir+self.name+'marks'+str(day)+'.mat'
			markmat = scipy.io.loadmat(markname,squeeze_me=True,struct_as_record=False)

			for ep in self.epochs:
				de_amps = pd.DataFrame()

				for tet in self.tetrodes:
					marktimes = markmat['marks'][day-1][ep-1][tet-1].times*self.Fs
					marktimes = marktimes.astype(np.int64,copy=False)
					marks = markmat['marks'][day-1][ep-1][tet-1].marks
					marks = marks.astype(np.int16,copy=False)
					tet_marks = SpikeFeatures.from_numpy_single_epoch_elec(day ,ep, tet, marktimes,marks,sampling_rate=self.Fs)
					if len(tet_marks.columns) == 4:
						tet_marks.columns=['c00','c01','c02','c03']
					if len(tet_marks.columns) == 3:
						tet_marks.columns=['c00','c01','c02']
					de_amps = de_amps.append(tet_marks)

				de_amps.sort_index(level='timestamp', inplace=True)
				print('duplicates found & removed: '+str(de_amps[de_amps.index.duplicated(keep='first')].size))
				de_amps = de_amps[~de_amps.index.duplicated(keep='first')]
				spk_amps = spk_amps.append(de_amps)

		spk_amps.sampling_rate = self.Fs
		return spk_amps

	def import_pos(self, encode_settings, xy = 'x'):

		allpos = pd.DataFrame()

		for day in self.days:
			for ep in self.epochs:
				posname = self.ff_dir+self.name+'pos'+str(day)+'.mat'
				posmat = scipy.io.loadmat(posname,squeeze_me=True,struct_as_record=False)
				pos_time = self.Fs*posmat['pos'][day-1][ep-1].data[:,0]
				pos_time = pos_time.astype(np.int64,copy=False)
				pos_runx = posmat['pos'][day-1][ep-1].data[:,5]
				pos_runy = posmat['pos'][day-1][ep-1].data[:,6]
				pos_vel = posmat['pos'][day-1][ep-1].data[:,8]

				if 'x' in xy:
					pos_obj = FlatLinearPosition.from_numpy_single_epoch(day, ep, pos_time, pos_runx, pos_vel, self.Fs,
                                                               encode_settings.arm_coordinates)
				if 'y' in xy:
					pos_obj = FlatLinearPosition.from_numpy_single_epoch(day, ep, pos_time, pos_runy, pos_vel, self.Fs,
                                                               encode_settings.arm_coordinates)
				allpos = allpos.append(pos_obj)

		allpos.sampling_rate = self.Fs
		return allpos

	def import_rips(self,pos_obj=None, velthresh=4):

		allrips = pd.DataFrame()

		for day in self.days:
			for ep in self.epochs:
				ripname = self.ff_dir+self.name+'ca1rippleskons'+str(day)+'.mat'
				ripmat = scipy.io.loadmat(ripname,squeeze_me=True,struct_as_record=False)
			#generate a pandas table with starttime, endtime, and maxthresh columns, then instantiate RippleTimes 
				ripdata = {'starttime':ripmat['ca1rippleskons'][day-1][ep-1].starttime,
    			        'endtime':ripmat['ca1rippleskons'][day-1][ep-1].endtime,
       				    'maxthresh':ripmat['ca1rippleskons'][day-1][ep-1].maxthresh}
				rippd = pd.DataFrame(ripdata,pd.MultiIndex.from_product([[day],[ep],
                        range(len(ripmat['ca1rippleskons'][day-1][ep-1].maxthresh))],
                        names=['day','epoch','event']))
			#reorder the fields 
				rippd = rippd[['starttime','endtime','maxthresh']]
				#rip_obj = RippleTimes.create_default(rippd, 1)

				#if pos_obj is not None:
				#add an additional field for velocity and filter out events exceeding velthresh
					#veltmp = pos_obj.get_irregular_resampled_old(self.Fs*rip_obj['starttime'])
					#rip_obj['vels'] = veltmp['linvel_flat'].values
					#rip_obj = rip_obj.iloc[rip_obj['vels'].values < 4]

				#allrips = allrips.append(rip_obj)

		allrips = allrips.append(rippd)
		return allrips

def convert_dan_posterior_to_xarray(posterior_df, tetrode_dictionary, velocity_filter, encode_settings, decode_settings, transition_matrix, shuffle_offset, trial_order, marks_time_shift_amount, position_bin_centers=None):
    '''Converts pandas dataframe from Dan's 1D decoder to xarray Dataset
    
    Parameters
    ----------
    posterior_df : pandas.DataFrame, shape (n_time, n_columns)
    position_bin_centers : None or ndarray, shape (n_position_bins,), optional
    
    Returns
    -------
    results : xarray.Dataset
    
    '''
    is_position_bin = posterior_df.columns.str.startswith('x')
    
    if position_bin_centers is None:
        n_position_bins = is_position_bin.sum()
        position_bin_centers = np.arange(n_position_bins)
        
    coords = dict(
        day=posterior_df.loc[:, 'day'].values,
        epoch=posterior_df.loc[:, 'epoch'].values,
        timestamp=posterior_df.loc[:, 'timestamp'].values,
        time=posterior_df.loc[:, 'time'].values,
        position=position_bin_centers,
        num_spikes=posterior_df.loc[:, 'num_spikes'].values,
        dec_bin=posterior_df.loc[:, 'dec_bin'].values,
        ripple_grp=posterior_df.loc[:, 'ripple_grp'].values,
    )

    return xr.Dataset(
	    {'posterior': (('time','position'), posterior_df.loc[:, is_position_bin].values),
	     'velocity filter encode': velocity_filter,
	     'velocity filter decode': velocity_filter,
	     'tetrodes': tetrode_dictionary,
	     'shuffle offset': shuffle_offset,
	     'marks_time_shift_amount': marks_time_shift_amount,
	     'trial_order': trial_order,
	     'sampling_rate': encode_settings['sampling_rate'],
	     'pos_bins': encode_settings['pos_bins'],
	     'pos_bin_edges': encode_settings['pos_bin_edges'],
	     'pos_bin_delta': encode_settings['pos_bin_delta'],
	     'pos_kernel': encode_settings['pos_kernel'],
	     'pos_kernel_std': encode_settings['pos_kernel_std'],
	     'mark_kernel_std': encode_settings['mark_kernel_std'],
	     'pos_num_bins': encode_settings['pos_num_bins'],
	     'pos_col_names': encode_settings['pos_col_names'],
	     'arm_coordinates': (encode_settings['arm_coordinates'][0]),
	     'trans_smooth_std': decode_settings['trans_smooth_std'],
	     'trans_uniform_gain': decode_settings['trans_uniform_gain'],
	     'time_bin_size': decode_settings['time_bin_size'],
	     'transition_matrix_name': 'flat powered',
	     'multiindex': ['day','epoch','timestamp','time'],
	     'transition_matrix': (('position','position'), transition_matrix)},
	    coords=coords)
