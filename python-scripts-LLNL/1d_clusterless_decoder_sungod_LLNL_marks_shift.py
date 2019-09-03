# python script to run 1d clusterless decoder on sungod data using spykshrk
# written by MEC from notebooks written by AKG and JG
# 3-6-19
# this version includes support for linearizing the whole epoch (use pos_all_linear instead of pos_subset)

#cell 1
# Setup and import packages
import sys
sys.path.append('/usr/workspace/wsb/coulter5/spykshrk_realtime')
import os 
import glob
from datetime import datetime
import trodes2SS
from trodes2SS import AttrDict, TrodesImport, convert_dan_posterior_to_xarray
import sungod_linearization
from sungod_linearization import createTrackGraph, hack_determinearmorder, turn_array_into_ranges, \
chunk_data, change_to_directory_make_if_nonexistent
import numpy as np
import scipy.io
import scipy as sp
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import holoviews as hv
import json
import functools
import dask
import dask.dataframe as dd
import dask.array as da
import networkx as nx
import loren_frank_data_processing as lfdp
import scipy.io as sio # for saving .mat files 
import inspect # for inspecting files (e.g. finding file source)
import multiprocessing 
import sys 
import pickle
from tempfile import TemporaryFile
from multiprocessing import Pool
import math 

print(datetime.now())
print(datetime.now(), file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))

# set path to folders where spykshrk core scripts live
path_main = '/usr/workspace/wsb/coulter5/spykshrk_realtime'
os.chdir(path_main)
from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, Posteriors, \
        EncodeSettings, pos_col_format, SpikeObservation, RippleTimes, DayEpochEvent, DayEpochTimeSeries
from spykshrk.franklab.pp_decoder.util import normal_pdf_int_lookup, gaussian, apply_no_anim_boundary, normal2D
from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder
#from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
from spykshrk.util import Groupby


#cell 2
# Import data

# Define path bases 
path_base_rawdata = '/p/lustre1/coulter5/remy/'

# Define parameters
# for epochs we want 2 and 4 for each day
rat_name = 'remy'
directory_temp = path_base_rawdata + rat_name + '/'
day_dictionary = {'remy':[20], 'gus':[28], 'bernard':[23], 'fievel':[19]}
epoch_dictionary = {'remy':[2], 'gus':[2], 'bernard':[4], 'fievel':[2]} 
#tetrodes_dictionary = {'remy': [4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
#                       'gus': [6,7,8,9,10,11,12,17,18,19,20,21,24,25,26,27,30], # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]
#                       'bernard': [1,2,3,4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
#                       'fievel': [1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,27,28,29]}

tetrodes_dictionary = {'remy': [4], 'gus': [6], 'bernard': [1], 'fievel': [1]}

# Maze information
os.chdir('/usr/workspace/wsb/coulter5/spykshrk_realtime/')
#maze_coordinates = scipy.io.loadmat('set_arm_nodes.mat',variable_names = 'linearcoord_NEW')
# new maze coordinates with only one segment for box
maze_coordinates = scipy.io.loadmat('set_arm_nodes.mat',variable_names = 'linearcoord_one_box')

print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]))
print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]), file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))

datasrc = TrodesImport(directory_temp, rat_name, day_dictionary[rat_name], 
                       epoch_dictionary[rat_name], tetrodes_dictionary[rat_name])
# Import marks
marks = datasrc.import_marks()
# # os.chdir('/data2/jguidera/data/')
# # np.load('marks.npy')

# Import position #? concerned about use of sampling rate in the definition for position
# Temporary small definition of encoding settings-- need 'arm_coordinates' to use datasrc.import_pos 
encode_settings = AttrDict({'arm_coordinates': [[0,0]]})
# Import position (#? concerned about use of sampling rate in the definition for position)
pos = datasrc.import_pos(encode_settings, xy='x')
posY = datasrc.import_pos(encode_settings, xy='y')

# Import ripples
rips = datasrc.import_rips(pos, velthresh=4)

# Define path bases
path_base_dayepoch = 'day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0])
#path_base_analysis = '/mnt/vortex/mcoulter/'
path_base_analysis = '/p/lustre1/coulter5/remy/maze_info/'

#cell 6
# linearize the whole epoch - should only have to do this once.

speed_threshold_save = 0; 

#new position variables for whole epoch
pos_all_linear = pos
posY1 = posY

#linear_start = pos.index.get_level_values('time')[encode_subset_start]
#linear_end = pos.index.get_level_values('time')[encode_subset_end]

# Define path base
#path_base_timewindow = str(int(round(linear_start))) + 'to' + str(int(round(linear_end))) + 'sec'
path_base_timewindow = 'whole_epoch_v2'
path_base_foranalysisofonesessionepoch = path_base_analysis + rat_name + '/' + path_base_dayepoch + '/' + path_base_timewindow

# Change to directory with saved linearization result
# Define folder for saved linearization result 
linearization_output_save_path = path_base_foranalysisofonesessionepoch + '/linearization_output/'
linearization_output_save_path
# Check if it exists, make if it doesn't
directory_path = linearization_output_save_path
change_to_directory_make_if_nonexistent(directory_path)

# Define name of linearization result
linearization_output1_save_filename = 'linearization_' + path_base_timewindow + '_speed' + str(speed_threshold_save) + '_linear_distance_arm_shift' + '.npy'
linearization_output2_save_filename = 'linearization_' + path_base_timewindow + '_speed' + str(speed_threshold_save) + '_track_segment_id_use' + '.npy'
# If linearization result doesn't exist, do linearization calculation
    
# If linearization result exists, load it 
print('Linearization result exists. Loading it.')
print("Linearization result exists. Loading it.", file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))
linear_distance_arm_shift = np.load(linearization_output1_save_filename)
track_segment_id_use = np.load(linearization_output2_save_filename)
#pos_subset['linpos_flat'] = linear_distance_arm_shift[(encode_subset_start-encode_subset_start):(encode_subset_end-encode_subset_start+1)]
#whole_epoch
pos_all_linear['linpos_flat']=linear_distance_arm_shift

#cell 7
# Define position bins #!!! HARD CODE: ASSUMES POSITION BIN OF WIDTH 1 !!!
# need to use the indices of the encoding time subset in this cell

# Initialize variables 
tracksegment_positionvalues_min_and_max = []
tracksegment_positionvalues_for_bin_edges = [] 

# Find min and max position for each track segment 
#tracksegments_temp = np.unique(track_segment_id_use[encode_subset_start:(encode_subset_end+1)])
#whole epoch
tracksegments_temp = np.unique(track_segment_id_use[0:len(linear_distance_arm_shift)])

for t_loop in tracksegments_temp: # for each track segment
    #indiceswewant_temp = track_segment_id_use[encode_subset_start:(encode_subset_end+1)] == t_loop
    #whole epoch
    indiceswewant_temp = track_segment_id_use[0:len(linear_distance_arm_shift)] == t_loop

    #tracksegment_positionvalues_temp = pos_subset.values[indiceswewant_temp,0] # second dimension of pos_subset: zero for position, 1 for velocity
    #whole epoch
    tracksegment_positionvalues_temp = pos_all_linear.values[indiceswewant_temp,0]

    tracksegment_positionvalues_min_and_max.append([tracksegment_positionvalues_temp.min(), tracksegment_positionvalues_temp.max()])
    # To define edges, floor mins and ceil maxes
    tracksegment_positionvalues_for_bin_edges.append([np.floor(tracksegment_positionvalues_temp.min()), np.ceil(tracksegment_positionvalues_temp.max())])

# Floor to get bins #? Is this right? Does 0 mean the bin spanning [0, 1]?
tracksegment_positionvalues_min_and_max_floor = np.floor(tracksegment_positionvalues_min_and_max)

# Find only bins in range of segments
binswewant_temp = []
for t_loop in tracksegment_positionvalues_min_and_max_floor: # for each track segment
    binswewant_temp.append(np.ndarray.tolist(np.arange(t_loop[0],t_loop[1] + 1))) # + 1 to account for np.arange not including last index
# Do same for edges
edgeswewant_temp = []
for t_loop in tracksegment_positionvalues_for_bin_edges: # for each track segment
    edgeswewant_temp.append(np.ndarray.tolist(np.arange(t_loop[0],t_loop[1] + 1))) # + 1 to account for np.arange not including last index

# Flatten (combine bins from segments)
binswewant_temp_flat = [y for x in binswewant_temp for y in x]
edgeswewant_temp_flat = [y for x in edgeswewant_temp for y in x]

# Find unique elements
arm_coords_wewant = (np.unique(binswewant_temp_flat))
edges_wewant = (np.unique(edgeswewant_temp_flat))

# Turn list of edges into ranges 
start_temp, end_temp = turn_array_into_ranges(edges_wewant)
arm_coordinates_WEWANT = np.column_stack((start_temp, end_temp))
print('Arm coordinates: ',arm_coordinates_WEWANT)
print('Arm coordinates: ',arm_coordinates_WEWANT, file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))

#cell 7.1
# this cell speeds up encoding with larger position bins
# try 5cm bins - do this by dividing position subset by 5 and arm coords by 5

#pos_subset['linpos_flat'] = (pos_subset['linpos_flat'])/5
#whole epoch
pos_all_linear['linpos_flat'] = (pos_all_linear['linpos_flat'])/5

arm_coordinates_WEWANT = arm_coordinates_WEWANT/5
arm_coordinates_WEWANT = np.around(arm_coordinates_WEWANT)
print('Arm coordinates: ',arm_coordinates_WEWANT)
print('Arm coordinates: ',arm_coordinates_WEWANT, file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))

#cell 9.1 randomize trial order within epoch
#read in trial times
trialsname = directory_temp+rat_name+'trials'+str(day_dictionary[rat_name][0])+'.mat'
trialsmat = scipy.io.loadmat(trialsname,squeeze_me=True,struct_as_record=False)
starttimes = trialsmat['trials'][day_dictionary[rat_name][0]-1][epoch_dictionary[rat_name][0]-1].starttime
starttimes = starttimes.astype(np.float64,copy=False)
endtimes = trialsmat['trials'][day_dictionary[rat_name][0]-1][epoch_dictionary[rat_name][0]-1].endtime
endtimes = endtimes.astype(np.float64,copy=False)
trialsindex = np.arange(starttimes.shape[0])
print('Number of trials: ',trialsindex.shape)
print('Number of trials: ',trialsindex.shape, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

# randomize trial order
indices = np.arange(starttimes.shape[0])
np.random.shuffle(indices)

#fixed random order
indices = [ 17,  92,   3,  98,  11,  78, 105, 100, 103,  37,  28,  62,  85,  59,  41,  93,  29, 102, 
6,  76,  13,  82,  18,  25,  64,  96,  20,  16,  65,  54,  12,  24,  56,   5,  74,  73, 
79,  89,  97,  70,  68,  46,   7,  40, 101,  48,  77,  63,  69, 108,  66,  15,  91,  33, 
45,  21,  51,  19,  30,  23,  72,  35,  42,  47,  95, 107, 104,  61,  43,  60,  67,  88, 
71,  14,  38,  32,  87,  57,  27,  31,   1,   2,  53,  86,  50,  49,   0,  52,  90,  10, 
44,  84,  55,  81, 106,  39,  75,  58,   9,  34,   4,   8,  26,  22,  94,  83,  36,  80, 99]

starttimes_shuffled = starttimes[indices]
endtimes_shuffled = endtimes[indices]
trialsindex_shuffled = trialsindex[indices]
print('Randomized trial order: ',trialsindex_shuffled)
print('Randomized trial order: ',trialsindex_shuffled, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

#to make a new position, marks and trial file with new start and end times:
#position
random_trial_pos_all = pos_all_linear.head(0)
for i in range(len(starttimes_shuffled)):
    random_trial_pos = pos_all_linear.loc[(pos_all_linear.index.get_level_values('time') <= endtimes_shuffled[i]) & (pos_all_linear.index.get_level_values('time') >= starttimes_shuffled[i])]
    random_trial_pos_all = random_trial_pos_all.append(random_trial_pos)
     
#marks
random_trial_marks_all = marks.head(0)
for i in range(len(starttimes_shuffled)):
    random_trial_marks = marks.loc[(marks.index.get_level_values('time') <= endtimes_shuffled[i]) & (marks.index.get_level_values('time') >= starttimes_shuffled[i])]
    random_trial_marks_all = random_trial_marks_all.append(random_trial_marks)

# filter for large negative marks and spike amplitude
marks_random_trial_non_negative = trodes2SS.threshold_marks_negative(random_trial_marks_all, negthresh=-999)
print('Original encode length: ',random_trial_marks_all.shape)
print('Original encode length: ',random_trial_marks_all.shape, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
print('Encoding marks non-negative filter: ',marks_random_trial_non_negative.shape)
print('Encoding marks non-negative filter: ',marks_random_trial_non_negative.shape, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

random_trial_spk_subset_sparse = trodes2SS.threshold_marks(marks_random_trial_non_negative, maxthresh=2000,minthresh=100)
print('original length: '+str(marks_random_trial_non_negative.shape[0]))
print('after filtering: '+str(random_trial_spk_subset_sparse.shape[0]))
print('original length: '+str(marks_random_trial_non_negative.shape[0]), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
print('after filtering: '+str(random_trial_spk_subset_sparse.shape[0]), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

# velocity filter to define encoding and decoding times
velocity_filter = 4
print('Velocity filter: ',velocity_filter)
print('Velocity filter: ',velocity_filter, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

#NOTE: to try marks shift on whole trials we need to do shift first, then velocity filter for encoding and decoding marks
# nope - cant do this, need to do velocity filter first

# #encoding spikes
linflat_obj = random_trial_pos_all.get_mapped_single_axis()

#linflat_obj = pos_all_linear.get_mapped_single_axis()
linflat_spkindex = linflat_obj.get_irregular_resampled(random_trial_spk_subset_sparse)
linflat_spkindex_encode_velthresh = linflat_spkindex.query('linvel_flat > @velocity_filter')

encode_spikes_random_trial = random_trial_spk_subset_sparse.loc[linflat_spkindex_encode_velthresh.index]

print('encoding spikes after velocity filter: '+str(encode_spikes_random_trial.shape[0]))
print('encoding spikes after velocity filter: '+str(encode_spikes_random_trial.shape[0]), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

# #decoding spikes
linflat_obj = random_trial_pos_all.get_mapped_single_axis()
#linflat_obj = pos.get_mapped_single_axis()
linflat_spkindex = linflat_obj.get_irregular_resampled(random_trial_spk_subset_sparse)
linflat_spkindex_decode_velthresh = linflat_spkindex.query('linvel_flat < @velocity_filter')

decode_spikes_random_trial = random_trial_spk_subset_sparse.loc[linflat_spkindex_decode_velthresh.index]

print('decoding spikes after velocity filter: '+str(decode_spikes_random_trial.shape[0]))
print('decoding spikes after velocity filter: '+str(decode_spikes_random_trial.shape[0]), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

#filter position for velocity
random_trial_pos_all_vel = random_trial_pos_all.loc[(random_trial_pos_all['linvel_flat']>velocity_filter)]
#random_trial_pos_all_vel = pos_all_linear.loc[(pos_all_linear['linvel_flat']>velocity_filter)]


#cell 9.2 shuffle encoding marks by set amount of time

# yes, we want to do this after the velocity filter for encoding spikes
# also what about the amplitude filter - can that still come before the shift? - yes

# caclulate epoch min and max times - need to do this on pre-filter marks
marks_for_epoch_time = marks.reset_index(level='time')
min_time = marks_for_epoch_time['time'].iloc[0]
max_time = marks_for_epoch_time['time'].iloc[-1]
epoch_time = max_time - min_time
print('Total spoch time (sec): ',epoch_time)
print('Total epoch time (sec) ',epoch_time, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))  

# # shift by 25% of the epoch time
shift_amt = 0
marks_index_target = min_time + shift_amt*epoch_time
encode_spikes_shift_query = encode_spikes_random_trial.reset_index()
marks_index_shift = encode_spikes_shift_query.iloc[(encode_spikes_shift_query['time']-marks_index_target).abs().argsort()[:1]].index.item()

# apply shift to tetrode channel columns in original dataframe
encode_spikes_random_trial = []
encode_spikes_random_trial = random_trial_spk_subset_sparse.loc[linflat_spkindex_encode_velthresh.index]
encode_spikes_random_trial['c00'] = np.roll(encode_spikes_random_trial['c00'],-(marks_index_shift))
encode_spikes_random_trial['c01'] = np.roll(encode_spikes_random_trial['c01'],-(marks_index_shift))
encode_spikes_random_trial['c02'] = np.roll(encode_spikes_random_trial['c02'],-(marks_index_shift))
encode_spikes_random_trial['c03'] = np.roll(encode_spikes_random_trial['c03'],-(marks_index_shift)) 

# output = position and encoding and decoding spikes
# save pandas dataframes as .npy files
# pos_all_linear_vel
# decode_spikes_random_trial
# encode_spikes_random_trial

print("End of script!")
print("End of script!", file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))
print(" ", file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))

