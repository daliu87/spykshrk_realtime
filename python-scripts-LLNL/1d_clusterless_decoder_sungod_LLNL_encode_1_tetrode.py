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

print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]))
print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]), file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))


# load saved position, encoding spikes (shifted), and decoding spikes
# variable names:
# pos_all_linear_vel
# decode_spikes_random_trial
# ecnode_spikes_random_trial

#cell 8
#define encoding settings
#max_pos = int(round(linear_distance_arm_shift.max()) + 20)

# if you are using 5cm position bins, use this max_pos instead
max_pos = int(round(linear_distance_arm_shift.max()/5)+5)

encode_settings = AttrDict({'sampling_rate': 3e4,
                            'pos_bins': np.arange(0,max_pos,1), # arm_coords_wewant
                            'pos_bin_edges': np.arange(0,max_pos + .1,1), # edges_wewant, 
                            'pos_bin_delta': 1, 
                            # 'pos_kernel': sp.stats.norm.pdf(arm_coords_wewant, arm_coords_wewant[-1]/2, 1),
                            'pos_kernel': sp.stats.norm.pdf(np.arange(0,max_pos,1), max_pos/2, 1), #note that the pos_kernel mean should be half of the range of positions (ie 180/90) # sp.stats.norm.pdf(np.arange(0,560,1), 280, 1),    
                            'pos_kernel_std': 1, 
                            'mark_kernel_std': int(20), 
                            'pos_num_bins': max_pos, # len(arm_coords_wewant)
                            'pos_col_names': [pos_col_format(ii, max_pos) for ii in range(max_pos)], # or range(0,max_pos,10)
                            'arm_coordinates': arm_coordinates_WEWANT}) # includes box, removes bins in the gaps 'arm_coordinates': [[0,max_pos]]})

print('Encode settings: ',encode_settings)
print('Encode settings: ',encode_settings, file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))

#cell 9
#define decode settings
decode_settings = AttrDict({'trans_smooth_std': 2,
                            'trans_uniform_gain': 0.0001,
                            'time_bin_size':60})

print('Decode settings: ',decode_settings)
print('Decode settings: ',decode_settings, file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))

#cell 10
# Run encoder
# these time-table lines are so that we can record the time it takes for encoder to run even if notebook disconnects
# look at the time stamps for the two files in /data2/mcoulter called time_stamp1 and time_stamp2
print('Starting encoder')
print("Starting encoder", file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))
time_table_data = {'age': [1, 2, 3, 4, 5]}
time_table = pd.DataFrame(time_table_data)
time_table.to_csv('/p/lustre1/coulter5/remy/time_stamp1.csv')

#for whole epoch: linflat=pos_all_linear_vel
#for subset: linflat=pos_subset
encoder = OfflinePPEncoder(linflat=pos_all_linear_vel, dec_spk_amp=decode_spikes_random_trial, encode_settings=encode_settings, 
                           decode_settings=decode_settings, enc_spk_amp=encode_spikes_random_trial, dask_worker_memory=1e9,
                           dask_chunksize = None)

#new output format to call results, prob_no_spike, and trans_mat for doing single tetrode encoding
encoder_output = encoder.run_encoder()
results = encoder_output['results']
prob_no_spike = encoder_output['prob_no_spike']
trans_mat = encoder_output['trans_mat']

# now we need to save these three variables - .npy files?
# results
# prob_no_spike
# trans_mat

time_table.to_csv('/p/lustre1/coulter5/remy/time_stamp2.csv')
print('Enocder finished!')
print('Encoder started at: ',datetime.fromtimestamp(os.path.getmtime('/p/lustre1/coulter5/remy/time_stamp1.csv')).strftime('%Y-%m-%d %H:%M:%S'))
print('Encoder finished at: ',datetime.fromtimestamp(os.path.getmtime('/p/lustre1/coulter5/remy/time_stamp2.csv')).strftime('%Y-%m-%d %H:%M:%S'))
print("Encoder finished!", file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))
print('Encoder started at: ',datetime.fromtimestamp(os.path.getmtime('/p/lustre1/coulter5/remy/time_stamp1.csv')).strftime('%Y-%m-%d %H:%M:%S'), file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))
print('Encoder finished at: ',datetime.fromtimestamp(os.path.getmtime('/p/lustre1/coulter5/remy/time_stamp2.csv')).strftime('%Y-%m-%d %H:%M:%S'), file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))


print("End of script!")
print("End of script!", file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))
print(" ", file=open("/p/lustre1/coulter5/remy/1d_decoder_log.txt","a"))
