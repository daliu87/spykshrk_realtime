# python script to linearize SunGod position using Eric's LFDP
# written by MEC from notebooks written by AKG and JG
# 4-4-19
# this version includes support for linearizing the whole epoch (use pos_all_linear instead of pos_subset)

#cell 1
# Setup and import packages
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

print("Linearization script!")
#print("Linearization script!", file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

print(datetime.now())
#print(datetime.now(), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

# set path to folders where spykshrk core scripts live
#path_main = '/home/mcoulter/spykshrk_realtime'
#os.chdir(path_main)
#from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, Posteriors, \
#        EncodeSettings, pos_col_format, SpikeObservation, RippleTimes, DayEpochEvent, DayEpochTimeSeries
#from spykshrk.franklab.pp_decoder.util import normal_pdf_int_lookup, gaussian, apply_no_anim_boundary, normal2D
#from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder
#from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
#from spykshrk.util import Groupby


#cell 2
# Import data

# Define path bases - set this to the directory with the position file
path_base_rawdata = '/data2/mcoulter/raw_data/'

# Define parameters
# for epochs we want 2 and 4 for each day
rat_name = 'remy'
directory_temp = path_base_rawdata + rat_name + '/'
day_dictionary = {'remy':[20], 'gus':[28], 'bernard':[12], 'fievel':[15]}
epoch_dictionary = {'remy':[2], 'gus':[4], 'bernard':[2], 'fievel':[2]} 
tetrodes_dictionary = {'remy': [4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
                       'gus': [6,7,8,9,10,11,12,17,18,19,20,21,24,25,26,27,30], # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]
                       'bernard': [1,2,3,4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
                       'fievel': [1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,27,28,29]}

# Maze information - set this to the directory with set_arm_nodes.mat
os.chdir('/home/mcoulter/spykshrk_realtime/')
#maze_coordinates = scipy.io.loadmat('set_arm_nodes.mat',variable_names = 'linearcoord_NEW')
# new maze coordinates with only one segment for box
maze_coordinates = scipy.io.loadmat('set_arm_nodes.mat',variable_names = 'linearcoord_one_box')

print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]))
#print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

datasrc = TrodesImport(directory_temp, rat_name, day_dictionary[rat_name], 
                       epoch_dictionary[rat_name], tetrodes_dictionary[rat_name])


# Import position 
# Temporary small definition of encoding settings-- need 'arm_coordinates' to use datasrc.import_pos 
encode_settings = AttrDict({'arm_coordinates': [[0,0]]})
pos = datasrc.import_pos(encode_settings, xy='x')
posY = datasrc.import_pos(encode_settings, xy='y')

# Define path bases - set this to directory where linearization results will be saved
path_base_dayepoch = 'day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0])
path_base_analysis = '/data2/mcoulter/maze_info/'

#cell 6
# Linearization
# linearize the whole epoch - should only have to do this once.

encode_settings = AttrDict({'arm_coordinates': [[0,0]]})
speed_threshold_save = 0
pos_all_linear = datasrc.import_pos(encode_settings, xy='x')
posY1 = datasrc.import_pos(encode_settings, xy='y')

# Define path base
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
if os.path.exists(linearization_output1_save_filename) == False:
    print('Linearization result doesnt exist. Doing linearization calculation')
    #print("Linearization result doesnt exist. Doing linearization calculation.", file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

    # Prepare for linearization 
    
    # Create graph elements
    track_graph, track_segments, center_well_id = createTrackGraph(maze_coordinates)
    #track_segments = lfdp.track_segment_classification.get_track_segments_from_graph(track_graph)

    # Define shift amounts 
    # 1-13-19 trying 10cm bins with flat transition matrix, set hardcode_shiftamount to 20
    # **** 
    hardcode_armorder = hack_determinearmorder(track_segments) # add progressive stagger in this order
    hardcode_shiftamount = 20 # add this stagger to sum of previous shifts
    # ****
    linearization_arm_lengths = []
    # Caculate length of outer arms, plot
    for track_segment in enumerate(track_segments): # for each track segment
        #plt.plot(track_segment[1][:,0],track_segment[1][:,1]) # plot track segment
        #plt.text(track_segment[1][0,0],track_segment[1][0,1],str(track_segment[0])) # label with segment number
        # Calculate and plot length of outer arms 
        if track_segment[0] < 8: # if an outer arm, calculate length 
            linearization_arm_lengths.append(np.linalg.norm(track_segment[1][0,:] - track_segment[1][1,:])) # calculate length
            #plt.text(track_segment[1][0,0],track_segment[1][0,1] - 4,str(linearization_arm_lengths[track_segment[0]])) # text to show length 
    # Define dictionary for shifts for each arm segment
    shift_linear_distance_by_arm_dictionary = dict() # initialize empty dictionary 
    for arm in enumerate(hardcode_armorder): # for each outer arm
        if arm[0] == 0: # if first arm, just shift hardcode_shiftamount
            temporary_variable_shift = hardcode_shiftamount 
        else: # if not first arm, add to hardcode_shiftamount length of previous arm 
            temporary_variable_shift = hardcode_shiftamount + linearization_arm_lengths[arm[0]] + shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm[0] - 1]]
        shift_linear_distance_by_arm_dictionary[arm[1]] = temporary_variable_shift
        
    
    # Pull node coordinates (store as node_coords)
    temp2 = [] # list for node coordinates
    for eachnode in track_graph.nodes: # for each node
        temp = list(track_graph.nodes[eachnode]['pos'])
        temp2.append(temp)
    node_coords = np.asarray(temp2)
    # Assign nodes to track segments
    track_segment_id_nodes = lfdp.track_segment_classification.find_nearest_segment(track_segments, node_coords)

    # Calculate linear distance of nodes to back well 
    linear_distance_nodes = lfdp.track_segment_classification.calculate_linear_distance(
            track_graph, track_segment_id_nodes, center_well_id, node_coords)

    # Linearize position
    #pos_subset_linear = pos.loc[(pos.index.get_level_values('time') <= linear_end) & (pos.index.get_level_values('time') >= linear_start)]
    #posY_subset_linear = posY.loc[(posY.index.get_level_values('time') <= linear_end) & (posY.index.get_level_values('time') >= linear_start)] 

    # Vector with position
    #simplepos = np.vstack([pos_subset_linear['linpos_flat'],posY_subset_linear['linpos_flat']]) # x pos, y pos
    # whole epoch
    simplepos = np.vstack([pos_all_linear['linpos_flat'],posY1['linpos_flat']])

    # Store under different name to plot for debugging 
    simplepos_original = simplepos 

    # Assign each position to a track segment
    # ****
    sensor_std_dev = 1 # 10
    assign_track_segments_one_is_Markov_two_is_naive = 2 # 1 for hidden markov model, 2 for naive
    # ****
    # Define back well
    #center_well_id = 17
    center_well_id = 16
    # HIDDEN MARKOV MODEL:
    # Assign position to track segment
    track_segment_id = lfdp.track_segment_classification.classify_track_segments(track_graph,
                                simplepos.T, sensor_std_dev=sensor_std_dev, route_euclidean_distance_scaling=1)
    # SIMPLER WAY: 
    #track_segments = lfdp.track_segment_classification.get_track_segments_from_graph(track_graph)
    track_segment_id_naive = lfdp.track_segment_classification.find_nearest_segment(track_segments, simplepos.T)
    # Choose track segment assignment 
    if assign_track_segments_one_is_Markov_two_is_naive == 1:
        track_segment_id_use = track_segment_id
    elif assign_track_segments_one_is_Markov_two_is_naive == 2:   
        track_segment_id_use = track_segment_id_naive
    # Find linear distance of position from back well 
    linear_distance = lfdp.track_segment_classification.calculate_linear_distance(track_graph, 
                                 track_segment_id_use, center_well_id, simplepos.T)

    # Modify: 1) collapse non-arm locations, 2) shift linear distance for the 8 arms
    newseg = np.copy(track_segment_id_use)
    # 1) Collapse non-arm locations
    # newseg[(newseg < 16) & (newseg > 7)] = 8
    # newseg[(newseg == 16)] = 9
    #try making one segment for box
    newseg[(newseg < 17) & (newseg > 7)] = 8
    
    # 2) Shift linear distance for each arm 
    linear_distance_arm_shift = np.copy(linear_distance)
    for seg in shift_linear_distance_by_arm_dictionary:
        linear_distance_arm_shift[(newseg==seg)]+=shift_linear_distance_by_arm_dictionary[seg]  
    # Incorporate modifications 

    #pos_subset['linpos_flat']=linear_distance_arm_shift[(encode_subset_start-encode_subset_start):(encode_subset_end-encode_subset_start+1)]
    #whole epoch
    pos_all_linear['linpos_flat']=linear_distance_arm_shift

    # Store some linearization results in python format for quick loading (pos_subset) 
    np.save(linearization_output1_save_filename, linear_distance_arm_shift)
    np.save(linearization_output2_save_filename, track_segment_id_use)
    
    # Save some linearization results in .mat file
    # Convert dictionary with shift for each arm to array since matlab can't read the dictionary 
    linearization_shift_segments_list = []
    for key in shift_linear_distance_by_arm_dictionary:
        temp = [key,shift_linear_distance_by_arm_dictionary[key]]
        linearization_shift_segments_list.append(temp)    
    # Change directory
    change_to_directory_make_if_nonexistent(linearization_output_save_path)
    # Define file name 
    file_name_temp = [rat_name + '_day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0]) + 
                      '_' + path_base_timewindow +
                      '_speed' + str(speed_threshold_save) + 
                      '_linearization_variables.mat']    

    # Store variables 
    export_this = AttrDict({'linearization_segments': track_segments,
                            'linearization_nodes_coordinates': node_coords,
                            'linearization_nodes_distance_to_back_well':linear_distance_nodes,
                            'linearization_shift_segments_list': linearization_shift_segments_list,
                            'linearization_position_segments':track_segment_id_use,
                            'linearization_position_distance_from_back_well':linear_distance,
                            'linearization_position_distance_from_back_well_arm_shift':linear_distance_arm_shift
                           })
    # Warn before overwriting file 
    if os.path.exists(file_name_temp[0]) == True:
        input("Press Enter to overwrite file")
        print('overwriting')
    # Save 
    print('saving file:',file_name_temp)
    #print('saving file:',file_name_temp, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
    sio.savemat(file_name_temp[0],export_this)
    
# If linearization result exists, load it 
else:
    print('Linearization result exists. Loading it.')
    #print("Linearization result exists. Loading it.", file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
    linear_distance_arm_shift = np.load(linearization_output1_save_filename)
    track_segment_id_use = np.load(linearization_output2_save_filename)
    #pos_subset['linpos_flat'] = linear_distance_arm_shift[(encode_subset_start-encode_subset_start):(encode_subset_end-encode_subset_start+1)]
    #whole_epoch
    pos_all_linear['linpos_flat']=linear_distance_arm_shift

print(datetime.now())
#print(datetime.now(), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

print("End of script!")
#print("End of script!", file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
#print(" ", file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
