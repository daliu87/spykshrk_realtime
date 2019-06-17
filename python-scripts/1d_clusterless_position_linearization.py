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
path_main = '/home/mcoulter/spykshrk_realtime'
os.chdir(path_main)
from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, Posteriors, \
        EncodeSettings, pos_col_format, SpikeObservation, RippleTimes, DayEpochEvent, DayEpochTimeSeries
from spykshrk.franklab.pp_decoder.util import normal_pdf_int_lookup, gaussian, apply_no_anim_boundary, normal2D
from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder
from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
from spykshrk.util import Groupby


#cell 2
# Import data

# Define path bases 
path_base_rawdata = '/mnt/vortex/mcoulter/raw_data/'

# Define parameters
# for epochs we want 2 and 4 for each day
rat_name = 'gus'
directory_temp = path_base_rawdata + rat_name + '/'
day_dictionary = {'remy':[20], 'gus':[30], 'bernard':[23], 'fievel':[19]}
epoch_dictionary = {'remy':[4], 'gus':[2], 'bernard':[4], 'fievel':[4]} 
tetrodes_dictionary = {'remy': [4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
                       'gus': [6,7,8,9,10,11,12,17,18,19,20,21,24,25,26,27,30], # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]
                       'bernard': [1,2,3,4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
                       'fievel': [1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,27,28,29]}
# Maze information
os.chdir('/home/mcoulter/spykshrk_realtime/')
# new maze coordinates with only one segment for box
linearcoord = scipy.io.loadmat('set_arm_nodes.mat')['linearcoord_one_box'][0]

print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]))

## Define path bases
path_base_dayepoch = 'day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0])
path_base_analysis = '/mnt/vortex/mcoulter/maze_info/'

speed_threshold_save = 0

# Define path base
#path_base_timewindow = str(int(round(linear_start))) + 'to' + str(int(round(linear_end))) + 'sec'
path_base_timewindow = 'whole_epoch_v3'
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

    from loren_frank_data_processing.position import _get_pos_dataframe 

    from loren_frank_data_processing import Animal
    ANIMALS = {'gus': Animal(directory='/mnt/vortex/mcoulter/raw_data/gus', short_name='gus')}
    epoch_key = ('gus', 30, 2)
    position_info = _get_pos_dataframe(epoch_key, ANIMALS)

    center_well_position = linearcoord[0][0]
    nodes = [center_well_position[np.newaxis, :]]

    for arm in linearcoord: 
        for point in arm[1:]:
            nodes.append(point[np.newaxis, :])
    nodes = np.concatenate(nodes)

    dist = []
    for arm in linearcoord:
        dist.append(np.linalg.norm(np.diff(arm, axis=0), axis=1))
    np.stack([*dist])

    edges = [(0, 1),(1, 2),(0, 3),(3, 4),(0, 5),(5, 6),(0, 7),(7, 8),
             (0, 9),(9, 10),(0, 11),(11, 12),(0, 13),(13, 14),(0, 15),(15, 16)]
    edge_distances = np.concatenate([*dist])

    import networkx as nx

    from loren_frank_data_processing.track_segment_classification import plot_track
    track_graph = nx.Graph()
    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))
    for edge, distance in zip(edges, edge_distances):
        track_graph.add_edge(edge[0], edge[1], distance=distance)
    plot_track(track_graph)

    from loren_frank_data_processing.track_segment_classification import classify_track_segments
    position = position_info.loc[:, ['x_position', 'y_position']].values
    #position = position[0:10000]
    print(position.shape)

    track_segment_id = classify_track_segments(
        track_graph, position,
        route_euclidean_distance_scaling=1,
        sensor_std_dev=1)
    print(track_segment_id.shape)

    from loren_frank_data_processing.track_segment_classification import calculate_linear_distance
    center_well_id = 0
    linear_distance = calculate_linear_distance(
            track_graph, track_segment_id, center_well_id, position)

    # this section calculates the shift amounts for each arm
    arm_distances = (edge_distances[1],edge_distances[3],edge_distances[5],edge_distances[7],
                    edge_distances[9],edge_distances[11],edge_distances[13],edge_distances[15])
    hardcode_shiftamount = 20

    shift_linear_distance_by_arm_dictionary = dict() # initialize empty dictionary 

    hardcode_armorder = [0,1,2,3,4,5,6,7]

    for arm in enumerate(hardcode_armorder): # for each outer arm
        if arm[0] == 0: # if first arm, just shift hardcode_shiftamount
            temporary_variable_shift = hardcode_shiftamount 
        else: # if not first arm, add to hardcode_shiftamount length of previous arm 
            temporary_variable_shift = hardcode_shiftamount + arm_distances[arm[0]] + shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm[0] - 1]]
        shift_linear_distance_by_arm_dictionary[arm[1]] = temporary_variable_shift

    # Modify: 1) collapse non-arm locations (segments 0-7), 
    # 2) shift linear distance for the 8 arms (segments 8-15)
    newseg = np.copy(track_segment_id)
    newseg[(newseg < 8)] = 0

    # 2) Shift linear distance for each arm 
    linear_distance_arm_shift = np.copy(linear_distance)
    for seg in shift_linear_distance_by_arm_dictionary:
        linear_distance_arm_shift[(newseg==(seg+8))]+=shift_linear_distance_by_arm_dictionary[(seg)]  
    # Incorporate modifications 

    #whole epoch - i dont think we need to change pos_all_linear if just generating the linearization files
    #pos_all_linear['linpos_flat']=linear_distance_arm_shift

    # Store some linearization results in python format for quick loading (pos_subset) 
    np.save(linearization_output1_save_filename, linear_distance_arm_shift)
    np.save(linearization_output2_save_filename, track_segment_id)
    
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
    export_this = AttrDict({'linearization_segments': edges,
                            'linearization_nodes_coordinates': nodes,
                            'linearization_nodes_distance_to_back_well':arm_distances,
                            'linearization_shift_segments_list': linearization_shift_segments_list,
                            'linearization_position_segments':track_segment_id,
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
    track_segment_id = np.load(linearization_output2_save_filename)
    #pos_subset['linpos_flat'] = linear_distance_arm_shift[(encode_subset_start-encode_subset_start):(encode_subset_end-encode_subset_start+1)]
    #whole_epoch
    pos_all_linear['linpos_flat']=linear_distance_arm_shift

print(datetime.now())
#print(datetime.now(), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

print("End of script!")
#print("End of script!", file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
#print(" ", file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
