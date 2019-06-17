# python script to run 1d clusterless decoder on sungod data using spykshrk
# written by MEC from notebooks written by AKG and JG
# 3-6-19
# this version includes support for linearizing the whole epoch (use pos_all_linear instead of pos_subset)

#cell 1
# Setup and import packages
import sys
sys.path.append('/usr/workspace/wsb/coulter5/spykshrk_realtime')
import os
import pdb
from datetime import datetime
import trodes2SS
from trodes2SS import AttrDict, TrodesImport, convert_dan_posterior_to_xarray
import sungod_linearization
from sungod_linearization import createTrackGraph, hack_determinearmorder, turn_array_into_ranges, chunk_data, change_to_directory_make_if_nonexistent
from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, Posteriors, EncodeSettings, pos_col_format, SpikeObservation, RippleTimes, DayEpochEvent, DayEpochTimeSeries
from spykshrk.franklab.pp_decoder.util import normal_pdf_int_lookup, gaussian, apply_no_anim_boundary, normal2D
from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder
#from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
from spykshrk.util import Groupby
import numpy as np
import scipy.io
import scipy as sp
import pandas as pd
import loren_frank_data_processing as lfdp
import scipy.io as sio # for saving .mat files 

def main(path_base_rawdata, rat_name, path_arm_nodes, path_base_analysis, shift_amt, path_out):
    # set log file name
    #log_file = '/p/lustre1/coulter5/remy/1d_decoder_log.txt'
    print(datetime.now())
    #print(datetime.now(), file=open(log_file,"a"))

    # set path to folders where spykshrk core scripts live
    #path_main = '/usr/workspace/wsb/coulter5/spykshrk_realtime'
    #os.chdir(path_main)

    #cell 2
    # Import data

    # Define path bases 
    #path_base_rawdata = '/p/lustre1/coulter5/remy/'

    # Define parameters
    # for epochs we want 2 and 4 for each day
    #shifts = [0, .10, .15, .20]
    #shifts = [0]
    #for shift_amt in shifts:
    #rat_name = 'remy'
    print(rat_name)
    #print(rat_name, file=open(log_file,"a"))
    
    directory_temp = path_base_rawdata + rat_name + '/'
    day_dictionary = {'remy':[20], 'gus':[28], 'bernard':[23], 'fievel':[19]}
    epoch_dictionary = {'remy':[2], 'gus':[4], 'bernard':[4], 'fievel':[4]} 
    tetrodes_dictionary = {'remy': [4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
                           'gus': [6,7,8,9,10,11,12,17,18,19,20,21,24,25,26,27,30], # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]
                           'bernard': [1,2,3,4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
                           'fievel': [1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,27,28,29]}
    #tetrodes_dictionary = {'remy': [4], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
     #                      'gus': [6], # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]
      #                     'bernard': [1],
       #                    'fievel': [1]}
                                                  
    # Maze information
    #os.chdir('/usr/workspace/wsb/coulter5/spykshrk_realtime/')
    #maze_coordinates = scipy.io.loadmat('set_arm_nodes.mat',variable_names = 'linearcoord_NEW')
    # new maze coordinates with only one segment for box
    maze_coordinates = scipy.io.loadmat(os.path.join(path_arm_nodes, 'set_arm_nodes.mat'), variable_names = 'linearcoord_one_box')

    print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]))
    #print('Lodaing raw data! '+str(rat_name)+' Day '+str(day_dictionary[rat_name])+' Epoch '+str(epoch_dictionary[rat_name]), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

    datasrc = TrodesImport(directory_temp, rat_name, day_dictionary[rat_name], 
                           epoch_dictionary[rat_name], tetrodes_dictionary[rat_name])
    # Import marks
    marks = datasrc.import_marks()
    # # os.chdir('/data2/jguidera/data/')
    # # np.load('marks.npy')

    # add print lines to show number of marks on each tetrode
    #print('Marks on tetrode 4: ', marks.xs(4,level='elec_grp_id').shape)
    #print('Marks on tetrode 4: ', marks.xs(4,level='elec_grp_id').shape, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
    #print('Marks on tetrode 28: ', marks.xs(28,level='elec_grp_id').shape)
    #print('Marks on tetrode 28: ', marks.xs(28,level='elec_grp_id').shape, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
    #print('Marks on tetrode 30: ', marks.xs(30,level='elec_grp_id').shape)
    #print('Marks on tetrode 30: ', marks.xs(30,level='elec_grp_id').shape, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

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
    #path_base_analysis = '/p/lustre1/coulter5/remy/maze_info/'

    #cell 3
    #filter ripples for velocity < 4
    #re-shape ripples input table into format for get_irregular_resample
    rips['timestamp'] = rips['starttime']
    rips['time'] = rips['starttime']
    rips.timestamp = rips.timestamp*30000
    rips['timestamp'] = rips['timestamp'].astype(int)
    rips.reset_index(level=['event'], inplace=True)
    rips.columns = ['event','starttime','endtime','maxthresh','timestamp','time']
    rips.set_index(['timestamp', 'time'], drop=True, append=True, inplace=True)

    #filter for velocity < 4 with get_irregular_resample
    linflat_obj = pos.get_mapped_single_axis()
    linflat_ripindex = linflat_obj.get_irregular_resampled(rips)
    linflat_ripindex_encode_velthresh = linflat_ripindex.query('linvel_flat < 4')

    #re-shape to RippleTimes format for plotting
    rips_vel_filt = rips.loc[linflat_ripindex_encode_velthresh.index]
    rips_vel_filt.reset_index(level=['timestamp','time'], inplace=True)
    rips_vel_filt.set_index(['event'], drop=True, append=True, inplace=True)
    rips_vel_filtered = RippleTimes.create_default(rips_vel_filt, 1)

    print('rips when animal velocity <= 4: '+str(linflat_ripindex_encode_velthresh.shape[0]))
    #print('rips when animal velocity <= 4: '+str(linflat_ripindex_encode_velthresh.shape[0]), file=open(log_file,"a"))

    #cell 4
    # dont run encoding or decdoing subset cells for the crossvalidation runs
    # the marks filtering happens right before running encoder

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
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    #change_to_directory_make_if_nonexistent(directory_path)

    # Define name of linearization result
    linearization_output1_save_filename = os.path.join(directory_path, 'linearization_' + path_base_timewindow + '_speed' + str(speed_threshold_save) + '_linear_distance_arm_shift' + '.npy')
    linearization_output2_save_filename = os.path.join(directory_path, 'linearization_' + path_base_timewindow + '_speed' + str(speed_threshold_save) + '_track_segment_id_use' + '.npy')
    # If linearization result doesn't exist, do linearization calculation
    if os.path.exists(linearization_output1_save_filename) == False:
        print('Linearization result doesnt exist. Doing linearization calculation')
        #print("Linearization result doesnt exist. Doing linearization calculation.", file=open(log_file,"a"))

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
        linear_distance_nodes = lfdp.track_segment_classification.calculate_linear_distance(track_graph, track_segment_id_nodes, center_well_id, node_coords)

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
        linear_distance = lfdp.track_segment_classification.calculate_linear_distance(track_graph, track_segment_id_use, center_well_id, simplepos.T)

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
        #change_to_directory_make_if_nonexistent(linearization_output_save_path)
        # Define file name 
        file_name_temp = os.path.join(linearization_output_save_path, rat_name + '_day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0]) + '_' + path_base_timewindow +'_speed' + str(speed_threshold_save) + '_linearization_variables.mat')

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
        #if os.path.exists(file_name_temp[0]) == True:
        #    input("Press Enter to overwrite file")
        #    print('overwriting')
        # Save 
        print('saving file:',file_name_temp)
        #print('saving file:',file_name_temp, file=open(log_file,"a"))
        sio.savemat(file_name_temp,export_this)
        
    # If linearization result exists, load it 
    else:
        print('Linearization result exists. Loading it.')
        #print("Linearization result exists. Loading it.", file=open(log_file,"a"))
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
    #print('Arm coordinates: ',arm_coordinates_WEWANT, file=open(log_file,"a"))

    #cell 7.1
    # this cell speeds up encoding with larger position bins
    # try 5cm bins - do this by dividing position subset by 5 and arm coords by 5

    #pos_subset['linpos_flat'] = (pos_subset['linpos_flat'])/5
    #whole epoch
    pos_all_linear['linpos_flat'] = (pos_all_linear['linpos_flat'])/5

    arm_coordinates_WEWANT = arm_coordinates_WEWANT/5
    arm_coordinates_WEWANT = np.around(arm_coordinates_WEWANT)
    print('Arm coordinates: ',arm_coordinates_WEWANT)
    #print('Arm coordinates: ',arm_coordinates_WEWANT, file=open(log_file,"a"))

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
                                'arm_coordinates': arm_coordinates_WEWANT,
                                'path_trans_mat': path_arm_nodes}) # includes box, removes bins in the gaps 'arm_coordinates': [[0,max_pos]]})

    print('Encode settings: ',encode_settings)
    #print('Encode settings: ',encode_settings, file=open(log_file,"a"))

    #cell 9
    #define decode settings
    decode_settings = AttrDict({'trans_smooth_std': 2,
                                'trans_uniform_gain': 0.0001,
                                'time_bin_size':60})

    print('Decode settings: ',decode_settings)
    #print('Decode settings: ',decode_settings, file=open(log_file,"a"))

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
    #print('Number of trials: ',trialsindex.shape, file=open(log_file,"a"))

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
    #print('Randomized trial order: ',trialsindex_shuffled, file=open(log_file,"a"))

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
    #print('Original encode length: ',random_trial_marks_all.shape, file=open(log_file,"a"))
    print('Encoding marks non-negative filter: ',marks_random_trial_non_negative.shape)
    #print('Encoding marks non-negative filter: ',marks_random_trial_non_negative.shape, file=open(log_file,"a"))

    random_trial_spk_subset_sparse = trodes2SS.threshold_marks(marks_random_trial_non_negative, maxthresh=2000,minthresh=100)
    print('original length: '+str(marks_random_trial_non_negative.shape[0]))
    print('after filtering: '+str(random_trial_spk_subset_sparse.shape[0]))
    #print('original length: '+str(marks_random_trial_non_negative.shape[0]), file=open(log_file,"a"))
    #print('after filtering: '+str(random_trial_spk_subset_sparse.shape[0]), file=open(log_file,"a"))

    # velocity filter to define encoding and decoding times
    velocity_filter = 4
    print('Velocity filter: ',velocity_filter)
    #print('Velocity filter: ',velocity_filter, file=open(log_file,"a"))

    #NOTE: to try marks shift on whole trials we need to do shift first, then velocity filter for encoding and decoding marks
    # nope - cant do this, need to do velocity filter first

    # #encoding spikes
    linflat_obj = random_trial_pos_all.get_mapped_single_axis()

    #linflat_obj = pos_all_linear.get_mapped_single_axis()
    linflat_spkindex = linflat_obj.get_irregular_resampled(random_trial_spk_subset_sparse)
    linflat_spkindex_encode_velthresh = linflat_spkindex.query('linvel_flat > @velocity_filter')

    encode_spikes_random_trial = random_trial_spk_subset_sparse.loc[linflat_spkindex_encode_velthresh.index]

    print('encoding spikes after velocity filter: '+str(encode_spikes_random_trial.shape[0]))
    #print('encoding spikes after velocity filter: '+str(encode_spikes_random_trial.shape[0]), file=open(log_file,"a"))

    # #decoding spikes
    linflat_obj = random_trial_pos_all.get_mapped_single_axis()
    #linflat_obj = pos.get_mapped_single_axis()
    linflat_spkindex = linflat_obj.get_irregular_resampled(random_trial_spk_subset_sparse)
    linflat_spkindex_decode_velthresh = linflat_spkindex.query('linvel_flat < @velocity_filter')

    decode_spikes_random_trial = random_trial_spk_subset_sparse.loc[linflat_spkindex_decode_velthresh.index]

    print('decoding spikes after velocity filter: '+str(decode_spikes_random_trial.shape[0]))
    #print('decoding spikes after velocity filter: '+str(decode_spikes_random_trial.shape[0]), file=open(log_file,"a"))

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
    #print('Total epoch time (sec) ',epoch_time, file=open(log_file,"a"))  

    # dont use firing rate to calculate the size of the shift, instead find the index at 25% of epoch, eg
    #tetrode_x_firing_rate = []
    #for i in range(len(starttimes_shuffled)):
    #    length = len(encode_spikes_all.loc[(encode_spikes_all.index.get_level_values('time') <= endtimes_shuffled[i]) & (encode_spikes_all.index.get_level_values('time') >= starttimes_shuffled[i])])
    #    tetrode_x_firing_rate.append(length/(endtimes_shuffled[i]-starttimes_shuffled[i]))
    #len(tetrode_x_firing_rate)
    
    #np.median(tetrode_x_firing_rate)

    # roll each tetrode channel by the shift amount 
    # shift amount: 5 mins + 5 sec (1/2 period) = 305 sec
    # shift amount for whole trial: 5 mins + 18 sec (1/2 period) = 318 - cant do this
    # new shift amount: 25% of epoch
    #marks_index_shift = int(305*np.median(tetrode_x_firing_rate))

    # # shift by 25% of the epoch time
    # note shift_amt is defined above in the outer loop
    marks_index_target = min_time + shift_amt*epoch_time
    encode_spikes_shift_query = encode_spikes_random_trial.reset_index()
    marks_index_shift = encode_spikes_shift_query.iloc[(encode_spikes_shift_query['time']-marks_index_target).abs().argsort()[:1]].index.item()

    print('Marks index shift: ',marks_index_shift)
    #print('Marks index shift: ',marks_index_shift, file=open(log_file,"a"))

    # save dataframe with both shifted and original marks
    encode_spikes_random_trial_save = encode_spikes_random_trial
    encode_spikes_random_trial_save['c00_shift'] = np.roll(encode_spikes_random_trial_save['c00'],-(marks_index_shift))
    encode_spikes_random_trial_save['c01_shift'] = np.roll(encode_spikes_random_trial_save['c01'],-(marks_index_shift))
    encode_spikes_random_trial_save['c02_shift'] = np.roll(encode_spikes_random_trial_save['c02'],-(marks_index_shift))
    encode_spikes_random_trial_save['c03_shift'] = np.roll(encode_spikes_random_trial_save['c03'],-(marks_index_shift)) 

    shifted_marks_file_name = '/p/lustre1/coulter5/remy/' + rat_name + '_' + str(day_dictionary[rat_name][0]) + '_' + str(epoch_dictionary[rat_name][0]) + '_vel4_convol_new_pos_marks_shuffle_' + str(marks_index_shift) + '_marks_4_15_19.nc'
    marks_time_shift2 = encode_spikes_random_trial_save.reset_index()
    marks_time_shift3 = marks_time_shift2.to_xarray()
    marks_time_shift3.to_netcdf(shifted_marks_file_name)
    print('Saved shifted marks to: '+shifted_marks_file_name)
    #print('Saved shifted marks to: '+shifted_marks_file_name, file=open(log_file,"a"))
 
    # apply shift to tetrode channel columns in original dataframe
    encode_spikes_random_trial = []
    encode_spikes_random_trial = random_trial_spk_subset_sparse.loc[linflat_spkindex_encode_velthresh.index]
    encode_spikes_random_trial['c00'] = np.roll(encode_spikes_random_trial['c00'],-(marks_index_shift))
    encode_spikes_random_trial['c01'] = np.roll(encode_spikes_random_trial['c01'],-(marks_index_shift))
    encode_spikes_random_trial['c02'] = np.roll(encode_spikes_random_trial['c02'],-(marks_index_shift))
    encode_spikes_random_trial['c03'] = np.roll(encode_spikes_random_trial['c03'],-(marks_index_shift))  

    # shift all marks - NOPE
    #marks_time_shift_all_input =  marks_time_shift.drop(columns=['timestamp_original','time_original'])

    print('Shifted marks shape: ',encode_spikes_random_trial.shape)
    #print('Shifted marks shape: ',encode_spikes_random_trial.shape, file=open(log_file,"a"))

    # re-order enoc

    #cell 9.25
    # apply velocity filter to shifted marks - nope cant do this - we want to decode the same spikes 

    #cell 9.3 shuffle position by set number of position time bins - use variable called shift_in_30Hz_time_bins
    # when we shifted position, shift was 9150
    offset_30Hz_time_bins = 0
    #print('Position shuffle offset: ',offset_30Hz_time_bins)
    #print('Position shuffle offset: ',offset_30Hz_time_bins, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

    #random_trial_pos_all_vel_shuffle = random_trial_pos_all_vel

    #comment out next line to turn off shuffle
    #random_trial_pos_all_vel_shuffle['linpos_flat'] = np.roll(random_trial_pos_all_vel_shuffle['linpos_flat'],-(offset_30Hz_time_bins))

    # version of shifted position for saving - includes original position
    # random_trial_pos_all_vel_shuffle_save = random_trial_pos_all_vel
    # random_trial_pos_all_vel_shuffle_save['linpos_flat_original'] = random_trial_pos_all_vel_shuffle_save['linpos_flat']
    # random_trial_pos_all_vel_shuffle_save['linpos_flat'] = np.roll(random_trial_pos_all_vel_shuffle_save['linpos_flat'],-(offset_30Hz_time_bins))
    # shifted_position_file_name = '/data2/mcoulter/linearized_position/' + rat_name + '_' + str(day_dictionary[rat_name][0]) + '_' + str(epoch_dictionary[rat_name][0]) + '_position_shuffle_9150_position_4_10_19.nc'
    # position_shift2 = random_trial_pos_all_vel_shuffle_save.reset_index()
    # position_shift3 = position_shift2.to_xarray()
    # position_shift3.to_netcdf(shifted_position_file_name)
    # print('Saved shifted position to: '+shifted_position_file_name)
    # print('Saved shifted position to: '+shifted_position_file_name, file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

    #Note: i think we want to use the shuffled position for occupancy normalization - this is most similar to the normal decoder

    #cell 10
    # Run encoder
    # these time-table lines are so that we can record the time it takes for encoder to run even if notebook disconnects
    # look at the time stamps for the two files in /data2/mcoulter called time_stamp1 and time_stamp2
    print('Starting encoder')
    #print("Starting encoder", file=open(log_file,"a"))
    #time_table_data = {'age': [1, 2, 3, 4, 5]}
    #time_table = pd.DataFrame(time_table_data)
    #time_table.to_csv('/p/lustre1/coulter5/remy/time_stamp1.csv')
    time_started = datetime.now()

    #for whole epoch: linflat=pos_all_linear_vel
    #for subset: linflat=pos_subset
    encoder = OfflinePPEncoder(linflat=random_trial_pos_all_vel, dec_spk_amp=decode_spikes_random_trial, encode_settings=encode_settings, decode_settings=decode_settings, enc_spk_amp=encode_spikes_random_trial, dask_worker_memory=1e9, dask_chunksize = None)

    #new output format to call results, prob_no_spike, and trans_mat for doing single tetrode encoding
    encoder_output = encoder.run_encoder()
    results = encoder_output['results']
    prob_no_spike = encoder_output['prob_no_spike']
    trans_mat = encoder_output['trans_mat']

    #results = encoder.run_encoder()

    #time_table.to_csv('/p/lustre1/coulter5/remy/time_stamp2.csv')
    time_finished =datetime.now()

    print('Enocder finished!')
    #print('Encoder started at: ',datetime.fromtimestamp(os.path.getmtime('/p/lustre1/coulter5/remy/time_stamp1.csv')).strftime('%Y-%m-%d %H:%M:%S'))
    print('Encoder started at: %s'%str(time_started))
    print('Encoder finished at: %s'%str(time_finished))
    #print("Encoder finished!", file=open(log_file,"a"))
    #print('Encoder started at: ',datetime.fromtimestamp(os.path.getmtime('/p/lustre1/coulter5/remy/time_stamp1.csv')).strftime('%Y-%m-%d %H:%M:%S'), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))
    #print('Encoder finished at: ',datetime.fromtimestamp(os.path.getmtime('/p/lustre1/coulter5/remy/time_stamp2.csv')).strftime('%Y-%m-%d %H:%M:%S'), file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

    #cell 11
    #make observations table from results
    # if the master script has the list of all tetrodes then this cell should be able to combine the results table from each tetrode

    tet_ids = np.unique(decode_spikes_random_trial.index.get_level_values('elec_grp_id'))
    observ_tet_list = []
    grp = decode_spikes_random_trial.groupby('elec_grp_id')
    for tet_ii, (tet_id, grp_spk) in enumerate(grp):
        tet_result = results[tet_ii]
        tet_result.set_index(grp_spk.index, inplace=True)
        observ_tet_list.append(tet_result)

    observ = pd.concat(observ_tet_list)
    observ_obj = SpikeObservation.create_default(observ.sort_index(level=['day', 'epoch', 'timestamp', 'elec_grp_id']), encode_settings)

    observ_obj['elec_grp_id'] = observ_obj.index.get_level_values('elec_grp_id')
    observ_obj.index = observ_obj.index.droplevel('elec_grp_id')

    # add a small offset to observations table to prevent division by 0 when calculating likelihoods
    # this is currently hard-coded for 5cm position bins -> 147 total bins
    observ_obj.loc[:,'x000':'x146'] = observ_obj.loc[:,'x000':'x146'].values + np.spacing(1)

    #cell 11.1
    #make prob_no_spike dictionary from individual tetrodes
    # if the master script has the list of all tetrodes then this cell should be able to combine the results table from each tetrode

    #this will take in prob_no_spike from several differnt encoder runs, each for a single tetrode, the dictionaries should be named
    # 'prob_no_spike_[tet number]'
    #tet_ids = [prob_no_spike_26,prob_no_spike_28,prob_no_spike_29,prob_no_spike_30]
    #prob_no_spike_all = tet_ids[0]
    #for tet_id in tet_ids[1:]:
    #    prob_no_spike_all.update(tet_id)

    #cell 13
    # save observations
    #observ_obj._to_hdf_store('/data2/mcoulter/fievel_19_2_observations_whole_epoch.h5','/analysis', 
    #                         'decode/clusterless/offline/observ_obj', 'observ_obj')
    #print('Saved observations to /data2/mcoulter/fievel_19_2_observations_whole_epoch.h5')
    #print('Saved observations to /data2/mcoulter/fievel_19_2_observations_whole_epoch.h5', file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

    #cell 14
    # load previously generated observations
    # hacky but reliable way to load a dataframe stored as hdf
    # Posteriors is imported from data_containers
    #observ_obj = Posteriors._from_hdf_store('/data2/mcoulter/remy_20_4_observ_obj_0_20000.h5','/analysis', 
    #                         'decode/clusterless/offline/observ_obj', 'observ_obj')

    #load prob_no_spike - this is a dictionary
    #probability_no_spike = np.load('/mnt/vortex/mcoulter/prob_no_spike.npy').item()

    #load transition matrix - this is an array
    #transition_matrix = np.load('/mnt/vortex/mcoulter/trans_mat.npy')

    #cell 15
    # Run PP decoding algorithm
    # NOTE 1-11-19 had to add spk_amp and vel to encode settings in order for decoding to run
    # what should these be set to? and why are they here now?
    time_bin_size = 60
    decode_settings = AttrDict({'trans_smooth_std': 2,
                                'trans_uniform_gain': 0.0001,
                                'time_bin_size':60})

    encode_settings = AttrDict({'sampling_rate': 3e4,
                                'pos_bins': np.arange(0,max_pos,1), # arm_coords_wewant
                                'pos_bin_edges': np.arange(0,max_pos + .1,1), # edges_wewant, 
                                'pos_bin_delta': 1, 
                                # 'pos_kernel': sp.stats.norm.pdf(arm_coords_wewant, arm_coords_wewant[-1]/2, 1),
                                'pos_kernel': sp.stats.norm.pdf(np.arange(0,max_pos,1), max_pos/2, 1), #note that the pos_kernel mean should be half of the range of positions (ie 180/90) # sp.stats.norm.pdf(np.arange(0,560,1), 280, 1),    
                                'pos_kernel_std': 1, 
                                'mark_kernel_std': int(20), 
                                'pos_num_bins': max_pos, # len(arm_coords_wewant)
                                'pos_col_names': [pos_col_format(ii, max_pos) for ii in range(max_pos)], # [pos_col_format(int(ii), len(arm_coords_wewant)) for ii in arm_coords_wewant],
                                'arm_coordinates': arm_coordinates_WEWANT, # 'arm_coordinates': [[0,max_pos]]})
                                'spk_amp': 60,
                                'vel': 0})

    #when running the encoder and decoder at same time: trans_mat=encoder.trans_mat['flat_powered']
    #AND  prob_no_spike=encoder.prob_no_spike
    #when loading a previsouly generated observations table use: trans_mat=transition_matrix
    # AND prob_no_spike=probability_no_spike

    print('Starting decoder')
    #print("Starting decoder", file=open(log_file,"a"))
    decoder = OfflinePPDecoder(observ_obj=observ_obj, trans_mat=encoder.trans_mat['flat_powered'], 
                               prob_no_spike=encoder.prob_no_spike,
                               encode_settings=encode_settings, decode_settings=decode_settings, 
                               time_bin_size=time_bin_size, all_linear_position=pos_all_linear, velocity_filter=4)

    posteriors = decoder.run_decoder()
    print('Decoder finished!')
    #print('Decoder finished!', file=open(log_file,"a"))
    print('Posteriors shape: '+ str(posteriors.shape))
    #print('Posteriors shape: '+ str(posteriors.shape), file=open(log_file,"a"))

    #cell 15.1
    # reorder posteriors and position to restore original trial order (undo the randomization)

    #cell 16
    #save posteriors with hdf
    #posteriors._to_hdf_store('/data2/mcoulter/posteriors/fievel_19_2_whole_epoch.h5','/analysis', 
    #                         'decode/clusterless/offline/posterior', 'learned_trans_mat')
    #print('Saved posteriors to /vortex/mcoulter/posteriors/fievel_19_2_whole_epoch.h5')
    #print('Saved posteriors to /vortex/mcoulter/posteriors/fievel_19_2_whole_epoch.h5', file=open("/data2/mcoulter/1d_decoder_log.txt","a"))

    #cell 17
    #load previously generated posteriors from hdf
    #posteriors = Posteriors._from_hdf_store('/data2/mcoulter/posteriors/remy_20_4_linearized_alltime_decode.h5','/analysis',
    #                                        'decode/clusterless/offline/posterior', 'learned_trans_mat')

    #cell 18 saving posteriors as netcdf instead of hdf
    # to export posteriors to MatLab
    # add ripple labels to posteriors and then convert posteriors to xarray then save as netcdf
    # this requires folding multiindex into posteriors dataframe first

    posterior_file_name = os.path.join(path_out,  rat_name + '_' + str(day_dictionary[rat_name][0]) + '_' + str(epoch_dictionary[rat_name][0]) + '_vel4_mask_convol_new_pos_yes_random_marks_shuffle_' + str(marks_index_shift) + '_posteriors_4_19_19.nc')

    post1 = posteriors.apply_time_event(rips_vel_filtered, event_mask_name='ripple_grp')
    post2 = post1.reset_index()
    #post3 = post2.to_xarray()
    post3 = convert_dan_posterior_to_xarray(post2, tetrodes_dictionary[rat_name], velocity_filter, encode_settings, decode_settings, trans_mat, offset_30Hz_time_bins, trialsindex_shuffled, marks_index_shift)
    #print(len(post3))
    post3.to_netcdf(posterior_file_name)
    print('Saved posteriors to '+posterior_file_name)
    #print('Saved posteriors to '+posterior_file_name, file=open(log_file,"a"))

    # to export linearized position to MatLab: again convert to xarray and then save as netcdf

    position_file_name = os.path.join(path_out, rat_name + '_' + str(day_dictionary[rat_name][0]) + '_' + str(epoch_dictionary[rat_name][0]) + '_vel4_mask_convol_new_pos_yes_random_marks_shuffle_' + str(marks_index_shift) + '_linearposition_4_19_19.nc')

    linearized_pos1 = pos_all_linear.apply_time_event(rips_vel_filtered, event_mask_name='ripple_grp')
    linearized_pos2 = linearized_pos1.reset_index()
    linearized_pos3 = linearized_pos2.to_xarray()
    linearized_pos3.to_netcdf(position_file_name)
    print('Saved linearized position to '+position_file_name)
    #print('Saved linearized position to '+position_file_name, file=open(log_file,"a"))

    print("End of script!")
    #print("End of script!", file=open(log_file,"a"))
    #print(" ", file=open(log_file,"a"))

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='path_base_rawdata', help='Base path to raw data')
    parser.add_argument('-n', action='store', dest='rat_name', help='Rat Name')
    parser.add_argument('-a', action='store', dest='path_arm_nodes', help='Path to directory with arm_nodes and simple_transition_matrix files')
    parser.add_argument('-l', action='store', dest='path_base_linearization', help='Base path to linearization')
    parser.add_argument('-s', action='store', dest='shift_amt', type=float, help='Shift amount')
    parser.add_argument('-o', action='store', dest='path_out', help='Path to output')
    results = parser.parse_args()

    main(results.path_base_rawdata, results.rat_name, results.path_arm_nodes, results.path_base_linearization, results.shift_amt, results.path_out)
