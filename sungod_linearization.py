# function definitions for linearizing SunGod
# based on Loren Frank Data Processing
# written by Jen 
# Jan 2019, MEC

import os
import numpy as np
import scipy as sp
import scipy.stats as ss
import scipy.io
import networkx as nx
import loren_frank_data_processing as lfdp
import scipy.io as sio # for saving .mat files 
import inspect # for inspecting files (e.g. finding file source)

import json
import functools


def createTrackGraph(maze_coordinates):
    #linearcoord = maze_coordinates['linearcoord_NEW'].squeeze()
    linearcoord = maze_coordinates['linearcoord_one_box'].squeeze()
    track_segments = [np.stack((arm[:-1], arm[1:]), axis=1) for arm in linearcoord]
    center_well_position = track_segments[0][0][0]
    track_segments, center_well_position = (np.unique(np.concatenate(track_segments), axis=0),
                center_well_position) # ## what does redefining center_well_position here do?
    nodes = np.unique(track_segments.reshape((-1, 2)), axis=0)
    edges = np.zeros(track_segments.shape[:2], dtype=int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id
    edge_distances = np.linalg.norm(
        np.diff(track_segments, axis=-2).squeeze(), axis=1)
    track_graph = nx.Graph()
    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))
    for edge, distance in zip(edges, edge_distances):
        track_graph.add_edge(edge[0], edge[1], distance=distance)
    center_well_id = np.unique(
        np.nonzero(np.isin(nodes, center_well_position).sum(axis=1) > 1)[0])[0]
    return track_graph, track_segments, center_well_id

def hack_determinearmorder(track_segments): 
    # order arm segments based on y position. ASSUMES CERTAIN LAYOUT OF TRACK_SEGMENTS. 
    d_temp = [] 
    for track_segment in enumerate(track_segments):
        if track_segment[0] < 8:
            d_temp.append(track_segment[1][1,1])

    rank = ss.rankdata(d_temp) - 1 # - 1 to account for python indexing

    order1 = [None]*len(rank)
    for r in enumerate(rank):
        order1[int(r[1])] = int(r[0])
    return(order1)

def turn_array_into_ranges(array1):
    array1_diff = np.ediff1d(array1)

    start_temp = [] 
    end_temp = []
    start_temp.append(array1[0]) 

    some_end_indices = np.where(array1_diff > 1)

    for i in range(len(some_end_indices[0])):
        # This is always an end index
        end_temp.append(array1[some_end_indices[0][i]])
        if array1[some_end_indices[0][i]] == start_temp[i]: # if this is the same as the last start index, it was already added as a start index-- don't need to add it again
            start_temp.append(array1[some_end_indices[0][i] + 1]) # define next start index   
        elif array1_diff[some_end_indices[0][i] - 1] > 1: # if last value was more than 1 away, this is also a start index
            start_temp.append(array1[some_end_indices[0][i]])    
        else: # if last value was NOT more than 1 away, this is JUST an end index, and next start index is next index
            start_temp.append(array1[some_end_indices[0][i] + 1])   
    # The last entry in array is always the last end index
    end_temp.append(array1[-1])  

    return start_temp, end_temp

# Function to define chunked data 
def chunk_data(data,number_of_chunks):
    print('chunking data of length',len(data),'samples into',str(number_of_chunks),'chunk(s)')
    # Takes 1D data and splits into number of chunks (as equally as possible)
    
    # Calculate number of data points per chunk
    datapoints_per_chunk = math.ceil(len(data)/number_of_chunks)
    print('datapoints_per_chunk:',datapoints_per_chunk)
    
    # Initialize empty list for chunked data
    chunked_data = [] 
    
    # Define chunks
    for chunk_number in range(number_of_chunks): # for each chunk
        chunked_data.append(data[chunk_number*datapoints_per_chunk:(chunk_number + 1)*datapoints_per_chunk]) 
    
    return chunked_data

    # Toy example
    #hi = np.concatenate((np.ones(5),np.ones(5)*2),axis=0)
    #print(hi)
    #chunked_data = []
    #chunk_number = 2
    #for i in range(10): 
    #    x = hi[chunk_number*i:chunk_number*(i + 1)]
    #    print(x)
    #    chunked_data.append(x)
    #print(chunked_data)

def change_to_directory_make_if_nonexistent(directory_path):
    # Make directory if it doesn't exist
    if os.path.exists(directory_path) == False:
        print('making path ' + directory_path)
        os.chdir('/')        
        os.makedirs(directory_path)
    # Change to directory 
    os.chdir(directory_path)
    # Print working directory
    os.getcwd()
    