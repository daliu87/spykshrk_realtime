import getopt
import sys
import os
import logging
import json
import spykshrk.realtime.binary_record_cy as bin_rec_cy
import multiprocessing as mp
import multiprocessing.sharedctypes
import pandas as pd
import uuid
import pickle
import cProfile
import time

run_config = None
hdf5_lock = None
shared_mem = None
shared_view = None


def init_shared_mem(config, arr, hdf5_lock_local):
    global run_config
    global shared_mem
    global shared_view
    global hdf5_lock
    run_config = config
    shared_mem = arr
    shared_view = memoryview(arr).cast('B')
    hdf5_lock = hdf5_lock_local


def binrec_to_pandas(binrec: bin_rec_cy.BinaryRecordsFileReader):

    binrec.start_record_reading()
    panda_dict = binrec.convert_pandas()

    return panda_dict


def merge_pandas(byte_range):
    pandas_item = pickle.loads(shared_view[byte_range[0]:byte_range[1]])
    rec_id = pandas_item[0]
    pandas = pandas_item[1]
    merged = pd.concat(pandas, ignore_index=True)
    merged = merged.apply(pd.to_numeric, errors='ignore')

    if 'timestamp' in merged.columns:
        merged.sort_values(['timestamp'], inplace=True)
        merged.reset_index(drop=True, inplace=True)

    hdf5_lock.acquire()

    hdf5_filename = os.path.join(run_config['files']['output_dir'],
                                 '{}.rec_merged.h5'.format(run_config['files']['prefix']))

    with pd.HDFStore(hdf5_filename, 'w') as hdf_store:
        hdf_store['rec_{}'.format(rec_id)] = merged

    hdf5_lock.release()

    # merged_bytes = pickle.dumps((rec_id, merged))
    # merged_bytes_size = len(merged_bytes)

    # shared_view[byte_range[0]:(byte_range[0] + merged_bytes_size)] = merged_bytes

    # return byte_range[0], (byte_range[0] + merged_bytes_size)


def main(argv):

    logging.getLogger().setLevel('INFO')

    try:
        opts, args = getopt.getopt(argv, "", ["config="])
    except getopt.GetoptError:
        logging.error('Usage: ...')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--config':
            config_filename = arg

    config = json.load(open(config_filename, 'r'))

    logging.info("Initializing BinaryRecordsFileReaders.")

    bin_list = []
    total_bin_size = 0
    for rec_mpi_rank in config['rank_settings']['enable_rec']:
        try:
            binrec = bin_rec_cy.BinaryRecordsFileReader(save_dir=config['files']['output_dir'],
                                                        file_prefix=config['files']['prefix'],
                                                        mpi_rank= rec_mpi_rank,
                                                        manager_label='state',
                                                        file_postfix=config['files']['rec_postfix'],
                                                        filemeta_as_col=False)
            bin_list.append(binrec)
            total_bin_size += binrec.getsize()
        except FileNotFoundError as ex:
            logging.warning('Binary record file not found, skipping: {}'.format(ex.filename))

    # Increase size for panda tables
    total_bin_size = int(total_bin_size * 1.25)

    shared_arr = mp.sharedctypes.RawArray('B', total_bin_size)
    shared_arr_view = memoryview(shared_arr).cast('B')

    hdf5_lock_local = mp.Lock()

    p = mp.Pool(20, initializer=init_shared_mem, initargs=[config, shared_arr, hdf5_lock_local], maxtasksperchild=1)
    logging.info("Converting binary record files into panda dataframes.")
    start_time = time.time()
    pandas_list = p.map(binrec_to_pandas, bin_list)
    end_time = time.time()
    logging.info("Done converting record files into dataframes"
                 ", took {:.01f} seconds ({:.02f} minutes).".format(end_time - start_time,
                                                                    (end_time - start_time)/60.))

    logging.info("Collecting individual record types and putting into shared memory.")
    start_time = time.time()

    pandas_dict = {rec_id: [] for rec_id in pandas_list[0].keys()}
    for rec_pandas in pandas_list:
        for rec_id, df in rec_pandas.items():
            pandas_dict[rec_id].append(df)

    start_byte = 0
    byte_ranges = []
    for pan_item in pandas_dict.items():
        pickled_pan_item = pickle.dumps(pan_item)
        pickled_pan_size = len(pickled_pan_item)
        shared_arr_view[start_byte:(start_byte+pickled_pan_size)] = pickled_pan_item
        byte_ranges.append((start_byte, start_byte + pickled_pan_size))
        # Set small gap between lists
        start_byte += int(1.005*pickled_pan_size)

    end_time = time.time()
    logging.info("Done collecting and putting dataframes into "
                 "shared memory, {:.01f} seconds ({:.02f} minutes).".format(end_time - start_time,
                                                                            (end_time - start_time)/60.))

    logging.info("Merging, sorting and saving each record type's dataframe.")
    start_time = time.time()
    p.map(merge_pandas, byte_ranges)
    #pandas_merged_address_ranges = p.map(merge_pandas, byte_ranges)
    end_time = time.time()
    logging.info("Done merging and sorting and saving all records,"
                 " {:.01f} seconds ({:.02f} minutes).".format(end_time - start_time,
                                                              (end_time - start_time)/60.))



if __name__ == '__main__':
    cProfile.runctx('main(sys.argv[1:])', globals=globals(), locals=locals(), filename='pstats')


