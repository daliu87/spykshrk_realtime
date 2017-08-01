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


def init_shared_mem(config, hdf5_lock_local):
    global run_config
    global hdf5_lock
    run_config = config
    hdf5_lock = hdf5_lock_local


def binrec_to_pandas(binrec: bin_rec_cy.BinaryRecordsFileReader):

    binrec.start_record_reading()
    panda_dict = binrec.convert_pandas()

    hdf5_temp_filename = os.path.join('/tmp', str(uuid.uuid4()) + '.h5')
    with pd.HDFStore(hdf5_temp_filename, 'w') as hdf5_store:
        filename_dict = {}
        for rec_id, df in panda_dict.items():
            if df.size > 0:
                filename_dict[rec_id] = hdf5_temp_filename
                hdf5_store['rec_'+str(rec_id)] = df

    return filename_dict


def merge_pandas(filename_items):

    rec_id = filename_items[0]
    filenames = filename_items[1]

    pandas = []

    for filename in filenames:
        store = pd.HDFStore(filename)
        pandas.append(store['rec_'+str(rec_id)])

    merged = pd.concat(pandas, ignore_index=True)
    merged = merged.apply(pd.to_numeric, errors='ignore')

    if 'timestamp' in merged.columns:
        merged.sort_values(['timestamp'], inplace=True)
        merged.reset_index(drop=True, inplace=True)

    hdf5_lock.acquire()

    hdf5_filename = os.path.join(run_config['files']['output_dir'],
                                 '{}.rec_merged.h5'.format(run_config['files']['prefix']))

    with pd.HDFStore(hdf5_filename) as hdf_store:
        hdf_store['rec_{}'.format(rec_id)] = merged

    hdf5_lock.release()


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
    # total_bin_size = int(total_bin_size * 1.25)

    # shared_arr = mp.sharedctypes.RawArray('B', total_bin_size)
    # shared_arr_view = memoryview(shared_arr).cast('B')

    hdf5_lock_local = mp.Lock()

    p = mp.Pool(20, initializer=init_shared_mem, initargs=[config, hdf5_lock_local], maxtasksperchild=1)
    logging.info("Converting binary record files into panda dataframes.")
    start_time = time.time()
    file_list = p.map(binrec_to_pandas, bin_list)
    end_time = time.time()
    logging.info("Done converting record files into dataframes"
                 ", took {:.01f} seconds ({:.02f} minutes).".format(end_time - start_time,
                                                                    (end_time - start_time)/60.))

    remapped_dict = {}
    for rec_files in file_list:
        for rec_id, filename in rec_files.items():
            rec_list = remapped_dict.setdefault(rec_id, [])
            rec_list.append(filename)

    logging.info("Merging, sorting and saving each record type's dataframe.")
    start_time = time.time()
    p.map(merge_pandas, remapped_dict.items())
    end_time = time.time()
    logging.info("Done merging and sorting and saving all records,"
                 " {:.01f} seconds ({:.02f} minutes).".format(end_time - start_time,
                                                              (end_time - start_time)/60.))


if __name__ == '__main__':
    cProfile.runctx('main(sys.argv[1:])', globals=globals(), locals=locals(), filename='pstats')


