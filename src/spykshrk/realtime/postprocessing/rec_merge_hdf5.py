import getopt
import sys
import os
import logging
import json
import spykshrk.realtime.binary_record_cy as bin_rec_cy
import multiprocessing
import pandas as pd
import pickle
import uuid
import cProfile

def binrec_to_pandas(binrec: bin_rec_cy.BinaryRecordsFileReader):

    binrec.start_record_reading()
    panda_dict = binrec.convert_pandas()
    filename = os.path.join('/tmp', str(uuid.uuid4()))
    file = open(filename, 'wb')
    pickle.dump(panda_dict, file=file)

    return filename


def merge_pandas(filename):
    file = open(filename, 'rb')
    pandas_item = pickle.load(file)
    rec_id = pandas_item[0]
    pandas = pandas_item[1]
    merged = pd.concat(pandas, ignore_index=True)
    merged = merged.apply(pd.to_numeric, errors='ignore')

    if 'timestamp' in merged.columns:
        merged.sort_values(['timestamp'], inplace=True)
        merged.reset_index(drop=True, inplace=True)

    output_filename = os.path.join('/tmp', str(uuid.uuid4()))
    output_file = open(output_filename, 'wb')
    pickle.dump((rec_id, merged), output_file)

    return output_filename


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "", ["config="])
    except getopt.GetoptError:
        logging.error('Usage: ...')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--config':
            config_filename = arg

    config = json.load(open(config_filename, 'r'))

    bin_list = []
    for rec_mpi_rank in config['rank_settings']['enable_rec']:
        try:
            print('Processing rank {}'.format(rec_mpi_rank))
            binrec = bin_rec_cy.BinaryRecordsFileReader(save_dir=config['files']['output_dir'],
                                                        file_prefix=config['files']['prefix'],
                                                        mpi_rank= rec_mpi_rank,
                                                        manager_label='state',
                                                        file_postfix=config['files']['rec_postfix'],
                                                        filemeta_as_col=False)
            bin_list.append(binrec)
        except FileNotFoundError as ex:
            logging.warning('Binary record file not found, skipping: {}'.format(ex.filename))

    p = multiprocessing.Pool(20)
    pickled_file_list = p.map(binrec_to_pandas, bin_list)

    pandas_list = []
    for filename in pickled_file_list:
        file = open(filename, 'rb')
        pandas_list.append(pickle.load(file))

    pandas_dict = {rec_id: [] for rec_id in pandas_list[0].keys()}
    for rec_pandas in pandas_list:
        for rec_id, df in rec_pandas.items():
            pandas_dict[rec_id].append(df)

    merge_filename_list = []
    for pan_item in pandas_dict.items():
        filename = os.path.join('/tmp', str(uuid.uuid4()))
        file = open(filename, 'wb')
        pickle.dump(pan_item, file)
        merge_filename_list.append(filename)

    pandas_merged_filenames = p.map(merge_pandas, merge_filename_list)

    pandas_merged = []
    for filename in pandas_merged_filenames:
        file = open(filename, 'rb')
        pandas_merged.append(pickle.load(file))

    hdf_store = pd.HDFStore(os.path.join(config['files']['output_dir'],
                                         '{}.rec_merged.h5'.format(config['files']['prefix'])))

    for rec_id, rec_df in pandas_merged:
        hdf_store['rec_{}'.format(rec_id)] = rec_df

    hdf_store.close()


if __name__ == '__main__':
    cProfile.runctx('main(sys.argv[1:])', globals=globals(), locals=locals(), filename='pstats')
