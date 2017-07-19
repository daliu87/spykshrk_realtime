import getopt
import sys
import os
import logging
import json
import spykshrk.realtime.binary_record_cy as bin_rec_cy
import multiprocessing
import pandas as pd


def binrec_to_pandas(binrec: bin_rec_cy.BinaryRecordsFileReader):
    binrec.start_record_reading()
    return binrec.convert_pandas()


def merge_pandas(pandas_item):
    rec_id = pandas_item[0]
    pandas = pandas_item[1]
    merged = pd.concat(pandas, ignore_index=True)
    merged = merged.apply(pd.to_numeric, errors='ignore')

    if 'timestamp' in merged.columns:
        merged.sort_values(['timestamp'], inplace=True)
        merged.reset_index(drop=True, inplace=True)

    return rec_id, merged


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
    pandas_list = p.map(binrec_to_pandas, bin_list)

    pandas_dict = {rec_id: [] for rec_id in pandas_list[0].keys()}
    for rec_pandas in pandas_list:
        for rec_id, df in rec_pandas.items():
            pandas_dict[rec_id].append(df)

    pandas_merged = p.map(merge_pandas, pandas_dict.items())

    hdf_store = pd.HDFStore(os.path.join(config['files']['output_dir'],
                                         '{}.rec_merged.h5'.format(config['files']['prefix'])))

    for rec_id, rec_df in pandas_merged:
        hdf_store['rec_{}'.format(rec_id)] = rec_df

    hdf_store.close()


if __name__ == '__main__':
    main(sys.argv[1:])
