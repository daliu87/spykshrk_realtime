import numpy as np
import sys
from time import time

from spykshrk.realtime.datatypes import LFPPoint, SpikePoint, LinPosPoint


class SimDataBuffer:
    """ This class implements the interleaving of multiple data streams
    based on each data stream sample's timestamps.

    Data streams are currently iterators that have a preset
    data return format.

    Each sample is then packaged in a unique message that can be iteratively
    streams in chronological order by setting up a generator.
    """

    def __init__(self, iter_list):
        self.iter_list = iter_list

    def __call__(self):
        data_point_list = [None] * len(self.iter_list)
        timestamp_list = [sys.maxsize] * len(self.iter_list)

        cur_iter_list = self.iter_list

        for ind, iter_item in enumerate(cur_iter_list):
            try:
                data_point_list[ind] = next(iter_item)
                timestamp_list[ind] = data_point_list[ind].timestamp
            except StopIteration:
                print('Iterator #{} finished.'.format(ind))
                del cur_iter_list[ind]
                del data_point_list[ind]
                del timestamp_list[ind]

        while cur_iter_list:
            try:
                min_ind = timestamp_list.index(min(timestamp_list))
                yield data_point_list[min_ind]

                data_point_list[min_ind] = next(cur_iter_list[min_ind])
                timestamp_list[min_ind] = data_point_list[min_ind].timestamp

            except StopIteration:
                print('Iterator #{} finished.'.format(min_ind))
                del cur_iter_list[min_ind]
                del data_point_list[min_ind]
                del timestamp_list[min_ind]


class SimDataBufferSlow:
    def __init__(self, spkdata_itr, eegdata_itr, posdata_itr):
        """
        Each iterator will be incremented as is, no error checking.
        Each iterator will be advanced in order of their returned element
        timestamp.

        Iterators must conform to a particular format.
        """
        self.spkdata_itr = spkdata_itr
        self.eegdata_itr = eegdata_itr
        self.posdata_itr = posdata_itr

    def __call__(self):
        """ Generator that will chonologically return spk, eeg and pos messages

            If timestamps are the same between different streams, assume the
            return message is in a random order from the streams

        Possible return types:
            spk_buffer
            eeg_buffer
            pos_buffer
        """

        # hard coded spk, eeg, and pos data streams.  Maybe change
        # to more flexible architecture if necessary

        # signals for when a stream is out of data
        spk_done_flag = False
        eeg_done_flag = False
        pos_done_flag = False

        spk_next = True
        eeg_next = True
        pos_next = True

        timestamp_list = [sys.maxsize] * 3

        print('TB_DATABUFFER: Begin streaming')
        while not spk_done_flag or not eeg_done_flag or not pos_done_flag:
            if spk_next and not spk_done_flag:
                try:
                    spk_tp = next(self.spkdata_itr)
                    timestamp_list[0] = spk_tp.timestamp
                    spk_next = False
                except StopIteration as ex:
                    spk_done_flag = True
                    spk_tp = SpikePoint(sys.maxsize, 0, 0)
                    timestamp_list[0] = spk_tp.timestamp
                    print('Spike data done streaming:')
            if eeg_next and not eeg_done_flag:
                try:
                    eeg_tp = next(self.eegdata_itr)
                    timestamp_list[1] = eeg_tp.timestamp
                    eeg_next = False
                except StopIteration as ex:
                    eeg_done_flag = True
                    eeg_tp = LFPPoint(sys.maxsize, 0, 0)
                    timestamp_list[1] = eeg_tp.timestamp
                    print('EEG data done streaming:')
            if pos_next and not pos_done_flag:
                try:
                    pos_tp = next(self.posdata_itr)
                    timestamp_list[2] = pos_tp.timestamp
                    pos_next = False
                except StopIteration as ex:
                    pos_done_flag = True
                    pos_tp = LinPosPoint(sys.maxsize, 0)
                    timestamp_list[2] = pos_tp.timestamp
                    print('Pos data done streaming:')

    #         next_data_idx = timestamp_list.index(min(timestamp_list))

    #         if next_data_idx == 0 and not spk_done_flag:
    #             spk_next = True
    #             yield spk_tp
    #         elif next_data_idx == 1 and not eeg_done_flag:
    #             eeg_next = True
    #             yield eeg_tp
    #         elif next_data_idx == 2 and not pos_done_flag:
    #             pos_next = True
    #             yield pos_tp
