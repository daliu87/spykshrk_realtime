from unittest import TestCase

from spykshrk.realtime.simulator.nspike_data import AnimalInfo, EEGDataStream, SpkDataStream, PosMatDataStream


class TestDataStreamTestAnimal(TestCase):

    def setUp(self):
        anim_dir = '/home/daliu/data/'

        timescale = 10000

        animal_name = 'test'
        days = [2]
        tetrodes = [5, 11, 12, 14, 19]

        epoch_encode = [1]
        new_data = True
        self.anim = AnimalInfo(base_dir=anim_dir,
                               name=animal_name,
                               days=days,
                               tetrodes=tetrodes,
                               epochs=epoch_encode,
                               timescale=timescale,
                               new_data=new_data)

    def test_EEGDataStream(self):

        eeg = EEGDataStream(self.anim)

        eeg_itr = eeg()
        buffer_count = 0
        last_timestamp = 0
        for buffer in eeg_itr:
            buffer_count += 1
            self.assertTrue(buffer.timestamp >= last_timestamp)
            self.assertTrue(buffer.ntrode_id in self.anim.tetrodes,
                            msg='ntrode {:} not in anim tetrode list.'.format(buffer.ntrode_id, self.anim.tetrodes))
            last_timestamp = buffer.timestamp
        self.assertTrue(buffer_count == 449914, 'buffer count was {:}, should have been 449914'.format(buffer_count))

    def test_SpkDataStream(self):

        spk = SpkDataStream(self.anim, 1000)

        spk_itr = spk()
        buffer_count = 0
        last_timestamp = 0
        for buffer in spk_itr:
            buffer_count += 1
            self.assertTrue(buffer.timestamp >= last_timestamp)
            last_timestamp = buffer.timestamp
        self.assertTrue(buffer_count == 10552)

    def test_PosMatDataStream(self):

        pos = PosMatDataStream(self.anim, 10000)

        pos_itr = pos()
        buffer_count = 0
        last_timestamp = 0
        for buffer in pos_itr:
            buffer_count += 1
            self.assertTrue(buffer.timestamp >= last_timestamp)
            last_timestamp = buffer.timestamp

        self.assertTrue(buffer_count == 2068)
