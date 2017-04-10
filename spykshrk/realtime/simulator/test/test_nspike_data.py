from unittest import TestCase

from trodes.FSData.nspike_data import AnimalInfo, EEGDataStream, SpkDataStream, PosMatDataStream


def animal_test_init():
    anim_dir = '/opt/data36/daliu/other/mkarlsso/'

    timescale = 10000

    animal_name = 'test'
    days = [2]
    tetrodes = [5, 11, 12, 14, 19]
    tetrodes_ca1 = [5, 11, 12, 14, 19]

    epoch_encode = [1]
    new_data = True
    anim = AnimalInfo(animal_dir=anim_dir,
                      animal_name=animal_name,
                      days=days,
                      tetrodes=tetrodes,
                      tetrodes_ca1=tetrodes_ca1,
                      epoch_encode=epoch_encode,
                      timescale=timescale,
                      new_data=new_data)
    return anim


class TestEEGDataStream(TestCase):

    def test_EEGDataStream(self):
        anim = animal_test_init()

        eeg = EEGDataStream(anim, 1000)

        eeg_itr = eeg()
        buffer_count = 0
        last_timestamp = 0
        for buffer in eeg_itr:
            buffer_count += 1
            self.assertTrue(buffer.timestamp >= last_timestamp)
            last_timestamp = buffer.timestamp
        self.assertTrue(buffer_count == 449914)


class TestSpkDataStream(TestCase):

    def test_SpkDataStream(self):
        anim = animal_test_init()

        spk = SpkDataStream(anim, 1000)

        spk_itr = spk()
        buffer_count = 0
        last_timestamp = 0
        for buffer in spk_itr:
            buffer_count += 1
            self.assertTrue(buffer.timestamp >= last_timestamp)
            last_timestamp = buffer.timestamp
        self.assertTrue(buffer_count == 10552)


class TestPosMatDataStream(TestCase):

    def test_PosMatDataStream(self):
        anim = animal_test_init()

        pos = PosMatDataStream(anim, 10000)

        pos_itr = pos()
        buffer_count = 0
        last_timestamp = 0
        for buffer in pos_itr:
            buffer_count += 1
            self.assertTrue(buffer.timestamp >= last_timestamp)
            last_timestamp = buffer.timestamp

        self.assertTrue(buffer_count == 2068)
