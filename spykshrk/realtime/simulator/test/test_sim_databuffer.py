from unittest import TestCase

from trodes.FSData.nspike_data import AnimalInfo, EEGDataStream, SpkDataStream, PosMatDataStream
from trodes.FSData.sim_databuffer import SimDataBuffer

from franklab.franklab_data import FrankAnimalInfo, LFPData, RawPosData, SpikeAmpData
from trodes.FSData.franklab_datastream import LFPDataStream, RawPosDataStream, SpikeDataStream


def nspike_animal_test_init():
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


class TestSimDataBuffer(TestCase):

    def test_nspike_SimDataBuffer(self):
        anim = FrankAnimalInfo('/opt/data36/jason/', 'kanye')

        eeg = EEGDataStream(anim, 100)
        spk = SpkDataStream(anim, 100)
        pos = PosMatDataStream(anim, 1000)

        databuffer = SimDataBuffer([eeg(), pos()])

        last_timestamp = 0
        for datacount, data in enumerate(databuffer()):
            self.assertTrue(data.timestamp >= last_timestamp)
            last_timestamp = data.timestamp

        self.assertTrue(datacount == 462559)

    def test_franklab_SimDataBuffer(self):
        anim = FrankAnimalInfo('/opt/data36/jason/', 'kanye')

        lfp_data = LFPData(anim, '20160426')
        lfp_stream = LFPDataStream(lfp_data=lfp_data, epoch=2, ntrodes=list(range(1, 17)))

        pos_data = RawPosData(anim, '20160426')
        pos_stream = RawPosDataStream(pos_data=pos_data, epoch=2, recon_type='online')

        databuffer = SimDataBuffer([lfp_stream(), pos_stream()])

        last_timestamp = 0
        for datacount, data in enumerate(databuffer()):
            self.assertTrue(data.timestamp >= last_timestamp)
            last_timestamp = data.timestamp

        print(datacount)
