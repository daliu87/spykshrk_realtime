import numpy as np
from mpi4py import MPI
from spikegadgets import trodesnetwork as tnp

from spykshrk.realtime import realtime_base as realtime_base, datatypes as datatypes
from spykshrk.realtime.simulator.simulator_process import SimulatorError


class TrodesDataReceiver(realtime_base.DataSourceReceiver):
    """Class that receives data from trodes using its network api
    """
    def __init__(self, comm: MPI.Comm, rank, config, datatype):
        super().__init__(comm=comm, rank=rank, config=config, datatype=datatype)
        self.start = False
        self.stop = False
        self.timestamp = 0
        self.network = config['trodes_network']['networkobject']
        self.channels = []
        if self.datatype is datatypes.Datatypes.LFP:
            self.DataPointCls = datatypes.LFPPoint

        elif self.datatype is datatypes.Datatypes.SPIKES:
            self.DataPointCls = datatypes.SpikePoint

        elif self.datatype is datatypes.Datatypes.LINEAR_POSITION:
            self.DataPointCls = datatypes.LinearPosPoint

        else:
            raise SimulatorError('{} is not a valid datatype.'.format(self.datatype))

    def register_datatype_channel(self, channel):
        self.channels.append(channel)

    def start_all_streams(self):
        if self.datatype is datatypes.Datatypes.LFP:
            chnls = [str(i) for i in self.channels]
            self.datastream = self.network.subscribeLFPData(300, chnls)
            self.datastream.initialize()
            self.buf = self.datastream.create_numpy_array()
            self.curntrode = -1
            self.subbedntrodes = len(chnls)

        elif self.datatype is datatypes.Datatypes.SPIKES:
            chnls = [str(i)+',0' for i in self.channels]
            self.datastream = self.network.subscribeSpikesData(300, chnls)
            self.datastream.initialize()
            self.buf = self.datastream.create_numpy_array()

        elif self.datatype is datatypes.Datatypes.LINEAR_POSITION:
            self.datastream = self.network.subscribeHighFreqData('PositionData', 'CameraModule', 20)
            self.datastream.initialize()
            ndtype = self.datastream.getDataType().dataFormat
            nbytesize = self.datastream.getDataType().byteSize
            bytesbuf = memoryview(bytes(nbytesize))
            self.buf = np.frombuffer(bytesbuf, dtype=np.dtype(ndtype))

        else:
            raise SimulatorError('{} is not a valid datatype.'.format(self.datatype))
        self.start = True

    def stop_all_streams(self):
        self.start = False

    def stop_iterator(self):
        self.stop = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration()

        if not self.start:
            return None

        #if is lfp and curntrode is less than total subbed ntrodes
        if self.datatype is datatypes.Datatypes.LFP:
            # if haven't gotten anything yet, curntrode = -1. if already sent all in one packet, curntrode = subbedntrodes
            if self.curntrode > 0 and self.curntrode < self.subbedntrodes:
                pt = datatypes.LFPPoint(self.timestamp.trodes_timestamp, self.channels[self.curntrode], self.channels[self.curntrode], self.buf[self.curntrode])
                self.curntrode = self.curntrode + 1
                return pt, None


        n = self.datastream.available(0)
        if n:
            byteswritten = 0
            systime = tnp.systemTimeMSecs()
            if self.datatype is datatypes.Datatypes.LFP:
                self.timestamp = self.datastream.getData()
                # Reset curntrode value. If lfp buffer is more than 1, then above code will read from buffer before reading from Trodes stream
                self.curntrode = 0
                pt = datatypes.LFPPoint(self.timestamp.trodes_timestamp, self.channels[self.curntrode], self.channels[self.curntrode], self.buf[self.curntrode])
                self.curntrode = 1
                return pt, None

            elif self.datatype is datatypes.Datatypes.SPIKES:
                self.timestamp = self.datastream.getData() #Data is [(ntrode, cluster, timestamp, [0-159 data - (i,data)] ) ]
                # Reshape data to look like what spykshrk expects
                d = self.buf[0][3][:,1]
                newshape = (int(len(d)/40), 40)
                return datatypes.SpikePoint(self.timestamp.trodes_timestamp, self.buf[0][0], np.reshape(d, newshape)), None

            elif self.datatype is datatypes.Datatypes.LINEAR_POSITION:
                byteswritten = self.datastream.readData(self.buf) #Data is [(timestamp, linear segment, position, x location, y location)]
                return datatypes.CameraModulePoint(self.buf[0][0], self.buf[0][1], self.buf[0][2], self.buf[0][3], self.buf[0][4]), None

            # # Option to return timing message but disabled
            # timing_message = None
            # data_message = self.DataPointCls
            # return self.DataPointCls, timing_message

        else:
            return None