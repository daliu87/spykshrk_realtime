from spykshrk.realtime import realtime_process

from mpi4py import MPI

class TimingMessage(realtime_process.RealtimeMessage):
    def __init__(self, label, start_rank):
        """
        
        Args:
            label: 
            start_rank: 
        """
        self.label = label
        self.timing_data = [(start_rank, MPI.Wtime())]

    def record_time(self, rank):
        self.timing_data.append((rank, MPI.Wtime()))

    def pack(self):
        pass

