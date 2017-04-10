# import trodes.FSData.fsDataMain as fsDataMain
import spykshrk.realtime.main_process as main_process
import spykshrk.realtime.ripple_process as ripple_process
import datetime
import logging
import logging.config
import cProfile
import sys
import getopt
from mpi4py import MPI
import time


def main(argv):
    # parse the command line arguments
    try:
        opts, args = getopt.getopt(argv, "", ["config="])
    except getopt.GetoptError:
        logging.error('Usage: ...')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--config':
            port = int(arg)

    # setup MPI
    comm = MPI.COMM_WORLD           # type: MPI.Comm
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # setup logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': ('%(asctime)s.%(msecs)03d [%(levelname)s] '
                           '(MPI-{:02d}) %(threadName)s %(name)s: %(message)s').format(rank),
                'datefmt': '%H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
            },
            'debug_file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'simple',
                'filename': '{}_MPI-{:02d}_debug.log'.format(datetime.datetime.now().
                                                             strftime('%Y-%m-%dT%H:%M:%S'), rank),
                'maxBytes': 10485760,
                'backupCount': 20,
                'encoding': 'utf8',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'debug_file_handler'],
                'level': 'NOTSET',
                'propagate': True,
            }
        }
    })

    logging.info('my name {}, my rank {}'.format(name, rank))

    if size == 1:
        # MPI is not running or is running on a single node.  Single processor mode
        pass

    # MPI node management

    supervisor_rank = 0
    latency_rank = 1
    ripple_ranks = [2, 3]
    if rank == supervisor_rank:
        # Supervisor node
        main_proc = main_process.MainProcess(comm=comm, rank=rank, ripple_ranks=ripple_ranks,
                                             latency_rank=latency_rank)
        main_proc.main_loop()

    if rank in ripple_ranks:
        ripple_proc = ripple_process.RippleProcess(comm, rank)
        ripple_proc.main_loop()

#cProfile.runctx('main(sys.argv[1:])', globals(), locals(), 'fsdatapy_profile')
main(sys.argv[1:])
