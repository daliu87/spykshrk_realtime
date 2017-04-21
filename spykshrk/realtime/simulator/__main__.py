# import trodes.FSData.fsDataMain as fsDataMain
import spykshrk.realtime.main_process as main_process
import spykshrk.realtime.ripple_process as ripple_process
import spykshrk.realtime.simulator.simulator_process as simulator_process
import datetime
import logging
import logging.config
import cProfile
import sys
import getopt
from mpi4py import MPI
import time
import json


def main(argv):
    # parse the command line arguments
    try:
        opts, args = getopt.getopt(argv, "", ["config="])
    except getopt.GetoptError:
        logging.error('Usage: ...')
        sys.exit(2)

    print(argv)
    print(opts)
    for opt, arg in opts:
        if opt == '--config':
            config_filename = arg

    config = json.load(open(config_filename, 'r'))

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

    if rank == config['rank']['supervisor']:
        # Supervisor node
        main_proc = main_process.MainProcess(comm=comm, rank=rank, config=config)
        main_proc.main_loop()

    if rank in config['rank']['ripples']:
        ripple_proc = ripple_process.RippleProcess(comm, rank, config=config)
        ripple_proc.main_loop()

    if rank == config['rank']['simulator']:
        simulator_proc = simulator_process.SimulatorProcess(comm, rank, config=config)
        simulator_proc.main_loop()

#cProfile.runctx('main(sys.argv[1:])', globals(), locals(), 'fsdatapy_profile')
main(sys.argv[1:])
