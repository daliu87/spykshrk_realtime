import logging
from . import pp_decoder
# from . import pipeline  # not ready for deployment
from . import binary_utils
from . import data_containers
from . import errors
from . import ff_data
from . import filterframework_util
from . import franklab_data
# from . import frank_preprocessing   # dependency on trodes python package
from . import print_util
from . import warnings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)
log_formater = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
log_handler.setFormatter(log_formater)
for handle in logger.handlers:
    logger.removeHandler(handle)
logger.addHandler(log_handler)
# Keeps messages from being passed to higher loggers.  All loggers for franklab packages should be setup here.
logger.propagate = 0
