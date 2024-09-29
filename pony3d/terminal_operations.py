import logging
import random
from datetime import datetime
from pony3d import __version__

def initialize_logging():
    """
    Initialize and configure the logger.

    Returns:
    logging.Logger: Configured logger instance.
    """
    date_time = datetime.now()
    timestamp = date_time.strftime('%d%m%Y_%H%M%S')
    logfile = f'pony3d_{timestamp}.log'

    logging.basicConfig(
        filename=logfile, level=logging.DEBUG,
        format='%(asctime)s:: %(levelname)-5s :: %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S ', force=True
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    return logger

def hello():
    logging.info(f'                                                 .d8888b.       888     ')
    logging.info(f'                                                d88P  Y88b      888     ')
    logging.info(f'                                                     .d88P      888     ')
    logging.info(f'            88888b.   .d88b.  88888b.  888  888     8888"   .d88888     ')
    logging.info(f'            888 "88b d88""88b 888 "88b 888  888      "Y8b. d88" 888     ')
    logging.info(f'            888  888 888  888 888  888 888  888 888    888 888  888     ')
    logging.info(f'            888 d88P Y88..88P 888  888 Y88b 888 Y88b  d88P Y88b 888     ')
    logging.info(f'            88888P"   "Y88P"  888  888  "Y88888  "Y8888P"   "Y88888     ')
    logging.info(f'            888                             888                         ')
    logging.info(f'            888                        Y8b d88P                         ')
    logging.info(f'            888                         "Y88P"         v{__version__}   ')

def spacer():
    """
    Add a spacer to the log for better readability.
    """
    logging.info('')
    logging.info('-' * 80)
    logging.info('')

