import logging
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
    """
    Print and log the welcome message, including the current version number.
    """
    logging.info('                       .oPYo.   .oPYo.   odYo.   o    o        ')
    logging.info("                       8    8   8    8   8' `8   8    8        ")
    logging.info('                       8    8   8    8   8   8   8    8        ')
    logging.info("                       8YooP'   `YooP'   8   8   `YooP8        ")
    logging.info('                       8                              8        ')
    logging.info(f"                       8                           ooP'   v{__version__} ")

def spacer():
    """
    Add a spacer to the log for better readability.
    """
    logging.info('')
    logging.info('-' * 80)
    logging.info('')

