import logging
import os
import sys
from datetime import datetime

from .constants import LOGS_FILE_PATH, LOGS_FORMATTER

FORMATTER = logging.Formatter(LOGS_FORMATTER)


def get_console_handler():
    """
    Get the console handler.

    Returns:
        logging.StreamHandler: The console handler object.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    """
    Get the file handler.

    Returns:
        logging.FileHandler: The file handler object.
    """
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    today_date = datetime.now().strftime("%d-%m-%Y")
    filename = LOGS_FILE_PATH.format(today_date)
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def init_loggers(logger_name: str = "Mindsql") -> logging.Logger:
    """
    Initialize the loggers.

    Parameters:
        logger_name (str): The name of the logger.

    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
        logger.propagate = False
    return logger
