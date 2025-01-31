import logging
import logging.config
import os

# Global logger
logger = logging.getLogger("gpplus")
logger.setLevel(logging.INFO)  # Default level is INFO
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_log_level = logging.INFO

def set_log_level(level):
    """
    Set log level.
    """
    global _log_level
    _log_level = level
    logger.setLevel(level)


def set_console_logger():
    """
    Set console logger
    """
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_formatter)
    logger.addHandler(console_handler)

_log_filename = "./gpplus.log"

def set_file_logger(filename):
    """
    Set file logger.
    """
    global _log_filename
    _log_filename = filename
    if not os.path.exists(os.path.dirname(_log_filename)):
        os.makedirs(os.path.dirname(_log_filename))  # Ensure the directory exists
    file_handler = logging.FileHandler(_log_filename)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)

def disable_logging():
    # Remove all handlers to fully disable logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

def enable_logging(console: bool = True, file: bool = True):
    # Restore the previous configuration or default to console
    set_log_level(logging.INFO)
    if console:
        set_console_logger()
    if file:
        set_file_logger(_log_filename)

# By default enable logging
enable_logging()
