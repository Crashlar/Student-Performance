"""
logger.py
----------
This module provides a reusable logging utility for the project.

It ensures:
- Logs are stored in the root /logs directory
- Each run generates a unique timestamped log file
- Logs are written to both file and console
"""

import logging
import os
import colorlog
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Create logs directory at project root 
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Create unique log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger.

    Parameters:
    ----------
    name : str
        Name of the logger (usually __name__)

    Returns:
    -------
    logging.Logger
        Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.propagate = False

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler (writes to logs folder)
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)

    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)

    color_formatter = colorlog.ColoredFormatter(
    "%(log_color)s[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

    console_handler.setFormatter(color_formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger