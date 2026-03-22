from src import CustomException
from src import logger
import sys


if __name__ == "__main__":
    try:
        logger.info("Application started")
        a = 1 / 0
    except Exception as e:
        logger.error("Error occurred")
        raise CustomException(e, sys)
    logger.info("Application ended")