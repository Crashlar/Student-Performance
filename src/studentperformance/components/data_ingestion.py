"""
data_ingestion.py
-----------------
This module is responsible for the data ingestion component of the ML pipeline.

It handles:
- Loading raw data from the data source (Kaggle via utility function)
- Saving the raw dataset locally
- Splitting the dataset into training and testing sets
- Persisting the split datasets into the artifacts directory

The module ensures that data is properly prepared and stored before
moving to the data transformation and model training stages.
"""

from src.studentperformance.logger import logging
from src.studentperformance.exception import CustomException
from src.studentperformance.utils import load_data
import sys
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.

    Attributes:
    ----------
    train_data_path : str
        File path where the training dataset will be saved.

    test_data_path : str
        File path where the testing dataset will be saved.

    raw_data_path : str
        File path where the raw dataset will be stored.
    """

    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    """
    Data Ingestion component responsible for:
    - Fetching the dataset
    - Saving raw data
    - Splitting into train and test sets
    - Saving processed splits into artifacts
    """

    def __init__(self):
        """
        Initializes the DataIngestion class with default configuration.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Executes the full data ingestion pipeline.

        Steps performed:
        1. Load dataset using utility function
        2. Save raw dataset locally
        3. Split dataset into training and testing sets
        4. Save train and test datasets into artifacts directory

        Returns:
        -------
        tuple
            Paths to the training and testing data files

        Raises:
        ------
        CustomException
            If any error occurs during the ingestion process
        """

        logger.info("Entered the data ingestion method or component")

        try:
            # Load dataset from Kaggle source
            df = load_data()
            logger.info("Read the dataset as dataframe")

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Train test split initiated")

            # Split dataset
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error("Exception occurred at data ingestion stage")
            raise CustomException(e, sys)