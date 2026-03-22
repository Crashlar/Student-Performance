"""
utils.py
--------------
This module is responsible for loading datasets from Kaggle using the kagglehub API.

It provides utility functions to:
- Download datasets from Kaggle
- Load CSV files into pandas DataFrames
- Return the loaded data in a structured format

The module is designed to handle multiple CSV files within a dataset
and return them as a dictionary of DataFrames.
"""

from src.studentperformance.logger import logging
from src.studentperformance.exception import CustomException
import sys
import pandas as pd
import os
import numpy as np
import kagglehub

logger = logging.getLogger(__name__)


def load_data():
    """
    Downloads and loads the student performance dataset from Kaggle.

    This function:
    - Downloads the dataset using kagglehub
    - Searches for CSV files in the downloaded directory
    - Reads the first CSV file into a pandas DataFrame
    - Returns the loaded DataFrame

    Returns:
    -------
    pandas.DataFrame
        The dataset loaded as a DataFrame

    Raises:
    ------
    FileNotFoundError
        If no CSV file is found in the downloaded dataset

    CustomException
        If any error occurs during download or data loading

    Notes:
    -----
    - Assumes the dataset contains at least one CSV file
    - Only the first CSV file found is loaded
    """

    try:
        path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
        logger.info(f"Dataset downloaded at: {path}")

        for file in os.listdir(path):
            if file.endswith(".csv"):
                file_path = os.path.join(path, file)

                # Load CSV into DataFrame
                df = pd.read_csv(file_path, low_memory=False)

                logger.info(f"Loaded {file} with shape {df.shape}")
                return df

        # Raise error if no CSV file is found
        raise FileNotFoundError("No CSV files found")

    except Exception as ex:
        # Wrap exception in custom exception for better traceability
        raise CustomException(ex, sys)

def save_object(file_path, obj):
    """
    Placeholder function for saving Python objects to disk.

    This function is intended to:
    - Serialize and save objects such as models, preprocessors, etc.
    - Persist trained artifacts for later use

    Parameters:
    ----------
    file_path : str
        Path where the object should be saved

    obj : any
        Python object to be saved

    Returns:
    -------
    None
    """
    pass