"""
utils.py
--------
This module provides utility functions used across the ML pipeline.

Key responsibilities:
- Downloading and loading datasets from Kaggle using kagglehub
- Cleaning and standardizing dataset column names
- Saving trained objects (models, preprocessors, etc.) to disk
- Evaluating multiple machine learning models and comparing their performance

These utilities support different stages of the pipeline such as data ingestion,
data transformation, and model training.
"""

from src.studentperformance.logger import logging
from src.studentperformance.exception import CustomException

import sys
import pandas as pd
import os
import pickle
import dill
import numpy as np
import kagglehub

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# Initialize logger for this module
logger = logging.getLogger(__name__)


def load_data():
    """
    Downloads and loads the student performance dataset from Kaggle.

    This function:
    - Downloads the dataset using kagglehub
    - Searches for CSV files in the downloaded directory
    - Loads the first CSV file into a pandas DataFrame
    - Standardizes column names (lowercase, no spaces, underscores)

    Returns:
    -------
    pandas.DataFrame
        Loaded and cleaned dataset

    Raises:
    ------
    FileNotFoundError
        If no CSV file is found in the dataset directory

    CustomException
        If any error occurs during dataset download or loading
    """

    try:
        # Download dataset from Kaggle
        path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
        logger.info(f"Dataset downloaded at: {path}")

        # Iterate through files in dataset directory
        for file in os.listdir(path):
            if file.endswith(".csv"):
                file_path = os.path.join(path, file)

                # Load CSV into DataFrame
                df = pd.read_csv(file_path, low_memory=False)

                # Standardize column names:
                # - Remove leading/trailing spaces
                # - Convert to lowercase
                # - Replace spaces and slashes with underscores
                df.columns = (
                    df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_")
                    .str.replace("/", "_")
                )

                logger.info(f"Loaded {file} with shape {df.shape}")
                return df

        # Raise error if no CSV found
        raise FileNotFoundError("No CSV files found")

    except Exception as ex:
        logger.error("Error occurred while loading data")
        raise CustomException(ex, sys)


def save_object(file_path, obj):
    """
    Saves a Python object to disk using dill serialization.

    This function is typically used to persist:
    - Trained machine learning models
    - Preprocessing pipelines
    - Any reusable artifacts

    Parameters:
    ----------
    file_path : str
        Destination path where the object will be saved

    obj : any
        Python object to serialize and store

    Returns:
    -------
    None

    Raises:
    ------
    CustomException
        If saving the object fails
    """

    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logger.info(f"Object saved at: {file_path}")

    except Exception as e:
        logger.error("Error occurred while saving object")
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains and evaluates multiple machine learning models.

    For each model:
    - Fit the model on training data
    - Predict on both training and testing data
    - Compute R² score for evaluation
    - Store the test score in a report dictionary

    Parameters:
    ----------
    X_train : array-like
        Training feature set

    y_train : array-like
        Training target values

    X_test : array-like
        Testing feature set

    y_test : array-like
        Testing target values

    models : dict
        Dictionary of model name (str) and model instance

    Returns:
    -------
    dict
        Dictionary containing model names as keys and their test R² scores as values

    Raises:
    ------
    CustomException
        If any error occurs during model training or evaluation
    """

    try:
        report = {}

        # Iterate through all models
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            # Train the model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate performance
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store only test score
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logger.error("Error occurred while evaluating models")
        raise CustomException(e, sys)
    

def load_object(file_path):
    """
    Load a serialized Python object (e.g., model or preprocessor)
    from a given file path using pickle.

    Args:
        file_path (str): Path to the .pkl file

    Returns:
        object: Loaded Python object (model, preprocessor, etc.)

    Raises:
        CustomException: If file is not found or loading fails
    """
    try:
        # Check if the file exists at the given path
        if not os.path.exists(file_path):
            raise Exception(f"File not found at {file_path}")

        # Open the file in binary read mode ("rb")
        with open(file_path, "rb") as file_obj:
            
            # Load and return the object using pickle
            return pickle.load(file_obj)

    except Exception as e:
        # Wrap and raise the exception using custom exception class
        raise CustomException(e, sys)