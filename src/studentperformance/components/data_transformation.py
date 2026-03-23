"""
data_transformation.py
----------------------
This module handles the data transformation stage of the ML pipeline.

It is responsible for:
- Defining preprocessing steps for numerical and categorical features
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Creating a unified preprocessing pipeline using ColumnTransformer

The transformed data is then ready for model training.
"""

from dataclasses import dataclass
import sys
import os 

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.studentperformance.exception import CustomException
from src.studentperformance.logger import logging
from src.studentperformance.utils import save_object

# Initialize logger for this module
logger = logging.getLogger(__name__)


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.

    Attributes:
    ----------
    preprocessor_obj_file_path : str
        File path where the preprocessing object will be saved
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """
    Data Transformation component responsible for:
    - Creating preprocessing pipelines
    - Handling missing values
    - Feature scaling and encoding
    """

    def __init__(self):
        """
        Initializes the DataTransformation class with configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing pipeline.

        The pipeline includes:
        - Numerical pipeline:
            * Missing value imputation (median)
            * Feature scaling (StandardScaler)
        - Categorical pipeline:
            * Missing value imputation (most frequent)
            * One-hot encoding
            * Feature scaling

        Returns:
        -------
        ColumnTransformer
            A preprocessor object that applies transformations
            to numerical and categorical features.

        Raises:
        ------
        CustomException
            If any error occurs during pipeline creation
        """

        try:
            # Define numerical features
            numeric_features = [
                # 'math_score',
                'reading_score',
                'writing_score'
            ]

            # Define categorical features
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # Log feature categories
            logger.info(f"Categorical columns: {categorical_features}")
            logger.info(f"Numerical columns: {numeric_features}")

            # Numerical pipeline:
            # Step 1: Fill missing values using median
            # Step 2: Scale features using standardization
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            logger.info("Numerical pipeline created (imputation + scaling)")

            # Categorical pipeline:
            # Step 1: Fill missing values using most frequent value
            # Step 2: Convert categories to numerical using OneHotEncoder
            # Step 3: Scale encoded features (without mean centering for sparse data)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logger.info("Categorical pipeline created (imputation + encoding + scaling)")

            # Combine numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            logger.info("Preprocessor object created successfully")

            return preprocessor

        except Exception as e:
            logger.error("Exception occurred in get_data_transformer_object")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        Executes the complete data transformation pipeline.

    This function:
    - Loads training and testing datasets from given file paths
    - Separates input features and target variable
    - Applies preprocessing (imputation, encoding, scaling)
    - Combines transformed features with target variable
    - Saves the preprocessing object for future use

    Parameters:
    ----------
    train_path : str
        Path to the training dataset CSV file

    test_path : str
        Path to the testing dataset CSV file

    Returns:
    -------
    tuple
        - Transformed training data (numpy array)
        - Transformed testing data (numpy array)
        - Path to saved preprocessing object

    Raises:
    ------
    CustomException
        If any error occurs during data transformation
    """

        try:
            # Load training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            # Get preprocessing pipeline
            logger.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column
            target_column_name = "math_score"

            # NOTE: These are remaining numeric features (excluding target)
            numerical_columns = ["writing_score", "reading_score"]

            # Split input features and target for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Split input features and target for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info(
                "Applying preprocessing object on training and testing dataframes"
            )

            # Fit on training data and transform
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Only transform test data (no fitting)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target column
            # np.c_ is used to concatenate arrays column-wise
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logger.info("Saving preprocessing object")

            # Save the preprocessing object for future inference
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logger.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logger.error("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)