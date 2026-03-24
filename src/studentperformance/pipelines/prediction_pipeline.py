# Import standard libraries
import os              # For file path operations
import sys             # For system-specific parameters and exception handling
import pandas as pd   # For creating DataFrame from input data

# Import custom modules
from src.studentperformance.exception import CustomException  # Custom exception class
from src.studentperformance.logger import logging             # Custom logging setup
from src.studentperformance.utils import load_object          # Utility to load saved objects

# Create a logger instance for this file
logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    This class is responsible for handling the prediction pipeline.
    It loads the trained model and preprocessor, transforms input data,
    and returns predictions.
    """

    def __init__(self):
        # Constructor (currently no initialization required)
        pass

    def predict(self, features):
        """
        Perform prediction using the trained model.

        Args:
            features (pd.DataFrame): Input data in DataFrame format

        Returns:
            np.array: Predicted results
        """
        try:
            # Define paths to model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")       

            # Load trained model from file
            model = load_object(file_path=model_path)

            # Load preprocessor (used during training)
            preprocessor = load_object(file_path=preprocessor_path)

            # Log successful loading
            logger.info("Model and preprocessor loaded successfully")

            # Transform input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions using the trained model
            preds = model.predict(data_scaled)

            # Return prediction results
            return preds

        except Exception as e: 
            # Log error if prediction fails
            logger.error("Error occurred during prediction")

            # Raise custom exception with system details
            raise CustomException(e, sys)


class CustomData:
    """
    This class is responsible for capturing user input data
    and converting it into a DataFrame format suitable for prediction.
    """

    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):
        # Assign input values to instance variables
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Convert the input data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing one row of input data
        """
        try:
            # Create a dictionary where each key matches training feature names
            # Values are wrapped in lists because DataFrame expects iterable values
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary to pandas DataFrame
            df = pd.DataFrame(custom_data_input_dict)

            # Return the created DataFrame
            return df

        except Exception as e:
            # Raise custom exception if DataFrame creation fails
            raise CustomException(e, sys)