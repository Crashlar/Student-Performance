"""
model_trainer.py
----------------
This module is responsible for training multiple machine learning models,
evaluating them, and selecting the best-performing model based on evaluation metrics.

Key responsibilities:
- Splitting transformed data into features and target
- Training multiple regression models
- Evaluating models using a utility function
- Selecting the best model based on R² score
- Saving the trained model for future use
"""

import os 
import sys

from src.studentperformance.exception import CustomException
from src.studentperformance.utils import save_object, evaluate_models
from src.studentperformance.logger import get_logger, logging

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (
    AdaBoostRegressor, 
    RandomForestRegressor, 
    GradientBoostingRegressor 
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from dataclasses import dataclass


# Initialize logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for Model Trainer.

    Attributes:
    ----------
    trained_model_file_path : str
        File path where the best trained model will be saved
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    ModelTrainer is responsible for:
    - Training multiple regression models
    - Evaluating their performance
    - Selecting the best model
    - Saving the final trained model
    """

    def __init__(self):
        """
        Initializes the ModelTrainer with configuration.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Trains multiple models and selects the best-performing model.

        Parameters:
        ----------
        train_array : numpy.ndarray
            Training data array (features + target)

        test_array : numpy.ndarray
            Testing data array (features + target)

        preprocessor_path : str
            Path to the saved preprocessing object

        Returns:
        -------
        float
            R² score of the best model on test data

        Raises:
        ------
        CustomException
            If any error occurs during model training or evaluation
        """

        try:
            logger.info("Splitting training and testing data into features and target")

            # Split features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define multiple regression models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate all models using utility function
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Get best model score
            best_model_score = max(sorted(model_report.values()))

            # Get corresponding model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Retrieve best model instance
            best_model = models[best_model_name]

            # Threshold check
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient performance")

            logger.info(f"Best model selected: {best_model_name}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate final model on test data
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logger.info(f"Model R2 Score on test data: {r2_square}")

            return r2_square

        except Exception as e:
            logger.error("Exception occurred in initiate_model_trainer")
            raise CustomException(e, sys)