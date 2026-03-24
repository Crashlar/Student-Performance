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

from sklearn.model_selection import GridSearchCV

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

            # Define hyperparameter grids
            param_grids = {
                    "Random Forest": {
                        "n_estimators": [50, 100],
                        "max_depth": [None, 10, 20],
                    },
                    "Decision Tree": {
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                    },
                    "Gradient Boosting": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 5],
                    },
                    "Linear Regression": {},
                    "XGBRegressor": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [3, 5],
                    },
                    "CatBoosting Regressor": {
                        "iterations": [50, 100],
                        "depth": [4, 6],
                        "learning_rate": [0.05, 0.1],
                    },
                    "AdaBoost Regressor": {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.05, 0.1],
                    },
                }

            best_models = {}
            model_report = {}

            for model_name, model in models.items():
                logger.info(f"Tuning hyperparameters for {model_name}")

                params = param_grids.get(model_name, {})

                if params:
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=5,
                        scoring="r2",
                        n_jobs=-1
                    )

                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                score = r2_score(y_test, y_pred)

                best_models[model_name] = best_model
                model_report[model_name] = score

                logger.info(f"{model_name} R2 Score: {score}")
                
            # Select best model based on highest R2 score
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = best_models[best_model_name]
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