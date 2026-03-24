"""
__init__.py
-----------
This module initializes and exposes commonly used utilities from the
studentperformance package for easier and centralized imports.

It aggregates key components such as:
- CustomException: for standardized error handling
- Logger utility: for consistent logging across modules
- Utility functions: for data loading and object persistence

By importing these here, other modules can access them directly from
the package without deep imports.
"""

from .studentperformance.exception import CustomException
from .studentperformance.logger import get_logger
from .studentperformance.utils import load_data
from .studentperformance.utils import evaluate_models
from .studentperformance.utils import load_object
from .studentperformance.utils import save_object
from .studentperformance.components.data_ingestion import DataIngestion , DataIngestionConfig
from .studentperformance.components.data_transformation import DataTransformation , DataTransformationConfig
from .studentperformance.components.model_trainer import ModelTrainer , ModelTrainerConfig
from .studentperformance.pipelines.prediction_pipeline import PredictionPipeline , CustomData


# Initialize a module-level logger for reuse across the application
logger = get_logger(__name__)