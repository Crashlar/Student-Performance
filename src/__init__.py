"""
Initialize project-wide utilities.

This module provides:
- Centralized access to the custom exception class
- Configured logger instance for consistent logging across the project

Note:
Avoid reinitializing the logger in multiple places.
Use this shared instance instead.
"""


from .studentperformance.exception import CustomException
from .studentperformance.logger import get_logger


# Initialize a module-level logger for reuse across the application
logger = get_logger(__name__)