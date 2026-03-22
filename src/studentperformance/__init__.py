from .exception import CustomException
from .logger import get_logger


# Initialize a module-level logger for reuse across the application
logger = get_logger(__name__)
