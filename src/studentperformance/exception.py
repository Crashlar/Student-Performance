"""
exception.py
-------------
This module defines custom exception handling utilities for the project.

It provides:
1. A helper function to extract detailed error information including
   file name and line number.
2. A CustomException class that enhances standard Python exceptions
   with more informative debugging details.

This helps in better debugging and logging during development
and production pipelines.
"""

import sys


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information from the exception.

    Parameters:
    ----------
    error : Exception
        The original exception object or error message.

    error_detail : sys
        The sys module, used to access exception traceback information.

    Returns:
    -------
    str
        A formatted error message containing:
        - File name where the error occurred
        - Line number of the error
        - Original error message

    Example:
    -------
    >>> try:
    >>>     1 / 0
    >>> except Exception as e:
    >>>     print(error_message_detail(e, sys))
    """

    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        "Error occurred in python script name [{0}] "
        "line number [{1}] error message [{2}]"
    ).format(file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    """
    Custom Exception class for enhanced error reporting.

    This class extends the built-in Exception class and provides
    detailed error messages including file name and line number.

    Attributes:
    ----------
    error_message : str
        A detailed error message generated using traceback information.

    Methods:
    -------
    __str__()
        Returns the formatted error message when the exception is printed.

    Example:
    -------
    >>> try:
    >>>     a = 1 / 0
    >>> except Exception as e:
    >>>     raise CustomException(e, sys)
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with detailed error info.

        Parameters:
        ----------
        error_message : Exception
            The original exception raised.

        error_detail : sys
            The sys module to extract traceback details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)



    def __str__(self) -> str:
        """
        Returns the formatted error message.

        Returns:
        -------
        str
            Detailed error message including file name, line number,
            and original error.
        """

        
        return self.error_message