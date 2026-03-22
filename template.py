"""
template.py
------------
This script initializes the project structure for a student performance prediction project.
It creates necessary directories and files for components, pipelines, utilities,
and root-level configuration. Boilerplate code is added to key files for clarity.
"""

import os
from pathlib import Path
import logging
from typing import List


# Define the project name
project_name = "studentperformance"

# List of files and directories to be created for the project structure
list_of_files = [

    f"src/{project_name}/__init__.py",

    # Components folder and files
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",

    # Pipelines folder and files
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",

    # Utility and core files
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",

    # Root-level files
    # "__init__.py",
    "app.py",
    "Dockerfile",   
    ".dockerignore",
    "requirements.txt",
    "stream.py",

    # Kaggle config 
    "~/.kaggle/kaggle.json",
    "~/.kaggle/.gitignore",
    
]

# Updated directories
extra_dirs = [
    "notebooks",
    "notebooks/data",
    # "notebooks/data/processed",
]


def create_project_structure(list : List , extra_dir : List):
    # Iterate through each file path in the list
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for the file {filename}")

        if (not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
            with open(filepath, "w") as f:
                pass
                logging.info(f"Creating empty file: {filename}")
        else:
            logging.info(f"{filename} already exists")

    for d in extra_dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Extra directory created: {d}")


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Initializing project structure...")
    create_project_structure(list_of_files , extra_dirs)
    logging.info("Project setup complete!")


if __name__ == "__main__":
    main()