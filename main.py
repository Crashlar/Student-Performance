from src import CustomException , logger , DataIngestion , DataIngestionConfig
import sys
import os 

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        obj.initiate_data_ingestion()

    except Exception as e:
        raise CustomException(e , sys )


