from src import (
    CustomException ,
    logger ,
    DataIngestion ,
    DataIngestionConfig ,
    DataTransformation ,
    DataTransformationConfig
)
import sys
import os 

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data , test_data = obj.initiate_data_ingestion()

        obj = DataTransformation()
        obj.initiate_data_transformation(train_data , test_data)

    except Exception as e:
        logger.error(e)
        raise CustomException(e , sys )


