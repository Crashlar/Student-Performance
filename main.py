from src import (
    CustomException ,
    logger ,
    DataIngestion ,
    DataIngestionConfig ,
    DataTransformation ,
    DataTransformationConfig,
    ModelTrainer , 
    ModelTrainerConfig
)
import sys
import os 

if __name__ == "__main__":
    try:
        train_data , test_data = DataIngestion().initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr , test_arr , preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data , test_data)
        model_trainer = ModelTrainer()
        
        model_trainer.initiate_model_trainer(train_arr , test_arr , preprocessor_obj_file_path)


    except Exception as e:
        logger.error(e)
        raise CustomException(e , sys )


