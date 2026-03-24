from src import (
    CustomException ,
    logger ,
    DataIngestion ,
    DataIngestionConfig ,
    DataTransformation ,
    DataTransformationConfig,
    ModelTrainer , 
    ModelTrainerConfig,
    load_object,
    CustomData,
    PredictionPipeline
)
import sys
import os 

if __name__ == "__main__":
    try:
        # Step 1: Create CustomData instance 
        data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72,
        writing_score=74
    )

        # Step 2: Convert to DataFrame
        pred_df = data.get_data_as_dataframe()

        # Step 3: Create pipeline
        pipeline = PredictionPipeline()

        # Step 4: Get prediction
        result = pipeline.predict(pred_df)

        print("Prediction:", result)
        
    except Exception as e:
        logger.error(e)
        raise CustomException(e , sys )


