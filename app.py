from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel , Field
from src import CustomException , logger
from typing import Literal , Annotated
import joblib
import pandas as pd 

# Step 1 load the files
with open("artifacts/model.pkl" , "rb") as f:
    model = joblib.load(f)

with open("artifacts/preprocessor.pkl" , "rb") as f:
    preprocessor = joblib.load(f)
    


app = FastAPI(
    title="Student Performance API",
    description="Predicts student exam performance using ML",
    version="0.1.0"
)


# step2 = schema of userInput
class UserInput(BaseModel):
    gender : Annotated[Literal["male" , "female"] , Field(...,description="Gender of the student")]
    race_ethnicity : Annotated[Literal["group A" , "group B" , "group C" , "group D" , "group E"] , Field(...,description="Race/ethnicity of the student")]
    parental_level_of_education : Annotated[Literal["some high school" , "high school" , "some college" , "associate's degree" , "bachelor's degree" , "master's degree"] , Field(...,description="Parental level of education of the student")]
    lunch : Annotated[Literal["standard" , "free/reduced"] , Field(...,description="Lunch type of the student")]
    test_preparation_course : Annotated[Literal["none" , "completed"] , Field(...,description="Test preparation course of the student")]    
    reading_score : Annotated[int , Field(... ,gt = 0 , lt = 100, description="Reading score of the student")  ]
    writing_score : Annotated[int , Field(... ,gt = 0 , lt = 100, description="Writing score of the student")]
    

# Step 3 : Prediction 
@app.post("/predict")
def predict_performance(user_input : UserInput):
    try:
        input_df = pd.DataFrame([{
            "gender" : user_input.gender,
            "race_ethnicity" : user_input.race_ethnicity,
            "parental_level_of_education" : user_input.parental_level_of_education,
            "lunch" : user_input.lunch,
            "test_preparation_course" : user_input.test_preparation_course,
            "reading_score" : user_input.reading_score,
            "writing_score" : user_input.writing_score
        
        }])
        
        prediction = model.predict(preprocessor.transform(input_df))
        
        return JSONResponse({"prediction" : prediction[0]})
    except Exception as e:
        return JSONResponse({"error" : str(e)})
    

@app.get("/")
def home():
    return {
        "message": "🎓 Student Performance Prediction API",
        "status": "running",
        "project": "Predict student performance using ML model",
        "version": "0.1.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict"
        },
        "usage": "Send POST request to /predict with student data"
    }
    
@app.get("/health")
def health():
    return(
        {
            "status" : "OK",
            "model Version" : "0.1.0"
        }
    )
 

