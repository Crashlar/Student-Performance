import streamlit as st
import requests

# Step 1 : keep fastapi running 

API_URL = "http://127.0.0.1:8000/predict"


# step 2 : design your ui 
st.title("Student Performance Prediction")
st.markdown("Enter your details below")


# step 3 : capture userInpput 

gender = st.selectbox("Gender" , ["male" , "female"])
race_ethnicity = st.selectbox("Race/Ethnicity" , ["group A" , "group B" , "group C" , "group D" , "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education" , ["some high school" , "high school" , "some college" , "associate's degree" , "bachelor's degree" , "master's degree"])
lunch = st.selectbox("Lunch" , ["standard" , "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course" , ["none" , "completed"])
reading_score = int(st.number_input("Reading Score" , min_value=0 , max_value=100))
writing_score = int(st.number_input("Writing Score" , min_value=0 , max_value=100))

if st.spinner("Predict"):
    input_data = {
        "gender" : gender,
        "race_ethnicity" : race_ethnicity,
        "parental_level_of_education" : parental_level_of_education,
        "lunch" : lunch,
        "test_preparation_course" : test_preparation_course,
        "reading_score" : reading_score,
        "writing_score" : writing_score
    }
    
    try:
        response = requests.post(API_URL , json = input_data)
        
        # step 4  : convert userinput to json format 
        result = response.json()
        
        # step 5 : send requests to fastapi 
        if response.status_code == 200 and "prediction" in result:
            prediction = result["prediction"]
            st.success(f"🎯 Predicted Score: {prediction}")
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")

