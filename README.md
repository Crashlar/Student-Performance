
# Student Performance Prediction

This project is an end-to-end machine learning application for predicting student performance, specifically their math scores, based on a variety of personal and academic factors.

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

## 🚀 Overview

The primary goal of this project is to build and deploy a reliable model that can predict a student's math score. The project follows a modular structure, encompassing data ingestion, data transformation, model training, and deployment via a web interface.

The application consists of two main parts:
1.  **A FastAPI backend** that serves the machine learning model via a REST API.
2.  **A Streamlit frontend** that provides an interactive user interface for making predictions.

## ✨ Features

- **Interactive UI**: A simple and intuitive web interface built with Streamlit to get predictions.
- **FastAPI Backend**: A robust and fast backend to serve the prediction model.
- **Modular Pipeline**: A well-structured ML pipeline for data processing and model training.
- **Data Version Control**: DVC is used to manage and version large data files.
- **Containerization**: Dockerfile is provided to containerize the application for easy deployment.

## 🛠️ Architecture

The application is designed with a clear separation of concerns:

1.  **Frontend (`stream.py`)**: The Streamlit application that captures user input through a web form.
2.  **Backend (`app.py`)**: The FastAPI application that receives the input from the frontend, preprocesses it, and passes it to the ML model.
3.  **ML Model (`artifacts/`)**: A pre-trained model and preprocessor that performs the prediction.
4.  **ML Pipeline (`src/studentperformance/`)**: The source code for the entire ML pipeline, including data ingestion, transformation, and model training.

<div align="center">
    <img src="https://i.imgur.com/your_architecture_diagram_link.png" alt="Architecture Diagram" width="700"/>
</div>

*Note: Replace the placeholder image with an actual architecture diagram if you have one.*

## ⚙️ How to Run

### Prerequisites

- Python 3.11
- Docker (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/Crashlar/Student-Performance.git
cd Student-Performance
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Run the Application

You need to run the FastAPI backend and the Streamlit frontend in two separate terminals.

**Terminal 1: Start the FastAPI Backend**

```bash
uvicorn app:app --reload
```

**Terminal 2: Start the Streamlit Frontend**

```bash
streamlit run stream.py
```

Now, open your browser and go to the address provided by Streamlit (usually `http://localhost:8501`) to access the application.

### 4. Using Docker (Alternative)

Build and run the application using Docker Compose:

```bash
docker-compose up --build
```

This will start both the FastAPI backend and the Streamlit frontend.

## 📁 Project Structure

```
.
├── .dvc/                   # DVC files for data versioning
├── .github/                # GitHub Actions workflows
├── artifacts/              # Trained model and preprocessor
├── notebooks/              # Jupyter notebooks for experimentation
├── src/
│   └── studentperformance/ # Source code for the ML pipeline
│       ├── components/     # Data ingestion, transformation, etc.
│       ├── pipelines/      # Training and prediction pipelines
│       └── ...
├── app.py                  # FastAPI application
├── stream.py               # Streamlit application
├── requirements.txt        # Project dependencies
├── Dockerfile              # Dockerfile for containerization
├── README.md               # You are here!
└── ...
```

## 📊 Model & Data

- **Model**: The model is trained using [CatBoost/XGBoost/Scikit-learn - specify which one]. The trained model and preprocessor are stored in the `artifacts/` directory.
- **Data**: The dataset used for training is sourced from [mention the source, e.g., Kaggle]. DVC is used to track the dataset, which can be pulled using `dvc pull`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

---
*This README was generated with the help of an AI assistant.*
