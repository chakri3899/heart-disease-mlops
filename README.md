Heart Disease Prediction - End-to-End MLOps Pipeline

MLOps Experimental Learning Assignment: End-to-End ML Model Development, CI/CD, and Production Deployment Experimental Learning

Objective: Design, develop, and deploy a scalable and reproducible machine learning solution utilising modern MLOps best practices. The assignment emphasises practical automation, experiment tracking, CI/CD pipelines, containerization, cloud deployment, and monitoring—mirroring real-world production scenarios.

Dataset: Title: Heart Disease UCI Dataset
Source: UCI Machine Learning Repository
• CSV containing 14+ features (age, sex, blood pressure, cholesterol, etc.) and a binary target (presence/absence of heart disease).

Problem Statement: Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

📌 Project Overview

This project implements a complete MLOps pipeline to predict heart disease using machine learning. It covers data preprocessing, model training, experiment tracking, API deployment, containerization, CI/CD automation, and Kubernetes deployment.

🧠 Goal:

Predict whether a patient has heart disease based on medical attributes such as age, cholesterol, blood pressure, etc.

📊 Dataset
Source: UCI Machine Learning Repository
Records: 303
Features: 13
Target: Presence of heart disease

⚙️ Tech Stack
Python
Scikit-learn
MLflow
FastAPI
Docker
GitHub Actions (CI/CD)
Kubernetes

🔄 Pipeline Architecture

User → FastAPI → Model → Prediction
    ↓
  Docker Container
    ↓
  Kubernetes Deployment
    ↓
  CI/CD (GitHub Actions)

🧪 Model Development

Two models were trained and evaluated:

1. Logistic Regression
Baseline linear model
2. Random Forest
Ensemble model with better performance

Evaluation Metrics:
Accuracy
Precision
Recall
ROC-AUC
Cross-validation (5-fold)

👉 Random Forest performed better and was selected as the final model.

🚀 How to Run
1. Setup Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Train Model
python -m src.train
3. Run API
uvicorn app.main:app --reload

Open:
http://localhost:8000/docs

🐳 Docker
docker build -t heart-api .
docker run -p 8000:8000 heart-api

☸️ Kubernetes
kubectl apply -f k8s/
kubectl port-forward service/heart-api-service 8000:80
🔗 API Usage

POST /predict

{
  "features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
}
🔄 CI/CD Pipeline

GitHub Actions pipeline:

Installs dependencies
Runs tests
Trains model
📈 Experiment Tracking

MLflow is used to:

Track model parameters
Log evaluation metrics
Store model artifacts

📌 Conclusion

This project demonstrates a complete MLOps workflow from data ingestion to deployment, ensuring scalability, reproducibility, and automation.