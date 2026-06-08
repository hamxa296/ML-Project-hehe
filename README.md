# Credit Card Fraud Detection - Ensemble Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1E88E5?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

An end-to-end machine learning pipeline implementing advanced credit card fraud classification. Focuses on ensembling tree-based classifiers and handling imbalanced datasets.

## Key Features
*   **Ensemble Modeling**: Trains and compares XGBoost, LightGBM, and CatBoost models.
*   **Imbalance Handling**: Features SMOTE and imbalanced-learn sampling techniques.
*   **MLOps Architecture**: Prefect orchestrator pipelines and Evidently analytics tracking.
*   **Vite Dashboard UI**: React/Vite web dashboard for real-time validation monitoring.

## File Structure
```text
├── data/              # Train/test datasets
├── src/               # Data processing and training code
├── notebooks/         # Model exploration & tuning notebooks
├── docker-compose.yml # Dev environment orchestrations
└── requirements.txt   # ML dependencies configurations
```
