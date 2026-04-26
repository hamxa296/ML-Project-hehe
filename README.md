# 🤖 IEEE-CIS Fraud Detection: Production MLOps System

## 📌 Project Overview
This project has evolved from a series of exploratory Jupyter Notebooks into a **Production-Grade MLOps Architecture**. We have successfully encapsulated complex fraud detection logic (Pruning, Feature Engineering, K-Means Clustering, and XGBoost) into a fully automated, deployable system.

---

## 🏗️ How the Architecture Works Together

This system is built using three core MLOps pillars:

### 1. Prefect (The Orchestrator)
Prefect is the "brain" of our training pipeline (`project/pipeline/prefect_flow.py`). When you run the Prefect script, it automatically:
- Loads the massive 500MB+ dataset.
- Executes the strictly-typed Scikit-Learn `Pipeline` which houses our custom `PruningTransformer`, `FeatureEngineeringTransformer`, `ClusteringTransformer`, and `XGBClassifier`.
- Computes critical quality metrics (like PR-AUC and ROC-AUC).
- **Quality Gate:** If the model fails to meet the 0.60 PR-AUC threshold, Prefect blocks the deployment.
- **Experiment Tracking & Versioning:** If successful, Prefect logs the hyperparameters to `results.csv` and securely saves the model as both a timestamped file and `model_latest.pkl`.

### 2. FastAPI (The Inference Engine)
FastAPI (`project/api/main.py`) acts as the bridge between our trained math and the real world. 
- It loads `model_latest.pkl` into memory the second it boots up.
- It uses **Pydantic** to strictly enforce data schemas, ensuring no malformed requests crash the server.
- It is equipped with **CORSMiddleware** so our React frontend can talk to it safely.
- It exposes two endpoints: `POST /predict` (for single transactions) and `POST /batch_predict` (for processing bulk CSV files).

### 3. Docker (The Deployment Vessel)
The entire FastAPI backend and its dependencies are packaged inside a `Dockerfile`. Docker guarantees that our application will run identically whether it's on your local Windows machine, an AWS ECS cluster, or a Render cloud instance. It entirely eliminates the "it works on my machine" problem.

---

## 🖥️ The React Dashboard: What is Live vs. Dummy Data?

Our React (Vite) frontend has been wired up to the FastAPI backend. Here is the exact breakdown of what is real and what is just UI mockup:

### 🟢 LIVE (Real API Connectivity)
- **Model Simulator (`/predict`):** When you adjust the Transaction Volume slider or input a Card ID and click "Run Prediction", that data is *actually* traveling over HTTP to FastAPI. FastAPI fills in the missing 162 columns, runs the XGBoost math, and returns the real probability. The Red/Yellow/Green UI glow is 100% driven by the live ML model.
- **Batch Inference (`/batch_predict`):** If you go to the Database tab and click `Upload CSV`, it actually sends that file to the backend, evaluates every row, and maps the results dynamically into the glowing Risk Matrix table.
- **Top Dashboard KPIs:** The four metric blocks at the top of your dashboard (Total Predictions, Fraud Detected, Safe Transactions, Avg Fraud Prob) are calculating *real-time statistics* based on the actual predictions you just made in the Simulator.

### 🟡 DUMMY (UI Placeholders)
- **The Charts (Epoch Accuracy & Feature Drift):** The AreaChart and BarChart on the main Dashboard are currently rendering placeholder data. Wiring live training epoch streams and drift calculations to a React frontend requires websockets or heavier monitoring tools (like Prometheus/Grafana), so these graphs currently just exist to showcase the UI design intent.

---

## 🛠️ How to Run the System

### 1. Train the Model
```bash
# Move into the project directory
cd project

# Run the Prefect Orchestrator
python pipeline/prefect_flow.py
```

### 2. Launch the Backend
```bash
# Start the FastAPI Server (Keep it running in its own terminal)
uvicorn api.main:app --port 8000
```

### 3. Launch the Frontend
```bash
# Open a new terminal, move to the frontend folder
cd frontend

# Start the React Vite Server
npm run dev
```

You can now open `http://localhost:5173` and interact with your production fraud detection system!
