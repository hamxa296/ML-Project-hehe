# Machine Learning Project Overview

Welcome to your ML project! If you're new to the tools used here, don't worry. This guide will walk you through exactly what is going on, what every folder does, and how the different puzzle pieces fit together to make a working app.

---

## 1. The Research Paper Connection

First, let's address your question about the research paper. I searched through your entire project and found several interesting comments and lines of code:

- **In `src/train.py`:** The code prints `"Training the Unified Paper-Exact Pipeline..."`
- **In `src/evaluate.py`:** The code compares its own performance against the paper's results:
  - `PR-AUC: ... (Paper XGBoost Benchmark: 0.834)`
  - `ROC-AUC: ... (Paper XGBoost Benchmark: 0.887)`
- **In `src/evaluate.py`:** A graph is generated with the title `'Confusion Matrix — Paper-Exact Pipeline'`

**What does this mean?**
This project was specifically designed to replicate the exact methodology and results of a published research paper on **Fraud Detection**. The code is built to use the exact same model (XGBoost) and tries to hit the paper's benchmark scores (0.834 for PR-AUC and 0.887 for ROC-AUC).

---

## 2. Which Model Are We Using?

We are using a model called **XGBoost (Extreme Gradient Boosting)**. 

**What is XGBoost?**
Imagine you are trying to guess the price of a house. You ask a friend, and they make a guess. Then you ask another friend to fix the first friend's mistakes. Then a third friend fixes the second friend's mistakes. 
XGBoost does this with "decision trees" (flowcharts of yes/no questions). It builds a bunch of small decision trees sequentially, where each new tree specifically focuses on correcting the errors made by the previous trees. It is incredibly powerful and is widely considered one of the best models in the world for handling tabular data (data that looks like an Excel spreadsheet).

---

## 3. The Big Three Frameworks: Docker, FastAPI, and Prefect

This project isn't just a Python script; it's a full "Production-Ready" system. Here is what the three main technologies do:

### 🐳 Docker (The Shipping Container)
**What it is:** A tool that packages your entire app (code, Python, libraries, settings) into a neat little box called a "Container."
**Why we use it:** Have you ever heard a programmer say, *"Well, it works on my machine!"*? Docker fixes this. By putting the app in a container, we guarantee that if it works on your computer, it will work exactly the same way on your friend's computer, or on a giant cloud server. It brings its own environment with it.

### ⚡ FastAPI (The Waiter)
**What it is:** A modern, super-fast web framework for Python. 
**Why we use it:** When your React frontend (the buttons and graphs you see on the screen) wants to predict if a transaction is fraud, it needs a way to talk to your XGBoost Python model. FastAPI acts like a waiter in a restaurant. The React frontend gives an order (data) to the waiter (FastAPI), the waiter takes it to the kitchen (your ML model), gets the food (the prediction), and brings it back to the customer. 

### 🌊 Prefect (The Manager / Orchestrator)
**What it is:** A workflow orchestration tool.
**Why we use it:** Training an ML model requires many steps in order: (1) Load Data, (2) Clean Data, (3) Train Model, (4) Evaluate Model, (5) Save Model. Prefect is the manager that ensures these tasks happen in the right order. If a task fails (like the data doesn't load), Prefect catches the error, maybe retries it automatically, and records exactly what went wrong on a nice dashboard. 

---

## 4. Folder Structure Breakdown

Here is exactly what every folder in your `project/` directory is responsible for:

### 📁 `api/` (The Backend Server)
- Contains `main.py`. This is where **FastAPI** lives. It defines the URLs (like `/predict` or `/batch_predict`) that the frontend uses to talk to the model.

### 📁 `frontend/` (The User Interface)
- Contains all the **React** and **JavaScript** code. This is what you see in the browser. It creates the beautiful dashboard and tables for the user to interact with.

### 📁 `pipeline/` (The Automation)
- Contains `prefect_flow.py`. This is where **Prefect** lives. It strings together the data loading, training, and evaluation scripts so that you can retrain the model automatically with one click.

### 📁 `src/` (The Machine Learning Brains)
This is where the actual Data Science happens:
- **`preprocess.py`**: Cleans up the raw data. It drops columns with too many missing values and removes useless data.
- **`features.py`**: Creates new, smarter data columns out of the old ones to help the model learn better.
- **`train.py`**: This is where the **XGBoost** model is created and trained on the data.
- **`predict.py`**: Contains helper functions to make predictions on new data.
- **`evaluate.py`**: Calculates the scores (like PR-AUC) and prints out the charts to see if the model is smart or stupid.

### 📁 `models/` (The Saved Brains)
- When the model finishes training, it gets saved as a `.pkl` file (like `model_latest.pkl`) in this folder. FastAPI then loads this saved brain to make predictions for the users.

### 📄 `Dockerfile` & `docker-compose.yml`
- The instruction manuals for **Docker** telling it exactly how to build the container and start up both FastAPI and Prefect at the same time.
