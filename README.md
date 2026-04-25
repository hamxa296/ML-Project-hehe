# 🤖 IEEE-CIS Fraud Detection Pipeline

## 📌 Project Overview
This project builds a production-grade, modular data preprocessing and machine learning pipeline for the **IEEE-CIS Fraud Detection** dataset. Due to the extreme class imbalance and high dimensionality of the dataset, this project implements advanced techniques including memory optimization, automated feature engineering, SMOTE balancing, PCA dimensionality reduction, and behavioural clustering to accurately detect fraudulent transactions.

## 🚀 Key Features

### 1. Object-Oriented Preprocessing (`src/preprocessing.py`)
A scalable, modular preprocessing pipeline built using `scikit-learn`'s `BaseEstimator` and `TransformerMixin`. It handles:
- **Memory Optimization:** Downcasts numerical data types to drastically reduce RAM usage (e.g., up to 68% reduction).
- **Automated Feature Engineering:** Extracts cyclical temporal features (Hour, Day of Week) from the raw `TransactionDT`.
- **Dynamic Null Pruning:** Drops features with a high percentage of missing values (e.g., >85% nulls).
- **Frequency Encoding:** Handles high-cardinality categorical features without exploding the dimensionality.

### 2. Robust Class Imbalance Handling
Addresses the severe class imbalance (approx. 3.5% fraud) using a two-phase `ClassImbalanceHandler`:
- **Phase 1 (Oversampling):** Uses SMOTE (Synthetic Minority Over-sampling Technique) to synthetically increase the fraud minority class to 20%.
- **Phase 2 (Undersampling):** Applies random undersampling to the majority class to achieve a 1:1 ratio.
- *Outcome:* The pipeline correctly applies resampling *only* to the training data to prevent leakage, outputting both a true-distribution dataset for EDA (`processed_train.csv`) and a balanced dataset for model training (`balanced_train.csv`).

### 3. Exploratory Data Analysis (EDA) (`notebooks/02_EDA.ipynb`)
Provides a suite of automated, dark-themed visualizations to uncover fraud patterns:
- Fraud rate distribution by time of day and day of week.
- Correlation heatmaps for continuous features.
- Density plots showing the distinction between legitimate and fraudulent transaction amounts.
- Categorical comparisons of product codes and card features.

### 4. Basic Baseline Models (`notebooks/03_Models_Basic.ipynb`)
Evaluates canonical baseline classifiers on the balanced dataset:
- **Logistic Regression:** Sets the linear baseline.
- **Random Forest:** Establishes a robust non-linear baseline.
- **XGBoost (Basic):** A gradient boosting model representing the industry standard for tabular data.
*Includes side-by-side ROC curve comparisons, confusion matrices, and feature importance bar charts.*

### 5. Advanced ML Pipeline (`notebooks/04_Models_Advanced.ipynb`)
Demonstrates three distinct machine learning paradigms in a unified architecture:
- **Stage A (Unsupervised Feature Engineering):**
  - **PCA (Dimensionality Reduction):** Compresses hundreds of highly correlated Vesta (V) features into ~50 principal components while retaining 95% variance.
  - **K-Means Clustering:** Groups transactions into distinct behavioural "archetypes" based on time and transaction amount, feeding a new `cluster_id` into downstream models.
- **Stage B (Supervised Classification):**
  - **Kernel SVM:** Trained exclusively on the PCA-reduced data using an RBF kernel to capture non-linear decision boundaries.
  - **XGBoost (Enhanced):** The champion model, trained on original features + PCA components + K-Means clusters.
- **Stage C (Supervised Regression):**
  - **Random Forest Regressor:** Trained solely on fraudulent rows to predict the actual financial loss (`TransactionAmt`), adding immense business triage value.

## 📁 Directory Structure
```
├── data/                       # Raw and processed CSV datasets
│   ├── processed_train.csv     # Unbalanced (for EDA)
│   └── balanced_train.csv      # 50/50 balanced via SMOTE (for Models)
├── docs/                       # Project documentation
│   └── preprocessing_report.md
├── notebooks/                  # Interactive Jupyter notebooks
│   ├── 01_Preprocessing.ipynb  # Pipeline orchestration
│   ├── 02_EDA.ipynb            # Exploratory Data Analysis
│   ├── 03_Models_Basic.ipynb   # Baseline models comparison
│   └── 04_Models_Advanced.ipynb# Advanced pipeline & regression
├── reports/
│   └── figures/                # Auto-generated plots & charts
├── src/                        # Source code
│   └── preprocessing.py        # Core transformer classes
├── requirements.txt            # Project dependencies
└── README.md                   # Project overview (this file)
```

## 🛠️ Setup & Installation
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the preprocessing script to generate the datasets (Note: running SMOTE may take 5-15 minutes):
   ```bash
   python src/preprocessing.py
   ```
3. Open the Jupyter notebooks to explore the EDA and train the models:
   ```bash
   jupyter notebook
   ```
