# 🧠 Main Model Pipeline Explanation: KMeans + XGBoost

This document provides a detailed breakdown of the exact machine learning pipeline currently deployed in the project. It maps out the flow from raw data ingestion through to the final classification, explicitly detailing the mathematical and programmatic logic at each step.

---

## 1. ✂️ Preprocessing & Pruning (`PruningTransformer`)
**File Location:** `src/preprocess.py`

The first stage of the pipeline strictly filters out noisy, redundant, and irrelevant features to ensure the model focuses only on strong signals. 

### Step-by-Step Pruning Logic:
1. **Missing Value Filter:**
   - Any feature with **> 95% missing values** is immediately dropped. This prevents the model from attempting to learn from overwhelmingly sparse noise.
2. **Zero-Variance Filter:**
   - Features that have exactly zero variance (a single constant value across all rows) are removed using `VarianceThreshold(threshold=0)`.
3. **Collinearity Filter (Correlation > 0.98):**
   - A correlation matrix is calculated. If two features are highly correlated (Pearson $r > 0.98$), one of them is arbitrarily dropped. This combats multicollinearity, making feature importance more reliable and training faster.
4. **Information Gain Filter:**
   - `mutual_info_classif` is calculated on a subset of the data. The algorithm ranks features by their dependency on the `isFraud` target variable.
   - The pipeline explicitly truncates the dataset to keep **only the top 167 most informative features**.
5. **Schema Locking:**
   - During `fit()`, the exact columns and data types are recorded. During `transform()` (inference), the pruner strictly enforces this schema—adding missing columns (as `0`), coercing bad types, and dropping unseen columns.

---

## 2. ⚙️ Feature Engineering (`FeatureEngineeringTransformer`)
**File Location:** `src/features.py`

This stage constructs derived features that capture the behavioral context of transactions, specifically tracking user velocity and temporal patterns.

### Derived Features:
1. **Missingness Indicators:**
   - For any column that had $> 20\%$ missing values during training, a new binary indicator column is created (`{col}_null`), signaling whether that field was intentionally left blank.
2. **Temporal Extraction:**
   - Using `TransactionDT` (seconds from a reference timestamp), the pipeline extracts:
     - `hour` (0-23): Captures time-of-day behavioral patterns.
     - `dow` (0-6): Captures day-of-week patterns (e.g., weekend vs. weekday fraud).
3. **User-Level Velocity Aggregations:**
   - A proxy "User ID" (`uid`) is constructed by concatenating `card1` and `addr1`.
   - `Amt_to_Median_User`: The transaction amount divided by the user's historical median amount. A highly anomalous ratio (e.g., 50x higher than their usual median) is a massive red flag.
   - `user_count`: The historical frequency/count of transactions by this specific user.
4. **Strict Imputation:**
   - The global median of every single column is recorded during training. All `NaN` values are imputed with these medians before passing to the model.

---

## 3. 🧩 Unsupervised Representation Learning (`ClusteringTransformer`)
**File Location:** `src/features.py`

This step embeds an unsupervised learning algorithm directly into the supervised pipeline to act as a powerful feature extractor.

### How it works:
- **KMeans (k=5)** is trained on the entirely cleaned and engineered dataset from the previous steps.
- The 5 clusters conceptually represent different "profiles" or archetypes of transactions in the dataset (e.g., *low-value domestic*, *high-value international*, *bot-like rapid transactions*).
- **The Output:** A brand new categorical feature called `cluster_label` (values 0-4) is appended to the dataset. The XGBoost model uses this cluster assignment as a macro-level feature to contextualize the micro-level variables.

---

## 4. 🚀 The Classifier (`XGBClassifier`)
**File Location:** `src/train.py`

The final estimator is an extreme gradient boosted decision tree (XGBoost). Because fraud datasets are notoriously imbalanced (usually < 3.5% fraud), the model is highly parameterized to combat this.

### Key Hyperparameters:
- **`scale_pos_weight`**: Calculates the ratio of Safe/Fraud transactions. This heavily penalizes the model for missing a fraud case, essentially forcing it to care about the minority class without needing to synthetically upsample (SMOTE) the data.
- **`n_estimators=500`**: A deep forest of 500 boosting rounds.
- **`max_depth=12`**: Relatively deep trees to capture highly complex, non-linear interactions between features (e.g., *If Country=X AND Hour=3AM AND Amt_to_Median > 5*).
- **`learning_rate=0.02`**: A slow learning rate combined with high estimators ensures the model converges smoothly without overfitting early.
- **`subsample=0.8` / `colsample_bytree=0.8`**: Randomly samples 80% of rows and 80% of columns per tree. This acts as strong regularization, preventing the model from over-relying on a single dominant feature.

---

## 🔁 Complete Data Flow Summary

1. `X_raw` (Raw CSV) $\rightarrow$ **PruningTransformer**
2. $\rightarrow$ Drops bad cols, keeps top 167 $\rightarrow$ **FeatureEngineeringTransformer**
3. $\rightarrow$ Adds `hour`, `dow`, user stats, imputes missing $\rightarrow$ **ClusteringTransformer**
4. $\rightarrow$ Adds `cluster_label` via KMeans $\rightarrow$ **XGBoost**
5. $\rightarrow$ **`isFraud` (0 or 1) & Probability Score**
