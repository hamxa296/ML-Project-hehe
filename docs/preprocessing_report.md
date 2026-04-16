# IEEE-CIS Fraud Detection: Advanced Data Preprocessing Guide

This document details the advanced, dynamic preprocessing methodology implemented for the Fraud Detection dataset. The entire pipeline is built in `src/preprocessing.py`.

## Designing for Scale: The Object-Oriented Pipeline Approach

For this undergraduate project, creating simple scripts with Pandas operations line-by-line is insufficient for a complex Machine Learning Engineering endeavor. The dataset is nearly 1.5 GB in size, and a haphazard approach leads to memory-leaks and data-leakage during validation.

Instead, we built a modular pipeline utilizing **scikit-learn custom Transformers**.

### Why use `BaseEstimator` and `TransformerMixin`?
1. **Dynamic Execution**: Each operation is packaged natively into an object having a `fit()` and `transform()` method.
2. **Preventing Data Leakage**: By fitting transformations (like Frequency Encoding) strictly on the Training set, we establish stateful transformations that safely apply to the Test Set.
3. **Reproducibility**: The sequence of transformations can be executed seamlessly over incoming data batches in production.

---

## Step 1: Memory Optimization (`reduce_mem_usage`)

### What is it?
This function loops through your dataframe columns, checks the minimum and maximum capacity required for integer and float features, and aggressively downcasts them (e.g., from an oversized `float64` to `float16` or `int64` to `int8`).

### Why do it?
Pandas defaults all numerics to 64-bit architectures indiscriminately, leading to immediate memory (RAM) bottlenecking when loading >500 million datapoints. Memory downcasting allows you to train computationally heavy gradient-boosting trees (XGBoost/LightGBM) locally on standard hardware without kernel crashes.

---

## Step 2: Fusing Transaction and Identity Sets (`DataMerger`)

### What is it?
It merges `transaction.csv` with `identity.csv` via a Left Join mapped on `TransactionID`. 

### Why do it?
Fraud footprints heavily rely on recognizing device information and categorical identity markers (like device OS or browser version). The merging unifies the datasets.
**Critical Correction**: We dynamically fix column discrepancies—the Kaggle test identity dataset is improperly named `id-01` to `id-38` whereas the training data uses `id_01` to `id_38`. If left unfixed, your trained model will fail during inference on the test set.

---

## Step 3: Temporal Feature Engineering (`TimeFeatureExtractor`)

### What is it?
Extracts human-readable cyclical structures out of `TransactionDT`.

### Why do it?
`TransactionDT` is distributed as abstract seconds from an arbitrary, hidden reference date. While the model could infer trends from a continuous counter over thousands of estimators, it's computationally wasteful. By using modulus mapping, we can derive the exact `Hour_Of_Day` and `Day_Of_Week`. Financial fraud exhibits drastic temporal correlations (e.g., nocturnal transactional spikes).

---

## Step 4: Intelligent Feature Pruning and Imputation (`DropHighNulls`)

### What is it?
Scans your column matrix dynamically and permanently drops features that exceed an 80% NaN (missing values) threshold. 

### Why do it?
Imputing a column where 90% of data is missing forces the model to learn your imputation (e.g. median) rather than the underlying data. This acts as a severe noise injector and exacerbates the Curse of Dimensionality.

---

## Step 5: Advanced Categorical Operations (`FrequencyEncoder`)

### What is it?
Encodes categorical features. Instead of random integers (Label Encoding) or exploding dimensionality (One-Hot Encoding), this maps each raw category to the frequency / density percentage of its occurrence.

### Why do it?
Fraud actors typically use rare configurations representing niche populations. Frequency Encoding exposes mathematically precise "rarified markers" directly to tree-based models like LightGBM to split on outlier device/network combinations optimally.
