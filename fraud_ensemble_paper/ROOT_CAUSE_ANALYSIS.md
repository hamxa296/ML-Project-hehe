# Root Cause Analysis: Feature Count Discrepancy (339 vs 167)

## Problem Statement
- **Expected:** 167 features (per paper Table 3 pruning pipeline)
- **Actual:** 339 features (current implementation)
- **Gap:** +202 features (+121%)

## Root Cause Analysis

### Pipeline Order Discrepancy

The **current implementation** processes features in this order:
```
Raw Data (431 features)
    ↓
[Feature Engineering - ADD new features]
    - Frequency encoding for categorical variables
    - Aggregation features (card1, card2, addr1 group statistics)
    - PCA on V-features (5 components)
    - Missing indicators
    ↓
[Feature Pruning - REMOVE low-value features]
    - Remove >95% missing
    - Remove zero-variance
    - Remove >0.98 correlated
    - Remove <0.001 MI (mutual information)
    ↓
339 features (pruned engineered set)
```

### What the Paper Specifies

The **paper's intended approach** (Section 3.2.1-3.2.2):
```
Raw Data (431 baseline features)
    ↓
[Feature Pruning - REMOVE low-value raw features] ← FIRST
    - Remove >95% missing: 431 → 298
    - Remove zero-variance: 298 → 276
    - Remove >0.98 corr: 276 → 203
    - Remove <0.001 MI: 203 → 167
    ↓
167 pruned baseline features
    ↓
[Feature Engineering - ADD new features on pruned baseline]  ← SECOND
    - Temporal features
    - Velocity features
    - Aggregation features
    - Interaction features
    ↓
~230-250 final features (pruned baseline + engineered)
```

## Code Evidence

### Current Implementation (`paper_pipeline.py` line 58-95)

```python
def fit(self, X, y):
    # STEP 1: Build features (engineering happens FIRST)
    df = self._build_features(X, fit=True)  # ← Adds 50+ engineered features
    
    # STEP 2: Then apply pruning
    self.keep_missing_ = ...  # Remove >95% missing
    self.keep_zero_var_ = ...  # Remove zero-variance
    self.keep_corr_ = ...      # Remove >0.98 corr
    self.keep_mi_ = ...        # Remove <0.001 MI
    
    return self
```

The `_build_features()` method (line 97-130) adds:
- Frequency encoding: `{col}_freq` for each categorical
- Aggregation features: `{group_col}__{stat}` for groups (card1, card2, addr1)
- PCA features: `V_pca_0` through `V_pca_4` (5 components)
- Missing indicators: `{col}_missing`
- Other features: `TransactionAmt_log`, `missing_count`

### Expected Implementation (Per Paper)

Should be:
1. Load raw 431 features
2. Apply 4-stage pruning to get 167 baseline features
3. Engineer new features on top of pruned 167
4. Final result: ~230-250 features

## Impact on Results

### Why Metrics Are Off

| Aspect | Current | Paper | Impact |
|--------|---------|-------|--------|
| Feature Set | 339 (engineered+pruned) | 167 (pruned) or ~250 (with FE) | ❌ Wrong feature set |
| Base Features | Mixed order | Deterministic pruning order | ❌ Different distributions |
| Information | Engineered features may add signal | Baseline + controlled FE | ❌ Possible over-specification |
| Reproducibility | Depends on FE order | Clear pruning → FE sequence | ❌ Can't verify paper |

**Expected impact on metrics:**
- Different feature set → different model decisions
- Wrong order → different pruning decisions (corr/MI scores change based on what's in the set)
- Engineering first masks which features are actually important in baseline

### Metrics Gap Explained

- **Current Stacking PR-AUC:** 0.329
- **Paper Target PR-AUC:** 0.891
- **Gap:** -0.562 (-63%)

**Likely reason:** Using 339 chaotic features vs 167 carefully pruned features means:
1. Model is overfitting to noisy features
2. Ensemble base learners are poorly calibrated
3. Stacking meta-learner is combining poor-quality predictions

## Solution

### Fix Implementation Order

**Change `PaperPreprocessor.fit()` to:**

```python
def fit(self, X: pd.DataFrame, y: pd.Series):
    df = X.copy()  # START with raw data ONLY
    
    # STEP 1: Apply pruning to raw features FIRST
    missing_rate = df.isna().mean()
    keep_missing = [c for c in df.columns if missing_rate[c] <= 0.95]
    df = df[keep_missing]
    
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = self.numeric_imputer_.fit_transform(numeric)
    
    # Remove zero-variance
    keep_zero_var = [c for c in df.columns if df[c].nunique() > 1]
    df = df[keep_zero_var]
    
    # Remove correlation >0.98
    corr_matrix = df.corr().abs()
    keep_corr = [...]  # Select features
    
    # Remove MI <0.001
    keep_mi = [...]  # Select features
    
    self.baseline_features_ = keep_mi  # Save 167 baseline features
    
    # STEP 2: Build features ONLY on pruned baseline
    df = df[self.baseline_features_]
    df = self._build_features(df, fit=True)  # Engineering happens SECOND
    
    return self
```

### Expected Results After Fix

1. **Feature Count:** 167 (baseline) + 50-80 (engineered) = ~220-240 features ✅
2. **Reproducibility:** Exact match to paper's sequence ✅
3. **Metrics:** Should improve significantly (target: PR-AUC ~0.891)
4. **Interpretability:** Clear which features are baseline vs engineered ✅

## Additional Issues to Address

### 1. Missing Base Models (LightGBM, CatBoost)
- Current: Only RF, XGBoost, LogReg (3 models)
- Paper: 5+ models including LightGBM, CatBoost
- **Fix:** Add LightGBM and CatBoost to ensemble

### 2. Incomplete Feature Engineering
- Current: Only basic aggregations and PCA
- Paper: Temporal features, velocity features, interaction features
- **Fix:** Expand `_build_features()` to match paper's full FE spec

### 3. Hyperparameter Tuning
- Current: Using defaults
- Paper: May have tuned hyperparameters
- **Fix:** Apply Optuna or extract paper-specified values

## Validation Plan

After fixing:
1. ✅ Verify feature count = 167 (baseline) after pruning
2. ✅ Verify total features ~220-240 after engineering
3. ✅ Run on 5k sample and check PR-AUC improves
4. ✅ Run on 590k full dataset
5. ✅ Compare all metrics to Table 4
6. ✅ Document gap analysis

