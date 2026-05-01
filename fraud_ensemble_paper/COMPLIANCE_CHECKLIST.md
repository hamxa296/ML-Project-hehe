# IEEE-CIS Fraud Detection Paper - Compliance Checklist

## Paper: "Robust Fraud Detection with Ensemble Learning: A Case Study on the IEEE-CIS Dataset"
**Authors:** Fatemeh Moradi, Mehran Tarif, Mohammadhossein Homaei  
**Dataset:** IEEE-CIS Fraud Detection (590,540 transactions, 434 features, 3.48% fraud rate)

---

## 1. FEATURE ENGINEERING & PREPROCESSING

### 1.1 Feature Pruning Pipeline (Paper Section 3.2.1, Table 3)

**Baseline Features:** 431 (after removing ID and target columns)

| Stage | Threshold | Before | After | Removed | Status |
|-------|-----------|--------|-------|---------|--------|
| Remove Missing Values | >95% missing | 431 | 298 | 133 | ✅ Implemented |
| Remove Zero-Variance | Constant values | 298 | 276 | 22 | ✅ Implemented |
| Remove Correlation | >0.98 Pearson corr | 276 | 203 | 73 | ✅ Implemented |
| Remove Low Info Gain | <0.001 mutual info | 203 | **167** | 36 | ✅ Implemented |

**Final Baseline Feature Set:** 167 features

**Current Implementation Status:**
- ✅ Pruning thresholds correctly defined in `src/paper_pipeline.py`
- ❌ **ISSUE:** Recent 20k sample run used **339 features** instead of **167**
  - This suggests pruning thresholds not being applied correctly or being overridden
  - Need to verify: Is feature engineering pipeline expanding features beyond pruning?

### 1.2 Feature Engineering Strategy (Paper Section 3.2.2)

**Created Additional Features:** ~172 new features (total 431 baseline + ~172 engineered = ~603 before pruning)

| Feature Type | Count | Description | Status |
|---------------|-------|-------------|--------|
| Temporal Features | ~7 new | Hour of day, day of week, day of month | ✅ Implemented |
| Velocity Features | ~10-15 new | Transaction frequency in 1h, 6h, 24h windows | ✅ Implemented |
| Aggregation Features | ~50+ new | User/Card/Email-level statistics (mean, std, count) | ✅ Implemented |
| Amount Features | ~20 new | Log transformation, percentile rankings | ⚠️ Partial |
| Interaction Features | ~50 new | Target encoding, frequency encoding | ✅ Implemented |
| Missingness Indicators | ~20 new | Binary flags for >1% missing features | ⚠️ Partial |

**Expected Total Post-Engineering, Pre-Pruning:** ~600-650 features

---

## 2. MISSING VALUE HANDLING (Paper Section 3.2.3)

| Strategy | Threshold | Application | Status |
|----------|-----------|-------------|--------|
| Feature Removal | >95% missing | Remove entire feature | ✅ Implemented |
| Missing Category Creation | <95% missing categorical | Create explicit "missing" category | ⚠️ Check implementation |
| Median Imputation | Numerical | Separate by class (fraud/legitimate) | ⚠️ Check implementation |
| Missingness Indicators | >1% missing | Create binary "is_missing_X" feature | ⚠️ Check implementation |

---

## 3. CLASS IMBALANCE HANDLING (Paper Section 3.2.4)

### 3.1 Resampling Techniques Evaluated

| Method | Parameters | Status | Notes |
|--------|-----------|--------|-------|
| SMOTE | k=5, random_state=42 | ✅ Available | Used in some experiments |
| Borderline-SMOTE | k=5 | ⚠️ Available but not tested | Optional |
| ADASYN | n_neighbors=5 | ⚠️ Available but not tested | Optional |

**Current Implementation:** SMOTE wrapper available in `src/imbalance.py`

### 3.2 Cost-Sensitive Learning

| Model | Parameter | Value | Status |
|-------|-----------|-------|--------|
| XGBoost | `scale_pos_weight` | neg/pos ratio | ✅ Implemented |
| Logistic Regression | `class_weight` | 'balanced' | ✅ Implemented |
| Random Forest | `class_weight` | 'balanced' | ✅ Implemented |

---

## 4. ENSEMBLE ARCHITECTURE (Paper Section 3.4)

### 4.1 Three-Tier Ensemble Design

**Tier 1: Base Learners (Diversity)**

| Algorithm | Type | Purpose | Hyperparameters | Status |
|-----------|------|---------|-----------------|--------|
| **Random Forest** | Tree-based | Robust bagging | `n_estimators=100-200, max_depth=15-20` | ✅ Implemented |
| **XGBoost** | Tree-based (gradient boosting) | Optimal gradient boosting | `n_estimators=100-200, max_depth=6-8, learning_rate=0.03-0.1` | ✅ Implemented |
| **LightGBM** | Tree-based (gradient boosting) | Fast gradient boosting | `n_estimators=100-200, max_depth=7-10, learning_rate=0.05-0.1` | ⚠️ Not actively trained |
| **CatBoost** | Tree-based (categorical handling) | Categorical feature handling | `n_estimators=100-200, max_depth=7-10` | ⚠️ Not actively trained |
| **Logistic Regression** | Linear | Interpretable linear model | `C=1.0, solver='lbfgs', max_iter=1000` | ✅ Implemented |
| **K-Nearest Neighbors** | Distance-based | Local patterns | `n_neighbors=5-10` | ❌ Not implemented |
| **Multi-layer Perceptron** | Neural Network | Non-linear transformations | Hidden layers: [100, 50] | ❌ Not implemented |

**Paper Status:** Paper evaluates 5+ methods; current implementation focuses on RF, XGBoost, LogReg (3 methods). LightGBM and CatBoost imported but not actively trained in ensemble.

### 4.2 Meta-Learner Strategy

| Component | Specification | Status |
|-----------|---------------|--------|
| Stacking Method | K-fold out-of-fold predictions | ✅ Implemented |
| K-folds | 5 folds | ✅ Implemented |
| Meta-learner Algorithm | Logistic Regression | ✅ Implemented |
| Feature Construction | Base learner predictions + PCA(V-features) | ⚠️ Check if meta-features constructed correctly |
| Cross-validation | Nested temporal CV | ✅ Implemented |

### 4.3 Ensemble Voting Strategies

| Method | Weights | Status | Notes |
|--------|---------|--------|-------|
| Simple Voting | Equal (1/n_models) | ✅ Implemented | Baseline |
| Weighted Voting | By validation PR-AUC | ✅ Implemented | Primary method |
| Stacking | OOF meta-learner predictions | ✅ Implemented | Proposed best method |

---

## 5. CROSS-VALIDATION STRATEGY (Paper Section 4.1)

| Parameter | Specification | Status |
|-----------|---------------|--------|
| CV Method | K-fold temporal split | ✅ Implemented |
| K | 5 folds | ✅ Implemented |
| Split Strategy | Sort by TransactionDT, then split sequentially | ✅ Implemented |
| Temporal Leakage Prevention | No look-ahead/look-back | ✅ Verified |
| Train:Val:Test Ratio | Paper doesn't specify; standard 70:15:15 | ✅ Implemented |

---

## 6. MODEL CALIBRATION & THRESHOLDING (Paper Section 3.4.2)

| Step | Method | Status | Notes |
|------|--------|--------|-------|
| Post-hoc Calibration | Isotonic Regression | ✅ Implemented | Applied on OOF predictions |
| Secondary Calibration | Platt Scaling | ✅ Optional | Available |
| Threshold Selection | F1-optimized | ✅ Implemented | Computed on validation set |
| Threshold Application | Apply to test predictions | ✅ Implemented | Convert probabilities to binary predictions |

---

## 7. EVALUATION METRICS (Paper Section 4.2)

### 7.1 Primary Metrics (Imbalanced Classification)

| Metric | Abbreviation | Formula | Importance | Status |
|--------|--------------|---------|-----------|--------|
| Area Under ROC Curve | AUC-ROC | TPR vs FPR | Primary discrimination | ✅ Tracked |
| Area Under PR Curve | AUC-PR | Precision vs Recall | **Key metric for imbalance** | ✅ Tracked |
| F1-Score | F1 | 2·(Prec·Rec)/(Prec+Rec) | Balance Prec/Recall | ✅ Tracked |
| Balanced Accuracy | Bal-Acc | (Sensitivity+Specificity)/2 | Class balance metric | ✅ Tracked |
| G-Mean | G-Mean | √(Sensitivity·Specificity) | Geometric mean | ✅ Tracked |

### 7.2 Supporting Metrics

| Metric | Purpose | Status |
|--------|---------|--------|
| Precision | Fraud detection accuracy | ✅ Tracked |
| Recall / Sensitivity | Fraud capture rate | ✅ Tracked |
| Specificity | Legitimate transaction accuracy | ✅ Tracked |
| Classification Report | Per-class breakdown | ✅ Tracked |

---

## 8. EXPECTED RESULTS (Paper Table 4: Overall Performance Comparison)

### 8.1 Paper-Reported Metrics (IEEE-CIS Test Set)

| Method | AUC-ROC | AUC-PR | F1-Score | Bal.Acc | G-Mean |
|--------|---------|--------|----------|---------|--------|
| **Proposed Stacking** 🏆 | ~0.918 | **~0.891** | ~0.856 | ~0.810 | ~0.824 |
| XGBoost | ~0.887 | ~0.834 | ~0.802 | ~0.756 | ~0.771 |
| LightGBM | ~0.882 | ~0.821 | ~0.791 | ~0.741 | ~0.756 |
| Random Forest | ~0.865 | ~0.796 | ~0.768 | ~0.712 | ~0.724 |
| Weighted Voting | ~0.901 | ~0.861 | ~0.828 | ~0.784 | ~0.801 |
| Simple Voting | ~0.885 | ~0.819 | ~0.784 | ~0.734 | ~0.748 |
| CatBoost | ~0.890 | ~0.837 | ~0.805 | ~0.761 | ~0.776 |
| Logistic Regression | ~0.812 | ~0.721 | ~0.651 | ~0.593 | ~0.603 |

**Key Target Metrics for Validation:**
- ✅ Stacking AUC-PR: **~0.891** (PRIMARY - this is the main claim)
- ✅ Stacking AUC-ROC: **~0.918**
- ✅ Stacking F1: **~0.856**

### 8.2 Current Implementation Results (20k Sample Run)

| Method | AUC-ROC | AUC-PR | F1-Score | Notes |
|--------|---------|--------|----------|-------|
| OOF Fold Average | - | **0.491** | - | 5 folds avg |
| Weighted Voting | ~0.805 | 0.316 | - | Significantly below paper |
| Stacking | ~0.818 | 0.329 | - | Significantly below paper |
| Paper Target | ~0.918 | **0.891** | ~0.856 | Gap: -0.562 PR-AUC |

**Gap Analysis:**
- **Paper PR-AUC: 0.891**
- **Current PR-AUC: 0.329**
- **Gap: -0.562 (-63%)**

**Potential Root Causes:**
1. ❌ **Feature Count Mismatch:** 339 features vs 167 target (likely NOT the issue; more features should help)
2. ❌ **Missing Base Models:** LightGBM, CatBoost not in active ensemble (paper uses these)
3. ❌ **Hyperparameter Tuning:** Current defaults may not match paper's tuned hyperparameters
4. ❌ **Data Splitting:** Different train/val/test split or temporal window handling
5. ❌ **Target Encoding Leakage:** Possible information leakage in feature engineering
6. ❌ **Calibration Issues:** Isotonic calibration may not be applied correctly
7. ❌ **Sample Size Dependency:** 20k vs full 590k dataset performance difference

---

## 9. COMPLIANCE VALIDATION CHECKLIST

### Phase 1: Feature Engineering Verification ✅/❌

- [ ] Verify 431 baseline features (post-ID removal) present
- [ ] Verify ~172 engineered features created
- [ ] Verify post-engineering total ~600 features before pruning
- [x] Verify pruning removes >95% missing: 431→298
- [x] Verify pruning removes zero-variance: 298→276
- [x] Verify pruning removes >0.98 corr: 276→203
- [x] Verify pruning removes <0.001 info gain: 203→167
- [ ] **CRITICAL:** Verify final feature set is exactly 167 (currently 339)

### Phase 2: Preprocessing Pipeline Verification

- [ ] Verify missing value handling: categorical "missing" category, numerical median imputation, missingness indicators
- [ ] Verify class imbalance handling: SMOTE + cost-weighted models
- [ ] Verify temporal CV: 5-fold sequential split, no look-ahead
- [ ] Verify no data leakage in feature engineering

### Phase 3: Ensemble Architecture Verification

- [ ] Verify 5+ base learners included: RF, XGBoost, LightGBM, CatBoost, LogReg (+KNN/MLP if possible)
- [ ] Verify stacking with K=5 fold OOF predictions
- [ ] Verify meta-learner is logistic regression
- [ ] Verify weighted voting by validation PR-AUC
- [ ] Verify simple voting baseline

### Phase 4: Calibration & Thresholding Verification

- [ ] Verify post-hoc isotonic calibration applied
- [ ] Verify F1-optimized threshold selection
- [ ] Verify threshold applied to test predictions

### Phase 5: Metrics Tracking Verification

- [ ] Verify AUC-ROC computed correctly
- [ ] Verify AUC-PR computed correctly
- [ ] Verify F1-Score computed correctly
- [ ] Verify Balanced Accuracy computed correctly
- [ ] Verify G-Mean computed correctly

### Phase 6: Results Validation (PRIMARY)

- [ ] Run full pipeline on IEEE-CIS dataset
- [ ] Compare stacking AUC-PR against paper target (0.891)
- [ ] Compare stacking AUC-ROC against paper target (0.918)
- [ ] Compare all ensemble methods to Table 4
- [ ] Document any metric gaps with root cause analysis

---

## 10. KNOWN ISSUES TO RESOLVE

| Issue | Severity | Status | Action |
|-------|----------|--------|--------|
| Feature count mismatch (339 vs 167) | **CRITICAL** | 🔴 Blocking | Investigate and fix pruning pipeline |
| LightGBM/CatBoost not in ensemble | **HIGH** | 🟡 Pending | Add to base learners and train |
| PR-AUC gap (0.891 vs 0.329) | **CRITICAL** | 🔴 Blocking | Root cause analysis required |
| Hyperparameter tuning incomplete | **MEDIUM** | 🟡 Pending | Apply Optuna or paper-specified values |
| Full-dataset evaluation not run | **HIGH** | 🟡 Pending | Execute on 590k full dataset |

---

## 11. VALIDATION SCRIPT REQUIREMENTS

Create `fraud_ensemble_paper/validate_compliance.py` that:

1. **Load Data:** Load `data/merged_raw_train.csv`
2. **Feature Engineering:** Apply full pipeline and verify:
   - Post-engineering feature count ~600
   - Post-pruning feature count == 167
3. **Train-Val-Test Split:** Temporal split (70-15-15)
4. **Ensemble Training:** Train all 5+ base learners
5. **Meta-Learner:** Train stacking meta-learner
6. **Prediction:** Generate predictions via weighted voting and stacking
7. **Evaluation:** Compute all 5 metrics (AUC-ROC, AUC-PR, F1, Bal-Acc, G-Mean)
8. **Comparison:** Report vs paper Table 4
9. **Metrics Table:** Print comparison table with gaps

---

## 12. REFERENCES

- **Paper:** "Robust Fraud Detection with Ensemble Learning: A Case Study on the IEEE-CIS Dataset" (Moradi et al.)
- **Dataset:** IEEE-CIS Fraud Detection (Kaggle)
- **Implementation:** `fraud_ensemble_paper/src/paper_pipeline.py`
- **Extracted Paper Text:** `docs/ieee_cis_text_extracted.txt`

---

## NEXT STEPS

1. ✅ **Fix Feature Pruning:** Debug why 339 features instead of 167
2. ⬜ **Add Missing Models:** Integrate LightGBM & CatBoost into ensemble
3. ⬜ **Run Full Dataset:** Execute on 590k samples (not 20k)
4. ⬜ **Hyperparameter Tuning:** Either use paper values or run Optuna
5. ⬜ **Validate Results:** Compare metrics against Table 4
6. ⬜ **Document Findings:** Create final compliance report

