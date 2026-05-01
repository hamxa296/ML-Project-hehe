# Gap Analysis: Project vs. AI221 Requirements

## Overall Status Summary

| Requirement Area | Status | Severity |
|---|---|---|
| Multiple ML task types | ❌ Missing | 🔴 Critical |
| FastAPI serving | ✅ Done | — |
| CI/CD - GitHub Actions | ⚠️ Partial | 🟡 Medium |
| Prefect orchestration | ⚠️ Partial | 🟡 Medium |
| Automated ML testing (DeepChecks) | ❌ Missing | 🔴 Critical |
| Docker + Docker Compose | ✅ Done | — |
| Error notifications (Discord/Email) | ❌ Missing | 🟠 High |
| Model versioning & comparison | ⚠️ Partial | 🟡 Medium |
| Success/failure notifications | ❌ Missing | 🟠 High |

---

## 🔴 CRITICAL GAPS

### 1. Multiple ML Task Types — Completely Missing
**Requirement:**
> "You have to include multiple machine learning tasks (classification, regression, dimensionality reduction, recommendation systems, time series analysis, clustering and association) in the same workflow."

**What you have:** Only **classification** (XGBoost fraud detection) + **clustering** (KMeans as a feature step, not a standalone task).

**What's missing:**
- ❌ **Regression** — e.g., predict transaction amount, fraud loss amount, or risk score
- ❌ **Dimensionality Reduction** — e.g., PCA or UMAP for visualization/compression (not as a pipeline step, but as a standalone demonstrated task)
- ❌ **Recommendation System** — e.g., flagging similar suspicious transactions
- ❌ **Time Series Analysis** — e.g., anomaly detection over time with TransactionDT
- ❌ **Association Rules** — e.g., Apriori/FP-Growth on product codes or card combinations

> [!CAUTION]
> This is the **single biggest gap**. The project statement says ALL of these must appear in the same workflow. You need at least a demonstration of each task type, even if simplified.

---

### 2. DeepChecks / ML Testing Framework — Missing
**Requirement:**
> "Using DeepChecks or equivalent ML testing framework: Test data integrity, Identify drift, Validate performance metrics, Detect issues during CI/CD automatically before deployment"

**What you have:** Manual pytest tests that simulate drift with hardcoded numbers (`drift_ratio < 0.2` with `100.0` vs `105.0`). That is **not** a real ML testing framework — it's a fake stub.

**What's missing:**
- ❌ `deepchecks` (or `evidently`, `whylogs`, `great_expectations`) installed and actually running checks
- ❌ Real data integrity checks on the actual dataset
- ❌ Real train/test distribution drift detection
- ❌ These checks running inside the CI/CD pipeline automatically

---

## 🟠 HIGH-PRIORITY GAPS

### 3. Prefect Notifications — Missing
**Requirement:**
> "Implement error handling, retry logic, and success/failure notifications (Discord/Email/Slack)"

**What you have:** Retry logic ✅ (`retries=2` on `load_task`). Error handling ✅ (quality gate raises `ValueError`).

**What's missing:**
- ❌ No Discord/Slack/Email notification on pipeline success
- ❌ No Discord/Slack/Email notification on pipeline failure
- These are explicitly called out in the rubric as separate requirements

---

### 4. CI/CD Pipeline — Incomplete
**Requirement:**
> "Automate: Code checks, Unit tests and ML tests, **Data validation**, Model training triggers, Container image building, **Deployment pipeline**"

**What you have:** test → train → docker build.

**What's missing:**
- ❌ **Code quality checks** — no `flake8`, `black`, or `pylint` step
- ❌ **Data validation** step in CI (separate from ML tests)
- ❌ **Deployment step** — the Docker image is built but never pushed to a registry (DockerHub, GHCR) or deployed anywhere
- ❌ **DeepChecks/ML validation** step explicitly in the workflow YAML

---

## 🟡 MEDIUM GAPS

### 5. Model Comparison / Experiment Logging — Weak
**Requirement:**
> "Run multiple ML experiments, Log results, Compare model versions (baseline vs improved), Provide observations on best-performing model, overfitting/underfitting, deployment speed improvements"

**What you have:** `results.csv` with append-only logging. Only **one model type** has ever been trained.

**What's missing:**
- ❌ No baseline model run (e.g., Logistic Regression or simple XGBoost with defaults vs. tuned)
- ❌ No explicit experiment comparison table in the code/artifacts
- The `results.csv` has runs but they're all the same model config — no variation demonstrated

### 6. Prefect — No Data Ingestion Step
**Requirement:**
> "Build a Prefect pipeline that includes: Data ingestion, Feature engineering, Model training, Evaluation, Saving and versioning"

**What you have:** The pipeline loads a local CSV. There's no data ingestion step that fetches from an external source.

**What's missing:**
- ⚠️ Ideally a `data_ingestion_task` that pulls from somewhere (can be a URL, API, or S3-like source) — currently it just reads a local file

---

## ✅ WHAT IS DONE WELL

| Area | Detail |
|---|---|
| FastAPI endpoints | `/predict`, `/batch_predict`, `/model_evaluations`, `/latest_metrics`, `/graphs/`, `/reload_model`, `/health` — comprehensive |
| Docker | Multi-stage Dockerfile (Node→React build + Python), well-structured |
| Docker Compose | 3 services (prefect, pipeline, api) with healthcheck and shared volumes |
| Prefect flow structure | Tasks properly decorated, quality gate, versioning, EDA tasks |
| Model saving & versioning | `model_latest.pkl` + `model_v_YYYYMMDD.pkl` versioned archive |
| GitHub Actions structure | test → train → docker chain with `needs:` dependencies |
| Clustering | KMeans integrated into sklearn Pipeline |

---

## Priority Fix Order

```
Priority 1 (Critical — lose major marks without these):
  → Add multiple ML task types (regression, time series, dimensionality reduction, association)
  → Add DeepChecks (or Evidently) real ML testing

Priority 2 (High — explicitly mentioned in rubric):
  → Add Prefect Discord/Slack notifications
  → Add code quality checks + deployment step to CI/CD

Priority 3 (Medium — strengthens the grade):
  → Add a baseline model experiment for comparison
  → Add a real data ingestion task in Prefect
```
