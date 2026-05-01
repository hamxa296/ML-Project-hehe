# 🗺️ Gap-Fixing Roadmap — AI221 ML Engineering Project

> Domain: **Economics & Finance** (IEEE-CIS Fraud Detection)
> Check boxes off as each item is completed.

---

## PHASE 1 — Multiple ML Task Types 🔴 CRITICAL
> The requirement says ALL task types must appear in the same workflow.
> We'll add them as new Prefect tasks inside the existing pipeline + expose them via FastAPI + show results in the frontend.

### 1A — Regression Task
- [x] **`src/regression.py`** — Ridge regression forecasting next-window fraud count from lag features
- [x] **Prefect task** `regression_task()` added to `prefect_flow.py`
- [x] **FastAPI endpoint** `GET /regression_results` — returns RMSE, R², forecast JSON
- [x] **Frontend** — Regression tab in `/ml-tasks` page with metric pills + Recharts line chart

### 1B — Dimensionality Reduction Task
- [x] **`src/dimensionality_reduction.py`** — PCA 2D scatter + variance explained charts
- [x] **Prefect task** `dimensionality_reduction_task()` added to flow
- [x] **Artifact saved** — `artifacts/pca_results.json` + `latest_pca_scatter.png` + `latest_pca_variance.png`
- [x] **FastAPI endpoint** `GET /pca_results`
- [x] **Frontend** — PCA tab with ScatterChart + BarChart variance in `/ml-tasks`

### 1C — Time Series Analysis Task
- [x] **`src/timeseries.py`** — Rolling averages, z-score anomaly detection, hour×day heatmap
- [x] **Prefect task** `timeseries_task()` added to flow
- [x] **Artifact saved** — `artifacts/timeseries_fraud_rate.json` + 2 PNG charts
- [x] **FastAPI endpoint** `GET /timeseries_results`
- [x] **Frontend** — Time Series tab with line chart + anomaly badges in `/ml-tasks`

### 1D — Clustering as Standalone Task (promote from hidden step)
- [x] **`src/clustering_analysis.py`** — Fraud rate per cluster, feature centroids, pie + bar charts
- [x] **Prefect task** `clustering_analysis_task()` added to flow
- [x] **Artifact saved** — `artifacts/cluster_profiles.json` + 2 PNG charts
- [x] **FastAPI endpoint** `GET /cluster_profiles`
- [x] **Frontend** — Clustering tab with BarChart + detail table in `/ml-tasks`

### 1E — Association Rules Task
- [x] **`src/association.py`** — FPGrowth on ProductCD, card4, card6, email domain, AmtBin
- [x] **Prefect task** `association_task()` added to flow
- [x] **Artifact saved** — `artifacts/association_rules.json` (top-20 rules by lift) + PNG
- [x] **FastAPI endpoint** `GET /association_rules`
- [x] **Frontend** — Association tab with full rules table + lift scores in `/ml-tasks`

---

## PHASE 2 — Real ML Testing with Evidently 🔴 CRITICAL
> Replace fake drift stub with a real ML testing framework.

- [ ] **Add `evidently` to `requirements.txt`**
- [ ] **`tests/test_deepchecks.py`** — New test file using Evidently:
  - [ ] Data integrity check (schema, nulls, duplicates)
  - [ ] Train vs test distribution drift report (feature drift score)
  - [ ] Model performance validation (AUC-PR above threshold)
  - [ ] Output HTML report saved to `artifacts/evidently_report.html`
- [ ] **`tests/test_ml.py`** — Replace fake drift stub (`100.0 vs 105.0`) with real Evidently drift detection call
- [ ] **CI/CD** — Add `pytest tests/test_deepchecks.py` as a dedicated `ml-checks` job in GitHub Actions (runs before `train`)
- [ ] **FastAPI endpoint** `GET /ml_report` — serves `evidently_report.html`
- [ ] **Frontend** — Link to ML health report page

---

## PHASE 3 — Prefect Notifications + CI/CD Hardening 🟠 HIGH

### 3A — Prefect Notifications
- [ ] **Discord webhook** set up (free, no account-gating unlike email SMTP)
- [ ] **`pipeline/notify.py`** — helper: `send_discord(message, color)` using `requests.post` to webhook URL
- [ ] **On success** — `training_pipeline` sends ✅ Discord embed with version, AUC-PR, AUC-ROC
- [ ] **On failure** — Prefect `on_failure` hook sends ❌ Discord embed with error message + task name
- [ ] **Environment variable** `DISCORD_WEBHOOK_URL` in `docker-compose.yml` + `.env.example`

### 3B — CI/CD: Code Quality Step
- [ ] **Add `flake8` to `requirements.txt`**
- [ ] **New job `lint`** in `ml_pipeline.yml` — runs `flake8 src/ api/ pipeline/ tests/ --max-line-length=120`
- [ ] **`lint` must pass before `test` job runs**

### 3C — CI/CD: Data Validation Step
- [ ] **New job `data-validate`** — runs Evidently data integrity checks in CI (uses a sample CSV or synthetic data if real data absent)
- [ ] Runs after `lint`, before `train`

### 3D — CI/CD: Docker Push to Registry
- [ ] **GitHub Secrets** — `DOCKER_USERNAME`, `DOCKER_PASSWORD` (or use GHCR)
- [ ] **New step in `docker` job** — `docker push` to DockerHub/GHCR after build succeeds
- [ ] This makes the CI/CD a complete **build → test → train → push → deploy** chain

---

## PHASE 4 — Baseline Experiment Comparison 🟡 MEDIUM

- [ ] **`src/train_baseline.py`** — Train a simple Logistic Regression baseline (no feature engineering, no clustering)
- [ ] **Prefect task** `train_baseline_task()` — runs before main training, logs to `results.csv` with `model_type = "LogisticRegression_baseline"`
- [ ] **`results.csv`** now has 2+ rows per session: baseline vs tuned XGBoost
- [ ] **Frontend** — Run history table already shows this automatically (it reads `results.csv`)
- [ ] **`artifacts/experiment_comparison.json`** — structured comparison: baseline vs best model

---

## Completion Tracker

| Phase | Items | Done | Status |
|---|---|---|---|
| 1A Regression | 4 | 4 | ✅ Complete |
| 1B Dimensionality Reduction | 5 | 5 | ✅ Complete |
| 1C Time Series | 5 | 5 | ✅ Complete |
| 1D Clustering (promote) | 5 | 5 | ✅ Complete |
| 1E Association Rules | 5 | 5 | ✅ Complete |
| 2 Evidently ML Testing | 7 | 0 | 🔴 Not started |
| 3A Notifications | 5 | 0 | 🔴 Not started |
| 3B Lint CI | 3 | 0 | 🔴 Not started |
| 3C Data Validate CI | 2 | 0 | 🔴 Not started |
| 3D Docker Push | 3 | 0 | 🔴 Not started |
| 4 Baseline Comparison | 5 | 0 | 🔴 Not started |

**Total: 24 / 49 items complete**
