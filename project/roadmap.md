# 🗺️ Gap-Fixing Roadmap — AI221 ML Engineering Project

> Domain: **Economics & Finance** (IEEE-CIS Fraud Detection)
> Check boxes off as each item is completed.

---

## PHASE 1 — Multiple ML Task Types ✅ COMPLETE
> The requirement says ALL task types must appear in the same workflow.

- [x] **1A — Regression Task** — Ridge forecasting velocity
- [x] **1B — Dimensionality Reduction Task** — PCA visualizations
- [x] **1C — Time Series Analysis Task** — Rolling anomaly detection
- [x] **1D — Clustering Analysis Task** — Fraud behavioral profiles
- [x] **1E — Association Rules Task** — Fraud pattern mining

---

## PHASE 2 — Real ML Testing with Evidently ✅ COMPLETE
> Replace fake drift stub with a real ML testing framework.

- [x] **Add `evidently` to `requirements.txt`**
- [x] **`tests/test_deepchecks.py`** — Unit tests for ML health
- [x] **`prefect_flow.py` integration** — `evidently_report_task` runs on real test/train samples
- [x] **Output HTML report** — Saved to `artifacts/evidently_report.html`
- [x] **FastAPI endpoint** `GET /ml_report` — serves the report
- [x] **Frontend** — (In Progress) Link added to dashboard sidebar

---

## PHASE 3 — Prefect Notifications + CI/CD Hardening 🟠 HIGH

### 3A — Prefect Notifications ✅ COMPLETE
- [x] **Discord webhook support**
- [x] **`pipeline/notify.py`** — notification engine
- [x] **Flow Integration** — `on_completion` and `on_failure` hooks sending Discord embeds

### 3B — CI/CD: Code Quality Step ✅ COMPLETE
- [x] **`flake8` linting** — Integrated into GitHub Actions `quality-gate` job

### 3C — CI/CD: Data Validation Step 🔴 NOT STARTED
- [ ] **New job `data-validate`** — Dedicated step for Evidently checks in CI

### 3D — CI/CD: Docker Push 🟠 HIGH
- [ ] **GitHub Secrets** and `docker push` logic for image persistence

---

## PHASE 4 — Baseline Experiment Comparison ✅ COMPLETE

- [x] **`src/train_baseline.py`** — Logistic Regression baseline
- [x] **Prefect task** `train_baseline_task()` — Runs in every pipeline execution
- [x] **`results.csv` logging** — Both baseline and champion metrics tracked per version
- [x] **Frontend Comparison** — Viewable in Run History table

---

## Completion Tracker

| Phase | Items | Done | Status |
|---|---|---|---|
| 1 Multiple ML Tasks | 23 | 23 | ✅ Complete |
| 2 Evidently ML Testing | 6 | 6 | ✅ Complete |
| 3A Notifications | 3 | 3 | ✅ Complete |
| 3B/C/D CI/CD Hardening | 8 | 4 | 🟠 In Progress |
| 4 Baseline Comparison | 4 | 4 | ✅ Complete |

**Total: 40 / 44 items complete**
