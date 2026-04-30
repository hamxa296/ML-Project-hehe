# ML Project Deep-Dive Analysis
## Fraud Detection Pipeline — Full System Map

---

## 1. Project Purpose

A **production-style MLOps system** that replicates an XGBoost fraud-detection research paper.
The model benchmarks against the paper's targets:
- PR-AUC target: **0.834** | Last run: **0.793**
- ROC-AUC target: **0.887** | Last run: **0.965** ✅

---

## 2. Full Directory Map

```
project/
├── api/            ← FastAPI backend (the HTTP server)
│   └── main.py
├── artifacts/      ← Shared volume: pipeline outputs consumed by API
│   ├── results.csv         ← Append-only run history log
│   └── latest_metrics.json ← JSON curve data for interactive Recharts
├── data/           ← Input CSVs (train_unbalanced.csv, test.csv)
├── frontend/       ← React (Vite) dashboard
│   └── src/
│       ├── pages/      ← 10 page components
│       ├── services/   ← api.js (fetch wrappers)
│       └── index.css   ← Design system (glassmorphism, dark theme)
├── models/         ← Shared volume: trained .pkl files
│   ├── model_latest.pkl    ← Stable pointer, overwritten each run
│   └── model_v_XXXXXXXX.pkl ← Versioned archive copy
├── pipeline/
│   └── prefect_flow.py ← Prefect orchestration (trains + evaluates)
├── results/
│   └── graphs/     ← Shared volume: PNG evaluation charts
│       ├── latest_roc_curve.png        ← Stable "latest" pointer
│       ├── latest_pr_curve.png
│       ├── latest_confusion_matrix.png
│       ├── latest_metric_summary.png
│       └── *_v_XXXXXXXX.png (versioned archive copies)
├── src/            ← ML logic library
│   ├── preprocess.py   ← PruningTransformer (feature selection)
│   ├── features.py     ← FeatureEngineeringTransformer + ClusteringTransformer
│   ├── train.py        ← XGBClassifier sklearn Pipeline builder
│   ├── predict.py      ← predict() helper
│   └── evaluate.py     ← Charts, JSON metrics, PNG graphs
├── tests/          ← pytest suite
├── .github/workflows/ml_pipeline.yml ← GitHub Actions CI/CD
├── Dockerfile          ← Multi-stage: Node→React build + Python backend
├── docker-compose.yml  ← 3 services: prefect, pipeline, api
└── requirements.txt    ← Python dependencies
```

---

## 3. The ML Pipeline (src/)

### sklearn Pipeline steps (executed in order):
```
raw CSV
  ↓ [1] PruningTransformer (preprocess.py)
      - Drop >95% missing columns
      - VarianceThreshold (zero-variance)
      - Correlation filter (>0.98 correlated pairs)
      - Mutual Info top-167 features
  ↓ [2] FeatureEngineeringTransformer (features.py)
      - Missingness indicator flags
      - Time features: hour-of-day, day-of-week (from TransactionDT)
      - User behavior: Amt_to_Median_User, user_count
      - Median imputation for missing values
  ↓ [3] ClusteringTransformer (features.py)
      - KMeans(n_clusters=5) on all features
      - Appends cluster_label as new feature
  ↓ [4] XGBClassifier
      - n_estimators=500, max_depth=12, learning_rate=0.02
      - scale_pos_weight=ratio (handles class imbalance)
      - tree_method='approx', subsample/colsample=0.8
```

### evaluate.py outputs per run:
| Output | Path | Purpose |
|---|---|---|
| results.csv | `artifacts/results.csv` | Append-only run log (API reads for history) |
| latest_metrics.json | `artifacts/latest_metrics.json` | ROC + PR curve JSON (100 subsampled pts) |
| latest_roc_curve.png | `results/graphs/` | PNG for Dashboard gallery |
| latest_pr_curve.png | `results/graphs/` | PNG for Dashboard gallery |
| latest_confusion_matrix.png | `results/graphs/` | PNG for Dashboard gallery |
| latest_metric_summary.png | `results/graphs/` | PNG for Dashboard gallery |
| model_latest.pkl | `models/` | Stable model pointer |
| model_v_XXXXXX.pkl | `models/` | Versioned archive |

---

## 4. Prefect Orchestration (pipeline/prefect_flow.py)

```python
@flow("Robust Fraud Detection Unified Pipeline")
def training_pipeline(config={"min_auc_pr": 0.6}):
    version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    X_train, X_test, y_train, y_test = load_task()       # @task retries=2
    pipeline = train_pipeline_task(X_train, y_train)      # @task
    auc_pr = evaluate_and_log_task(pipeline, X_test, ...)  # @task
    if auc_pr < 0.6:
        raise ValueError("Quality gate failed!")           # ← ML quality gate
    save_model_task(pipeline, version)                     # @task
```

**Key behavior:**
- Prefect v2 (pinned `<3.0.0`) runs as a server on port 4200
- Each task decorated with `@task` → visible in Prefect UI
- `retries=2` on `load_task` for resilience
- **Quality gate**: if AUC-PR < 0.6, the flow fails & model is NOT saved
- `version` string = `v_YYYYMMDD_HHMMSS` — ties all artifacts together
- Pipeline runs as a **one-shot container** that exits when done

---

## 5. FastAPI Backend (api/main.py)

### Endpoints:
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns 503 if model not loaded |
| POST | `/predict` | Single transaction fraud prediction |
| POST | `/batch_predict` | CSV upload → probabilities + optional metrics |
| GET | `/model_evaluations` | Returns full `results.csv` as JSON array |
| GET | `/latest_metrics` | Returns `latest_metrics.json` (curve data) |
| GET | `/graph_list` | Lists `latest_*.png` filenames in `/results/graphs` |
| GET | `/graphs/{filename}` | Serves a PNG with `no-cache` headers |
| POST | `/reload_model` | Hot-reloads model from disk (no container restart) |
| GET | `/*` | SPA fallback → serves React `frontend/dist/index.html` |

### Path resolution strategy:
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # /app in Docker
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH    = PROJECT_ROOT / "models" / "model_latest.pkl"
GRAPHS_DIR    = PROJECT_ROOT / "results" / "graphs"
```
All paths anchored to file location → works regardless of launch CWD.

### Model lifecycle:
- **Startup**: loads `model_latest.pkl` if it exists
- **Hot-reload**: `POST /reload_model` re-reads the pkl in-place
- **Caching**: graph responses have `Cache-Control: no-store` → browser always re-fetches

---

## 6. Docker Architecture

### Dockerfile (multi-stage):
```dockerfile
# Stage 1 — build React app
FROM node:20-slim AS frontend-builder
COPY frontend/package*.json → npm install → npm run build
→ produces frontend/dist/

# Stage 2 — Python image ships BOTH backend + built frontend
FROM python:3.11-slim
pip install -r requirements.txt
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key:** The same Docker image is reused for both `pipeline` and `api` services.

### docker-compose.yml — 3 services:
```
┌─────────────────────────────────────────────────────────────┐
│  prefect  (prefecthq/prefect:2-python3.11)  port 4200      │
│    └─ prefect server start --host 0.0.0.0                  │
│                                                             │
│  pipeline  (build: .)  ← one-shot, exits when done         │
│    └─ python pipeline/prefect_flow.py                       │
│    volumes:                                                 │
│      ./models   → /app/models                              │
│      ./results  → /app/results                             │
│      ./data     → /app/data                                │
│      ./artifacts→ /app/artifacts                           │
│                                                             │
│  api  (build: .)  port 8000  ← long-running server         │
│    └─ uvicorn api.main:app (via CMD in Dockerfile)          │
│    volumes:                                                 │
│      ./models   → /app/models   (reads model_latest.pkl)   │
│      ./results  → /app/results  (serves PNG graphs)        │
│      ./artifacts→ /app/artifacts (reads JSON + CSV)        │
└─────────────────────────────────────────────────────────────┘
```

### Volume sharing strategy:
- `pipeline` container WRITES to shared volumes (`models/`, `artifacts/`, `results/`)
- `api` container READS from the same shared volumes
- `depends_on: prefect` ensures Prefect starts first (but does NOT wait for readiness)
- After pipeline finishes, frontend calls `POST /reload_model` to hot-load the new pkl

---

## 7. React Frontend (frontend/)

**Tech:** Vite + React + React Router + Recharts + Lucide Icons

### Routes:
| Path | Component | Purpose |
|---|---|---|
| `/` | `Landing.jsx` | Hero landing page |
| `/dashboard` | `Dashboard.jsx` | KPIs, Recharts curves, PNG gallery, run history |
| `/pipeline` | `PipelineLog.jsx` | Table of all pipeline runs (from `/model_evaluations`) |
| `/pipeline/:id` | `RunDetail.jsx` | Detail view for a single run |
| `/eda` | `EDA.jsx` | Exploratory data analysis |
| `/evaluator` | `DatasetEvaluator.jsx` | CSV upload → batch predict |

### API URL resolution (`services/api.js`):
```js
const API_URL = import.meta.env.VITE_API_URL 
  || (import.meta.env.DEV ? 'http://localhost:8000' : '');
// In production (Docker), API_BASE = '' → relative URLs work
// because FastAPI serves the SPA at the root mount point
```

### Dashboard data flow:
```
useEffect → fetchData() → Promise.allSettled([
  GET /model_evaluations → evaluations table
  GET /latest_metrics    → Recharts ROC + PR curve data
  GET /graph_list        → list of latest_*.png filenames
])
→ graphs served from GET /graphs/{filename}?t=cache_bust
```

---

## 8. CI/CD (GitHub Actions)

### Workflow: `.github/workflows/ml_pipeline.yml`
```
on: push to main / pull_request

Job 1: test (always runs)
  → pip install -r requirements.txt
  → pytest tests/ -v

Job 2: train (only on main, needs: test)
  → python pipeline/prefect_flow.py
  → echo "Warning: Data not found" if fails (graceful skip)

Job 3: docker (only on main, needs: train)
  → docker build -t fraud-api:latest .
```

> [!WARNING]
> The `train` job in CI doesn't have the actual CSVs (`data/` is gitignored).
> It gracefully skips training but still validates the Docker build.

---

## 9. Known Issues & Current State

### What's working:
- ✅ One successful pipeline run recorded in `artifacts/results.csv`
- ✅ `artifacts/latest_metrics.json` has real ROC/PR curve data
- ✅ FastAPI backend fully implemented with all endpoints
- ✅ Dashboard displays interactive Recharts curves and PNG gallery
- ✅ Multi-stage Docker build compiles React + ships with Python

### Known pain points from conversation history:
1. **Docker timeout issues** — `pip install` in Docker can time out; `--default-timeout=100` was added
2. **Path resolution bugs** — Fixed: `PROJECT_ROOT` now correctly anchored to `Path(__file__).parent.parent` in both `api/main.py` and `pipeline/prefect_flow.py`
3. **Volume mount ordering** — Docker can create a *directory* instead of a file if the host file doesn't exist yet; solved by mounting the parent `artifacts/` directory, not individual files
4. **`depends_on` limitation** — `depends_on: prefect` only waits for container START, not until the Prefect server is actually ready to accept connections

### The `GRAPH_LABELS` mismatch:
In `Dashboard.jsx`, the label map uses:
```js
'roc_curve.png': '📈 ROC Curve'   // ← bare filename
```
But `/graph_list` returns `latest_roc_curve.png` — so labels currently fall through to the raw filename. Minor cosmetic bug.

---

## 10. Running the Full Stack

### Local dev (without Docker):
```bash
# Terminal 1 — Start Prefect server
prefect server start

# Terminal 2 — Train the model  
cd project
python pipeline/prefect_flow.py

# Terminal 3 — Start FastAPI
cd project
uvicorn api.main:app --reload --port 8000

# Terminal 4 — Start React dev server
cd project/frontend
npm run dev  # → http://localhost:5173
```

### Docker (full containerized):
```bash
cd project
docker-compose up --build
# → FastAPI + React SPA at http://localhost:8000
# → Prefect UI at http://localhost:4200
# → pipeline container trains and exits automatically
```

### Commands to run again (re-train):
```bash
docker-compose run pipeline python pipeline/prefect_flow.py
# Then hot-reload the API:
curl -X POST http://localhost:8000/reload_model
```
