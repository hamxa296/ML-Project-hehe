from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
from io import StringIO
from pathlib import Path
import os
import json
import shutil
from datetime import datetime, timezone
from prefect.client.orchestration import get_client
from prefect.server.schemas.states import State, StateType

app = FastAPI(title="Robust Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All paths are anchored to the project root (/app inside Docker)
# so they always resolve correctly regardless of launch CWD.
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"      # maps to ./artifacts volume
MODELS_DIR     = PROJECT_ROOT / "models"
MODEL_PATH     = MODELS_DIR / "model_latest.pkl"
RESULTS_PATH   = ARTIFACTS_DIR / "results.csv"
METRICS_PATH   = ARTIFACTS_DIR / "latest_metrics.json"
GRAPHS_DIR     = PROJECT_ROOT / "results" / "graphs"
EDA_RAW_DIR    = ARTIFACTS_DIR / "eda_raw"
EDA_PROC_DIR   = ARTIFACTS_DIR / "eda_processed"
EDA_RAW_JSON   = EDA_RAW_DIR  / "raw_eda_data.json"
EDA_PROC_JSON  = EDA_PROC_DIR / "processed_eda_data.json"

model_pipeline = None

@app.on_event("startup")
def load_artifacts():
    global model_pipeline
    if MODEL_PATH.exists():
        model_pipeline = joblib.load(MODEL_PATH)
        print("Unified Pipeline loaded successfully.")
    else:
        print(f"Warning: Pipeline not found at {MODEL_PATH}. Run Prefect flow first.")

class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., description="Transaction Amount in USD")
    card1: Optional[float] = Field(0.0, description="Payment card identifier")
    addr1: Optional[float] = Field(0.0, description="Billing region")
    TransactionDT: Optional[float] = Field(0.0, description="Transaction delta time")
    P_emaildomain: Optional[str] = Field("", description="Purchaser email domain")

    model_config = {
        "extra": "allow"  # Allows arbitrary extra fields dynamically processed by sklearn pipeline
    }

class PredictionResponse(BaseModel):
    is_fraud: int
    probability: float

@app.get("/health")
def health():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded. Train the model first.")
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
def predict_single(payload: TransactionInput):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        df = pd.DataFrame([payload.model_dump()])
        prob = float(model_pipeline.predict_proba(df)[:, 1][0])
        pred = int(prob > 0.5)
        return {"is_fraud": pred, "probability": prob}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
def batch_predict(file: UploadFile = File(...)):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    try:
        contents = file.file.read()
        s = str(contents, 'utf-8')
        df = pd.read_csv(StringIO(s))

        has_labels = 'isFraud' in df.columns
        y_true = None
        if has_labels:
            y_true = df['isFraud'].astype(int)

        probs = model_pipeline.predict_proba(df)[:, 1]
        preds = (probs > 0.5).astype(int)

        results_table = []
        if has_labels:
            y_true_list = y_true.tolist()
            for i, (pred, prob, t) in enumerate(zip(preds, probs, y_true_list)):
                results_table.append({
                    "index": i,
                    "prediction": int(pred),
                    "probability": float(prob),
                    "true_label": int(t)
                })
        else:
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                results_table.append({
                    "index": i,
                    "prediction": int(pred),
                    "probability": float(prob)
                })

        response = {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "results_table": results_table,
            "batch_results": [{"probability": float(p), "is_fraud_pred": int(f)} for p, f in zip(probs, preds)]
        }

        if has_labels:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            response["metrics"] = {
                "accuracy":  float(accuracy_score(y_true, preds)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall":    float(recall_score(y_true, preds, zero_division=0)),
                "f1":        float(f1_score(y_true, preds, zero_division=0)),
                "roc_auc":   float(roc_auc_score(y_true, probs))
            }

        return response
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")

@app.get("/model_evaluations")
def get_model_evaluations():
    if not RESULTS_PATH.exists():
        return {"evaluations": []}
    try:
        df = pd.read_csv(RESULTS_PATH)
        df = df.fillna("")
        return {"evaluations": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")

@app.get("/latest_metrics")
def get_latest_metrics():
    if not METRICS_PATH.exists():
        return {"roc_curve": [], "pr_curve": []}
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {str(e)}")

@app.get("/graph_list")
def get_graph_list():
    """Returns only the stable 'latest_*' graph filenames (one per graph type).
    Versioned copies also exist on disk for audit purposes."""
    if not GRAPHS_DIR.exists():
        return {"graphs": []}
    # Only expose the 'latest_' prefix files — these are atomically overwritten each run
    graphs = sorted(f.name for f in GRAPHS_DIR.iterdir()
                    if f.suffix == ".png" and f.name.startswith("latest_"))
    return {"graphs": graphs}

@app.get("/graphs/{filename}")
def get_graph(filename: str):
    """Serves a specific PNG graph file by name. No-cache headers force browsers
    to re-fetch after each pipeline run instead of showing stale images."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    graph_path = GRAPHS_DIR / filename
    if not graph_path.exists() or graph_path.suffix != ".png":
        raise HTTPException(status_code=404, detail=f"Graph '{filename}' not found. Run the pipeline first.")
    return FileResponse(
        str(graph_path),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        }
    )

@app.get("/eda_list")
def get_eda_list():
    """Returns all EDA plots grouped by category (raw / processed)."""
    def _list(d: Path):
        if not d.exists():
            return []
        return sorted(f.name for f in d.iterdir() if f.suffix == ".png")
    return {
        "raw":       _list(EDA_RAW_DIR),
        "processed": _list(EDA_PROC_DIR),
    }

@app.get("/eda/{category}/{filename}")
def get_eda_plot(category: str, filename: str):
    """Serves an EDA PNG. category must be 'raw' or 'processed'."""
    if category not in ("raw", "processed"):
        raise HTTPException(status_code=400, detail="category must be 'raw' or 'processed'")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    base_dir = EDA_RAW_DIR if category == "raw" else EDA_PROC_DIR
    path = base_dir / filename
    if not path.exists() or path.suffix != ".png":
        raise HTTPException(status_code=404, detail=f"EDA plot '{filename}' not found. Run the pipeline first.")
    return FileResponse(
        str(path),
        media_type="image/png",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )

@app.get("/eda_stats/raw")
def get_raw_eda_stats():
    """Returns the computed raw EDA statistics as JSON for interactive frontend charts."""
    if not EDA_RAW_JSON.exists():
        return {"available": False, "message": "Run the pipeline first to generate EDA stats."}
    try:
        with open(EDA_RAW_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading raw EDA data: {str(e)}")

@app.get("/eda_stats/processed")
def get_processed_eda_stats():
    """Returns the computed processed EDA statistics as JSON for interactive frontend charts."""
    if not EDA_PROC_JSON.exists():
        return {"available": False, "message": "Run the pipeline first to generate EDA stats."}
    try:
        with open(EDA_PROC_JSON, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading processed EDA data: {str(e)}")

@app.post("/reload_model")
def reload_model():
    """Hot-reloads the latest trained model without restarting the API container.
    Call this from the frontend after a pipeline run completes."""
    global model_pipeline
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No model file found. Run the pipeline first.")
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        return {"status": "reloaded", "model_path": str(MODEL_PATH)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Queries the Prefect API for the current state of the ML pipeline."""
    try:
        async with get_client() as client:
            # Look for recent flow runs
            runs = await client.read_flow_runs(limit=5)
            if not runs:
                return {"status": "IDLE"}
            
            # Find any run that is not in a terminal state
            active_runs = [r for r in runs if r.state_name not in ("Completed", "Failed", "Cancelled", "Crashed")]
            
            if active_runs:
                r = active_runs[0]
                return {
                    "status": "RUNNING",
                    "state": r.state_name,
                    "name": r.name,
                    "id": str(r.id),
                    "start_time": r.start_time.isoformat() if r.start_time else None
                }
            
            return {"status": "IDLE", "last_run": runs[0].state_name}
    except Exception as e:
        return {"status": "UNKNOWN", "error": str(e)}

@app.post("/pipeline/cancel/{run_id}")
async def cancel_pipeline(run_id: str):
    """Cancels an active flow run in Prefect."""
    try:
        async with get_client() as client:
            # We set the state to CANCELLED. Prefect's engine will pick this up
            # and send a SIGTERM to the pipeline process.
            await client.set_flow_run_state(
                flow_run_id=run_id,
                state=State(type=StateType.CANCELLED, name="Cancelled")
            )
            return {"status": "cancelled", "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel run: {str(e)}")

@app.get("/models")
def list_models():
    """Returns a list of all versioned models available in the registry."""
    if not MODELS_DIR.exists():
        return {"models": []}
    
    models = []
    # Identify which actual file model_latest.pkl is pointing to (by comparing content or just having it)
    # Since we use shutil.copy2, we can't easily check symlink. 
    # But we can check results.csv to see which version corresponds to which metrics.
    
    for f in MODELS_DIR.iterdir():
        if f.suffix == ".pkl" and f.name != "model_latest.pkl":
            stats = f.stat()
            models.append({
                "name": f.name,
                "size_mb": round(stats.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat(),
            })
    
    # Sort by date descending (newest first)
    models.sort(key=lambda x: x["created_at"], reverse=True)
    return {"models": models}

@app.post("/models/activate/{name}")
def activate_model(name: str):
    """Hot-swaps the active model in memory and updates model_latest.pkl for persistence."""
    global model_pipeline
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid model name")
        
    target = MODELS_DIR / name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Model {name} not found")
    
    try:
        # 1. Load into memory for instant hot-swap
        model_pipeline = joblib.load(target)
        
        # 2. Overwrite the 'latest' pointer so it persists across container restarts
        shutil.copy2(target, MODEL_PATH)
        
        return {"status": "activated", "model": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate model: {str(e)}")

# Serve static files from the 'frontend/dist' directory (Docker build step)
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")

    @app.exception_handler(404)
    async def custom_404_handler(request, __):
        return FileResponse(str(frontend_dist / "index.html"))
