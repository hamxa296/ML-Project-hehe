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
MODEL_PATH     = PROJECT_ROOT / "models" / "model_latest.pkl"
RESULTS_PATH   = ARTIFACTS_DIR / "results.csv"
METRICS_PATH   = ARTIFACTS_DIR / "latest_metrics.json"
GRAPHS_DIR     = PROJECT_ROOT / "results" / "graphs"

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
            # Prevent browser caching — critical for dynamic graph updates
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        }
    )

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

# Serve static files from the 'frontend/dist' directory (Docker build step)
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")

    @app.exception_handler(404)
    async def custom_404_handler(request, __):
        return FileResponse(str(frontend_dist / "index.html"))
