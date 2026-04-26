from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib
from io import StringIO
import os

app = FastAPI(title="Robust Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/model_latest.pkl"
model_pipeline = None

@app.on_event("startup")
def load_artifacts():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print("Unified Pipeline loaded successfully.")
    else:
        print("Warning: Pipeline not found. Run Prefect flow first.")

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
        # Convert pydantic strict model + dynamic extra kwargs to DataFrame
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
            "batch_results": [{"probability": float(p), "is_fraud_pred": int(f)} for p, f in zip(probs, preds)] # backward compatibility
        }
        
        if has_labels:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            response["metrics"] = {
                "accuracy": float(accuracy_score(y_true, preds)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, probs))
            }
            
        return response
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")
