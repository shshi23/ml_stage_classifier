from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .model_loader import manager
from .preprocessing import preprocess_input

app = FastAPI(
    title="Clinical Stage Predaiction API",
    description="ML Service for predicting clinical stages (1-8)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

class PredictionResponse(BaseModel):
    stage: int
    confidence: float
    probabilities: Dict[str, float]
    shap_explanation: List[Dict[str, Any]]

@app.on_event("startup")
def startup_event():
    manager.load()
    if not manager.is_loaded:
        raise RuntimeError("Не удалось загрузить модель. Проверьте папку models/")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": manager.is_loaded}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Dict[str, Any]):
    if not manager.is_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        raw_data = request
        X_processed = preprocess_input(raw_data, manager.feature_names)

        pred_encoded = int(manager.model.predict(X_processed)[0])
        proba_array = manager.model.predict_proba(X_processed)[0]
        
        pred_stage = pred_encoded + manager.stage_offset
        confidence = float(proba_array[pred_encoded])

        probabilities = {
            str(i + manager.stage_offset): float(p) 
            for i, p in enumerate(proba_array)
        }

        shap_raw = manager.explainer.shap_values(X_processed)

        # Извлекаем значения для предсказанного класса
        if isinstance(shap_raw, list):
            shap_row = shap_raw[pred_encoded][0]
        elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
            shap_row = shap_raw[0, :, pred_encoded]
        elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 2:
            shap_row = shap_raw[0]
        else:
            shap_row = np.zeros(len(manager.feature_names))

        abs_shap = np.abs(shap_row).flatten()

        feature_names_list = [str(name) for name in manager.feature_names]
        
        top_indices = sorted(range(len(abs_shap)), key=lambda i: abs_shap[i], reverse=True)[:3]
        
        shap_explanation = []
        for idx in top_indices:
            feature_name = feature_names_list[idx]
            feature_val = X_processed.iloc[0, idx]
            impact = float(abs_shap[idx])
            
            if pd.isna(feature_val):
                val_out = None
            elif isinstance(feature_val, (np.integer, np.floating)):
                val_out = feature_val.item()
            elif isinstance(feature_val, (int, float)):
                val_out = float(feature_val)
            else:
                val_out = str(feature_val)
            
            shap_explanation.append({
                "feature": feature_name,
                "value": val_out,
                "impact": impact 
            })

        return {
            "stage": pred_stage,
            "confidence": confidence,
            "probabilities": probabilities,
            "shap_explanation": shap_explanation
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))