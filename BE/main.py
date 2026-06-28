import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from preprocess import extract_features

BASE_DIR = os.path.dirname(__file__)

def load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

svm_model  = load("model_svm.pkl")
rf_model   = load("model_rf.pkl")
xgb_model  = load("model_xgb.pkl")

MODELS = {}
if svm_model:  MODELS["SVM"]           = svm_model
if rf_model:   MODELS["Random Forest"] = rf_model
if xgb_model:  MODELS["XGBoost"]       = xgb_model

print(f"Models loaded: {list(MODELS.keys())}")

app = FastAPI(title="Malaria Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}

@app.get("/")
def root():
    return {"message": "Malaria Detection API", "models": list(MODELS.keys())}

@app.get("/models")
def get_models():
    return {"models": list(MODELS.keys())}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "Random Forest"
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Format file harus PNG atau JPG.")

    if not MODELS:
        raise HTTPException(status_code=503, detail="Tidak ada model tersedia.")

    if model_name not in MODELS:
        raise HTTPException(status_code=400,
            detail=f"Model '{model_name}' tidak ditemukan. Pilihan: {list(MODELS.keys())}")

    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 10MB.")

    try:
        features, feature_summary = extract_features(img_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model  = MODELS[model_name]
    X      = features.reshape(1, -1)
    pred   = model.predict(X)[0]
    proba  = model.predict_proba(X)[0]

    label      = "Parasitized" if pred == 1 else "Uninfected"
    confidence = float(proba[int(pred)]) * 100

    # Dapatkan indeks dan skor fitur yang terpilih dari pipeline model
    try:
        selector = model.named_steps['selector']
        support = selector.get_support()
        scores = selector.scores_
        selected_indices = [int(i) for i, selected in enumerate(support) if selected]
        selected_scores = [float(scores[i]) for i in selected_indices]
    except Exception:
        try:
            estimator = (
                model.named_steps.get('rf') or
                model.named_steps.get('xgb') or
                model.named_steps.get('svm')
            )
            importances = estimator.feature_importances_
            selected_indices = [int(i) for i, v in enumerate(importances) if v > 0]
            selected_scores  = [float(importances[i]) for i in selected_indices]
        except Exception:
            selected_indices = []
            selected_scores  = []

    return JSONResponse({
        "label":      label,
        "confidence": round(confidence, 2),
        "model_used": model_name,
        "probabilities": {
            "Uninfected":  round(float(proba[0]) * 100, 2),
            "Parasitized": round(float(proba[1]) * 100, 2),
        },
        "features": feature_summary,
        "selected_indices": selected_indices,
        "selected_scores": selected_scores
    })