"""FastAPI serving layer for churn inference."""

from datetime import datetime
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from model import MLPClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    total_orders: int = Field(..., ge=0)
    total_revenue: float = Field(..., gt=0.0)
    avg_revenue: float = Field(..., gt=0.0)
    std_revenue: float = Field(default=0.0, ge=0.0)
    total_profit: float = Field(...)
    avg_profit: float = Field(...)
    avg_discount: float = Field(..., ge=0.0, le=1.0)
    total_quantity: int = Field(..., ge=0)
    avg_quantity: float = Field(..., ge=0.0)
    days_since_last_purchase: int = Field(..., ge=0)
    customer_lifetime_days: int = Field(..., ge=1)
    purchase_frequency: float = Field(..., ge=0.0)
    Region_encoded: int = Field(..., ge=0)
    Product_Category_encoded: int = Field(..., ge=0)
    Customer_Segment_encoded: int = Field(..., ge=0)
    Payment_Method_encoded: int = Field(..., ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: CustomerInput


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    prediction_label: str
    confidence: float
    timestamp: str


app = FastAPI(
    title="MLP Churn Classifier API",
    description="FastAPI service for customer churn prediction",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = None
SCALER = None
FEATURE_NAMES = list(CustomerInput.model_fields.keys())
SCHEMA_FEATURES = list(CustomerInput.model_fields.keys())


def _load_checkpoint_model(model_path: Path) -> MLPClassifier:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict'")

    state_dict = checkpoint["model_state_dict"]
    linear_layers = [
        key for key in state_dict.keys() if key.startswith("network") and key.endswith("weight")
    ]
    layer_indices = sorted({int(key.split(".")[1]) for key in linear_layers})
    hidden_dims = []
    for layer_idx in layer_indices[:-1]:
        hidden_dims.append(state_dict[f"network.{layer_idx}.weight"].shape[0])

    input_dim = state_dict[f"network.{layer_indices[0]}.weight"].shape[1]
    model = MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=0.3)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@app.on_event("startup")
async def startup_event():
    global MODEL, SCALER, FEATURE_NAMES

    candidates = [
        ROOT_DIR / "artifacts" / "mlp_churn_classifier_final.pth",
        ROOT_DIR / "serving" / "mlp_churn_classifier_final.pth",
        ROOT_DIR / "mlp_classifier.pth",
    ]

    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        raise RuntimeError("No model file found for FastAPI startup")

    MODEL = _load_checkpoint_model(model_path)
    logger.info("Loaded model from %s", model_path)

    scaler_path = ROOT_DIR / "artifacts" / "scaler.pkl"
    feature_names_path = ROOT_DIR / "artifacts" / "feature_names.pkl"

    if scaler_path.exists():
        SCALER = joblib.load(scaler_path)
    if feature_names_path.exists():
        loaded_feature_names = joblib.load(feature_names_path)
        if isinstance(loaded_feature_names, list) and all(
            isinstance(name, str) for name in loaded_feature_names
        ):
            if set(loaded_feature_names).issubset(set(SCHEMA_FEATURES)):
                FEATURE_NAMES = loaded_feature_names
            else:
                logger.warning(
                    "Loaded feature_names are incompatible with strict request schema; "
                    "falling back to schema feature order"
                )
                FEATURE_NAMES = SCHEMA_FEATURES
        else:
            FEATURE_NAMES = SCHEMA_FEATURES


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": MODEL is not None,
        "service": "churn-fastapi",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features_dict = request.features.model_dump()
    missing = [feature for feature in FEATURE_NAMES if feature not in features_dict]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    x = np.array([[features_dict[name] for name in FEATURE_NAMES]], dtype=np.float32)
    if SCALER is not None:
        x = SCALER.transform(x)

    with torch.no_grad():
        probability = float(MODEL(torch.FloatTensor(x)).item())

    prediction = probability >= 0.5
    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=prediction,
        prediction_label="Churned" if prediction else "Active",
        confidence=round(max(probability, 1 - probability), 4),
        timestamp=datetime.utcnow().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.app:app", host="0.0.0.0", port=5000, reload=False)
