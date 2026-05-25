import os
import time
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MNC Real-Estate Pricing Engine Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RealEstateInputSchema(BaseModel):
    BHK: int = Field(..., ge=1, le=10)
    Size_in_SqFt: float = Field(..., gt=150, lt=100000)
    Year_Built: int = Field(2020, ge=1950, le=2026)
    Property_Type: str
    States: str
    City: str
    Locality: str

MEMORY_REGISTRY = {}

@app.on_event("startup")
def load_production_pipeline_binary():
    model_asset_path = "india_housing_production_pipeline.joblib"
    if os.path.exists(model_asset_path):
        MEMORY_REGISTRY["engine"] = joblib.load(model_asset_path)
        print("🚀 [MLOPS SUCCESS] Unified pipeline weights successfully cached in RAM.")
    else:
        # Fallback to alternative filename if first path misses
        alt_path = "xgboost_housing_pipeline.joblib"
        if os.path.exists(alt_path):
            MEMORY_REGISTRY["engine"] = joblib.load(alt_path)
            print("🚀 [MLOPS SUCCESS] Baseline pipeline weights cached in RAM.")
        else:
            print("⚠️ Warning: No .joblib model binary files detected in root catalog.")

@app.post("/api/v1/predict")
def run_live_inference(payload: RealEstateInputSchema):
    start_time = time.time()
    pipeline = MEMORY_REGISTRY.get("engine")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction engine uninitialized or model binary missing.")
        
    raw_payload_dict = {
        'BHK': payload.BHK,
        'Size_in_SqFt': payload.Size_in_SqFt,
        'Year_Built': payload.Year_Built,
        'Property_Type': payload.Property_Type,
        'States': payload.States,
        'City': payload.City,
        'Locality': payload.Locality
    }
    input_df = pd.DataFrame([raw_payload_dict])
    
    input_df['sqft_per_bhk'] = input_df['Size_in_SqFt'] / input_df['BHK']
    input_df['spatial_density_interaction'] = input_df['BHK'] * input_df['Size_in_SqFt']
    
    try:
        predicted_valuation = pipeline.predict(input_df)[0]
        execution_latency = (time.time() - start_time) * 1000
        return {
            "status": "success",
            "valuation": f"₹{predicted_valuation:,.2f} Lakhs",
            "telemetry": {
                "latency_ms": round(execution_latency, 2),
                "engine_variant": "XGBoost_Log_Stabilized"
            }
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Pipeline inference failure: {str(err)}")
