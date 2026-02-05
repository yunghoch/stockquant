# lasps/api/main.py

"""LASPS v7a FastAPI Server.

REST API for stock prediction using the Sector-Aware Fusion Model.
Endpoints: health check, prediction, and sector listing.

Usage:
    uvicorn lasps.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch

app = FastAPI(title="LASPS v7a", version="7.0.0a1")


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    stock_codes: List[str]


class PredictionResponse(BaseModel):
    """Response body for a single stock prediction."""
    code: str
    name: str
    sector: str
    prediction: str
    confidence: float
    probabilities: List[float]
    llm_analysis: Optional[str] = None


# Global model/service (initialized on startup)
_predictor = None
_collector = None


@app.on_event("startup")
async def startup() -> None:
    """Load model checkpoint and initialize services on startup."""
    global _predictor
    # Checkpoint loading logic - connects to trained model
    pass


@app.get("/health")
async def health() -> dict:
    """Health check endpoint.

    Returns:
        Dict with status and model_loaded flag.
    """
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest) -> List[PredictionResponse]:
    """Run prediction for given stock codes.

    Args:
        request: PredictionRequest with list of stock codes.

    Returns:
        List of PredictionResponse for each stock.

    Raises:
        HTTPException: 503 if model is not loaded.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Actual prediction logic connects after service integration
    return []


@app.get("/sectors")
async def list_sectors() -> dict:
    """List all available sector codes with IDs and names.

    Returns:
        Dict mapping sector code to {id, name}.
    """
    from lasps.config.sector_config import SECTOR_CODES
    return {
        code: {"id": sid, "name": name}
        for code, (sid, name, _) in SECTOR_CODES.items()
    }
