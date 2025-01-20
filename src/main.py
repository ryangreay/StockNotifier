from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime

from .model import StockPredictor
from .notifier import StockNotifier
from .data_collector import get_latest_data
from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    PREDICTION_THRESHOLD,
    API_KEY
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key_header

# Initialize components
predictor = StockPredictor()
notifier = StockNotifier()

class TrainRequest(BaseModel):
    symbol: str
    test_size: Optional[float] = 0.2

class PredictionRequest(BaseModel):
    symbol: str
    notify: bool = True

class PredictionResponse(BaseModel):
    symbol: str
    prediction: int
    confidence: float
    timestamp: str
    current_price: float
    notification_sent: Optional[bool] = None
    notification_error: Optional[str] = None

@app.post("/train")
async def train_model(request: TrainRequest, api_key: str = Depends(get_api_key)):
    """Train the model on historical data for a given stock symbol."""
    try:
        predictor.train(request.symbol, request.test_size)
        return {"status": "success", "message": f"Model trained successfully for {request.symbol}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
    """Make prediction for a given stock symbol."""
    try:
        # Get latest data
        latest_data = get_latest_data(request.symbol)
        if latest_data is None:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Make prediction
        prediction, probabilities = predictor.predict(latest_data)
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        current_price = latest_data['Close'].iloc[-1]
        
        response = PredictionResponse(
            symbol=request.symbol,
            prediction=int(prediction),
            confidence=float(confidence),
            timestamp=datetime.now().isoformat(),
            current_price=float(current_price)
        )
        
        # Send notification if requested and confidence exceeds threshold
        if request.notify and confidence >= PREDICTION_THRESHOLD:
            success, message = await notifier.send_notification(
                request.symbol, prediction, probabilities, current_price
            )
            response.notification_sent = success
            if not success:
                response.notification_error = message
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    """Check if the service is healthy."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 