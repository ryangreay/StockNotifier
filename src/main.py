from uuid import uuid4
from fastapi import FastAPI, HTTPException, Security, Depends, APIRouter, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security.api_key import APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from . import auth, models, schemas
from .database import get_db
from telegram import Bot, Update
from telegram.ext import CallbackContext

from .model import StockPredictor
from .notifier import StockNotifier
from .data_collector import get_latest_data
from .telegram import TelegramBot
from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    PREDICTION_THRESHOLD,
    API_KEY,
    TELEGRAM_BOT_TOKEN
)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

router = APIRouter()

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
telegram_bot = TelegramBot()  # Initialize the TelegramBot instance

class TrainRequest(BaseModel):
    symbol: str
    test_size: Optional[float] = 0.2

class UntrainRequest(BaseModel):
    symbols: List[str]

class PredictionRequest(BaseModel):
    symbol: str
    notify: bool = True

class PredictionResponse(BaseModel):
    symbol: str
    prediction: int
    confidence: float
    predicted_movement: str
    up_probability: float
    down_probability: float
    movement_exceeds_threshold: bool
    timestamp: str
    current_price: float
    notification_sent: Optional[bool] = None
    notification_error: Optional[str] = None

class TelegramConnectRequest(BaseModel):
    connection_token: str
    user_id: int



@app.post("/connect-telegram")
async def connect_telegram(
    request: TelegramConnectRequest,
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """Connect user's Telegram account using token."""
    try:
        # Verify token and get chat_id
        connection = telegram_bot.pending_connections.get(request.connection_token)
        if not connection:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired connection token"
            )
        
        if datetime.now() > connection['expires_at']:
            telegram_bot.pending_connections.pop(request.connection_token)
            raise HTTPException(
                status_code=400,
                detail="Connection token has expired"
            )
        
        # Get user from database
        user = db.query(models.User).filter(models.User.id == request.user_id).first()
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        # Create or update telegram connection
        telegram_conn = db.query(models.UserTelegramConnection).filter(
            models.UserTelegramConnection.user_id == user.id
        ).first()
        
        if telegram_conn:
            telegram_conn.telegram_chat_id = connection['chat_id']
            telegram_conn.is_active = True
        else:
            telegram_conn = models.UserTelegramConnection(
                user_id=user.id,
                telegram_chat_id=connection['chat_id'],
                is_active=True
            )
            db.add(telegram_conn)
        
        db.commit()
        
        # Remove used token
        telegram_bot.pending_connections.pop(request.connection_token)
        
        return {
            "status": "success",
            "message": "Telegram account connected successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainRequest, api_key: str = Depends(get_api_key)):
    """Train the model on historical data for a given stock symbol."""
    try:
        predictor.train(request.symbol, request.test_size)
        return {"status": "success", "message": f"Model trained successfully for {request.symbol}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/untrain")
async def untrain_model(request: UntrainRequest, api_key: str = Depends(get_api_key)):
    """Remove specified symbols from the model's training data."""
    try:
        predictor.untrain(request.symbols)
        return {
            "status": "success", 
            "message": f"Symbols removed from training: {', '.join(request.symbols)}",
            "remaining_symbols": sorted(list(predictor.trained_symbols))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(
    request: PredictionRequest,
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db)
):
    """Get prediction for a stock symbol."""
    try:
        # Get user's stock subscription and settings first
        stock_sub = db.query(models.UserStock, models.UserSettings).join(
            models.UserSettings,
            models.UserStock.user_id == models.UserSettings.user_id
        ).filter(
            models.UserStock.symbol == request.symbol,
            models.UserStock.enabled == True
        ).first()
        
        if not stock_sub:
            raise HTTPException(
                status_code=404,
                detail="No active subscription found for this stock"
            )
        
        settings = stock_sub.UserSettings
        
        # Get latest data and prepare features using user settings
        raw_data = get_latest_data(
            request.symbol,
            prediction_window=settings.prediction_window,
            movement_threshold=settings.significant_movement_threshold
        )
        features = prepare_features(raw_data)
        
        # Make prediction
        prediction, probabilities = predictor.predict(request.symbol, features)
        current_price = raw_data['Close'].iloc[-1]
        
        # Extract probabilities for each class
        down_prob = probabilities[0]  # Probability of downward movement (class 0)
        up_prob = probabilities[1]    # Probability of upward movement (class 1)
        
        # Get the confidence (probability of predicted class)
        confidence = up_prob if prediction == 1 else down_prob
        
        notification_sent = None
        notification_error = None
        
        # Send notification if:
        # 1. Notification is requested
        # 2. Confidence exceeds prediction threshold
        # 3. Predicted movement exceeds significant movement threshold
        should_notify = (
            request.notify and
            confidence >= settings.prediction_threshold and
            (
                (prediction == 1 and up_prob >= settings.significant_movement_threshold) or
                (prediction == 0 and down_prob >= settings.significant_movement_threshold)
            )
        )
        
        if should_notify:
            success, error = await notifier.send_notification(
                user_id=stock_sub.UserStock.user_id,
                symbol=request.symbol,
                prediction=prediction,
                probability=probabilities,
                current_price=current_price,
                db=db
            )
            notification_sent = success
            notification_error = error
        
        return {
            "symbol": request.symbol,
            "prediction": prediction,
            "confidence": confidence,
            "predicted_movement": "up" if prediction == 1 else "down",
            "up_probability": up_prob,
            "down_probability": down_prob,
            "movement_exceeds_threshold": (
                (prediction == 1 and up_prob >= settings.significant_movement_threshold) or
                (prediction == 0 and down_prob >= settings.significant_movement_threshold)
            ),
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "notification_sent": notification_sent,
            "notification_error": notification_error
        }
        
    except Exception as e:
        # Send error notification if this was a subscribed stock
        if stock_sub:
            await notifier.send_error_notification(
                user_id=stock_sub.UserStock.user_id,
                symbol=request.symbol,
                error_message=str(e),
                db=db
            )
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    """Check if the service is healthy."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.post("/register", response_model=schemas.Token)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create tokens
    access_token = auth.create_access_token(
        data={"sub": str(db_user.id)},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = auth.create_refresh_token(db_user.id, db)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/token", response_model=schemas.Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Find user
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = auth.create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = auth.create_refresh_token(user.id, db)
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/google-login", response_model=schemas.Token)
async def google_login(
    token_data: schemas.GoogleToken,
    db: Session = Depends(get_db)
):
    # Verify Google token
    google_data = await auth.verify_google_token(token_data.token)
    
    # Find or create user
    user = db.query(models.User).filter(models.User.email == google_data['email']).first()
    if not user:
        user = models.User(
            email=google_data['email'],
            full_name=google_data['name'],
            google_id=google_data['sub']
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Create tokens
    access_token = auth.create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = auth.create_refresh_token(user.id, db)
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/refresh-token", response_model=schemas.Token)
async def refresh_token(
    token: schemas.RefreshToken,
    db: Session = Depends(get_db)
):
    try:
        # Verify refresh token
        payload = jwt.decode(token.refresh_token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Invalid refresh token")
        
        # Get stored token
        db_token = db.query(models.RefreshToken).filter(
            models.RefreshToken.token == token.refresh_token,
            models.RefreshToken.expires_at > datetime.utcnow()
        ).first()
        
        if not db_token:
            raise HTTPException(status_code=400, detail="Invalid or expired refresh token")
        
        # Create new tokens
        access_token = auth.create_access_token(
            data={"sub": str(db_token.user_id)},
            expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        new_refresh_token = auth.create_refresh_token(db_token.user_id, db)
        
        # Delete old refresh token
        db.delete(db_token)
        db.commit()
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
        
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid refresh token")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 