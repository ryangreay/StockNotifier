from uuid import uuid4
from fastapi import FastAPI, HTTPException, Security, Depends, APIRouter, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
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
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import and_, func
from zoneinfo import ZoneInfo

from .model import StockPredictor
from .notifier import StockNotifier
from .data_collector import get_latest_data, prepare_features
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
    version=API_VERSION,
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
    },
    swagger_ui_parameters={"persistAuthorization": True}
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://stock-notifier.fly.dev"],  # Add both development and production URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Security schemes for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",  # Updated to match the actual endpoint
    scheme_name="OAuth2",
    auto_error=False
)

bearer_scheme = HTTPBearer(
    scheme_name="JWT",
    description="Enter your JWT token",
    auto_error=False
)

# Initialize components
predictor = StockPredictor()
notifier = StockNotifier()
telegram_bot = TelegramBot()

# Create router for auth endpoints
auth_router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
)

# Create routers
stocks_router = APIRouter(prefix="/stocks", tags=["stocks"])
user_stocks_router = APIRouter(prefix="/users/stocks", tags=["user stocks"])

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        user_id = int(payload.get("sub"))
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

    user = db.query(models.User).filter(
        and_(
            models.User.id == user_id,
            models.User.is_active == True,
            models.User.is_deleted == False
        )
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Account is inactive or has been deleted"
        )
        
    return user

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

@app.post("/connect-telegram")
async def connect_telegram(
    request: TelegramConnectRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Connect user's Telegram account using token."""
    try:
        # Find and verify pending connection
        pending_conn = db.query(models.PendingTelegramConnection).filter(
            models.PendingTelegramConnection.token == request.connection_token,
            models.PendingTelegramConnection.expires_at > datetime.now()
        ).first()
        
        if not pending_conn:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired connection token"
            )
        
        # Check if this chat is already connected to another user
        existing_connection = db.query(models.UserTelegramConnection).filter(
            models.UserTelegramConnection.telegram_chat_id == pending_conn.telegram_chat_id,
            models.UserTelegramConnection.user_id != current_user.id,
            models.UserTelegramConnection.is_active == True
        ).first()
        
        if existing_connection:
            raise HTTPException(
                status_code=400,
                detail="This Telegram chat is already connected to another user"
            )
        
        # Create or update telegram connection
        telegram_conn = db.query(models.UserTelegramConnection).filter(
            models.UserTelegramConnection.user_id == current_user.id
        ).first()
        
        if telegram_conn:
            telegram_conn.telegram_chat_id = pending_conn.telegram_chat_id
            telegram_conn.is_active = True
        else:
            telegram_conn = models.UserTelegramConnection(
                user_id=current_user.id,
                telegram_chat_id=pending_conn.telegram_chat_id,
                is_active=True
            )
            db.add(telegram_conn)
        
        # Delete the used pending connection
        db.delete(pending_conn)
        db.commit()
        
        return {
            "status": "success",
            "message": "Telegram account connected successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(
    request: TrainRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Train model for a stock symbol."""
    try:
        # Verify user has an active subscription for this symbol
        stock_sub = db.query(models.UserStock).filter(
            models.UserStock.user_id == current_user.id,
            models.UserStock.symbol == request.symbol,
            models.UserStock.enabled == True
        ).first()
        
        if not stock_sub:
            raise HTTPException(
                status_code=404,
                detail="No active subscription found for this stock"
            )
        
        # Train the model
        metrics = predictor.train(
            user_id=current_user.id,
            symbols=[request.symbol],
            test_size=request.test_size,
            db=db
        )
        
        return {
            "status": "success",
            "message": f"Model trained successfully for {request.symbol}",
            "metrics": metrics,
            "symbol": request.symbol,
            "test_size": request.test_size
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )

@app.post("/untrain")
async def untrain_model(
    request: UntrainRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove specified symbols from the model's training data."""
    try:
        # Verify user has active subscriptions for all symbols
        subscribed_symbols = db.query(models.UserStock.symbol).filter(
            models.UserStock.user_id == current_user.id,
            models.UserStock.symbol.in_(request.symbols),
            models.UserStock.enabled == True
        ).all()
        subscribed_symbols = [s[0] for s in subscribed_symbols]
        
        unsubscribed = set(request.symbols) - set(subscribed_symbols)
        if unsubscribed:
            raise HTTPException(
                status_code=404,
                detail=f"No active subscriptions found for symbols: {', '.join(unsubscribed)}"
            )
            
        predictor.untrain(current_user.id, request.symbols)
        
        # Get updated trained symbols for this user
        _, trained_symbols = predictor.get_user_model(current_user.id)
        
        return {
            "status": "success", 
            "message": f"Symbols removed from training: {', '.join(request.symbols)}",
            "remaining_symbols": sorted(list(trained_symbols))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(
    request: PredictionRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get prediction for a stock symbol."""
    try:
        # Get user's stock subscription and settings
        stock_sub = db.query(models.UserStock, models.UserSettings).join(
            models.UserSettings,
            models.UserStock.user_id == models.UserSettings.user_id
        ).filter(
            models.UserStock.user_id == current_user.id,
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
            timeframe=settings.training_timeframe,
            prediction_window=settings.prediction_window
        )
        features = prepare_features(raw_data)
        
        # Make prediction using user's model
        prediction, probabilities = predictor.predict(
            user_id=current_user.id,
            symbol=request.symbol,
            features=features
        )
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
        should_notify = request.notify and confidence >= settings.prediction_threshold
        
        if should_notify:
            success, error = await notifier.send_notification(
                user_id=current_user.id,
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
            "movement_exceeds_threshold": prediction == 1,  # Model already accounts for threshold
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "notification_sent": notification_sent,
            "notification_error": notification_error
        }
        
    except Exception as e:
        # Send error notification if this was a subscribed stock
        if stock_sub:
            await notifier.send_error_notification(
                user_id=current_user.id,
                symbol=request.symbol,
                error_message=str(e),
                db=db
            )
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is healthy (public endpoint)."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/auth")
async def health_check_auth(current_user: models.User = Depends(get_current_user)):
    """Check if the service is healthy (authenticated endpoint)."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "full_name": current_user.full_name
        }
    }

@app.post("/users/stocks", response_model=List[schemas.UserStockResponse])
async def add_user_stocks(
    request: schemas.UserStockCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add stocks to user's watchlist.
    
    - **symbols**: List of stock symbols to add (e.g. ["AAPL", "GOOGL"])
    - Maximum of 5 active stocks allowed per user
    
    Returns a list of added stocks with their status.
    """
    # Check current number of enabled stocks for the user
    current_stock_count = db.query(models.UserStock).filter(
        models.UserStock.user_id == current_user.id,
        models.UserStock.enabled == True
    ).count()

    # Calculate how many more stocks can be added
    stocks_limit = 5
    remaining_slots = stocks_limit - current_stock_count

    if remaining_slots <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"You have reached the maximum limit of {stocks_limit} stocks. Please remove some stocks before adding new ones."
        )

    if len(request.symbols) > remaining_slots:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot add {len(request.symbols)} stocks. You can only add {remaining_slots} more stock(s) to reach the limit of {stocks_limit}."
        )

    # First verify all stocks exist in available_stocks and are enabled
    available_stocks = db.query(models.AvailableStock).filter(
        models.AvailableStock.symbol.in_(request.symbols),
        models.AvailableStock.enabled == True
    ).all()
    
    # Check if any symbols are not available or not enabled
    available_symbols = {stock.symbol for stock in available_stocks}
    invalid_symbols = set(request.symbols) - available_symbols
    if invalid_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"The following symbols are not available or not enabled: {', '.join(invalid_symbols)}"
        )
    
    added_stocks = []
    for symbol in request.symbols:
        # Check if stock already exists for user
        existing_stock = db.query(models.UserStock).filter(
            models.UserStock.user_id == current_user.id,
            models.UserStock.symbol == symbol
        ).first()
        
        if existing_stock:
            # If exists but disabled, enable it
            if not existing_stock.enabled:
                # Check limit again when enabling a disabled stock
                if current_stock_count >= stocks_limit:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot enable more stocks. You have reached the limit of {stocks_limit} stocks."
                    )
                existing_stock.enabled = True
                db.commit()
                db.refresh(existing_stock)
                current_stock_count += 1
            added_stocks.append(existing_stock)
            continue
            
        # Create new stock subscription
        new_stock = models.UserStock(
            user_id=current_user.id,
            symbol=symbol,
            enabled=True
        )
        
        db.add(new_stock)
        db.commit()
        db.refresh(new_stock)
        added_stocks.append(new_stock)
    
    return added_stocks

@app.get("/users/stocks", response_model=List[schemas.UserStockResponse])
async def get_user_stocks(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all active stocks in user's watchlist.
    
    Returns a list of stocks that the user is currently monitoring.
    """
    stocks = db.query(models.UserStock).filter(
        models.UserStock.user_id == current_user.id,
        models.UserStock.enabled == True
    ).order_by(models.UserStock.symbol).all()
    
    return stocks

@app.delete("/users/stocks", response_model=List[schemas.UserStockResponse])
async def remove_user_stocks(
    request: schemas.UserStockRemove,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove stocks from user's watchlist.
    
    - **symbols**: List of stock symbols to remove (e.g. ["AAPL", "GOOGL"])
    
    Returns a list of removed stocks.
    """
    removed_stocks = []
    
    for symbol in request.symbols:
        # Find the stock subscription
        stock = db.query(models.UserStock).filter(
            models.UserStock.user_id == current_user.id,
            models.UserStock.symbol == symbol,
            models.UserStock.enabled == True
        ).first()
        
        if stock:
            # Disable the stock subscription instead of deleting it
            stock.enabled = False
            db.commit()
            db.refresh(stock)
            removed_stocks.append(stock)
    
    if not removed_stocks:
        raise HTTPException(
            status_code=404,
            detail="No active subscriptions found for the specified symbols"
        )
    
    return removed_stocks

@app.put("/users/settings", response_model=schemas.UserSettings)
async def update_user_settings(
    settings: schemas.UserSettings,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user settings.
    
    Settings that can be updated:
    - **prediction_threshold**: Confidence threshold for predictions (0.0-1.0)
    - **significant_movement_threshold**: Price movement threshold (0.0-1.0)
    - **prediction_window**: Hours to look ahead for predictions (>= 1)
    - **historical_days**: Days of historical data to use (>= 1)
    - **training_timeframe**: Data timeframe ('1h', '6h', '1d', '1wk', '1mo')
    - **notification_days**: Binary string for days to notify (e.g. '1111100' for Mon-Fri)
    - **notify_market_open**: Whether to notify at market open
    - **notify_midday**: Whether to notify at midday
    - **notify_market_close**: Whether to notify at market close
    - **timezone**: User's timezone (e.g. 'America/New_York')
    """
    # Get existing settings or create new
    user_settings = db.query(models.UserSettings).filter(
        models.UserSettings.user_id == current_user.id
    ).first()
    
    if not user_settings:
        user_settings = models.UserSettings(user_id=current_user.id)
        db.add(user_settings)
    
    # Update settings
    for field, value in settings.dict().items():
        setattr(user_settings, field, value)
    
    db.commit()
    db.refresh(user_settings)
    
    return user_settings

@app.get("/users/settings", response_model=schemas.UserSettings)
async def get_user_settings(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user settings.
    
    Returns the current settings for the authenticated user.
    If no settings exist, returns default settings.
    """
    # Get existing settings
    user_settings = db.query(models.UserSettings).filter(
        models.UserSettings.user_id == current_user.id
    ).first()
    
    if not user_settings:
        # Create default settings if none exist
        user_settings = models.UserSettings(user_id=current_user.id)
        db.add(user_settings)
        db.commit()
        db.refresh(user_settings)
    
    return user_settings

@auth_router.post("/register", response_model=schemas.Token)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.
    
    - **email**: Valid email address
    - **full_name**: User's full name
    - **password**: Password (minimum 8 characters)
    """
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

@auth_router.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Get access token using username (email) and password.
    
    The token can then be used in the Authorize button at the top:
    1. Click Authorize
    2. In the "Value" field enter: Bearer token
    3. Click Authorize
    """
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

@auth_router.post("/google-login", response_model=schemas.Token)
async def google_login(
    token_data: schemas.GoogleToken,
    db: Session = Depends(get_db)
):
    """
    Login with Google OAuth token.
    
    - **token**: Google OAuth ID token
    """
    google_data = await auth.verify_google_token(token_data.token)
    
    # First try to find user by google_id (most reliable method)
    user = db.query(models.User).filter(
        models.User.google_id == google_data['sub']
    ).first()
    
    if not user:
        # If no user found by google_id, check email
        existing_user = db.query(models.User).filter(
            models.User.email == google_data['email']
        ).first()
        
        if existing_user:
            if existing_user.hashed_password:
                # User exists with password (email signup) - don't allow Google login
                raise HTTPException(
                    status_code=400,
                    detail="Email already registered with password. Please use email login."
                )
            elif existing_user.google_id and existing_user.google_id != google_data['sub']:
                # User exists with different Google account
                raise HTTPException(
                    status_code=400,
                    detail="Email already registered with different Google account."
                )
            else:
                # Update existing user with google_id if missing
                existing_user.google_id = google_data['sub']
                user = existing_user
        else:
            # Create new user
            user = models.User(
                email=google_data['email'],
                full_name=google_data['name'],
                google_id=google_data['sub'],
                hashed_password=None  # Explicitly set None for Google users
            )
            db.add(user)
    
    # Update user info if needed
    if user.full_name != google_data['name']:
        user.full_name = google_data['name']
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    # Create tokens
    access_token = auth.create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = auth.create_refresh_token(user.id, db)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@auth_router.post("/refresh-token", response_model=schemas.Token)
async def refresh_token(
    token: schemas.RefreshToken,
    db: Session = Depends(get_db)
):
    """
    Get new access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    """
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

@auth_router.get("/check-deleted-account/{email}", response_model=schemas.DeletedAccountResponse)
async def check_deleted_account(email: str, db: Session = Depends(get_db)):
    """
    Check if a deleted account exists and can be reactivated.
    
    - **email**: Email address of the deleted account
    
    Returns information about the deleted account including reactivation deadline.
    """
    # Find the deleted user
    user = db.query(models.User).filter(
        models.User.email == email,
        models.User.is_deleted == True
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Calculate reactivation deadline (30 days from deletion)
    reactivation_deadline = user.deleted_at + timedelta(days=30)
    # Use UTC for comparison
    now = datetime.now(ZoneInfo("UTC"))
    can_reactivate = now < reactivation_deadline
    
    return {
        "email": user.email,
        "deletion_date": user.deleted_at,
        "can_reactivate": can_reactivate,
        "reactivation_deadline": reactivation_deadline,
        "deletion_type": "google" if user.google_id else "password"
    }

@auth_router.post("/reactivate", response_model=schemas.Token)
async def reactivate_account(
    reactivation: schemas.ReactivateAccountRequest,
    db: Session = Depends(get_db)
):
    """
    Reactivate a deleted account.
    
    - **email**: Email address of the account
    - **password**: Password (for password-based accounts)
    - **google_token**: Google OAuth token (for Google-based accounts)
    
    Returns new access and refresh tokens if successful.
    """
    # Find the deleted user
    user = db.query(models.User).filter(
        models.User.email == reactivation.email,
        models.User.is_deleted == True
    ).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Check if within reactivation window
    reactivation_deadline = user.deleted_at + timedelta(days=30)
    now = datetime.now(ZoneInfo("UTC"))
    if now > reactivation_deadline:
        raise HTTPException(
            status_code=400,
            detail="Reactivation period has expired. Please create a new account."
        )
    
    # Verify credentials
    if user.google_id:
        if not reactivation.google_token:
            raise HTTPException(
                status_code=400,
                detail="This account was created with Google. Please use Google sign-in to reactivate."
            )
        # Verify Google token
        google_data = await auth.verify_google_token(reactivation.google_token)
        if google_data['sub'] != user.google_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid Google account"
            )
    else:
        if not reactivation.password:
            raise HTTPException(
                status_code=400,
                detail="Password is required for reactivation"
            )
        if not auth.verify_password(reactivation.password, user.hashed_password):
            raise HTTPException(
                status_code=400,
                detail="Invalid password"
            )
    
    # Reactivate the account
    user.is_deleted = False
    user.is_active = True
    user.deleted_at = None
    user.last_login = datetime.utcnow()
    
    # Reactivate user settings and stocks if they exist
    db.query(models.UserStock).filter(
        models.UserStock.user_id == user.id
    ).update({"enabled": True})
    
    db.commit()
    
    # Create new tokens
    access_token = auth.create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = auth.create_refresh_token(user.id, db)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.get("/telegram-status")
async def get_telegram_status(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Check if user has an active Telegram connection."""
    telegram_conn = db.query(models.UserTelegramConnection).filter(
        models.UserTelegramConnection.user_id == current_user.id,
        models.UserTelegramConnection.is_active == True
    ).first()
    
    return {
        "is_connected": telegram_conn is not None,
        "connected_at": telegram_conn.connected_at if telegram_conn else None
    }

@app.get("/stocks", response_model=List[schemas.AvailableStockResponse])
async def get_available_stocks(
    enabled: Optional[bool] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of available stocks.
    
    Parameters:
    - **enabled**: Optional filter for enabled/disabled stocks
    
    Returns list of stocks with their symbols, names, and enabled status.
    """
    query = db.query(models.AvailableStock)
    if enabled is not None:
        query = query.filter(models.AvailableStock.enabled == enabled)
    return query.order_by(models.AvailableStock.name).all()

@app.delete("/users/me", status_code=204)
async def delete_account(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Soft delete the current user's account.
    This will:
    1. Mark the account as deleted
    2. Deactivate all stocks
    3. Remove Telegram connections
    4. Invalidate refresh tokens
    """
    try:
        # Mark user as deleted
        current_user.is_deleted = True
        current_user.is_active = False
        current_user.deleted_at = func.now()
        
        # Deactivate all user's stocks
        db.query(models.UserStock).filter(
            models.UserStock.user_id == current_user.id
        ).update({"enabled": False})
        
        # Remove Telegram connections
        db.query(models.UserTelegramConnection).filter(
            models.UserTelegramConnection.user_id == current_user.id
        ).update({"is_active": False})
        
        # Invalidate all refresh tokens
        db.query(models.RefreshToken).filter(
            models.RefreshToken.user_id == current_user.id
        ).delete()
        
        db.commit()
        return None
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to delete account"
        )

# Add routers to app
app.include_router(auth_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 