from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from typing_extensions import Annotated

class UserBase(BaseModel):
    """Base user schema with common attributes."""
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    """Schema for user registration."""
    password: Annotated[str, Field(min_length=8)]  # Password must be at least 8 characters

class UserUpdate(BaseModel):
    """Schema for updating user information."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[Annotated[str, Field(min_length=8)]] = None

class UserResponse(UserBase):
    """Schema for user response data."""
    id: int
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    """Schema for authentication tokens."""
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    """Schema for token payload data."""
    sub: Optional[str] = None

class RefreshToken(BaseModel):
    """Schema for refresh token requests."""
    refresh_token: str

class GoogleToken(BaseModel):
    """Schema for Google OAuth token."""
    token: str

class UserSettings(BaseModel):
    """Schema for user settings."""
    prediction_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    significant_movement_threshold: float = Field(default=0.025, ge=0.0, le=1.0)
    prediction_window: int = Field(default=12, ge=1)
    historical_days: int = Field(default=700, ge=1)
    training_timeframe: str = Field(default='1h', pattern='^(1h|6h|1d|1wk|1mo)$')
    notification_days: str = Field(default='1111100', pattern='^[01]{7}$')
    notify_market_open: bool = True
    notify_midday: bool = False
    notify_market_close: bool = True
    timezone: str = Field(default='America/New_York')

    class Config:
        from_attributes = True

class UserStockCreate(BaseModel):
    """Schema for adding user stocks."""
    symbols: list[str] = Field(..., min_items=1)  # At least one symbol required

class UserStockRemove(BaseModel):
    """Schema for removing user stocks."""
    symbols: list[str] = Field(..., min_items=1)  # At least one symbol required

class UserStockResponse(BaseModel):
    """Schema for user stock response."""
    user_id: int
    symbol: str
    enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True

class TelegramConnectRequest(BaseModel):
    """Schema for connecting Telegram account."""
    connection_token: str 