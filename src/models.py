from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, UniqueConstraint, Index
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import BIT
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    google_id = Column(String, unique=True, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserTelegramConnection(Base):
    __tablename__ = "user_telegram_connections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    telegram_chat_id = Column(String(100))
    connected_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True)
    prediction_threshold = Column(Float, default=0.85)
    significant_movement_threshold = Column(Float, default=0.025)
    prediction_window = Column(Integer, default=12)
    historical_days = Column(Integer, default=700)
    training_timeframe = Column(String(2), default='1H')
    notification_days = Column(BIT(7), default='1111100')
    notify_market_open = Column(Boolean, default=True)
    notify_midday = Column(Boolean, default=False)
    notify_market_close = Column(Boolean, default=True)
    timezone = Column(String(50), default='America/New_York')
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_user_settings_notifications', 
              'notification_days', 'notify_market_open', 
              'notify_midday', 'notify_market_close'),
    )

class UserStock(Base):
    __tablename__ = "user_stocks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String(10))
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('user_id', 'symbol'),
        Index('idx_user_stocks_enabled', 'user_id', 'enabled'),
    ) 