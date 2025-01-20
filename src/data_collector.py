import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import HISTORICAL_DAYS, FEATURE_COLUMNS

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset."""
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    std = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA_20'] + (std * 2)
    df['Lower_Band'] = df['MA_20'] - (std * 2)
    
    return df

def create_target_variable(df, threshold):
    """Create target variable based on future price movements."""
    future_returns = df['Close'].pct_change(periods=24).shift(-24)  # 24-hour future returns
    df['target'] = (future_returns.abs() > threshold).astype(int)
    return df

def get_historical_data(symbol, days=None):
    """Fetch historical data for a given symbol."""
    days = days or HISTORICAL_DAYS
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date, interval='1h')
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        df = calculate_technical_indicators(df)
        df = create_target_variable(df, threshold=0.02)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    except Exception as e:
        raise Exception(f"Error fetching data for {symbol}: {str(e)}")

def prepare_features(df):
    """Prepare features for model training/prediction."""
    return df[FEATURE_COLUMNS]

def get_latest_data(symbol):
    """Get the most recent data for prediction."""
    df = get_historical_data(symbol, days=5)  # Get last 5 days of hourly data
    return df.iloc[-1:] if not df.empty else None 