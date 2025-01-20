import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
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

def fetch_data_with_retry(symbol, start_date, end_date, max_retries=3, retry_delay=5):
    """Fetch data with retry logic."""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date, interval='1h')
            
            if not df.empty:
                return df
            
            print(f"Attempt {attempt + 1}: Empty data received for {symbol}, retrying...")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error fetching data for {symbol}: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    raise ValueError(f"Failed to fetch data for {symbol} after {max_retries} attempts")

def get_historical_data(symbol, days=None):
    """Fetch historical data for a given symbol."""
    days = days or HISTORICAL_DAYS
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Try to fetch data with retries
        df = fetch_data_with_retry(symbol, start_date, end_date)
        
        # Verify we have enough data
        if len(df) < 24:  # Need at least 24 hours of data
            raise ValueError(f"Insufficient data points for {symbol}: got {len(df)}, need at least 24")
        
        df = calculate_technical_indicators(df)
        df = create_target_variable(df, threshold=0.02)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if df.empty:
            raise ValueError(f"No valid data points after processing for {symbol}")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error processing data for {symbol}: {str(e)}")

def prepare_features(df):
    """Prepare features for model training/prediction."""
    return df[FEATURE_COLUMNS]

def get_latest_data(symbol):
    """Get the most recent data for prediction."""
    try:
        # Try to get more data than needed to ensure we have enough after processing
        df = get_historical_data(symbol, days=7)  # Get last 7 days of hourly data
        if df is None or df.empty:
            raise ValueError(f"No data available for {symbol}")
            
        # Return the most recent complete data point
        return df.iloc[-1:] if not df.empty else None
    except Exception as e:
        print(f"Error in get_latest_data for {symbol}: {str(e)}")
        raise 