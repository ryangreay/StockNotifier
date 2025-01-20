import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from src.config import (
    HISTORICAL_DAYS,
    FEATURE_COLUMNS,
    SIGNIFICANT_MOVEMENT_THRESHOLD,
    PREDICTION_WINDOW
)

def get_real_time_price(symbol):
    """Get real-time price data for a symbol."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.get_info()
        return {
            'currentPrice': info.get('regularMarketPrice', None),
            'open': info.get('regularMarketOpen', None),
            'high': info.get('regularMarketDayHigh', None),
            'low': info.get('regularMarketDayLow', None),
            'volume': info.get('regularMarketVolume', None)
        }
    except Exception as e:
        print(f"Error getting real-time price for {symbol}: {str(e)}")
        return None

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

def create_target_variable(df):
    """Create target variable based on future price movements."""
    # Use PREDICTION_WINDOW for the look-ahead period
    future_returns = df['Close'].pct_change(periods=PREDICTION_WINDOW).shift(-PREDICTION_WINDOW)
    df['target'] = (future_returns.abs() > SIGNIFICANT_MOVEMENT_THRESHOLD).astype(int)
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
        if len(df) < PREDICTION_WINDOW:  # Need at least PREDICTION_WINDOW hours of data
            raise ValueError(f"Insufficient data points for {symbol}: got {len(df)}, need at least {PREDICTION_WINDOW}")
        
        df = calculate_technical_indicators(df)
        df = create_target_variable(df)
        
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
    """Get the most recent data for prediction with real-time price."""
    try:
        # Get historical data for technical indicators
        df = get_historical_data(symbol, days=7)  # Get last 7 days of hourly data
        if df is None or df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Get real-time price data
        real_time = get_real_time_price(symbol)
        if real_time and real_time['currentPrice']:
            # Update the last row with real-time data
            latest_data = df.iloc[-1:].copy()
            latest_data.loc[latest_data.index[-1], 'Close'] = real_time['currentPrice']
            if real_time['open']:
                latest_data.loc[latest_data.index[-1], 'Open'] = real_time['open']
            if real_time['high']:
                latest_data.loc[latest_data.index[-1], 'High'] = real_time['high']
            if real_time['low']:
                latest_data.loc[latest_data.index[-1], 'Low'] = real_time['low']
            if real_time['volume']:
                latest_data.loc[latest_data.index[-1], 'Volume'] = real_time['volume']
            
            # Recalculate technical indicators for the updated data
            latest_data = calculate_technical_indicators(pd.concat([df[:-1], latest_data])).iloc[-1:]
            return latest_data
            
        return df.iloc[-1:]
    except Exception as e:
        print(f"Error in get_latest_data for {symbol}: {str(e)}")
        raise 