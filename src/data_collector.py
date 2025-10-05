import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import redis
import json
import requests
from src.config import (
    HISTORICAL_DAYS,
    FEATURE_COLUMNS,
    SIGNIFICANT_MOVEMENT_THRESHOLD,
    PREDICTION_WINDOW,
    REDIS_URL,
    ALPHA_VANTAGE_API_KEY,
    FINNHUB_API_KEY
)

# Initialize Redis client for caching
redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None

# Cache configuration with longer expiration times for Alpha Vantage data
CACHE_EXPIRY = {
    '1h': 3600,     # 1 hour for hourly data
    '6h': 7200,     # 2 hours for 6-hour data
    '1d': 86400,    # 24 hours for daily data
    '1wk': 172800,  # 48 hours for weekly data
    '1mo': 259200   # 72 hours for monthly data
}

timeframe_map = {
        '1H': '1h',     # 1 hour data
        '6H': '6h',     # 6 hour data
        '1D': '1d',     # daily data
        '1W': '1wk',    # weekly data
        '1M': '1mo'     # monthly data
    }

# Maximum days of historical data available for each timeframe
timeframe_max_days = {
    '1h': 729,     # 2 years of hourly data
    '6h': 729,     # 2 years of 6-hour data
    '1d': 7299,    # 20 years of daily data
    '1wk': 7299,   # 20 years of weekly data
    '1mo': 7299    # 20 years of monthly data
}

# Alpha Vantage API configuration
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_FUNCTIONS = {
    '1h': 'TIME_SERIES_INTRADAY',
    '1d': 'TIME_SERIES_DAILY',
    '1wk': 'TIME_SERIES_WEEKLY',
    '1mo': 'TIME_SERIES_MONTHLY'
}

def get_timeframe_window(interval: str) -> int:
    """Get the maximum number of days available for a given timeframe."""
    return timeframe_max_days.get(interval, 729)  # Default to 730 days if unknown interval

def get_alpha_vantage_data(symbol: str, interval: str = '1d') -> pd.DataFrame:
    """Fetch data from Alpha Vantage API."""
    # Check if we've exceeded the daily limit
    daily_limit_key = 'alpha_vantage_daily_count'
    current_count = int(redis_client.get(daily_limit_key) or 0)
    
    if current_count >= 25:  # Daily limit reached
        print("Alpha Vantage daily limit reached, using Yahoo Finance fallback")
        return None
        
    try:
        # Map interval to Alpha Vantage function
        function = ALPHA_VANTAGE_FUNCTIONS.get(interval, 'TIME_SERIES_DAILY')
        
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'outputsize': 'full'
        }
        
        # Add interval parameter for intraday data
        if interval == '1h':
            params['interval'] = '60min'
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            print(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
            return None
            
        # Check for rate limit message
        if 'Note' in data and 'API call frequency' in data['Note']:
            print(f"Alpha Vantage rate limit warning for {symbol}: {data['Note']}")
            return None
        
        # Find the time series key
        time_series_keys = [k for k in data.keys() if 'Time Series' in k]
        if not time_series_keys:
            print(f"No time series data found in Alpha Vantage response for {symbol}. Response keys: {list(data.keys())}")
            return None
            
        time_series_key = time_series_keys[0]
        time_series_data = data[time_series_key]
        
        if not time_series_data:
            print(f"Empty time series data from Alpha Vantage for {symbol}")
            return None
        
        # Increment API call counter
        pipe = redis_client.pipeline()
        pipe.incr(daily_limit_key)
        # Set expiry for counter if it doesn't exist
        pipe.expire(daily_limit_key, 86400)  # 24 hours
        pipe.execute()
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series_data, orient='index')
        
        # Rename columns to match our format
        column_map = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }
        df = df.rename(columns=column_map)
        
        # Convert string values to float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort index in ascending order
        df = df.sort_index()
        
        return df
        
    except requests.RequestException as e:
        print(f"Network error fetching Alpha Vantage data for {symbol}: {str(e)}")
        return None
    except ValueError as e:
        print(f"Error parsing Alpha Vantage data for {symbol}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching Alpha Vantage data for {symbol}: {str(e)}")
        return None

def get_cached_data(symbol: str, interval: str):
    """Get data from cache if available."""
    if not redis_client:
        return None
        
    cache_key = f"stock_data:{symbol}:{interval}"
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        try:
            data_dict = json.loads(cached_data)
            return pd.DataFrame.from_dict(data_dict)
        except Exception as e:
            print(f"Error loading cached data: {str(e)}")
    
    return None

def cache_data(symbol: str, interval: str, df: pd.DataFrame):
    """Cache stock data in Redis with longer expiration times."""
    if not redis_client:
        return
        
    try:
        cache_key = f"stock_data:{symbol}:{interval}"
        data_dict = df.to_dict()
        redis_client.setex(
            cache_key,
            CACHE_EXPIRY[interval],
            json.dumps(data_dict)
        )
    except Exception as e:
        print(f"Error caching data: {str(e)}")

def get_alpha_vantage_quote(symbol: str) -> dict:
    """Get real-time quote from Alpha Vantage."""
    try:
        # Check if we've exceeded the daily limit
        daily_limit_key = 'alpha_vantage_daily_count'
        current_count = int(redis_client.get(daily_limit_key) or 0)
        
        if current_count >= 25:  # Daily limit reached
            print("Alpha Vantage daily limit reached, using Yahoo Finance fallback")
            return None
            
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            print(f"Alpha Vantage error: {data['Error Message']}")
            return None
            
        # Check for rate limit message
        if 'Note' in data and 'API call frequency' in data['Note']:
            print("Alpha Vantage rate limit warning")
            return None
        
        # Increment API call counter
        pipe = redis_client.pipeline()
        pipe.incr(daily_limit_key)
        pipe.expire(daily_limit_key, 86400)  # 24 hours
        pipe.execute()
        
        quote = data.get('Global Quote', {})
        if quote:
            return {
                'currentPrice': float(quote.get('05. price', 0)),
                'open': float(quote.get('02. open', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'volume': float(quote.get('06. volume', 0))
            }
        return None
    except Exception as e:
        print(f"Error getting Alpha Vantage quote: {str(e)}")
        return None

def get_finnhub_quote(symbol: str) -> dict:
    """Get real-time quote from Finnhub."""
    try:
        # Check rate limit in Redis
        rate_limit_key = 'finnhub_minute_count'
        current_count = int(redis_client.get(rate_limit_key) or 0)
        
        if current_count >= 60:  # Minute limit reached
            print("Finnhub rate limit reached, using fallback")
            return None
            
        headers = {'X-Finnhub-Token': FINNHUB_API_KEY}
        response = requests.get(
            f'https://finnhub.io/api/v1/quote?symbol={symbol}',
            headers=headers
        )
        
        # Increment and set expiry for rate limit counter
        pipe = redis_client.pipeline()
        pipe.incr(rate_limit_key)
        pipe.expire(rate_limit_key, 60)  # Reset counter after 1 minute
        pipe.execute()
        
        data = response.json()
        
        if 'c' in data:  # Current price exists
            return {
                'currentPrice': data['c'],
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'volume': data.get('v', 0)  # Some symbols might have volume
            }
        return None
    except Exception as e:
        print(f"Error getting Finnhub quote: {str(e)}")
        return None

def get_real_time_price(symbol: str) -> dict:
    """Get real-time price data with smart caching to minimize API calls."""
    cache_key = f"realtime_price:{symbol}"
    
    # Try to get cached data first
    if redis_client:
        cached_price = redis_client.get(cache_key)
        if cached_price:
            return json.loads(cached_price)
    
    # Get current market status
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    is_market_hours = (
        (current_hour > 9 or (current_hour == 9 and current_minute >= 30)) and 
        current_hour < 16
    )
    is_pre_market = current_hour >= 4 and (current_hour < 9 or (current_hour == 9 and current_minute < 30))
    is_after_hours = current_hour >= 16 and current_hour < 20
    
    # Try Alpha Vantage first
    price_data = get_alpha_vantage_quote(symbol)
    
    # Fall back to Yahoo Finance if Alpha Vantage fails
    if not price_data:
        try:
            stock = yf.Ticker(symbol)
            info = stock.get_info()
            price_data = {
                'currentPrice': info.get('regularMarketPrice', None),
                'open': info.get('regularMarketOpen', None),
                'high': info.get('regularMarketDayHigh', None),
                'low': info.get('regularMarketDayLow', None),
                'volume': info.get('regularMarketVolume', None)
            }
        except Exception as e:
            print(f"Error getting Yahoo Finance price: {str(e)}")
            return None
    
    # Cache successful response with smart duration
    if price_data and redis_client:
        # Default cache duration
        cache_duration = 300  # 5 minutes
        
        # Adjust cache duration based on market hours
        if is_market_hours:
            cache_duration = 60  # 1 minute during market hours
        elif is_pre_market or is_after_hours:
            cache_duration = 300  # 5 minutes during extended hours
        else:
            cache_duration = 1800  # 30 minutes outside trading hours
            
        # Special case: Near market open/close, cache for shorter duration
        if (current_hour == 9 and current_minute >= 15) or (current_hour == 15 and current_minute >= 45):
            cache_duration = 30  # 30 seconds near market open/close
            
        redis_client.setex(cache_key, cache_duration, json.dumps(price_data))
    
    return price_data

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
    # MACD Histogram and z-score normalization to reduce price correlation
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    macd_hist_mean = df['MACD_Hist'].rolling(window=60, min_periods=1).mean()
    macd_hist_std = df['MACD_Hist'].rolling(window=60, min_periods=1).std()
    # Avoid division by zero for MACD_Hist_Z
    safe_std = macd_hist_std.replace(0, np.nan)
    df['MACD_Hist_Z'] = (df['MACD_Hist'] - macd_hist_mean) / safe_std
    
    # Bollinger Bands
    std = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA_20'] + (std * 2)
    df['Lower_Band'] = df['MA_20'] - (std * 2)
    # Bollinger %B indicator
    bb_width = (df['Upper_Band'] - df['Lower_Band'])
    # Avoid division by zero for PercentB
    safe_bb_width = bb_width.replace(0, np.nan)
    df['PercentB'] = (df['Close'] - df['Lower_Band']) / safe_bb_width
    
    # Replace inf with NaN to be dropped later
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    

    return df

def create_target_variable(df, prediction_window: int, movement_threshold: float):
    """Create target variable based on future price movements."""
    # Use user's prediction window for the look-ahead period
    future_returns = df['Close'].pct_change(periods=prediction_window).shift(-prediction_window)
    df['target'] = (future_returns.abs() > movement_threshold).astype(int)
    return df

def fetch_data_with_retry(symbol, interval='1h', max_retries=3, initial_delay=5):
    """Fetch data with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            # First check cache
            cached_data = get_cached_data(symbol, interval)
            if cached_data is not None:
                
                return cached_data
            
            # Try Alpha Vantage first
            df = get_alpha_vantage_data(symbol, interval)
            
            # If Alpha Vantage returns too little intraday data, force Yahoo fallback
            if df is not None and interval in ['1h', '6h']:
                min_rows_required = 1000 if interval == '1h' else 400
                if len(df) < min_rows_required:
                    
                    max_days = get_timeframe_window(interval)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=max_days)
                    if attempt > 0:
                        delay = initial_delay * (2 ** (attempt - 1))
                        
                        time.sleep(delay)
                    stock = yf.Ticker(symbol)
                    df = stock.history(start=start_date, end=end_date, interval=interval)
            
            # Fall back to Yahoo Finance if Alpha Vantage fails or limit reached
            if df is None:
                
                # Get the maximum window size for this interval
                max_days = get_timeframe_window(interval)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=max_days)
                
                # Add delay with exponential backoff
                if attempt > 0:
                    delay = initial_delay * (2 ** (attempt - 1))
                    
                    time.sleep(delay)
                
                stock = yf.Ticker(symbol)
                df = stock.history(start=start_date, end=end_date, interval=interval)
            
            if not df.empty:
                # Cache the successful response
                cache_data(symbol, interval, df)
                return df
            
            
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error fetching data for {symbol}: {str(e)}")
            
            # If this is the last attempt, raise the error
            if attempt == max_retries - 1:
                raise
    
    raise ValueError(f"Failed to fetch data for {symbol} after {max_retries} attempts")

def get_historical_data(symbol: str, timeframe: str = '1h', prediction_window: int = 12, movement_threshold: float = 0.025):
    """Fetch historical data for a given symbol with specified timeframe."""
    
    try:
        # Try to fetch data with retries
        df = fetch_data_with_retry(symbol, interval=timeframe)
        
        # Verify we have enough data
        min_periods = max(24, prediction_window) if timeframe in ['1h', '6h'] else max(7, prediction_window // 24)
        if len(df) < min_periods:
            raise ValueError(f"Insufficient data points for {symbol}: got {len(df)}, need at least {min_periods}")
        
        df = calculate_technical_indicators(df)
        df = create_target_variable(df, prediction_window, movement_threshold)
        
        # Drop rows with NaN values (including ones from indicator guards)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if df.empty:
            raise ValueError(f"No valid data points after processing for {symbol}")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error processing data for {symbol}: {str(e)}")

def prepare_features(df):
    """Prepare features for model training/prediction."""
    return df[FEATURE_COLUMNS]

def get_latest_data(symbol: str, timeframe: str = '1h', prediction_window: int = 12):
    """Get the most recent data for prediction with real-time price."""
    try:
        
        # Fetch data with retries
        df = fetch_data_with_retry(symbol, interval=timeframe)
        
        # Verify we have enough data
        min_periods = max(24, prediction_window) if timeframe in ['1h', '6h'] else max(7, prediction_window // 24)
        if len(df) < min_periods:
            raise ValueError(f"Insufficient data points for {symbol}: got {len(df)}, need at least {min_periods}")
        
        # Calculate technical indicators (we don't need target variable for prediction)
        df = calculate_technical_indicators(df)
        
        # Drop rows with NaN values (including ones from indicator guards)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if df.empty:
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