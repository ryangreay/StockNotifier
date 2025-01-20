import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Security
API_KEY = os.getenv('API_KEY')

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Model Configuration
MODEL_PATH = 'models/stock_predictor.joblib'
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA_5', 'MA_20', 'RSI', 'MACD', 'MACD_Signal',
    'Upper_Band', 'Lower_Band'
]

# Trading Configuration
PREDICTION_THRESHOLD = 0.75  # Confidence threshold for predictions
SIGNIFICANT_MOVEMENT_THRESHOLD = 0.04  # 2% movement threshold
HISTORICAL_DAYS = 500  # Days of historical data for training
PREDICTION_WINDOW = 24  # Hours to look ahead for prediction

# API Configuration
API_TITLE = "Stock Movement Predictor"
API_DESCRIPTION = "Predicts significant stock movements using ML"
API_VERSION = "1.0.0" 