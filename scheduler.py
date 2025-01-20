import asyncio
import schedule
import time
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv('API_KEY')

def is_market_hours():
    """Check if it's currently market hours (9:30 AM - 4:00 PM Eastern Time)."""
    eastern = pytz.timezone('America/New_York')
    now = datetime.now(eastern)
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if now.weekday() > 4:
        return False
    
    # Market hours are 9:30 AM - 4:00 PM Eastern
    market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_start <= now <= market_end

class PredictionScheduler:
    def __init__(self, base_url, symbols):
        if not API_KEY:
            raise ValueError("API_KEY environment variable is not set")
            
        self.base_url = base_url
        self.symbols = symbols
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }

    def make_prediction(self, symbol):
        """Make prediction for a single symbol."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    headers=self.headers,
                    json={"symbol": symbol, "notify": True},
                    timeout=30  # Add timeout
                )
                
                if response.status_code == 200:
                    print(f"[{datetime.now(pytz.timezone('America/Los_Angeles'))}] Successfully predicted {symbol}")
                    return True
                else:
                    print(f"[{datetime.now(pytz.timezone('America/Los_Angeles'))}] Error predicting {symbol}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"[{datetime.now(pytz.timezone('America/Los_Angeles'))}] Timeout predicting {symbol}")
            except Exception as e:
                print(f"[{datetime.now(pytz.timezone('America/Los_Angeles'))}] Exception predicting {symbol}: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"Retrying {symbol} in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        return False

    def run_predictions(self):
        """Run predictions for all symbols."""
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        
        # Only run during market hours
        if not is_market_hours():
            print(f"[{current_time}] Skipping predictions - outside market hours")
            return
            
        print(f"[{current_time}] Running scheduled predictions...")
        for symbol in self.symbols:
            success = self.make_prediction(symbol)
            if not success:
                print(f"[{current_time}] Failed to predict {symbol} after all retries")
            time.sleep(2)  # Add delay between symbols to avoid rate limiting

def convert_to_utc(local_time_str, local_timezone_str):
    """Convert local time to UTC for scheduling."""
    local_tz = pytz.timezone(local_timezone_str)
    now = datetime.now(local_tz)
    
    # Parse the target time
    hour, minute = map(int, local_time_str.split(':'))
    target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # Convert to UTC
    utc_time = target_time.astimezone(pytz.UTC)
    return f"{utc_time.hour:02d}:{utc_time.minute:02d}"

def main():
    # Stock symbols to monitor
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    # Get API URL from environment or use default
    BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

    current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
    print(f"[{current_time}] Starting scheduler, using API at: {BASE_URL}")
    
    try:
        scheduler = PredictionScheduler(BASE_URL, SYMBOLS)
        
        # Convert 8 AM Pacific to UTC for scheduling
        utc_time = convert_to_utc("06:31", "America/Los_Angeles")
        schedule.every().day.at(utc_time).do(scheduler.run_predictions)
        
        print(f"[{current_time}] Scheduled daily predictions at 6:31 AM Pacific Time")
        
        # Keep the script running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled tasks
            
    except Exception as e:
        print(f"[{current_time}] Fatal error in scheduler: {str(e)}")
        raise

if __name__ == "__main__":
    main() 