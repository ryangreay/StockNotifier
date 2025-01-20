import asyncio
import schedule
import time
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv('API_KEY')

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
                    print(f"[{datetime.now()}] Successfully predicted {symbol}")
                    return True
                else:
                    print(f"[{datetime.now()}] Error predicting {symbol}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"[{datetime.now()}] Timeout predicting {symbol}")
            except Exception as e:
                print(f"[{datetime.now()}] Exception predicting {symbol}: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"Retrying {symbol} in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        return False

    def run_predictions(self):
        """Run predictions for all symbols."""
        print(f"[{datetime.now()}] Running scheduled predictions...")
        for symbol in self.symbols:
            success = self.make_prediction(symbol)
            if not success:
                print(f"[{datetime.now()}] Failed to predict {symbol} after all retries")
            time.sleep(2)  # Add delay between symbols to avoid rate limiting

def main():
    # Stock symbols to monitor
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    # Get API URL from environment or use default
    BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

    print(f"[{datetime.now()}] Starting scheduler, using API at: {BASE_URL}")
    
    try:
        scheduler = PredictionScheduler(BASE_URL, SYMBOLS)
        
        # Schedule predictions every 2 hours
        schedule.every(2).hours.do(scheduler.run_predictions)
        
        # Run once immediately on startup
        scheduler.run_predictions()
        
        # Keep the script running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled tasks
            
    except Exception as e:
        print(f"[{datetime.now()}] Fatal error in scheduler: {str(e)}")
        raise

if __name__ == "__main__":
    main() 