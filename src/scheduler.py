import asyncio
import schedule
import time
from datetime import datetime
import requests
from src.config import API_KEY

class PredictionScheduler:
    def __init__(self, base_url, symbols):
        self.base_url = base_url
        self.symbols = symbols
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }

    def make_prediction(self, symbol):
        """Make prediction for a single symbol."""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                headers=self.headers,
                json={"symbol": symbol, "notify": True}
            )
            if response.status_code == 200:
                print(f"[{datetime.now()}] Successfully predicted {symbol}")
            else:
                print(f"[{datetime.now()}] Error predicting {symbol}: {response.text}")
        except Exception as e:
            print(f"[{datetime.now()}] Exception predicting {symbol}: {str(e)}")

    def run_predictions(self):
        """Run predictions for all symbols."""
        print(f"[{datetime.now()}] Running scheduled predictions...")
        for symbol in self.symbols:
            self.make_prediction(symbol)

def main():
    # Configure your stock symbols here
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA"]  # Add more symbols as needed
    BASE_URL = "http://localhost:8000"  # Change this to your deployed URL

    scheduler = PredictionScheduler(BASE_URL, SYMBOLS)
    
    # Schedule predictions every 10 minutes
    schedule.every(2).hours.do(scheduler.run_predictions)
    
    # Run once immediately on startup
    scheduler.run_predictions()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute for scheduled tasks

if __name__ == "__main__":
    main() 