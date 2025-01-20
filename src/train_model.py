import argparse
from src.model import StockPredictor

def main():
    parser = argparse.ArgumentParser(description='Train the stock movement predictor model')
    parser.add_argument('symbol', type=str, help='Stock symbol to train on (e.g., AAPL)')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    print(f"Training model for {args.symbol}...")
    predictor = StockPredictor()
    predictor.train(args.symbol, args.test_size)
    print("Training complete!")

if __name__ == "__main__":
    main() 