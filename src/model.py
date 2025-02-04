import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from google.cloud import storage
from src.config import (
    GCS_BUCKET_NAME,
    GCS_MODEL_PATH,
    LOCAL_MODEL_PATH,
    FEATURE_COLUMNS
)
from src.data_collector import get_historical_data, prepare_features

class StockPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.is_trained = False
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(GCS_BUCKET_NAME)
        self.trained_symbols = set()  # Keep track of trained symbols
        self.training_data = {}  # Store training data for each symbol
    
    def save_model_to_gcs(self):
        """Save model and trained symbols to GCS."""
        # Save model locally first
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        
        # Save both model and trained symbols
        save_data = {
            'model': self.model,
            'trained_symbols': self.trained_symbols,
            'training_data': self.training_data
        }
        joblib.dump(save_data, LOCAL_MODEL_PATH)
        
        # Upload to GCS
        blob = self.bucket.blob(GCS_MODEL_PATH)
        blob.upload_from_filename(LOCAL_MODEL_PATH)
        
        # Clean up local file
        os.remove(LOCAL_MODEL_PATH)
    
    def load_model_from_gcs(self):
        """Load model and trained symbols from GCS."""
        try:
            # Download from GCS
            blob = self.bucket.blob(GCS_MODEL_PATH)
            os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
            blob.download_to_filename(LOCAL_MODEL_PATH)
            
            # Load model and trained symbols
            save_data = joblib.load(LOCAL_MODEL_PATH)
            self.model = save_data['model']
            self.trained_symbols = save_data.get('trained_symbols', set())
            self.training_data = save_data.get('training_data', {})
            self.is_trained = True
            
            # Clean up local file
            os.remove(LOCAL_MODEL_PATH)
            print(f"Model loaded successfully. Trained on symbols: {', '.join(sorted(self.trained_symbols))}")
            return True
        except Exception as e:
            print(f"Error loading model from GCS: {str(e)}")
            return False

    def untrain(self, symbols):
        """Remove specified symbols from the model's training data and retrain."""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        if not self.is_trained:
            if not self.load_model_from_gcs():
                raise ValueError("No trained model found to untrain")

        # Remove symbols from trained set
        self.trained_symbols -= set(symbols)
        
        if not self.trained_symbols:
            print("No symbols remaining after untrain. Model needs to be trained on new data.")
            self.is_trained = False
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.save_model_to_gcs()
            return

        # Retrain model on remaining symbols
        print(f"Retraining model on remaining symbols: {', '.join(sorted(self.trained_symbols))}")
        self.train(list(self.trained_symbols))
    
    def train(self, symbols, test_size=0.2):
        """Train the model on multiple stock symbols."""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            print(f"\nProcessing data for {symbol}...")
            try:
                # Get historical data
                df = get_historical_data(symbol)
                
                # Prepare features and target
                X = prepare_features(df)
                y = df['target']
                
                # Add symbol as a feature
                X['symbol'] = symbol
                
                all_features.append(X)
                all_targets.append(y)
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        if not all_features:
            raise ValueError("No valid data available for training")
        
        # Combine all data
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)
        
        # Convert symbol to categorical
        X = pd.get_dummies(X, columns=['symbol'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # If model was previously trained, update with new data
        if self.is_trained:
            print("\nUpdating existing model with new data...")
        else:
            print("\nTraining new model...")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Update trained symbols
        self.trained_symbols.update(symbols)
        
        # Print model performance
        y_pred = self.model.predict(X_test)
        print("\nModel Performance Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model to GCS
        self.save_model_to_gcs()
        print(f"\nModel saved to GCS: {GCS_BUCKET_NAME}/{GCS_MODEL_PATH}")
        print(f"Model is now trained on symbols: {', '.join(sorted(self.trained_symbols))}")
        
        return self.model
    
    def predict(self, symbol: str, features):
        """Make predictions using the trained model."""
        if not self.is_trained:
            if not self.load_model_from_gcs():
                raise ValueError("Model not trained. Please train the model first.")
        
        # Create a copy of features to avoid modifying the original
        features_copy = features.copy()
        
        # Add symbol as a feature
        features_copy['symbol'] = symbol
        
        # Create dummy variables for all known symbols
        for trained_symbol in self.trained_symbols:
            features_copy[f'symbol_{trained_symbol}'] = 1 if symbol == trained_symbol else 0
        
        # Ensure features match the expected columns
        missing_cols = set(self.model.feature_names_in_) - set(features_copy.columns)
        if missing_cols:
            for col in missing_cols:
                features_copy[col] = 0
        
        # Select only the columns the model was trained on
        features_copy = features_copy[self.model.feature_names_in_]
        
        # Make prediction and get probability
        prediction = self.model.predict(features_copy)
        probability = self.model.predict_proba(features_copy)
        
        return prediction[0], probability[0]

    def get_feature_importance(self):
        """Get feature importance scores."""
        if not self.is_trained:
            if not self.load_model_from_gcs():
                raise ValueError("Model not trained. Please train the model first.")
        
        importance = dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 