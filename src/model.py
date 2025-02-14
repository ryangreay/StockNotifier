import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from google.cloud import storage
from datetime import datetime
from collections import OrderedDict
from src.config import (
    GCS_BUCKET_NAME,
    GCS_MODEL_PATH,
    LOCAL_MODEL_PATH,
    FEATURE_COLUMNS
)
from src.data_collector import get_historical_data, prepare_features
from .database import SessionLocal
from .models import UserSettings

class StockPredictor:
    def __init__(self, max_models_in_memory=10):
        """
        Initialize the StockPredictor with user-specific models.
        
        Args:
            max_models_in_memory (int): Maximum number of models to keep in memory
        """
        self.models = OrderedDict()  # user_id -> (model, last_used)
        self.max_models = max_models_in_memory
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(GCS_BUCKET_NAME)
        
    def _get_model_path(self, user_id: int) -> str:
        """Get the GCS path for a user's model."""
        return f"models/user_{user_id}/stock_predictor.joblib"
    
    def _get_local_path(self, user_id: int) -> str:
        """Get the local path for a user's model."""
        return f"/tmp/stock_predictor_{user_id}.joblib"
    
    def _create_new_model(self) -> RandomForestClassifier:
        """Create a new RandomForestClassifier instance."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    
    def _manage_memory(self):
        """Remove least recently used models if memory limit is reached."""
        while len(self.models) > self.max_models:
            # Remove the least recently used model
            self.models.popitem(last=False)
    
    def get_user_model(self, user_id: int) -> tuple:
        """
        Get or load a user's model.
        
        Args:
            user_id (int): The user's ID
            
        Returns:
            tuple: (model, trained_symbols)
        """
        # Check if model is in memory
        if user_id in self.models:
            model_data = self.models[user_id]
            # Update last used time
            self.models.move_to_end(user_id)
            return model_data['model'], model_data['trained_symbols']
        
        # Try to load from GCS
        try:
            model_data = self.load_model_from_gcs(user_id)
            if model_data:
                self._manage_memory()  # Ensure we don't exceed memory limit
                self.models[user_id] = {
                    'model': model_data['model'],
                    'trained_symbols': model_data['trained_symbols'],
                    'last_used': datetime.now()
                }
                return model_data['model'], model_data['trained_symbols']
        except Exception as e:
            print(f"Error loading model for user {user_id}: {str(e)}")
        
        # Create new model if none exists
        model = self._create_new_model()
        self._manage_memory()
        self.models[user_id] = {
            'model': model,
            'trained_symbols': set(),
            'last_used': datetime.now()
        }
        return model, set()
    
    def save_model_to_gcs(self, user_id: int):
        """Save user's model and trained symbols to GCS."""
        if user_id not in self.models:
            raise ValueError(f"No model found for user {user_id}")
            
        model_data = self.models[user_id]
        local_path = self._get_local_path(user_id)
        gcs_path = self._get_model_path(user_id)
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Save model data
        save_data = {
            'model': model_data['model'],
            'trained_symbols': model_data['trained_symbols']
        }
        joblib.dump(save_data, local_path)
        
        # Upload to GCS
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        
        # Clean up local file
        os.remove(local_path)
    
    def load_model_from_gcs(self, user_id: int) -> dict:
        """Load user's model and trained symbols from GCS."""
        try:
            local_path = self._get_local_path(user_id)
            gcs_path = self._get_model_path(user_id)
            
            # Download from GCS
            blob = self.bucket.blob(gcs_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            
            # Load model data
            model_data = joblib.load(local_path)
            
            # Clean up local file
            os.remove(local_path)
            
            print(f"Model loaded successfully for user {user_id}. "
                  f"Trained on symbols: {', '.join(sorted(model_data['trained_symbols']))}")
            return model_data
            
        except Exception as e:
            print(f"Error loading model for user {user_id} from GCS: {str(e)}")
            return None

    def untrain(self, user_id: int, symbols: list):
        """Remove specified symbols from the user's model training data and retrain."""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        model, trained_symbols = self.get_user_model(user_id)
        
        # Remove symbols from trained set
        trained_symbols -= set(symbols)
        
        if not trained_symbols:
            print(f"No symbols remaining after untrain for user {user_id}. Creating new model.")
            self.models[user_id] = {
                'model': self._create_new_model(),
                'trained_symbols': set(),
                'last_used': datetime.now()
            }
            self.save_model_to_gcs(user_id)
            return
        
        # Retrain model on remaining symbols
        print(f"Retraining model for user {user_id} on remaining symbols: "
              f"{', '.join(sorted(trained_symbols))}")
        self.train(user_id, list(trained_symbols))
    
    def train(self, user_id: int, symbols: list, test_size=0.2, db=None):
        """Train the user's model on multiple stock symbols."""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Get user settings from database
        if not db:
            db = SessionLocal()
            should_close_db = True
        else:
            should_close_db = False
            
        try:
            # Get user settings
            user_settings = db.query(UserSettings).filter(
                UserSettings.user_id == user_id
            ).first()
            
            if not user_settings:
                raise ValueError(f"No settings found for user {user_id}")
            
            all_features = []
            all_targets = []
            
            for symbol in symbols:
                print(f"\nProcessing data for user {user_id}, symbol {symbol}...")
                try:
                    # Get historical data using user settings
                    df = get_historical_data(
                        symbol,
                        timeframe=user_settings.training_timeframe,
                        prediction_window=user_settings.prediction_window,
                        movement_threshold=user_settings.significant_movement_threshold
                    )
                    
                    # Prepare features and target
                    X = prepare_features(df)
                    y = df['target']
                    
                    # Add symbol as a feature
                    X = X.copy()  # Create an explicit copy
                    X.loc[:, 'symbol'] = symbol
                    
                    all_features.append(X)
                    all_targets.append(y)
                    
                except Exception as e:
                    print(f"Error processing {symbol} for user {user_id}: {str(e)}")
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
            
            # Get or create user's model
            model, trained_symbols = self.get_user_model(user_id)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Update trained symbols
            trained_symbols.update(symbols)
            
            # Update model in memory
            self.models[user_id] = {
                'model': model,
                'trained_symbols': trained_symbols,
                'last_used': datetime.now()
            }
            
            # Print model performance
            y_pred = model.predict(X_test)
            print(f"\nModel Performance Report for user {user_id}:")
            print(classification_report(y_test, y_pred))
            
            # Save the model to GCS
            self.save_model_to_gcs(user_id)
            print(f"\nModel saved to GCS for user {user_id}")
            print(f"Model is now trained on symbols: {', '.join(sorted(trained_symbols))}")
            
            return model
            
        finally:
            if should_close_db:
                db.close()
    
    def predict(self, user_id: int, symbol: str, features):
        """Make predictions using the user's trained model."""
        model, trained_symbols = self.get_user_model(user_id)
        
        # Create a copy of features to avoid modifying the original
        features_copy = features.copy()
        
        # Add symbol as a feature
        features_copy['symbol'] = symbol
        
        # Create dummy variables for all known symbols
        for trained_symbol in trained_symbols:
            features_copy[f'symbol_{trained_symbol}'] = 1 if symbol == trained_symbol else 0
        
        # Ensure features match the expected columns
        missing_cols = set(model.feature_names_in_) - set(features_copy.columns)
        if missing_cols:
            for col in missing_cols:
                features_copy[col] = 0
        
        # Select only the columns the model was trained on
        features_copy = features_copy[model.feature_names_in_]
        
        # Make prediction and get probability
        prediction = model.predict(features_copy)
        probability = model.predict_proba(features_copy)
        
        # Update last used time
        self.models.move_to_end(user_id)
        
        return prediction[0], probability[0]

    def get_feature_importance(self, user_id: int):
        """Get feature importance scores for user's model."""
        model, _ = self.get_user_model(user_id)
        importance = dict(zip(model.feature_names_in_, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 