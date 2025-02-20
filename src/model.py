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
            performance_metrics = {}
            
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
                    
                    # Calculate stock performance metrics
                    total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                    daily_returns = df['Close'].pct_change()
                    volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
                    max_drawdown = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100
                    avg_volume = df['Volume'].mean()
                    
                    # Handle date formatting safely
                    try:
                        start_date = pd.to_datetime(df.index[0]).strftime('%Y-%m-%d') if isinstance(df.index[0], str) else df.index[0].strftime('%Y-%m-%d')
                        end_date = pd.to_datetime(df.index[-1]).strftime('%Y-%m-%d') if isinstance(df.index[-1], str) else df.index[-1].strftime('%Y-%m-%d')
                    except Exception as e:
                        print(f"Warning: Error formatting dates for {symbol}: {str(e)}")
                        start_date = str(df.index[0])
                        end_date = str(df.index[-1])
                    
                    # Store performance metrics
                    performance_metrics[symbol] = {
                        'total_return': round(total_return, 2),
                        'volatility': round(volatility, 2),
                        'max_drawdown': round(max_drawdown, 2),
                        'avg_volume': int(avg_volume),
                        'data_points': len(df),
                        'date_range': {
                            'start': start_date,
                            'end': end_date
                        },
                        'current_price': round(float(df['Close'].iloc[-1]), 2),
                        'significant_movements': round(float((df['target'] == 1).mean() * 100), 2)  # Percentage of significant movements
                    }
                    
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
            
            # Calculate model performance metrics
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            
            classification_metrics = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Get feature importance
            feature_importance = dict(zip(model.feature_names_in_, model.feature_importances_))
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Calculate class distribution for weighted metrics
            class_distribution = {
                'no_movement': float((y_test == 0).mean()),
                'significant_movement': float((y_test == 1).mean())
            }
            
            model_metrics = {
                'accuracy': round(classification_metrics['accuracy'] * 100, 2),
                'precision': round(classification_metrics['weighted avg']['precision'] * 100, 2),
                'recall': round(classification_metrics['weighted avg']['recall'] * 100, 2),
                'f1_score': round(classification_metrics['weighted avg']['f1-score'] * 100, 2),
                'roc_auc': round(roc_auc * 100, 2),
                'class_metrics': {
                    'no_movement': {
                        'precision': round(classification_metrics['0']['precision'] * 100, 2),
                        'recall': round(classification_metrics['0']['recall'] * 100, 2),
                        'f1_score': round(classification_metrics['0']['f1-score'] * 100, 2),
                        'support': int(classification_metrics['0']['support'])
                    },
                    'significant_movement': {
                        'precision': round(classification_metrics['1']['precision'] * 100, 2),
                        'recall': round(classification_metrics['1']['recall'] * 100, 2),
                        'f1_score': round(classification_metrics['1']['f1-score'] * 100, 2),
                        'support': int(classification_metrics['1']['support'])
                    }
                },
                'class_distribution': {
                    'no_movement': round(class_distribution['no_movement'] * 100, 2),
                    'significant_movement': round(class_distribution['significant_movement'] * 100, 2)
                },
                'confusion_matrix': {
                    'true_negative': int(conf_matrix[0][0]),
                    'false_positive': int(conf_matrix[0][1]),
                    'false_negative': int(conf_matrix[1][0]),
                    'true_positive': int(conf_matrix[1][1])
                },
                'top_features': {k: round(v * 100, 2) for k, v in top_features.items()},
                'training_samples': len(y_train),
                'test_samples': len(y_test)
            }
            
            # Save the model to GCS
            self.save_model_to_gcs(user_id)
            print(f"\nModel saved to GCS for user {user_id}")
            print(f"Model is now trained on symbols: {', '.join(sorted(trained_symbols))}")
            
            # Return comprehensive metrics
            return {
                'model_performance': model_metrics,
                'stock_performance': performance_metrics,
                'training_info': {
                    'timeframe': user_settings.training_timeframe,
                    'prediction_window': user_settings.prediction_window,
                    'movement_threshold': user_settings.significant_movement_threshold,
                    'trained_symbols': list(trained_symbols)
                }
            }
            
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