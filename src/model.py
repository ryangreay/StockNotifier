import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
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
        Initialize the StockPredictor with user- and symbol-specific models.
        
        Args:
            max_models_in_memory (int): Maximum number of models to keep in memory
        """
        # Keyed by (user_id, symbol) -> { 'model': RandomForestClassifier, 'last_used': datetime }
        self.models = OrderedDict()
        self.max_models = max_models_in_memory
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(GCS_BUCKET_NAME)
        
    def _get_model_path(self, user_id: int, symbol: str) -> str:
        """Get the GCS path for a user's model for a specific symbol."""
        return f"models/user_{user_id}/symbols/{symbol}.joblib"
    
    def _get_local_path(self, user_id: int, symbol: str) -> str:
        """Get the local path for a user's model for a specific symbol."""
        return f"/tmp/stock_predictor_{user_id}_{symbol}.joblib"
    
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
    
    def get_model(self, user_id: int, symbol: str) -> RandomForestClassifier:
        """Get or load a user's model for a specific symbol."""
        key = (user_id, symbol)
        if key in self.models:
            self.models.move_to_end(key)
            return self.models[key]['model']
        
        # Try to load from GCS
        try:
            model = self.load_model_from_gcs(user_id, symbol)
            if model is not None:
                self._manage_memory()
                self.models[key] = {
                    'model': model,
                    'last_used': datetime.now()
                }
                return model
        except Exception as e:
            print(f"Error loading model for user {user_id}, symbol {symbol}: {str(e)}")
        
        # Create new model if none exists
        model = self._create_new_model()
        self._manage_memory()
        self.models[key] = {
            'model': model,
            'last_used': datetime.now()
        }
        return model
    
    def save_model_to_gcs(self, user_id: int, symbol: str):
        """Save user's model for a symbol to GCS."""
        key = (user_id, symbol)
        if key not in self.models:
            raise ValueError(f"No model found for user {user_id}, symbol {symbol}")
        
        model = self.models[key]['model']
        local_path = self._get_local_path(user_id, symbol)
        gcs_path = self._get_model_path(user_id, symbol)
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, local_path)
        
        # Upload to GCS
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        
        # Clean up local file
        os.remove(local_path)
    
    def load_model_from_gcs(self, user_id: int, symbol: str):
        """Load user's model for a symbol from GCS."""
        try:
            local_path = self._get_local_path(user_id, symbol)
            gcs_path = self._get_model_path(user_id, symbol)
            
            # Download from GCS
            blob = self.bucket.blob(gcs_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            
            # Load model
            model = joblib.load(local_path)
            
            # Clean up local file
            os.remove(local_path)
            
            print(f"Model loaded successfully for user {user_id}, symbol {symbol}.")
            return model
            
        except Exception as e:
            print(f"Error loading model for user {user_id}, symbol {symbol} from GCS: {str(e)}")
            return None

    def delete_model_from_gcs(self, user_id: int, symbol: str) -> bool:
        """Delete a user's model for a symbol from GCS."""
        try:
            gcs_path = self._get_model_path(user_id, symbol)
            blob = self.bucket.blob(gcs_path)
            if blob.exists():
                blob.delete()
                return True
            return False
        except Exception as e:
            print(f"Error deleting model for user {user_id}, symbol {symbol} from GCS: {str(e)}")
            return False

    def untrain(self, user_id: int, symbols: list):
        """Delete models for the specified symbols for this user."""
        if isinstance(symbols, str):
            symbols = [symbols]
        
        for symbol in symbols:
            key = (user_id, symbol)
            if key in self.models:
                del self.models[key]
            self.delete_model_from_gcs(user_id, symbol)
    
    def train(self, user_id: int, symbols: list, test_size=0.2, db=None):
        """Train one model per symbol for the user. Returns metrics for the last symbol if one provided, otherwise a dict per symbol."""
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
            
            all_symbol_results = {}
            last_result = None
            
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
                    
                    # Prepare features and target
                    X = prepare_features(df)
                    y = df['target']
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, shuffle=True
                    )
                    
                    # Create and train model for this symbol
                    model = self._create_new_model()
                    model.fit(X_train, y_train)
                    
                    # Store in memory and save to GCS
                    key = (user_id, symbol)
                    self.models[key] = {
                        'model': model,
                        'last_used': datetime.now()
                    }
                    self._manage_memory()
                    self.save_model_to_gcs(user_id, symbol)
                    print(f"Model saved to GCS for user {user_id}, symbol {symbol}")
                    
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
                    
                    stock_performance = {
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
                        'significant_movements': round(float((df['target'] == 1).mean() * 100), 2)
                    }
                    
                    result = {
                        'model_performance': model_metrics,
                        'stock_performance': stock_performance,
                        'training_info': {
                            'symbol': symbol,
                            'timeframe': user_settings.training_timeframe,
                            'prediction_window': user_settings.prediction_window,
                            'movement_threshold': user_settings.significant_movement_threshold
                        }
                    }
                    all_symbol_results[symbol] = result
                    last_result = result
                except Exception as e:
                    print(f"Error processing {symbol} for user {user_id}: {str(e)}")
                    continue
            
            if last_result is None:
                raise ValueError("No valid data available for training")
            
            # If single symbol requested, return that symbol's result; else return dict of results
            return last_result if len(symbols) == 1 else all_symbol_results
        finally:
            if should_close_db:
                db.close()

    def tune_hyperparameters(
        self,
        user_id: int,
        symbol: str,
        n_iter: int = 20,
        cv: int = 5,
        scoring: str = 'roc_auc',
        test_size: float = 0.2,
        db=None
    ):
        """Run randomized search with K-fold CV to tune RF hyperparameters for a single symbol.

        Saves the best model to GCS and updates in-memory model. Returns tuning summary and holdout metrics.
        """
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

            # Fetch data
            df = get_historical_data(
                symbol,
                timeframe=user_settings.training_timeframe,
                prediction_window=user_settings.prediction_window,
                movement_threshold=user_settings.significant_movement_threshold
            )

            # Basic validations
            if 'target' not in df.columns:
                raise ValueError("Prepared dataset missing 'target' column")
            if df['target'].nunique() < 2:
                raise ValueError("Target variable has only one class; cannot run stratified CV")

            # Prepare features/target
            X = prepare_features(df)
            y = df['target']

            # Train/holdout split for unbiased final evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
            )

            # Define model and parameter distributions
            base_model = self._create_new_model()
            param_distributions = {
                'n_estimators': list(range(50, 301)),
                'max_depth': list(range(3, 21)),
                'max_features': ['sqrt', 'log2', None]
            }

            # Stratified K-Fold for classification
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

            # Randomized search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=-1,
                cv=cv_splitter,
                random_state=42,
                verbose=1,
                refit=True
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            # Store and persist best model
            key = (user_id, symbol)
            self.models[key] = {
                'model': best_model,
                'last_used': datetime.now()
            }
            self._manage_memory()
            self.save_model_to_gcs(user_id, symbol)

            # Evaluate on holdout
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)

            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

            classification_metrics = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

            feature_importance = dict(zip(best_model.feature_names_in_, best_model.feature_importances_))
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])

            class_distribution = {
                'no_movement': float((y_test == 0).mean()),
                'significant_movement': float((y_test == 1).mean())
            }

            holdout_metrics = {
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

            # Calculate stock performance overview (reuse previous logic)
            total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
            daily_returns = df['Close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            max_drawdown = ((df['Close'].cummax() - df['Close']) / df['Close'].cummax()).max() * 100
            avg_volume = df['Volume'].mean()

            try:
                start_date = pd.to_datetime(df.index[0]).strftime('%Y-%m-%d') if isinstance(df.index[0], str) else df.index[0].strftime('%Y-%m-%d')
                end_date = pd.to_datetime(df.index[-1]).strftime('%Y-%m-%d') if isinstance(df.index[-1], str) else df.index[-1].strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Warning: Error formatting dates for {symbol}: {str(e)}")
                start_date = str(df.index[0])
                end_date = str(df.index[-1])

            stock_performance = {
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
                'significant_movements': round(float((df['target'] == 1).mean() * 100), 2)
            }

            return {
                'tuning': {
                    'best_params': search.best_params_,
                    'best_cv_score': search.best_score_,
                    'cv_folds': cv,
                    'n_iter': n_iter,
                    'scoring': scoring
                },
                'holdout_performance': holdout_metrics,
                'stock_performance': stock_performance,
                'training_info': {
                    'symbol': symbol,
                    'timeframe': user_settings.training_timeframe,
                    'prediction_window': user_settings.prediction_window,
                    'movement_threshold': user_settings.significant_movement_threshold
                }
            }
        finally:
            if should_close_db:
                db.close()
    
    def predict(self, user_id: int, symbol: str, features):
        """Make predictions using the user's trained model for a specific symbol."""
        model = self.get_model(user_id, symbol)
        
        # Create a copy of features to avoid modifying the original
        features_copy = features.copy()
        
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
        key = (user_id, symbol)
        if key in self.models:
            self.models.move_to_end(key)
        
        return prediction[0], probability[0]

    def get_feature_importance(self, user_id: int, symbol: str):
        """Get feature importance scores for user's model for a symbol."""
        model = self.get_model(user_id, symbol)
        importance = dict(zip(model.feature_names_in_, model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def list_user_symbols(self, user_id: int) -> list:
        """List symbols that have a trained model for this user (by checking GCS)."""
        prefix = f"models/user_{user_id}/symbols/"
        symbols = set()
        try:
            for blob in self.bucket.list_blobs(prefix=prefix):
                name = blob.name
                # Expecting paths like models/user_{id}/symbols/{SYMBOL}.joblib
                if name.startswith(prefix) and name.endswith('.joblib'):
                    symbol = name[len(prefix):-7]  # remove prefix and .joblib
                    if symbol:
                        symbols.add(symbol)
        except Exception as e:
            print(f"Error listing models for user {user_id}: {str(e)}")
        # Include any in-memory models not yet saved
        for (uid, sym) in self.models.keys():
            if uid == user_id:
                symbols.add(sym)
        return sorted(symbols)