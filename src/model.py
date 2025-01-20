import joblib
import numpy as np
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
    
    def save_model_to_gcs(self):
        """Save model to GCS."""
        # Save model locally first
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, LOCAL_MODEL_PATH)
        
        # Upload to GCS
        blob = self.bucket.blob(GCS_MODEL_PATH)
        blob.upload_from_filename(LOCAL_MODEL_PATH)
        
        # Clean up local file
        os.remove(LOCAL_MODEL_PATH)
    
    def load_model_from_gcs(self):
        """Load model from GCS."""
        try:
            # Download from GCS
            blob = self.bucket.blob(GCS_MODEL_PATH)
            os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
            blob.download_to_filename(LOCAL_MODEL_PATH)
            
            # Load model
            self.model = joblib.load(LOCAL_MODEL_PATH)
            self.is_trained = True
            
            # Clean up local file
            os.remove(LOCAL_MODEL_PATH)
            return True
        except Exception as e:
            print(f"Error loading model from GCS: {str(e)}")
            return False
    
    def train(self, symbol, test_size=0.2):
        """Train the model on historical data."""
        # Get historical data
        df = get_historical_data(symbol)
        
        # Prepare features and target
        X = prepare_features(df)
        y = df['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print model performance
        y_pred = self.model.predict(X_test)
        print("\nModel Performance Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model to GCS
        self.save_model_to_gcs()
        print(f"\nModel saved to GCS: {GCS_BUCKET_NAME}/{GCS_MODEL_PATH}")
        
        return self.model
    
    def predict(self, features):
        """Make predictions using the trained model."""
        if not self.is_trained:
            if not self.load_model_from_gcs():
                raise ValueError("Model not trained. Please train the model first.")
        
        # Ensure features match the expected columns
        missing_cols = set(FEATURE_COLUMNS) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")
        
        # Make prediction and get probability
        prediction = self.model.predict(features[FEATURE_COLUMNS])
        probability = self.model.predict_proba(features[FEATURE_COLUMNS])
        
        return prediction[0], probability[0]

    def get_feature_importance(self):
        """Get feature importance scores."""
        if not self.is_trained:
            if not self.load_model_from_gcs():
                raise ValueError("Model not trained. Please train the model first.")
        
        importance = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 