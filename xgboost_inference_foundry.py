#!/usr/bin/env python3
"""
XGBoost Standby Activation Model - Inference Script for Azure AI Foundry
========================================================================

This script loads a pre-trained XGBoost model and provides prediction capabilities
for standby activation rates in Azure AI Foundry environment.

Usage:
1. Upload xgboost_model_*.json and model_metadata_*.json to Azure AI Foundry
2. Upload new data CSV for predictions
3. Run this script to get predictions

Author: Data Science Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import json
import warnings
import os
import glob
from typing import Optional, Dict, List
warnings.filterwarnings('ignore')

# XGBoost
import xgboost as xgb

# =============================================================================
# ROBUST FILE FINDING FOR AZURE AI FOUNDRY
# =============================================================================

def find_model_files(search_path: str = "/mnt/data") -> tuple:
    """Robustly locate model and metadata JSON files in Azure AI Foundry."""
    # Check if we're in Azure environment
    if os.path.exists("/mnt/data"):
        base_path = "/mnt/data"
    else:
        base_path = "."
    
    print(f"🔍 Searching for model files in: {base_path}")
    print(f"📁 Contents: {os.listdir(base_path)}")
    
    # Find model files using glob patterns
    model_candidates = sorted(glob.glob(f"{base_path}/*xgboost_model*.json"))
    meta_candidates = sorted(glob.glob(f"{base_path}/*model_metadata*.json"))
    
    print(f"📊 Model candidates: {model_candidates}")
    print(f"📋 Metadata candidates: {meta_candidates}")
    
    # Pick the newest (last in sorted list)
    model_path = model_candidates[-1] if model_candidates else None
    metadata_path = meta_candidates[-1] if meta_candidates else None
    
    print(f"✅ Selected MODEL_PATH: {model_path}")
    print(f"✅ Selected METADATA_PATH: {metadata_path}")
    
    return model_path, metadata_path

class XGBoostStandbyPredictor:
    """XGBoost model for standby activation prediction in Azure AI Foundry."""
    
    def __init__(self):
        self.model = None
        self.features_used = []
        self.metadata = {}
        self.model_loaded = False
    
    def load_model(self, model_path: str = None, metadata_path: str = None) -> bool:
        """
        Load XGBoost model and metadata from JSON files.
        
        Args:
            model_path: Path to the XGBoost model JSON file (auto-detected if None)
            metadata_path: Path to the model metadata JSON file (auto-detected if None)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Auto-detect files if paths not provided
            if model_path is None or metadata_path is None:
                auto_model_path, auto_metadata_path = find_model_files()
                model_path = model_path or auto_model_path
                metadata_path = metadata_path or auto_metadata_path
            
            # Validate paths
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not metadata_path or not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            # Load XGBoost model
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            print(f"✅ Model loaded from: {model_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.features_used = self.metadata['features_used']
            print(f"✅ Metadata loaded from: {metadata_path}")
            print(f"📊 Model trained with {len(self.features_used)} features")
            print(f"📅 Model date: {self.metadata.get('timestamp', 'Unknown')}")
            
            # Display model performance
            if 'metrics_val' in self.metadata and self.metadata['metrics_val']:
                val_metrics = self.metadata['metrics_val']
                print(f"🎯 Validation Performance:")
                print(f"   • MAE: {val_metrics['MAE']:.6f}")
                print(f"   • RMSE: {val_metrics['RMSE']:.6f}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def add_special_day_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add special day flags and temporal features (same as training)."""
        df = df.copy()
        if 'Date' not in df.columns:
            return df
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month Number'] = df['Date'].dt.month
        df['is_weekend'] = df['Date'].dt.weekday >= 5
        
        # Special day flags
        df['is_fixed_holiday'] = df['Date'].apply(self.is_fixed_holiday)
        df['is_festival'] = df['Date'].apply(self.is_festival)
        df['is_payday_proximity'] = df['Date'].apply(self.is_payday_proximity)
        df['is_fy_end_proximity'] = df['Date'].apply(
            lambda d: (d.month == 3 and d.day >= 24) if pd.notna(d) else False
        )
        
        # Add Season (0=Summer, 1=Winter)
        df['Season'] = ((df['Month Number'] >= 11) | (df['Month Number'] <= 3)).astype(int)
        
        return df
    
    def is_payday_proximity(self, date):
        """Check if date is near payday (end/start of month)."""
        if pd.isna(date):
            return False
        end_of_month = (date + pd.offsets.MonthEnd(0)).date()
        return (date.date() == end_of_month) or (1 <= date.day <= 7)
    
    def is_fixed_holiday(self, date):
        """Check if date is a fixed holiday."""
        if pd.isna(date):
            return False
        FIXED_HOLIDAYS = {(1, 26), (8, 15), (10, 2), (12, 25)}
        return (date.month, date.day) in FIXED_HOLIDAYS
    
    def is_festival(self, date):
        """Check if date is a festival."""
        if pd.isna(date):
            return False
        FESTIVAL_DATES = {
            (2023, 11, 12), (2023, 3, 8), (2023, 4, 22),
            (2024, 11, 1), (2024, 3, 25), (2024, 4, 11),
            (2025, 10, 20), (2025, 3, 14), (2025, 3, 31)
        }
        return (date.year, date.month, date.day) in FESTIVAL_DATES
    
    def add_fatigue_kpis(self, df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
        """Add rolling window fatigue KPIs (same as training)."""
        df = df.copy()
        group_keys = ['Station', 'Rank', 'Duty Window Number']
        
        if 'Date' in df.columns:
            df = df.sort_values(group_keys + ['Date'])
        
        # Rolling averages and volatility
        for metric in ['Activation Rate', 'Pairing Start Count']:
            if metric in df.columns:
                df[f'{metric} MA{window_days}'] = df.groupby(group_keys)[metric].transform(
                    lambda s: s.rolling(window_days, min_periods=3).mean()
                )
                df[f'{metric} Vol{window_days}'] = df.groupby(group_keys)[metric].transform(
                    lambda s: s.rolling(window_days, min_periods=3).std()
                )
        
        # Above mean flag
        if 'Activation Rate' in df.columns:
            ma_col = f'Activation Rate MA{window_days}'
            df['AR_AboveMean_Flag'] = (
                (df[ma_col].notna()) & 
                (df['Activation Rate'].notna()) & 
                (df['Activation Rate'] > df[ma_col])
            ).astype(int)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for prediction (apply same transformations as training).
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame ready for prediction
        """
        print(f"🔧 Preprocessing {len(df)} rows...")
        
        # Add special day flags
        df_processed = self.add_special_day_flags(df)
        
        # Ensure numeric types for key columns
        numeric_cols = [
            'Pairing Start Count', 'Multi-Day Pairing Count', 'INT Pairing Count',
            'Standby Activation Count', 'Multi-Day Pairing Ratio', 'INT Pairing Ratio'
        ]
        
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Add fatigue KPIs if not present
        needed_fatigue = [
            'Activation Rate MA7', 'Pairing Start Count MA7', 
            'Activation Rate Vol7', 'Pairing Start Count Vol7', 'AR_AboveMean_Flag'
        ]
        
        if not all(col in df_processed.columns for col in needed_fatigue):
            print("   🔄 Computing fatigue KPIs...")
            df_processed = self.add_fatigue_kpis(df_processed, window_days=7)
        
        print(f"   ✅ Preprocessing complete. Shape: {df_processed.shape}")
        return df_processed
    
    def predict(self, data: pd.DataFrame, include_features: bool = False) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            data: Input DataFrame with crew data
            include_features: Whether to include feature values in output
            
        Returns:
            pd.DataFrame: Original data with predictions added
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"🔮 Making predictions on {len(data)} samples...")
        
        # Preprocess data
        df_processed = self.preprocess_data(data)
        
        # Extract features for prediction (with fallback for missing features)
        X_pred = pd.DataFrame()
        missing_features = []
        
        for feature in self.features_used:
            if feature in df_processed.columns:
                X_pred[feature] = df_processed[feature]
            else:
                missing_features.append(feature)
                # Use fallback values for missing features (INSIDE the loop)
                if 'MA7' in feature or 'Vol7' in feature:
                    X_pred[feature] = 0.0
                elif feature == 'AR_AboveMean_Flag':
                    X_pred[feature] = 0
                elif feature in ['Year', 'Month Number', 'Season']:
                    X_pred[feature] = df_processed.get(feature, 0)
                else:
                    X_pred[feature] = 0.0
        
        if missing_features:
            print(f"   ⚠️  Using fallback values for missing features: {missing_features}")
        
        # Handle any remaining NaN values
        X_pred = X_pred.fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X_pred)
        
        # Clip predictions to reasonable range [0, 1]
        predictions = np.clip(predictions, 0, 1)
        
        # Create output DataFrame
        result_df = data.copy()
        result_df['predicted_activation_rate'] = predictions
        
        # Optionally include feature values
        if include_features:
            for feature in self.features_used:
                result_df[f'feature_{feature}'] = X_pred[feature]
        
        print(f"   ✅ Predictions complete")
        print(f"   📊 Predicted range: {predictions.min():.4f} to {predictions.max():.4f}")
        print(f"   📊 Mean prediction: {predictions.mean():.4f}")
        
        return result_df
    
    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        return {
            "features_count": len(self.features_used),
            "features_used": self.features_used,
            "model_params": self.metadata.get('model_params', {}),
            "training_metrics": {
                "validation": self.metadata.get('metrics_val'),
                "test": self.metadata.get('metrics_test')
            },
            "dataset_info": self.metadata.get('dataset_info', {}),
            "model_timestamp": self.metadata.get('timestamp')
        }

# =============================================================================
# CONVENIENCE FUNCTIONS FOR AZURE AI FOUNDRY
# =============================================================================

def load_and_predict(model_path: str, metadata_path: str, data_path: str, 
                    output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load model and make predictions in one call.
    
    Args:
        model_path: Path to XGBoost model JSON file
        metadata_path: Path to model metadata JSON file  
        data_path: Path to input CSV data file
        output_path: Optional path to save predictions CSV
        
    Returns:
        pd.DataFrame: Data with predictions
    """
    print("🚀 XGBoost Standby Activation Prediction")
    print("=" * 50)
    
    # Initialize predictor
    predictor = XGBoostStandbyPredictor()
    
    # Load model
    if not predictor.load_model(model_path, metadata_path):
        return None
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        print(f"📊 Loaded data: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Make predictions
    try:
        predictions_df = predictor.predict(data)
        
        # Save output if path provided
        if output_path:
            predictions_df.to_csv(output_path, index=False)
            print(f"💾 Predictions saved to: {output_path}")
        
        return predictions_df
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None

def predict_single_record(model_path: str = None, metadata_path: str = None, 
                         date: str = "2025-12-31", station: str = "DEL", 
                         rank: str = "CP", duty_window: int = 1,
                         pairing_start_count: int = 5, multi_day_pairing_count: int = 2,
                         int_pairing_count: int = 1, standby_activation_count: int = 3) -> float:
    """
    Predict activation rate for a single record.
    
    Args:
        model_path: Path to XGBoost model JSON file (auto-detected if None)
        metadata_path: Path to model metadata JSON file (auto-detected if None)
        date: Date string (e.g., '2025-12-31')
        station: Station code (e.g., 'DEL')
        rank: Rank (e.g., 'CP')
        duty_window: Duty window number (1-6)
        pairing_start_count: Number of pairing starts
        multi_day_pairing_count: Number of multi-day pairings
        int_pairing_count: Number of international pairings
        standby_activation_count: Number of standby activations
        
    Returns:
        float: Predicted activation rate
    """
    # Create single record DataFrame
    record = pd.DataFrame([{
        'Date': date,
        'Station': station,
        'Rank': rank,
        'Duty Window Number': duty_window,
        'Pairing Start Count': pairing_start_count,
        'Multi-Day Pairing Count': multi_day_pairing_count,
        'INT Pairing Count': int_pairing_count,
        'Standby Activation Count': standby_activation_count,
        'Multi-Day Pairing Ratio': multi_day_pairing_count / pairing_start_count if pairing_start_count > 0 else 0,
        'INT Pairing Ratio': int_pairing_count / pairing_start_count if pairing_start_count > 0 else 0,
        'Activation Rate': standby_activation_count / pairing_start_count if pairing_start_count > 0 else 0,
        'Fleet': 'Airbus',  # Default
        'Season': 1,  # Default Winter
        'IrOps': 1  # Default
    }])
    
    try:
        # Direct prediction without unnecessary load_and_predict call
        predictor = XGBoostStandbyPredictor()
        if predictor.load_model(model_path, metadata_path):
            result = predictor.predict(record)
            return float(result['predicted_activation_rate'].iloc[0])
        return None
    except Exception as e:
        print(f"❌ Error in single prediction: {e}")
        return None

# =============================================================================
# MAIN EXECUTION FOR AZURE AI FOUNDRY
# =============================================================================

if __name__ == "__main__":
    print("🎯 XGBoost Standby Activation Inference Script")
    print("Ready for Azure AI Foundry!")
    
    # Test auto-detection
    try:
        model_path, metadata_path = find_model_files()
        if model_path and metadata_path:
            # Quick validation test
            predictor = XGBoostStandbyPredictor()
            success = predictor.load_model()
            if success:
                print(f"🎉 Auto-detection successful!")
                print(f"📊 Features: {len(predictor.features_used)}")
                
                # Test single prediction for DEL/CP on 31-12-2025
                test_prediction = predict_single_record(
                    date="2025-12-31", station="DEL", rank="CP", duty_window=1
                )
                if test_prediction is not None:
                    print(f"✅ Test prediction for DEL/CP/DW1 on 2025-12-31: {test_prediction:.4f}")
            else:
                print("❌ Model loading failed")
        else:
            print("❌ Model files not found")
    except Exception as e:
        print(f"❌ Auto-detection failed: {e}")
    
    print("\n📋 Usage Examples:")
    print("1. Auto-detection:")
    print("   predictor = XGBoostStandbyPredictor()")
    print("   predictor.load_model()  # Auto-finds files")
    print("\n2. Single prediction:")
    print("   rate = predict_single_record(date='2025-12-31', station='DEL', rank='CP')")
    print("\n3. Batch prediction:")
    print("   predictions = load_and_predict(None, None, 'data.csv')  # Auto-finds model files")