"""
Model Comparison Script
======================
Compares original vs enhanced model performance.

Author: NASA Space Apps Team
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

def load_original_data():
    """Load original processed data."""
    return pd.read_csv('Nasa_Space_Apps/Exoplanets/processed_exoplanet_data.csv')

def load_enhanced_data():
    """Load enhanced processed data."""
    return pd.read_csv('Nasa_Space_Apps/Exoplanets/enhanced_processed_exoplanet_data.csv')

def train_original_model():
    """Train model with original 7 features."""
    print("=== ORIGINAL MODEL (7 Features) ===")
    
    # Load original data
    data = load_original_data()
    
    # Features used for prediction
    features = [
        'pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist'
    ]
    
    X = data[features]
    y = data['label']
    
    # Map to binary: 1 = exoplanet, 0 = not exoplanet
    y_bin = y.apply(lambda v: 1 if v == 1.0 else 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    
    return accuracy

def train_enhanced_model():
    """Train model with enhanced 25 features."""
    print("\n=== ENHANCED MODEL (25 Features) ===")
    
    # Load enhanced data
    data = load_enhanced_data()
    
    # Get selected features
    try:
        selected_features = joblib.load('Nasa_Space_Apps/Exoplanets/selected_features.pkl')
    except FileNotFoundError:
        print("Selected features not found, using all features")
        selected_features = [col for col in data.columns if col != 'label']
    
    X = data[selected_features]
    y = data['label']
    
    # Map to binary: 1 = exoplanet, 0 = not exoplanet
    y_bin = y.apply(lambda v: 1 if v == 2.0 else 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    # Train XGBoost with optimized parameters
    class_counts = np.bincount(y_train)
    class_weights = {0: class_counts[1] / class_counts[0], 1: 1.0}
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        scale_pos_weight=class_weights[0]
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    
    return accuracy

def main():
    """Compare both models."""
    print("🪐 Exoplanet Model Comparison")
    print("=" * 50)
    
    # Train and evaluate both models
    original_acc = train_original_model()
    enhanced_acc = train_enhanced_model()
    
    # Calculate improvement
    improvement = enhanced_acc - original_acc
    improvement_pct = (improvement / original_acc) * 100
    
    print("\n" + "=" * 50)
    print("📊 COMPARISON RESULTS")
    print("=" * 50)
    print(f"Original Model Accuracy:  {original_acc:.4f} ({original_acc*100:.2f}%)")
    print(f"Enhanced Model Accuracy:  {enhanced_acc:.4f} ({enhanced_acc*100:.2f}%)")
    print(f"Improvement:              +{improvement:.4f} (+{improvement_pct:.1f}%)")
    print("=" * 50)
    
    if improvement > 0:
        print("🎉 Enhanced model performs better!")
    else:
        print("⚠️  Original model performs better (unexpected)")

if __name__ == "__main__":
    main()
