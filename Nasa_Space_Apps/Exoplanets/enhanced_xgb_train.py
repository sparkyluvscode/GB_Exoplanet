"""
Enhanced XGBoost Exoplanet Classifier
=====================================
Advanced training with 58 features, feature selection, hyperparameter tuning,
and class balancing for maximum accuracy.

Usage:
    python enhanced_xgb_train.py

Dependencies:
    pandas, xgboost, scikit-learn, joblib, imbalanced-learn

Author: NASA Space Apps Team
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE for class balancing
try:
    from imbalanced_learn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available. Install imbalanced-learn for better class balancing.")

def load_enhanced_data():
    """Load the enhanced processed dataset."""
    # Try enhanced data first, fallback to original
    try:
        data = pd.read_csv('Nasa_Space_Apps/Exoplanets/enhanced_processed_exoplanet_data.csv')
        print("Loaded enhanced processed data")
    except FileNotFoundError:
        data = pd.read_csv('Nasa_Space_Apps/Exoplanets/processed_exoplanet_data.csv')
        print("Loaded original processed data")
    
    return data

def prepare_features_and_target(data):
    """Prepare features and target with proper encoding."""
    # Separate features and target
    target_col = 'label'
    
    # Get all columns except target
    feature_cols = [col for col in data.columns if col != target_col]
    
    # Handle categorical features
    categorical_features = []
    for col in feature_cols:
        if data[col].dtype == 'object' or data[col].dtype.name == 'category':
            categorical_features.append(col)
            # Encode categorical variables
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
    
    X = data[feature_cols]
    y = data[target_col]
    
    # Convert to binary classification: 1 = exoplanet (confirmed), 0 = not exoplanet
    y_binary = (y == 2.0).astype(int)
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y_binary)}")
    
    return X, y_binary, feature_cols

def apply_feature_selection(X, y, k=25):
    """Apply feature selection to reduce noise."""
    print(f"Applying feature selection (top {k} features)...")
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    print(f"Selected features: {list(selected_features)}")
    
    return X_selected, selected_features

def balance_classes(X, y, method='smote'):
    """Balance classes using various methods."""
    print(f"Balancing classes using {method}...")
    
    if method == 'smote' and SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == 'class_weight':
        # XGBoost will handle this internally
        X_balanced, y_balanced = X, y
    else:
        # No balancing
        X_balanced, y_balanced = X, y
    
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

def optimize_xgb_hyperparameters(X_train, y_train):
    """Optimize XGBoost hyperparameters using grid search."""
    print("Optimizing XGBoost hyperparameters...")
    
    # Create XGBoost classifier with class weights
    class_counts = np.bincount(y_train)
    class_weights = {0: class_counts[1] / class_counts[0], 1: 1.0}
    
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=class_weights[0],
        n_jobs=-1
    )
    
    # Use smaller grid for faster execution
    param_grid = {
        'n_estimators': [1000],
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.9],
        'colsample_bytree': [0.9]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_models(X_train, X_test, y_train, y_test, use_optimization=True):
    """Train and evaluate multiple models."""
    models = {}
    
    # XGBoost with optimization
    if use_optimization:
        xgb_model = optimize_xgb_hyperparameters(X_train, y_train)
    else:
        # Simple XGBoost with good defaults
        class_counts = np.bincount(y_train)
        class_weights = {0: class_counts[1] / class_counts[0], 1: 1.0}
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=class_weights[0],
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
    
    models['XGBoost'] = xgb_model
    
    # Random Forest for comparison
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models."""
    results = {}
    
    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
    
    return results

def main():
    """Main training pipeline."""
    print("=== Enhanced XGBoost Exoplanet Classifier ===")
    
    # Load data
    data = load_enhanced_data()
    X, y, feature_names = prepare_features_and_target(data)
    
    # Feature selection
    X_selected, selected_features = apply_feature_selection(X, y, k=25)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Balance classes
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, method='smote')
    
    # Train models
    models = train_models(X_train_balanced, X_test, y_train_balanced, y_test, use_optimization=True)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    # Save model and feature names
    joblib.dump(best_model, 'Nasa_Space_Apps/Exoplanets/enhanced_xgb_model.pkl')
    joblib.dump(selected_features, 'Nasa_Space_Apps/Exoplanets/selected_features.pkl')
    
    print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    print("Model saved as enhanced_xgb_model.pkl")
    print("Selected features saved as selected_features.pkl")
    
    return results

if __name__ == "__main__":
    results = main()
