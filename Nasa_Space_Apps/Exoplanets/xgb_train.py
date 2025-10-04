"""
XGBoost Exoplanet Classifier
================================
Loads preprocessed data, trains a simple XGBoost model, evaluates, and saves the model.

Usage:
    python xgb_train.py

Dependencies:
    pandas, xgboost, scikit-learn, joblib

Author: NASA Space Apps Team
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Path to preprocessed data CSV
DATA_PATH = 'Nasa_Space_Apps/Exoplanets/processed_exoplanet_data.csv'

# Load the preprocessed dataset
data = pd.read_csv(DATA_PATH)

# Features used for prediction
features = [
    'pl_orbper',  # Orbital period (days)
    'pl_rade',    # Planetary radius (Earth radii)
    'pl_bmasse',  # Planetary mass (Earth masses)
    'st_teff',    # Star effective temperature (Kelvin)
    'st_rad',     # Star radius (Solar radii)
    'st_mass',    # Star mass (Solar masses)
    'sy_dist'     # System distance (parsecs)
]

# X = features, y = label (1=exoplanet, 0=false positive, 2=candidate)
X = data[features]
y = data['label']

# Map to binary: 1 = exoplanet, 0 = not exoplanet (candidate or false positive)
y_bin = y.apply(lambda v: 1 if v == 1.0 else 0)

# Split data into training and test sets (stratify for balanced classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# Train Random Forest model (simple, robust, no external dependencies)
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=6,           # Limit tree depth for generalization
    random_state=42
)

xgb_model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=42)

xgb_model.fit(X_train, y_train)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

print('Accuracy: {:.4f}'.format(acc))
print(classification_report(y_test, y_pred, digits=4))

print('XGBoost Accuracy: {:.4f}'.format(xgb_acc))
print(classification_report(y_test, xgb_pred, digits=4))


# Save the trained model for later use in the web app
joblib.dump(model, 'Nasa_Space_Apps/Exoplanets/rf_exoplanet_model.pkl')
print('Model saved as rf_exoplanet_model.pkl')
