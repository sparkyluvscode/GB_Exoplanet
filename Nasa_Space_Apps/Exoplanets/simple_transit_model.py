"""
Simple Transit Detection Model - Uses Only Available Data
Uses only the 8 parameters actually present in the training data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def create_simple_features(df):
    """Create features using only the 8 available parameters."""
    print("Creating simple features from available data...")
    
    # Basic transit physics features (using only available data)
    
    # 1. Transit depth (planet radius / star radius)
    # pl_rade is Earth radii, st_rad is Solar radii
    # Convert: Rp/Rs = pl_rade / (st_rad * 109.2)
    df['transit_depth'] = (df['pl_rade'] / (df['st_rad'] * 109.2)) ** 2
    df['transit_depth_log'] = np.log1p(df['transit_depth'])
    
    # 2. Semi-major axis from orbital period (Kepler's 3rd law)
    # a_AU = ((pl_orbper/365.25)^(2/3)) * (st_mass^(1/3))
    df['semi_major_axis_AU'] = ((df['pl_orbper'] / 365.25) ** (2/3)) * (df['st_mass'] ** (1/3))
    df['semi_major_axis_AU_log'] = np.log1p(df['semi_major_axis_AU'])
    
    # 3. Transit duration proxy (R* / a)
    df['transit_duration_ratio'] = df['st_rad'] / (np.pi * df['semi_major_axis_AU'])
    df['transit_duration_ratio_log'] = np.log1p(df['transit_duration_ratio'])
    
    # 4. Signal-to-noise proxy (transit depth * stellar temperature)
    df['snr_proxy'] = df['transit_depth'] * np.sqrt(df['st_teff'])
    
    # 5. Planetary density (mass / volume)
    df['pl_density'] = df['pl_bmasse'] / (df['pl_rade'] ** 3)
    df['pl_density_log'] = np.log1p(df['pl_density'])
    
    # 6. Surface gravity (mass / radius^2)
    df['surface_gravity'] = df['pl_bmasse'] / (df['pl_rade'] ** 2)
    df['surface_gravity_log'] = np.log1p(df['surface_gravity'])
    
    # 7. Stellar luminosity proxy
    df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    df['stellar_luminosity_log'] = np.log1p(df['stellar_luminosity'])
    
    # 8. Orbital velocity (Kepler's laws)
    df['orbital_velocity'] = np.sqrt(df['st_mass']) / np.sqrt(df['pl_orbper'])
    df['orbital_velocity_log'] = np.log1p(df['orbital_velocity'])
    
    # 9. Log transformations for skewed features
    df['pl_orbper_log'] = np.log1p(df['pl_orbper'])
    df['st_teff_log'] = np.log1p(df['st_teff'])
    df['st_rad_log'] = np.log1p(df['st_rad'])
    df['st_mass_log'] = np.log1p(df['st_mass'])
    
    # 10. Planet-star size ratio
    df['size_ratio'] = df['pl_rade'] / (df['st_rad'] * 109.2)
    df['size_ratio_log'] = np.log1p(df['size_ratio'])
    
    # 11. Mass-radius relationship
    df['mass_radius_ratio'] = df['pl_bmasse'] / df['pl_rade']
    df['mass_radius_ratio_log'] = np.log1p(df['mass_radius_ratio'])
    
    # 12. Stellar properties ratios
    df['st_teff_st_mass_ratio'] = df['st_teff'] / (df['st_mass'] * 5778)  # Normalized by solar
    df['st_rad_st_mass_ratio'] = df['st_rad'] / df['st_mass']
    
    print(f"Created {len([col for col in df.columns if col not in ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist', 'label']])} derived features")
    return df

def train_simple_model():
    """Train a model using only available data."""
    print("=== TRAINING SIMPLE TRANSIT MODEL ===")
    
    # Load data
    data = pd.read_csv('Nasa_Space_Apps/Exoplanets/processed_exoplanet_data.csv')
    print(f"Loaded {len(data)} samples")
    
    # Create features
    data = create_simple_features(data)
    
    # Prepare features (exclude original columns and label)
    original_cols = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist', 'label']
    feature_cols = [col for col in data.columns if col not in original_cols]
    
    print(f"Using {len(feature_cols)} derived features")
    print("Features:", feature_cols)
    
    # Prepare X and y
    X = data[feature_cols]
    y = (data['label'] == 2.0).astype(int)  # Planets vs non-planets
    
    print(f"Planets: {y.sum()}, Non-planets: {len(y) - y.sum()}")
    
    # Handle any missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n=== RESULTS ===")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_imputed, y, cv=5, scoring='accuracy')
    print(f"Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test predictions
    y_pred = model.predict(X_test)
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['Non-planet', 'Planet']))
    
    # Save model and features
    joblib.dump(model, 'Nasa_Space_Apps/Exoplanets/simple_transit_model.pkl')
    joblib.dump(feature_cols, 'Nasa_Space_Apps/Exoplanets/simple_transit_features.pkl')
    joblib.dump(imputer, 'Nasa_Space_Apps/Exoplanets/simple_transit_imputer.pkl')
    
    print(f"\n=== MODEL SAVED ===")
    print("Files saved:")
    print("- simple_transit_model.pkl")
    print("- simple_transit_features.pkl") 
    print("- simple_transit_imputer.pkl")
    
    return model, feature_cols, imputer

if __name__ == "__main__":
    train_simple_model()
