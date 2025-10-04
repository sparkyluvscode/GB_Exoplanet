"""
Corrected Physics Model for NASA Space Apps
==========================================
Fixes critical physics errors identified by ChatGPT feedback:
1. Correct transit depth units (Earth radii / Solar radii conversion)
2. Proper Kepler scalings (years, AU units)
3. Correct orbital mechanics
4. Remove spurious observational bias features
5. Handle missing values properly
6. Deterministic feature ordering

Author: NASA Space Apps Team
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib

def create_corrected_physics_features(df):
    """Create features with correct physics and units."""
    print("Creating corrected physics features...")
    
    # CRITICAL FIX 1: Correct transit depth units
    # pl_rade is Earth radii, st_rad is Solar radii
    # Convert: Rp/Rs = pl_rade / (st_rad * 109.2)
    df['transit_depth_corrected'] = (df['pl_rade'] / (df['st_rad'] * 109.2)) ** 2
    df['transit_depth_corrected_log'] = np.log1p(df['transit_depth_corrected'])
    
    # CRITICAL FIX 2: Correct semi-major axis (use years, AU units)
    # a_AU = ((pl_orbper/365.25)^(2/3)) * (st_mass^(1/3))
    df['semi_major_axis_AU'] = ((df['pl_orbper'] / 365.25) ** (2/3)) * (df['st_mass'] ** (1/3))
    df['semi_major_axis_AU_log'] = np.log1p(df['semi_major_axis_AU'])
    
    # CRITICAL FIX 3: Correct transit duration ratio
    # Use R*/(π a) - dimensionally correct proxy
    df['transit_duration_ratio_corrected'] = df['st_rad'] / (np.pi * df['semi_major_axis_AU'])
    df['transit_duration_ratio_corrected_log'] = np.log1p(df['transit_duration_ratio_corrected'])
    
    # CORRECTED SNR proxy (with proper transit depth)
    df['snr_proxy_corrected'] = df['transit_depth_corrected'] * np.sqrt(df['st_teff'])
    
    # Planetary density (this was correct)
    df['pl_density'] = df['pl_bmasse'] / (df['pl_rade'] ** 3)
    df['pl_density_log'] = np.log1p(df['pl_density'])
    
    # Surface gravity (this was correct)
    df['surface_gravity'] = df['pl_bmasse'] / (df['pl_rade'] ** 2)
    df['surface_gravity_log'] = np.log1p(df['surface_gravity'])
    
    # Habitable zone distance (this was correct)
    df['hab_zone_distance'] = np.abs(df['pl_insol'] - 1.0)
    df['hab_zone_distance_log'] = np.log1p(df['hab_zone_distance'])
    
    # Stellar luminosity (this was correct)
    df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    df['stellar_luminosity_log'] = np.log1p(df['stellar_luminosity'])
    
    # Orbital velocity (Kepler's laws - this was correct)
    df['orbital_velocity'] = np.sqrt(df['st_mass']) / np.sqrt(df['pl_orbper'])
    df['orbital_velocity_log'] = np.log1p(df['orbital_velocity'])
    
    # Temperature ratio (this was correct)
    df['temp_ratio'] = df['pl_eqt'] / df['st_teff']
    
    # Log transformations for highly skewed features
    df['pl_orbper_log'] = np.log1p(df['pl_orbper'])
    df['sy_dist_log'] = np.log1p(df['sy_dist'])
    
    print(f"Added {len([col for col in df.columns if col not in ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist', 'label']])} corrected features")
    return df

def select_physics_features(df, target_col='label'):
    """Select features based on exoplanet physics, removing observational bias."""
    print("Selecting physics-based features (removing observational bias)...")
    
    # TIER 1: Core transit method physics (highest priority)
    tier1_features = [
        'transit_depth_corrected', 'transit_depth_corrected_log',
        'transit_duration_ratio_corrected', 'transit_duration_ratio_corrected_log',
        'snr_proxy_corrected', 'pl_orbper', 'pl_rade', 'pl_bmasse'
    ]
    
    # TIER 2: Planetary physics
    tier2_features = [
        'pl_density', 'pl_density_log', 'surface_gravity', 'surface_gravity_log',
        'st_teff', 'st_rad', 'st_mass'
    ]
    
    # TIER 3: Orbital mechanics
    tier3_features = [
        'semi_major_axis_AU', 'semi_major_axis_AU_log',
        'orbital_velocity', 'orbital_velocity_log',
        'pl_orbper_log'
    ]
    
    # TIER 4: Habitability and stellar properties
    tier4_features = [
        'hab_zone_distance', 'hab_zone_distance_log',
        'stellar_luminosity', 'stellar_luminosity_log',
        'temp_ratio', 'pl_insol', 'pl_eqt'
    ]
    
    # TIER 5: Stellar classification
    tier5_features = [
        'st_logg', 'st_met'
    ]
    
    # REMOVED: Observational bias features (sy_dist, magnitudes, etc.)
    # These encode observational selection effects, not planet physics
    
    # Select features from all tiers
    selected_features = []
    for tier in [tier1_features, tier2_features, tier3_features, tier4_features, tier5_features]:
        selected_features.extend([f for f in tier if f in df.columns])
    
    # Add any missing original features that are physics-based
    physics_original = ['pl_insol', 'pl_eqt', 'st_logg', 'st_met']
    selected_features.extend([f for f in physics_original if f in df.columns and f not in selected_features])
    
    # DETERMINISTIC ORDERING (no more set() randomization)
    selected_features = sorted(list(set(selected_features)))
    
    print(f"Selected {len(selected_features)} physics-based features (removed observational bias)")
    print(f"Features: {selected_features}")
    return selected_features

def train_corrected_model(df, selected_features):
    """Train model with corrected physics and proper handling."""
    print("Training corrected physics model...")
    
    X = df[selected_features]
    y = df['label']
    
    # CORRECT LABEL MAPPING: 2.0 = Exoplanet (1), others = Not Exoplanet (0)
    y_binary = (y == 2.0).astype(int)
    print(f"Label distribution: {np.bincount(y_binary)}")
    
    # HANDLE MISSING VALUES (Critical fix)
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Train GradientBoosting with corrected parameters
    model = GradientBoostingClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Corrected model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(classification_report(y_test, y_pred, digits=4))
    
    return model, selected_features, accuracy, imputer

def main():
    """Main corrected physics pipeline."""
    print("=== CORRECTED PHYSICS MODEL FOR NASA SPACE APPS ===")
    print("Fixing critical physics errors identified by ChatGPT")
    print()
    
    # Load enhanced data
    data = pd.read_csv('Nasa_Space_Apps/Exoplanets/enhanced_processed_exoplanet_data.csv')
    print(f"Loaded data: {data.shape}")
    
    # Create corrected physics features
    data = create_corrected_physics_features(data)
    print(f"After corrected feature engineering: {data.shape}")
    
    # Select physics-based features (remove observational bias)
    selected_features = select_physics_features(data)
    
    # Train corrected model
    model, features, accuracy, imputer = train_corrected_model(data, selected_features)
    
    # Save corrected model
    joblib.dump(model, 'Nasa_Space_Apps/Exoplanets/corrected_physics_model.pkl')
    joblib.dump(features, 'Nasa_Space_Apps/Exoplanets/corrected_physics_features.pkl')
    joblib.dump(imputer, 'Nasa_Space_Apps/Exoplanets/corrected_physics_imputer.pkl')
    
    print(f"\n🚀 CORRECTED PHYSICS MODEL RESULTS:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Features: {len(features)} physics-based (no observational bias)")
    print("Model saved as corrected_physics_model.pkl")
    
    return model, features, accuracy

if __name__ == "__main__":
    model, features, accuracy = main()
