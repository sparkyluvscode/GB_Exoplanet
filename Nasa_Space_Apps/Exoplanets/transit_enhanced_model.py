import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import os

# --- Configuration ---
DATA_PATH = 'Nasa_Space_Apps/Exoplanets/processed_exoplanet_data.csv'
MODEL_DIR = 'Nasa_Space_Apps/Exoplanets/'
MODEL_NAME = 'transit_enhanced_model.pkl'
FEATURES_NAME = 'transit_enhanced_features.pkl'
IMPUTER_NAME = 'transit_enhanced_imputer.pkl'
RANDOM_STATE = 42

# --- Enhanced Transit Feature Engineering ---
def create_transit_enhanced_features(df):
    """
    Create enhanced features including the specific transit features requested:
    - Rp/Rs (Planet radius / Star radius ratio)
    - Transit Depth
    - Transit Duration
    Plus other scientifically meaningful transit observables.
    """
    print("Creating enhanced transit features...")

    # Ensure essential columns exist
    required_cols = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # === CORE TRANSIT FEATURES (as requested) ===
    
    # 1. Rp/Rs - Planet to Star radius ratio
    # pl_rade is Earth radii, st_rad is Solar radii
    # Convert: Rp/Rs = pl_rade / (st_rad * 109.2)
    df['rp_rs_ratio'] = df['pl_rade'] / (df['st_rad'] * 109.2)
    df['rp_rs_ratio_log'] = np.log1p(df['rp_rs_ratio'])

    # 2. Transit Depth - (Rp/Rs)²
    # This is the fractional decrease in stellar flux during transit
    df['transit_depth'] = df['rp_rs_ratio'] ** 2
    df['transit_depth_log'] = np.log1p(df['transit_depth'])

    # 3. Transit Duration (proxy)
    # Use semi-major axis and stellar radius to estimate transit duration
    # Semi-major axis from Kepler's 3rd law: a = ((P/365.25)^(2/3)) * (M*^(1/3))
    df['semi_major_axis_AU'] = ((df['pl_orbper'] / 365.25) ** (2/3)) * (df['st_mass'] ** (1/3))
    
    # Transit duration proxy: T_dur ∝ R*/a
    df['transit_duration_proxy'] = df['st_rad'] / (np.pi * df['semi_major_axis_AU'])
    df['transit_duration_proxy_log'] = np.log1p(df['transit_duration_proxy'])

    # === ADDITIONAL TRANSIT OBSERVABLES ===
    
    # 4. Signal-to-Noise proxy
    # Transit depth * sqrt(stellar temperature) - higher temp = better SNR
    df['snr_proxy'] = df['transit_depth'] * np.sqrt(df['st_teff'])
    df['snr_proxy_log'] = np.log1p(df['snr_proxy'])

    # 5. Impact parameter proxy
    # Related to transit shape - grazing vs central transits
    # Estimated from orbital period and stellar properties
    df['impact_parameter_proxy'] = np.sqrt(df['pl_orbper']) / (df['st_rad'] * 10)
    df['impact_parameter_proxy_log'] = np.log1p(df['impact_parameter_proxy'])

    # 6. Planetary density
    df['pl_density'] = df['pl_bmasse'] / (df['pl_rade'] ** 3)
    df['pl_density_log'] = np.log1p(df['pl_density'])

    # 7. Surface gravity
    df['surface_gravity'] = df['pl_bmasse'] / (df['pl_rade'] ** 2)
    df['surface_gravity_log'] = np.log1p(df['surface_gravity'])

    # 8. Stellar luminosity proxy
    df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    df['stellar_luminosity_log'] = np.log1p(df['stellar_luminosity'])

    # 9. Orbital velocity (Kepler's laws)
    df['orbital_velocity'] = np.sqrt(df['st_mass']) / np.sqrt(df['pl_orbper'])
    df['orbital_velocity_log'] = np.log1p(df['orbital_velocity'])

    # 10. Transit probability
    # Probability of seeing a transit: P ∝ R*/a
    df['transit_probability'] = df['st_rad'] / df['semi_major_axis_AU']
    df['transit_probability_log'] = np.log1p(df['transit_probability'])

    # 11. Log transformations for skewed features
    df['pl_orbper_log'] = np.log1p(df['pl_orbper'])
    df['st_teff_log'] = np.log1p(df['st_teff'])
    df['st_rad_log'] = np.log1p(df['st_rad'])
    df['st_mass_log'] = np.log1p(df['st_mass'])
    df['pl_rade_log'] = np.log1p(df['pl_rade'])
    df['pl_bmasse_log'] = np.log1p(df['pl_bmasse'])

    # 12. Planet-star property ratios
    df['mass_radius_ratio'] = df['pl_bmasse'] / df['pl_rade']
    df['mass_radius_ratio_log'] = np.log1p(df['mass_radius_ratio'])

    # 13. Stellar property ratios
    df['st_teff_st_mass_ratio'] = df['st_teff'] / (df['st_mass'] * 5778)  # Normalized by solar
    df['st_rad_st_mass_ratio'] = df['st_rad'] / df['st_mass']

    # 14. Transit observability metrics
    # Larger planets around smaller stars = easier to detect
    df['transit_observability'] = df['rp_rs_ratio'] * np.sqrt(df['st_teff'])
    df['transit_observability_log'] = np.log1p(df['transit_observability'])

    # 15. Physical plausibility checks
    # Unphysical combinations might indicate false positives
    df['density_sanity_check'] = np.where(df['pl_density'] > 50, 0, 1)  # Flag unrealistic densities
    df['size_sanity_check'] = np.where(df['rp_rs_ratio'] > 1.0, 0, 1)  # Flag planets larger than star

    print(f"Created {len([col for col in df.columns if col not in required_cols + ['label']])} enhanced transit features")
    return df

# --- Main Training Function ---
def train_transit_enhanced_model():
    print("=== TRAINING TRANSIT ENHANCED MODEL ===")
    
    # Load data
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(data)} samples")

    # Convert 'label' to binary: 1 for planet, 0 for non-planet
    data['label'] = data['label'].apply(lambda x: 1 if x == 2.0 else 0)

    # Feature Engineering
    df_features = create_transit_enhanced_features(data.copy())

    # Define features (X) and target (y)
    # Exclude original raw features that are directly used to create derived features
    base_features_to_exclude = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist']
    
    # Ensure 'label' is not in features
    X = df_features.drop(columns=base_features_to_exclude + ['label'], errors='ignore')
    y = df_features['label']

    # Save the list of features for consistent inference
    feature_columns = X.columns.tolist()
    print(f"Using {len(feature_columns)} enhanced transit features")
    print(f"Key features: rp_rs_ratio, transit_depth, transit_duration_proxy, snr_proxy")
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, FEATURES_NAME))

    # Handle class imbalance
    print(f"Planets: {y.sum()}, Non-planets: {len(y) - y.sum()}")

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
    joblib.dump(imputer, os.path.join(MODEL_DIR, IMPUTER_NAME))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Train model with optimized parameters
    print("Training enhanced transit model...")
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== RESULTS ===")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_imputed_df, y, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), 
        scoring='accuracy'
    )
    print(f"Cross-validation: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(10))

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, MODEL_NAME))
    print(f"\n=== MODEL SAVED ===")
    print(f"Files saved:")
    print(f"- {MODEL_NAME}")
    print(f"- {FEATURES_NAME}")
    print(f"- {IMPUTER_NAME}")
    
    return model, feature_importance

if __name__ == "__main__":
    train_transit_enhanced_model()
