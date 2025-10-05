import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'processed_exoplanet_data.csv')
MODEL_DIR = SCRIPT_DIR
MODEL_NAME = 'corrected_transit_model.pkl'
FEATURES_NAME = 'corrected_transit_features.pkl'
IMPUTER_NAME = 'corrected_transit_imputer.pkl'
RANDOM_STATE = 42

def create_corrected_transit_features(df):
    """
    Create scientifically accurate transit features.
    Based on actual transit method physics without mass dependencies.
    """
    print("Creating SCIENTIFICALLY CORRECTED transit features...")

    # Ensure essential columns exist
    required_cols = ['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'sy_dist']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # === CORE TRANSIT OBSERVABLES (PHYSICALLY ACCURATE) ===
    
    # 1. Rp/Rs - Planet to Star radius ratio (THE fundamental transit observable)
    # Convert Earth radii to Solar radii: 1 R_earth = 0.009167 R_sun
    df['rp_rs_ratio'] = df['pl_rade'] * 0.009167 / df['st_rad']
    df['rp_rs_ratio_log'] = np.log1p(df['rp_rs_ratio'])

    # 2. Transit Depth - (Rp/Rs)² (the actual brightness decrease)
    df['transit_depth'] = df['rp_rs_ratio'] ** 2
    df['transit_depth_log'] = np.log1p(df['transit_depth'])

    # 3. Transit Duration (physically correct)
    # For circular orbits: T_dur = (R* * P) / (π * a)
    # Using Kepler's 3rd law: a ∝ P^(2/3) (assuming stellar mass ~ 1 solar mass)
    # T_dur ∝ (R* * P^(1/3)) / π
    df['transit_duration'] = (df['st_rad'] * (df['pl_orbper'] ** (1/3))) / np.pi
    df['transit_duration_log'] = np.log1p(df['transit_duration'])

    # 4. Transit Probability (geometric probability)
    # P_transit = R* / a, where a ∝ P^(2/3)
    # P_transit ∝ R* / P^(2/3)
    df['transit_probability'] = df['st_rad'] / (df['pl_orbper'] ** (2/3))
    df['transit_probability_log'] = np.log1p(df['transit_probability'])

    # 5. Signal-to-Noise Ratio proxy
    # SNR ∝ (transit_depth) * sqrt(stellar_flux) / sqrt(noise)
    # Approximating stellar flux ∝ R*² * T_eff⁴ and noise ∝ sqrt(period)
    stellar_flux_proxy = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    noise_proxy = np.sqrt(df['pl_orbper'])
    df['snr_proxy'] = df['transit_depth'] * np.sqrt(stellar_flux_proxy) / noise_proxy
    df['snr_proxy_log'] = np.log1p(df['snr_proxy'])

    # 6. Stellar Luminosity (Stefan-Boltzmann law)
    # L ∝ R*² * T_eff⁴
    df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    df['stellar_luminosity_log'] = np.log1p(df['stellar_luminosity'])

    # 7. Transit Observability (distance-dependent detectability)
    # How observable is this transit given the distance?
    # Brightness decreases as 1/distance², so observability ∝ transit_depth / distance
    df['transit_observability'] = df['transit_depth'] / df['sy_dist']
    df['transit_observability_log'] = np.log1p(df['transit_observability'])

    # 8. Transit Frequency (how often transits occur)
    # Frequency = 1 / orbital period
    df['transit_frequency'] = 1.0 / df['pl_orbper']
    df['transit_frequency_log'] = np.log1p(df['transit_frequency'])

    # 9. Stellar Properties (normalized to solar values)
    df['st_teff_normalized'] = df['st_teff'] / 5778  # Normalized to solar temperature
    df['st_rad_normalized'] = df['st_rad']  # Already in solar radii
    df['stellar_temp_radius_ratio'] = df['st_teff'] / df['st_rad']

    # 10. Log transformations for skewed features
    df['pl_orbper_log'] = np.log1p(df['pl_orbper'])
    df['pl_rade_log'] = np.log1p(df['pl_rade'])
    df['st_teff_log'] = np.log1p(df['st_teff'])
    df['st_rad_log'] = np.log1p(df['st_rad'])
    df['sy_dist_log'] = np.log1p(df['sy_dist'])

    # 11. Physical Sanity Checks
    df['size_sanity_check'] = np.where(df['rp_rs_ratio'] > 1.0, 0, 1)  # Planet can't be bigger than star
    df['period_sanity_check'] = np.where((df['pl_orbper'] < 0.1) | (df['pl_orbper'] > 10000), 0, 1)
    df['temperature_sanity_check'] = np.where((df['st_teff'] < 2000) | (df['st_teff'] > 10000), 0, 1)
    df['transit_depth_sanity'] = np.where((df['transit_depth'] < 1e-6) | (df['transit_depth'] > 0.1), 0, 1)

    # 12. Distance-based SNR degradation
    # SNR decreases with distance due to photon noise
    df['distance_snr_degradation'] = df['snr_proxy'] / np.sqrt(df['sy_dist'])
    df['distance_snr_degradation_log'] = np.log1p(df['distance_snr_degradation'])

    # 13. Transit Impact Parameter Proxy
    # Related to transit geometry (how the planet crosses the star)
    # Impact parameter affects transit shape and duration
    df['impact_parameter_proxy'] = np.sqrt(df['pl_orbper']) / (df['st_rad'] * 10)
    df['impact_parameter_proxy_log'] = np.log1p(df['impact_parameter_proxy'])

    print(f"Created {len([col for col in df.columns if col not in required_cols + ['label']])} scientifically corrected transit features")
    return df

def train_corrected_transit_model():
    print("=== TRAINING SCIENTIFICALLY CORRECTED TRANSIT MODEL ===")
    
    # Load data
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(data)} samples")

    # Filter for realistic exoplanet candidates
    print("Filtering for realistic transit candidates...")
    print(f"Original dataset: {len(data)} samples")
    
    # Remove obvious false positives (label = 1.0)
    data = data[data['label'] != 1.0]
    print(f"Filtered dataset: {len(data)} samples")
    print(f"Removed: {len(pd.read_csv(DATA_PATH)) - len(data)} samples ({(len(pd.read_csv(DATA_PATH)) - len(data))/len(pd.read_csv(DATA_PATH))*100:.1f}%)")

    # Convert 'label' to binary: 1 for planet, 0 for non-planet
    data['label'] = data['label'].apply(lambda x: 1 if x == 2.0 else 0)

    # Feature Engineering (NO MASS FEATURES)
    df_features = create_corrected_transit_features(data.copy())

    # Define features (X) and target (y)
    # Exclude ALL mass-related features and raw input features
    base_features_to_exclude = ['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'sy_dist']
    mass_features_to_exclude = ['pl_bmasse', 'st_mass']  # Explicitly exclude mass features
    
    X = df_features.drop(columns=base_features_to_exclude + mass_features_to_exclude + ['label'], errors='ignore')
    y = df_features['label']

    # Save the list of features for consistent inference
    feature_columns = X.columns.tolist()
    print(f"Using {len(feature_columns)} scientifically corrected transit features")
    print(f"Features: {feature_columns}")
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, FEATURES_NAME))

    # Handle class imbalance
    print(f"Planets: {y.sum()}, Non-planets: {len(y) - y.sum()}")

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
    joblib.dump(imputer, os.path.join(MODEL_DIR, IMPUTER_NAME))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_imputed_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Train XGBoost model
    print("Training scientifically corrected XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== RESULTS ===")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_imputed_df, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring='accuracy')
    print(f"Cross-validation: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, MODEL_NAME))
    print(f"\n=== MODEL SAVED ===")
    print(f"Files saved:\n- {MODEL_NAME}\n- {FEATURES_NAME}\n- {IMPUTER_NAME}")

    # Display top features
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n=== TOP 15 MOST IMPORTANT FEATURES ===")
        print(feature_importance.head(15).to_string(index=False))

    print("\n=== SCIENTIFICALLY CORRECTED MODEL SUMMARY ===")
    print("✅ Uses ONLY transit method observables (NO MASS):")
    print("   - Orbital period (pl_orbper)")
    print("   - Planet radius (pl_rade)")
    print("   - Stellar temperature (st_teff)")
    print("   - Stellar radius (st_rad)")
    print("   - System distance (sy_dist)")
    print("✅ Physically accurate calculations:")
    print("   - Correct Earth/Solar radii conversion (0.009167)")
    print("   - Proper transit duration formula")
    print("   - Accurate transit probability")
    print("   - Realistic SNR calculations")
    print("❌ NO MASS MEASUREMENTS REQUIRED!")
    print("🎯 Perfect for transit surveys (Kepler, K2, TESS)")

    return accuracy, np.mean(cv_scores)

if __name__ == "__main__":
    accuracy, cv_score = train_corrected_transit_model()
    
    print(f"\n🏆 SCIENTIFICALLY CORRECTED TRANSIT MODEL PERFORMANCE:")
    print(f"   • Test Accuracy: {accuracy:.2%}")
    print(f"   • Cross-validation: {cv_score:.2%}")
    
    if accuracy >= 0.95:
        print("\n🎉 EXCELLENT! Achieved 95%+ accuracy with correct physics!")
    elif accuracy >= 0.90:
        print("\n✅ GOOD! Achieved 90%+ accuracy with correct physics!")
    else:
        print("\n⚠️  May need further tuning...")
