"""
Transit-Focused Exoplanet Detection Model
=========================================
A scientifically grounded model that focuses on features actually measurable
by transit surveys (K2, TESS, Kepler) without assuming planetary mass is available.

Author: Team Grizzlies
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib

def create_transit_features(df):
    """
    Create features based on what transit surveys actually measure.
    Focus on observable quantities, not derived parameters.
    """
    print("Creating transit-focused features...")
    
    # CORE TRANSIT OBSERVABLES (what telescopes actually see)
    # 1. Transit depth in ppm (corrected units)
    df['transit_depth_ppm'] = (df['pl_rade'] / (df['st_rad'] * 109.2)) ** 2 * 1e6
    df['transit_depth_log'] = np.log1p(df['transit_depth_ppm'])
    
    # 2. Orbital period (fundamental observable)
    df['orbital_period_days'] = df['pl_orbper']
    df['orbital_period_log'] = np.log1p(df['pl_orbper'])
    
    # 3. Transit duration proxy (R*/a, where a comes from Kepler's law)
    # a = (P^2 * M_star)^(1/3) in AU (using years for P)
    period_years = df['pl_orbper'] / 365.25
    semi_major_axis = (period_years ** 2 * df['st_mass']) ** (1/3)
    df['transit_duration_proxy'] = df['st_rad'] / semi_major_axis
    df['transit_duration_proxy_log'] = np.log1p(df['transit_duration_proxy'])
    
    # 4. Signal-to-noise proxy (transit depth * sqrt(stellar temp))
    df['snr_proxy'] = df['transit_depth_ppm'] * np.sqrt(df['st_teff'])
    df['snr_proxy_log'] = np.log1p(df['snr_proxy'])
    
    # STELLAR PROPERTIES (needed to interpret transits)
    # 5. Stellar temperature (affects noise and interpretation)
    df['stellar_temp_log'] = np.log(df['st_teff'])
    
    # 6. Stellar radius (affects transit depth)
    df['stellar_radius_log'] = np.log(df['st_rad'])
    
    # 7. Stellar mass (affects orbital dynamics)
    df['stellar_mass_log'] = np.log(df['st_mass'])
    
    # 8. Stellar density proxy (M/R^3)
    df['stellar_density_proxy'] = df['st_mass'] / (df['st_rad'] ** 3)
    df['stellar_density_proxy_log'] = np.log1p(df['stellar_density_proxy'])
    
    # PHYSICAL VALIDATION (using only radius when mass is available)
    # 9. Planetary radius (directly measured)
    df['planetary_radius_log'] = np.log(df['pl_rade'])
    
    # 10. Radius ratio (Rp/Rs) - fundamental transit parameter
    df['radius_ratio'] = df['pl_rade'] / (df['st_rad'] * 109.2)
    df['radius_ratio_log'] = np.log1p(df['radius_ratio'])
    
    # 11. Planetary density (only when mass is available, otherwise NaN)
    # This will create NaN for many entries, which is scientifically correct
    mass_available = ~df['pl_bmasse'].isna()
    df['planetary_density'] = np.nan
    df.loc[mass_available, 'planetary_density'] = (
        df.loc[mass_available, 'pl_bmasse'] / 
        (df.loc[mass_available, 'pl_rade'] ** 3)
    )
    df['planetary_density_log'] = np.log1p(df['planetary_density'].fillna(0))
    
    # 12. Surface gravity proxy (only when mass is available)
    df['surface_gravity_proxy'] = np.nan
    df.loc[mass_available, 'surface_gravity_proxy'] = (
        df.loc[mass_available, 'pl_bmasse'] / 
        (df.loc[mass_available, 'pl_rade'] ** 2)
    )
    df['surface_gravity_proxy_log'] = np.log1p(df['surface_gravity_proxy'].fillna(0))
    
    # ORBITAL MECHANICS (from observable period and stellar mass)
    # 13. Semi-major axis in AU (Kepler's 3rd law)
    df['semi_major_axis_AU'] = semi_major_axis
    df['semi_major_axis_AU_log'] = np.log1p(df['semi_major_axis_AU'])
    
    # 14. Orbital velocity proxy (sqrt(M_star/a))
    df['orbital_velocity_proxy'] = np.sqrt(df['st_mass'] / df['semi_major_axis_AU'])
    df['orbital_velocity_proxy_log'] = np.log1p(df['orbital_velocity_proxy'])
    
    # 15. Tidal radius (distance where tidal forces become significant)
    # This helps identify unrealistic orbital configurations
    df['tidal_radius_proxy'] = df['st_rad'] * (df['st_mass'] / df['pl_rade']) ** (1/3)
    df['tidal_radius_proxy_log'] = np.log1p(df['tidal_radius_proxy'])
    
    # PHYSICAL REALITY CHECKS
    # 16. Roche limit check (planet too close to star)
    df['roche_limit_violation'] = (df['semi_major_axis_AU'] < 2.456 * df['st_rad'] * 
                                  (df['st_mass'] / (df['pl_rade'] ** 3)) ** (1/3))
    
    # 17. Transit depth sanity check (too deep = likely stellar companion)
    df['transit_too_deep'] = df['transit_depth_ppm'] > 10000  # 1% transit depth
    
    # 18. Orbital period sanity check (too short = unphysical)
    df['period_too_short'] = df['pl_orbper'] < 0.5  # Less than 12 hours
    
    # REMOVE OBSERVATIONAL BIAS FEATURES
    # Don't use distance, magnitudes, etc. as they encode selection effects
    
    print(f"Created {len([col for col in df.columns if col not in ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist', 'label']])} transit-focused features")
    return df

def select_transit_features(df, target_col='label'):
    """
    Select the most scientifically meaningful features for transit detection.
    """
    # Features that are always measurable
    core_features = [
        'transit_depth_ppm', 'transit_depth_log',
        'orbital_period_days', 'orbital_period_log',
        'transit_duration_proxy', 'transit_duration_proxy_log',
        'snr_proxy', 'snr_proxy_log',
        'stellar_temp_log', 'stellar_radius_log', 'stellar_mass_log',
        'stellar_density_proxy', 'stellar_density_proxy_log',
        'planetary_radius_log', 'radius_ratio', 'radius_ratio_log',
        'semi_major_axis_AU', 'semi_major_axis_AU_log',
        'orbital_velocity_proxy', 'orbital_velocity_proxy_log',
        'tidal_radius_proxy', 'tidal_radius_proxy_log',
        'roche_limit_violation', 'transit_too_deep', 'period_too_short'
    ]
    
    # Features that are only available when mass is known
    mass_dependent_features = [
        'planetary_density_log', 'surface_gravity_proxy_log'
    ]
    
    # Check which mass-dependent features are available
    available_features = []
    for feature in core_features:
        if feature in df.columns:
            available_features.append(feature)
    
    for feature in mass_dependent_features:
        if feature in df.columns and not df[feature].isna().all():
            available_features.append(feature)
    
    print(f"Selected {len(available_features)} transit-focused features")
    return available_features

def train_transit_model():
    """
    Train the transit-focused model with proper feature selection.
    """
    print("Loading data...")
    data = pd.read_csv('Nasa_Space_Apps/Exoplanets/processed_exoplanet_data.csv')
    
    print(f"Original data shape: {data.shape}")
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    # Create transit-focused features
    data = create_transit_features(data)
    
    # Select scientifically meaningful features
    feature_cols = select_transit_features(data)
    
    # Prepare features and target
    X = data[feature_cols].fillna(0)  # Fill NaN with 0 for mass-dependent features
    y = data['label']
    
    # Convert to binary classification
    y_binary = (y == 2.0).astype(int)
    
    print(f"Binary label distribution:\n{pd.Series(y_binary).value_counts()}")
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Train model
    print("Training transit-focused model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("TRANSIT-FOCUSED MODEL RESULTS")
    print("="*50)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test set performance
    test_accuracy = (y_pred == y_test).mean()
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = np.trapz(precision, recall)
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model and features
    joblib.dump(model, 'Nasa_Space_Apps/Exoplanets/transit_focused_model.pkl')
    joblib.dump(feature_cols, 'Nasa_Space_Apps/Exoplanets/transit_focused_features.pkl')
    
    # Save scaler (if needed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'Nasa_Space_Apps/Exoplanets/transit_focused_scaler.pkl')
    
    print(f"\nModel saved with {len(feature_cols)} features")
    print("Files saved:")
    print("- transit_focused_model.pkl")
    print("- transit_focused_features.pkl") 
    print("- transit_focused_scaler.pkl")
    
    return model, feature_cols, test_accuracy, roc_auc, pr_auc

if __name__ == "__main__":
    model, features, accuracy, roc_auc, pr_auc = train_transit_model()
