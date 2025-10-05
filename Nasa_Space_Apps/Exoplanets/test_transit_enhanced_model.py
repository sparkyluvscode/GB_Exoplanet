import pandas as pd
import numpy as np
import joblib
import os

def create_transit_features(pl_orbper, pl_rade, pl_bmasse, st_teff, st_rad, st_mass, sy_dist):
    """Create enhanced transit features from input parameters."""
    
    # Create a DataFrame with the input data
    df = pd.DataFrame({
        'pl_orbper': [pl_orbper],
        'pl_rade': [pl_rade],
        'pl_bmasse': [pl_bmasse],
        'st_teff': [st_teff],
        'st_rad': [st_rad],
        'st_mass': [st_mass],
        'sy_dist': [sy_dist]
    })
    
    # === CORE TRANSIT FEATURES ===
    
    # 1. Rp/Rs - Planet to Star radius ratio
    df['rp_rs_ratio'] = df['pl_rade'] / (df['st_rad'] * 109.2)
    df['rp_rs_ratio_log'] = np.log1p(df['rp_rs_ratio'])

    # 2. Transit Depth - (Rp/Rs)²
    df['transit_depth'] = df['rp_rs_ratio'] ** 2
    df['transit_depth_log'] = np.log1p(df['transit_depth'])

    # 3. Transit Duration (proxy)
    df['semi_major_axis_AU'] = ((df['pl_orbper'] / 365.25) ** (2/3)) * (df['st_mass'] ** (1/3))
    df['transit_duration_proxy'] = df['st_rad'] / (np.pi * df['semi_major_axis_AU'])
    df['transit_duration_proxy_log'] = np.log1p(df['transit_duration_proxy'])

    # === ADDITIONAL TRANSIT OBSERVABLES ===
    
    # 4. Signal-to-Noise proxy
    df['snr_proxy'] = df['transit_depth'] * np.sqrt(df['st_teff'])
    df['snr_proxy_log'] = np.log1p(df['snr_proxy'])

    # 5. Impact parameter proxy
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
    df['st_teff_st_mass_ratio'] = df['st_teff'] / (df['st_mass'] * 5778)
    df['st_rad_st_mass_ratio'] = df['st_rad'] / df['st_mass']

    # 14. Transit observability metrics
    df['transit_observability'] = df['rp_rs_ratio'] * np.sqrt(df['st_teff'])
    df['transit_observability_log'] = np.log1p(df['transit_observability'])

    # 15. Physical plausibility checks
    df['density_sanity_check'] = np.where(df['pl_density'] > 50, 0, 1)
    df['size_sanity_check'] = np.where(df['rp_rs_ratio'] > 1.0, 0, 1)

    return df

def test_exoplanet(name, pl_orbper, pl_rade, pl_bmasse, st_teff, st_rad, st_mass, sy_dist, expected_result, description):
    """Test a single exoplanet with the enhanced transit model."""
    
    # Load model
    try:
        model = joblib.load('Nasa_Space_Apps/Exoplanets/transit_enhanced_model.pkl')
        features = joblib.load('Nasa_Space_Apps/Exoplanets/transit_enhanced_features.pkl')
        imputer = joblib.load('Nasa_Space_Apps/Exoplanets/transit_enhanced_imputer.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create features
    df_features = create_transit_features(pl_orbper, pl_rade, pl_bmasse, st_teff, st_rad, st_mass, sy_dist)
    
    # Prepare data for model
    base_features_to_exclude = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist']
    X = df_features.drop(columns=base_features_to_exclude, errors='ignore')
    
    # Ensure feature order matches training
    X_ordered = X[features]
    
    # Impute missing values
    X_imputed = imputer.transform(X_ordered)
    
    # Make prediction
    prediction = model.predict(X_imputed)[0]
    probability = model.predict_proba(X_imputed)[0]
    
    # Calculate key transit features
    rp_rs_ratio = df_features['rp_rs_ratio'].iloc[0]
    transit_depth = df_features['transit_depth'].iloc[0]
    transit_duration_proxy = df_features['transit_duration_proxy'].iloc[0]
    snr_proxy = df_features['snr_proxy'].iloc[0]
    transit_probability = df_features['transit_probability'].iloc[0]
    pl_density = df_features['pl_density'].iloc[0]
    
    # Determine result
    result = "✅ CORRECT" if prediction == expected_result else "❌ INCORRECT"
    
    return {
        'name': name,
        'description': description,
        'expected': expected_result,
        'predicted': prediction,
        'probability': probability[1] * 100,
        'result': result,
        'rp_rs_ratio': rp_rs_ratio,
        'transit_depth': transit_depth,
        'transit_duration_proxy': transit_duration_proxy,
        'snr_proxy': snr_proxy,
        'transit_probability': transit_probability,
        'pl_density': pl_density
    }

def test_enhanced_model():
    """Test the enhanced transit model on various example exoplanets."""
    
    print("=== TESTING ENHANCED TRANSIT MODEL ON EXAMPLE EXOPLANETS ===\n")
    
    # Test cases: (name, pl_orbper, pl_rade, pl_bmasse, st_teff, st_rad, st_mass, sy_dist, expected_result, description)
    test_cases = [
        # Real confirmed exoplanets
        ("K2-18 b", 32.94, 2.61, 8.63, 3457, 0.402, 0.36, 124, 1, "Super-Earth in habitable zone"),
        ("HD 209458 b", 3.52, 15.4, 220.0, 6075, 1.2, 1.15, 159, 1, "First exoplanet discovered by transit"),
        ("Kepler-452 b", 384.8, 5.2, 5.0, 5757, 1.05, 1.04, 1402, 1, "Earth-like planet in habitable zone"),
        ("WASP-12 b", 1.09, 18.7, 461.0, 6300, 1.6, 1.35, 1419, 1, "Ultra-hot Jupiter"),
        ("TRAPPIST-1 b", 1.51, 1.12, 1.02, 2559, 0.12, 0.08, 12.4, 1, "Rocky planet in TRAPPIST-1 system"),
        
        # False positive cases (should be 0)
        ("Brown Dwarf", 1.5, 12.0, 6000.0, 6500, 1.46, 1.32, 270, 0, "Brown dwarf (too massive for planet)"),
        ("Stellar Variability", 0.6, 1.0, 1.0, 5800, 1.00, 1.00, 100, 0, "Stellar variability mimic"),
        ("Grazing EB", 0.8, 10.0, 20.0, 5000, 0.90, 0.90, 200, 0, "Grazing eclipsing binary"),
        
        # Edge cases
        ("Hot Jupiter", 3.2, 13.0, 318.0, 6100, 1.20, 1.15, 200, 1, "Hot Jupiter (should be planet)"),
        ("Mini-Neptune", 30.0, 2.6, 8.0, 5200, 0.90, 0.90, 120, 1, "Mini-Neptune (should be planet)"),
        ("Ultra-short Period", 0.6, 1.0, 1.0, 5800, 1.00, 1.00, 100, 0, "Ultra-short period (likely false)"),
    ]
    
    results = []
    
    for test_case in test_cases:
        result = test_exoplanet(*test_case)
        if result:
            results.append(result)
    
    # Display results
    print("📊 TEST RESULTS")
    print("=" * 80)
    
    correct_predictions = 0
    total_predictions = len(results)
    
    for result in results:
        print(f"\n🪐 {result['name']}")
        print(f"   Description: {result['description']}")
        print(f"   Expected: {'Planet' if result['expected'] == 1 else 'False Positive'}")
        print(f"   Predicted: {'Planet' if result['predicted'] == 1 else 'False Positive'}")
        print(f"   Confidence: {result['probability']:.1f}%")
        print(f"   Result: {result['result']}")
        
        print(f"   Key Transit Features:")
        print(f"     Rp/Rs Ratio: {result['rp_rs_ratio']:.4f}")
        print(f"     Transit Depth: {result['transit_depth']:.6f}")
        print(f"     Transit Duration Proxy: {result['transit_duration_proxy']:.4f}")
        print(f"     SNR Proxy: {result['snr_proxy']:.2f}")
        print(f"     Transit Probability: {result['transit_probability']:.4f}")
        print(f"     Planetary Density: {result['pl_density']:.2f} g/cm³")
        
        if result['predicted'] == result['expected']:
            correct_predictions += 1
    
    # Summary
    print(f"\n📈 SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions/total_predictions*100:.1f}%")
    
    # Analyze by category
    confirmed_planets = [r for r in results if r['expected'] == 1]
    false_positives = [r for r in results if r['expected'] == 0]
    
    print(f"\n🎯 BREAKDOWN BY CATEGORY:")
    print(f"Confirmed Planets: {len(confirmed_planets)}")
    print(f"False Positives: {len(false_positives)}")
    
    # Accuracy by category
    correct_planets = sum(1 for r in confirmed_planets if r['predicted'] == 1)
    correct_false_positives = sum(1 for r in false_positives if r['predicted'] == 0)
    
    print(f"\n📊 ACCURACY BY CATEGORY:")
    print(f"Planet Detection: {correct_planets}/{len(confirmed_planets)} ({correct_planets/len(confirmed_planets)*100:.1f}%)")
    print(f"False Positive Rejection: {correct_false_positives}/{len(false_positives)} ({correct_false_positives/len(false_positives)*100:.1f}%)")
    
    # Feature analysis
    print(f"\n🔬 TRANSIT FEATURE ANALYSIS:")
    print("=" * 80)
    
    # Group by prediction
    predicted_planets = [r for r in results if r['predicted'] == 1]
    predicted_false_positives = [r for r in results if r['predicted'] == 0]
    
    if predicted_planets:
        avg_transit_depth_planets = np.mean([r['transit_depth'] for r in predicted_planets])
        avg_rp_rs_planets = np.mean([r['rp_rs_ratio'] for r in predicted_planets])
        avg_snr_planets = np.mean([r['snr_proxy'] for r in predicted_planets])
        
        print(f"Predicted Planets (n={len(predicted_planets)}):")
        print(f"  Average Transit Depth: {avg_transit_depth_planets:.6f}")
        print(f"  Average Rp/Rs Ratio: {avg_rp_rs_planets:.4f}")
        print(f"  Average SNR Proxy: {avg_snr_planets:.2f}")
    
    if predicted_false_positives:
        avg_transit_depth_fp = np.mean([r['transit_depth'] for r in predicted_false_positives])
        avg_rp_rs_fp = np.mean([r['rp_rs_ratio'] for r in predicted_false_positives])
        avg_snr_fp = np.mean([r['snr_proxy'] for r in predicted_false_positives])
        
        print(f"\nPredicted False Positives (n={len(predicted_false_positives)}):")
        print(f"  Average Transit Depth: {avg_transit_depth_fp:.6f}")
        print(f"  Average Rp/Rs Ratio: {avg_rp_rs_fp:.4f}")
        print(f"  Average SNR Proxy: {avg_snr_fp:.2f}")
    
    return results

if __name__ == "__main__":
    test_enhanced_model()
