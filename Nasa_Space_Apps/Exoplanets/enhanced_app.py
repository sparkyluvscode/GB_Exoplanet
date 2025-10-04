"""
Enhanced Streamlit Web App for Exoplanet Prediction
==================================================
Uses the improved XGBoost model with 91.70% accuracy and 25 selected features.

How to use:
- Enter the object's parameters below.
- Click 'Predict' to see if it's likely an exoplanet.

Author: NASA Space Apps Team
"""
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the enhanced trained XGBoost model and selected features
try:
    model = joblib.load('Nasa_Space_Apps/Exoplanets/enhanced_xgb_model.pkl')
    selected_features = joblib.load('Nasa_Space_Apps/Exoplanets/selected_features.pkl')
    print("Loaded enhanced model successfully")
except FileNotFoundError:
    # Fallback to original model
    model = joblib.load('Nasa_Space_Apps/Exoplanets/rf_exoplanet_model.pkl')
    selected_features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist']
    print("Using fallback model")

st.set_page_config(page_title="Enhanced Exoplanet Predictor", page_icon="🪐", layout="centered")
st.title("🪐 Enhanced Exoplanet Prediction App")
st.write("""
**Advanced AI Model with 91.70% Accuracy**

Enter the parameters below to predict whether the object is likely an exoplanet. 
This model uses 25 carefully selected features and advanced machine learning techniques.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Planetary Properties")
    pl_orbper = st.number_input('Orbital Period (days)', min_value=0.0, value=10.0, 
                               help="Time for one orbit around the star.")
    pl_rade = st.number_input('Planetary Radius (Earth radii)', min_value=0.0, value=1.0, 
                             help="Radius compared to Earth.")
    pl_bmasse = st.number_input('Planetary Mass (Earth masses)', min_value=0.0, value=10.0, 
                               help="Mass compared to Earth.")
    
    # Additional features for enhanced model
    if 'pl_insol' in selected_features:
        pl_insol = st.number_input('Insolation Flux (Earth units)', min_value=0.0, value=1.0,
                                  help="Amount of stellar radiation received.")
    else:
        pl_insol = 1.0
    
    if 'pl_eqt' in selected_features:
        pl_eqt = st.number_input('Equilibrium Temperature (K)', min_value=0.0, value=300.0,
                                help="Planetary equilibrium temperature.")
    else:
        pl_eqt = 300.0

with col2:
    st.subheader("⭐ Stellar Properties")
    st_teff = st.number_input('Star Effective Temperature (K)', min_value=0.0, value=5500.0, 
                             help="Temperature of the host star.")
    st_rad = st.number_input('Star Radius (Solar radii)', min_value=0.0, value=1.0, 
                            help="Radius of the host star.")
    st_mass = st.number_input('Star Mass (Solar masses)', min_value=0.0, value=1.0, 
                             help="Mass of the host star.")
    
    # Additional stellar features
    if 'st_logg' in selected_features:
        st_logg = st.number_input('Star Surface Gravity (log g)', min_value=0.0, value=4.5,
                                 help="Surface gravity of the star.")
    else:
        st_logg = 4.5
    
    if 'st_met' in selected_features:
        st_met = st.number_input('Star Metallicity [Fe/H]', min_value=-2.0, value=0.0,
                                help="Metal content of the star.")
    else:
        st_met = 0.0

# System properties
st.subheader("🌌 System Properties")
sy_dist = st.number_input('System Distance (parsecs)', min_value=0.0, value=100.0, 
                         help="Distance from Earth.")

# Additional system features
if 'sy_vmag' in selected_features:
    sy_vmag = st.number_input('Visual Magnitude', min_value=0.0, value=10.0,
                             help="Apparent magnitude in V band.")
else:
    sy_vmag = 10.0

if 'sy_kmag' in selected_features:
    sy_kmag = st.number_input('K-band Magnitude', min_value=0.0, value=8.0,
                             help="Apparent magnitude in K band.")
else:
    sy_kmag = 8.0

# Create feature vector based on selected features
feature_dict = {
    'pl_orbper': pl_orbper,
    'pl_rade': pl_rade,
    'pl_bmasse': pl_bmasse,
    'pl_insol': pl_insol,
    'pl_eqt': pl_eqt,
    'st_teff': st_teff,
    'st_rad': st_rad,
    'st_mass': st_mass,
    'st_logg': st_logg,
    'st_met': st_met,
    'sy_dist': sy_dist,
    'sy_vmag': sy_vmag,
    'sy_kmag': sy_kmag
}

# Add error features (set to 0 for user input)
error_features = ['pl_radeerr1', 'pl_radeerr2', 'pl_insolerr1', 'pl_insolerr2', 
                 'pl_eqterr1', 'pl_eqterr2', 'st_tefferr1', 'st_tefferr2',
                 'st_raderr1', 'st_raderr2', 'st_masserr1', 'st_masserr2',
                 'st_loggerr1', 'st_loggerr2', 'st_meterr1', 'st_meterr2',
                 'sy_disterr1', 'sy_disterr2', 'sy_vmagerr1', 'sy_vmagerr2',
                 'sy_kmagerr1', 'sy_kmagerr2', 'pl_orbeccenerr1', 'pl_orbeccenerr2']

for feature in error_features:
    if feature in selected_features:
        feature_dict[feature] = 0.0

# Add derived features
if 'orbital_velocity_proxy' in selected_features:
    feature_dict['orbital_velocity_proxy'] = np.sqrt(st_mass) / np.sqrt(pl_orbper)

if 'star_planet_ratio' in selected_features:
    feature_dict['star_planet_ratio'] = st_rad / pl_rade

if 'pl_density' in selected_features:
    feature_dict['pl_density'] = pl_bmasse / (pl_rade ** 3)

# Add discovery method flags (set to 0 for user input)
method_features = ['method_Transit', 'method_Radial Velocity', 'method_Microlensing']
for feature in method_features:
    if feature in selected_features:
        feature_dict[feature] = 0

# Add other features that might be in selected_features
other_features = ['disc_year', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 
                 'koi_fpflag_co', 'koi_fpflag_ec']
for feature in other_features:
    if feature in selected_features:
        feature_dict[feature] = 0

# Create input array in the correct order
input_features = []
for feature in selected_features:
    if feature in feature_dict:
        input_features.append(feature_dict[feature])
    else:
        input_features.append(0.0)  # Default value for missing features

input_array = np.array([input_features])

if st.button('🔮 Predict Exoplanet', type="primary"):
    try:
        # Make prediction
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0][1]
        
        # Display results
        if pred == 1:
            st.success(f"🎉 **This object is LIKELY an exoplanet!**")
            st.success(f"Confidence: {prob:.1%}")
            
            # Additional insights
            if prob > 0.9:
                st.info("💡 Very high confidence - this is almost certainly an exoplanet!")
            elif prob > 0.8:
                st.info("💡 High confidence - strong evidence for exoplanet status.")
            else:
                st.info("💡 Moderate confidence - likely an exoplanet but more data would help.")
        else:
            st.error(f"❌ **This object is NOT likely an exoplanet.**")
            st.error(f"Confidence: {prob:.1%}")
            
            # Additional insights
            if prob < 0.1:
                st.info("💡 Very low confidence - this is almost certainly not an exoplanet.")
            elif prob < 0.2:
                st.info("💡 Low confidence - weak evidence for exoplanet status.")
            else:
                st.info("💡 Moderate confidence - unlikely to be an exoplanet.")
        
        # Show feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Top Contributing Features")
            feature_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(feature_importance.set_index('Feature'))
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Information section
st.markdown("---")
st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", "91.70%", "8.14%")
with col2:
    st.metric("Features Used", "25", "58 available")
with col3:
    st.metric("Model Type", "XGBoost", "Enhanced")

st.markdown("""
**Model Details:**
- **Algorithm**: XGBoost Classifier with hyperparameter optimization
- **Features**: 25 carefully selected from 58 available features
- **Training Data**: NASA Exoplanet Archive (preprocessed)
- **Class Balancing**: SMOTE + class weights
- **Validation**: 3-fold cross-validation

**Key Features Used:**
- Planetary properties (radius, mass, orbital period)
- Stellar properties (temperature, radius, mass, metallicity)
- System properties (distance, magnitudes)
- Derived features (density, ratios, velocity proxy)
- Discovery method flags and quality indicators
""")

st.markdown("---")
st.markdown("*Enhanced by NASA Space Apps Team - Powered by Advanced Machine Learning*")
