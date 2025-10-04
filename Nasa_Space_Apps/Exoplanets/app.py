"""
Minimal Streamlit Web App for Exoplanet Prediction
Loads trained XGBoost model and provides a user-friendly interface for predictions.

How to use:
- Enter the object's parameters below.
- Click 'Predict' to see if it's likely an exoplanet.

Author: NASA Space Apps Team
"""
import streamlit as st
import numpy as np
import joblib

# Load the trained Random Forest model
MODEL_PATH = 'Nasa_Space_Apps/Exoplanets/rf_exoplanet_model.pkl'
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Exoplanet Predictor", page_icon="🪐", layout="centered")
st.title("🪐 Exoplanet Prediction App")
st.write("""
Enter the parameters below to predict whether the object is likely an exoplanet. All values should be positive and realistic for best results.
""")

# Input fields for all features (with explanations)
pl_orbper = st.number_input('Orbital Period (days)', min_value=0.0, value=10.0, help="Time for one orbit around the star.")
pl_rade = st.number_input('Planetary Radius (Earth radii)', min_value=0.0, value=1.0, help="Radius compared to Earth.")
pl_bmasse = st.number_input('Planetary Mass (Earth masses)', min_value=0.0, value=10.0, help="Mass compared to Earth.")
st_teff = st.number_input('Star Effective Temperature (K)', min_value=0.0, value=5500.0, help="Temperature of the host star.")
st_rad = st.number_input('Star Radius (Solar radii)', min_value=0.0, value=1.0, help="Radius of the host star.")
st_mass = st.number_input('Star Mass (Solar masses)', min_value=0.0, value=1.0, help="Mass of the host star.")
sy_dist = st.number_input('System Distance (parsecs)', min_value=0.0, value=100.0, help="Distance from Earth.")

if st.button('Predict'):
    # Prepare input in the order expected by the model
    input_features = np.array([[pl_orbper, pl_rade, pl_bmasse, st_teff, st_rad, st_mass, sy_dist]])
    pred = model.predict(input_features)[0]
    prob = model.predict_proba(input_features)[0][1]
    if pred == 1:
        st.success(f"This object is likely an exoplanet! (Confidence: {prob:.2%})")
    else:
        st.error(f"This object is NOT likely an exoplanet. (Confidence: {prob:.2%})")

st.markdown("---")
st.markdown("""
*Model: Random Forest | Data: NASA Exoplanet Archive (preprocessed)*
""")
