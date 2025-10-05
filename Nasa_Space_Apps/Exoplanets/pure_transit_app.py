import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Custom CSS for better styling
st.set_page_config(
    page_title="Team Grizzlies - Pure Transit Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and feature information
@st.cache_data
def load_model():
    """Load the pure transit model and related files."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(script_dir, 'pure_transit_model.pkl')
        features_path = os.path.join(script_dir, 'pure_transit_features.pkl')
        imputer_path = os.path.join(script_dir, 'pure_transit_imputer.pkl')
        
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        imputer = joblib.load(imputer_path)
        
        return model, features, imputer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def create_pure_transit_features(pl_orbper, pl_rade, st_teff, st_rad, sy_dist):
    """Create pure transit features from input parameters (NO MASS REQUIRED)."""
    
    # Create a DataFrame with the input data
    df = pd.DataFrame({
        'pl_orbper': [pl_orbper],
        'pl_rade': [pl_rade],
        'st_teff': [st_teff],
        'st_rad': [st_rad],
        'sy_dist': [sy_dist]
    })
    
    # === CORE TRANSIT OBSERVABLES ===
    
    # 1. Rp/Rs - Planet to Star radius ratio (THE fundamental transit observable)
    df['rp_rs_ratio'] = df['pl_rade'] / (df['st_rad'] * 109.2)  # Convert Earth radii to Solar radii
    df['rp_rs_ratio_log'] = np.log1p(df['rp_rs_ratio'])

    # 2. Transit Depth - (Rp/Rs)² (the actual brightness decrease)
    df['transit_depth'] = df['rp_rs_ratio'] ** 2
    df['transit_depth_log'] = np.log1p(df['transit_depth'])

    # 3. Transit Duration (without mass - using period and stellar properties)
    df['transit_duration_proxy'] = df['st_rad'] * (df['pl_orbper'] ** (1/3))
    df['transit_duration_proxy_log'] = np.log1p(df['transit_duration_proxy'])

    # 4. Signal-to-Noise proxy (transit depth × stellar brightness proxy)
    df['snr_proxy'] = df['transit_depth'] * np.sqrt(df['st_teff'])
    df['snr_proxy_log'] = np.log1p(df['snr_proxy'])

    # 5. Stellar luminosity proxy (without mass)
    df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    df['stellar_luminosity_log'] = np.log1p(df['stellar_luminosity'])

    # 6. Transit probability proxy (without mass)
    df['transit_probability_proxy'] = df['st_rad'] / (df['pl_orbper'] ** (2/3))
    df['transit_probability_proxy_log'] = np.log1p(df['transit_probability_proxy'])

    # 7. Transit observability (how easy to detect)
    df['transit_observability'] = df['rp_rs_ratio'] * np.sqrt(df['st_teff'])
    df['transit_observability_log'] = np.log1p(df['transit_observability'])

    # 8. Impact parameter proxy (without mass)
    df['impact_parameter_proxy'] = np.sqrt(df['pl_orbper']) / (df['st_rad'] * 10)
    df['impact_parameter_proxy_log'] = np.log1p(df['impact_parameter_proxy'])

    # 9. Stellar properties ratios
    df['st_teff_normalized'] = df['st_teff'] / 5778  # Normalized to solar
    df['st_rad_normalized'] = df['st_rad']  # Already in solar radii
    df['st_teff_st_rad_ratio'] = df['st_teff'] / df['st_rad']

    # 10. Log transformations for skewed features
    df['pl_orbper_log'] = np.log1p(df['pl_orbper'])
    df['pl_rade_log'] = np.log1p(df['pl_rade'])
    df['st_teff_log'] = np.log1p(df['st_teff'])
    df['st_rad_log'] = np.log1p(df['st_rad'])
    df['sy_dist_log'] = np.log1p(df['sy_dist'])

    # 11. Physical plausibility checks (transit-only)
    df['size_sanity_check'] = np.where(df['rp_rs_ratio'] > 1.0, 0, 1)  # Planet can't be bigger than star
    df['period_sanity_check'] = np.where((df['pl_orbper'] < 0.1) | (df['pl_orbper'] > 10000), 0, 1)
    df['temperature_sanity_check'] = np.where((df['st_teff'] < 2000) | (df['st_teff'] > 10000), 0, 1)

    # 12. Distance-based observability
    df['distance_observability'] = df['transit_depth'] / (df['sy_dist'] ** 0.5)
    df['distance_observability_log'] = np.log1p(df['distance_observability'])

    # 13. Transit frequency (how often transits occur)
    df['transit_frequency'] = 1.0 / df['pl_orbper']  # Transits per day
    df['transit_frequency_log'] = np.log1p(df['transit_frequency'])

    return df

def get_preset_data():
    """Get preset exoplanet data for testing (NO MASS REQUIRED)."""
    return {
        'K2-18 b': {
            'pl_orbper': 32.94,
            'pl_rade': 2.61,
            'st_teff': 3457,
            'st_rad': 0.402,
            'sy_dist': 124,
            'status': 'CONFIRMED',
            'description': 'Super-Earth in habitable zone of M dwarf',
            'discovery': 'K2 mission (2015)'
        },
        'HD 209458 b': {
            'pl_orbper': 3.52,
            'pl_rade': 15.4,
            'st_teff': 6075,
            'st_rad': 1.2,
            'sy_dist': 159,
            'status': 'CONFIRMED',
            'description': 'First exoplanet discovered by transit method',
            'discovery': 'Ground-based transit (1999)'
        },
        'Kepler-452 b': {
            'pl_orbper': 384.8,
            'pl_rade': 5.2,
            'st_teff': 5757,
            'st_rad': 1.05,
            'sy_dist': 1402,
            'status': 'CONFIRMED',
            'description': 'Earth-like planet in habitable zone',
            'discovery': 'Kepler mission (2015)'
        }
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🌌 Team Grizzlies - Pure Transit Exoplanet Hunter</h1>', unsafe_allow_html=True)
    
    # Load model
    model, features, imputer = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check that the model files exist.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("🎯 Transit Parameters (NO MASS REQUIRED!)")
    
    # Preset selection
    preset_data = get_preset_data()
    selected_preset = st.sidebar.selectbox("Choose a preset:", ["Custom"] + list(preset_data.keys()))
    
    # Initialize session state for preset loading
    if 'preset_loaded' not in st.session_state:
        st.session_state.preset_loaded = False
    if 'selected_preset_name' not in st.session_state:
        st.session_state.selected_preset_name = None
    
    # Load preset button
    if st.sidebar.button("Load Preset", disabled=(selected_preset == "Custom")):
        st.session_state.preset_loaded = True
        st.session_state.selected_preset_name = selected_preset
    
    # Initialize session state for form values
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {
            'pl_orbper': 365.25,
            'pl_rade': 1.0,
            'st_teff': 5778,
            'st_rad': 1.0,
            'sy_dist': 100
        }
    
    # Load preset values if button was clicked
    if st.session_state.preset_loaded and st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom":
        preset_values = preset_data[st.session_state.selected_preset_name]
        st.session_state.form_values = {
            'pl_orbper': preset_values['pl_orbper'],
            'pl_rade': preset_values['pl_rade'],
            'st_teff': preset_values['st_teff'],
            'st_rad': preset_values['st_rad'],
            'sy_dist': preset_values['sy_dist']
        }
        st.session_state.preset_loaded = False
        st.rerun()  # Force rerun to update the form
    
    # Input form
    with st.sidebar.form("exoplanet_form"):
        st.subheader("🪐 Planetary Properties")
        pl_orbper = st.number_input('Orbital Period (days)', min_value=0.1, max_value=1000.0, value=st.session_state.form_values['pl_orbper'], step=1.0, format="%.2f")
        pl_rade = st.number_input('Planet Radius (Earth radii)', min_value=0.1, max_value=50.0, value=st.session_state.form_values['pl_rade'], step=0.1, format="%.2f")
        
        st.subheader("⭐ Stellar Properties")
        st_teff = st.number_input('Star Temperature (K)', min_value=2000, max_value=10000, value=st.session_state.form_values['st_teff'], step=50)
        st_rad = st.number_input('Star Radius (Solar radii)', min_value=0.1, max_value=10.0, value=st.session_state.form_values['st_rad'], step=0.1, format="%.2f")
        
        st.subheader("🌍 System Properties")
        sy_dist = st.number_input('System Distance (pc)', min_value=1, max_value=10000, value=st.session_state.form_values['sy_dist'], step=10)
        
        submitted = st.form_submit_button("🔍 Analyze Exoplanet")
    
    # Update session state with current form values
    st.session_state.form_values = {
        'pl_orbper': pl_orbper,
        'pl_rade': pl_rade,
        'st_teff': st_teff,
        'st_rad': st_rad,
        'sy_dist': sy_dist
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Pure Transit Analysis")
        
        if submitted or (st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom"):
            # Create features
            df_features = create_pure_transit_features(pl_orbper, pl_rade, st_teff, st_rad, sy_dist)
            
            # Prepare data for model (exclude original columns)
            base_features_to_exclude = ['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'sy_dist']
            X = df_features.drop(columns=base_features_to_exclude, errors='ignore')
            X = X[features]  # Ensure correct feature order
            
            # Impute and predict
            X_imputed = imputer.transform(X)
            prediction = model.predict(X_imputed)[0]
            probability = model.predict_proba(X_imputed)[0]
            
            # Display results
            if prediction == 1:
                st.markdown('<div class="prediction-card"><h2>🎉 EXOPLANET DETECTED!</h2><h3>This appears to be a real exoplanet</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-card"><h2>❌ Not an Exoplanet</h2><h3>This appears to be a false positive or stellar variability</h3></div>', unsafe_allow_html=True)
            
            # Confidence
            confidence = max(probability) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Transit features
            st.subheader("🔬 Calculated Transit Features")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="feature-card"><strong>Rp/Rs Ratio:</strong><br>{df_features["rp_rs_ratio"].iloc[0]:.4f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="feature-card"><strong>Transit Depth:</strong><br>{df_features["transit_depth"].iloc[0]:.6f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="feature-card"><strong>Transit Duration:</strong><br>{df_features["transit_duration_proxy"].iloc[0]:.3f}</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f'<div class="feature-card"><strong>Signal-to-Noise:</strong><br>{df_features["snr_proxy"].iloc[0]:.3f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="feature-card"><strong>Transit Probability:</strong><br>{df_features["transit_probability_proxy"].iloc[0]:.4f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="feature-card"><strong>Transit Observability:</strong><br>{df_features["transit_observability"].iloc[0]:.3f}</div>', unsafe_allow_html=True)
            
            # Sanity checks
            st.subheader("✅ Physical Sanity Checks")
            sanity_checks = []
            if df_features['size_sanity_check'].iloc[0] == 1:
                sanity_checks.append("✅ Planet radius reasonable")
            else:
                sanity_checks.append("⚠️ Planet may be too large")
            
            if df_features['period_sanity_check'].iloc[0] == 1:
                sanity_checks.append("✅ Orbital period reasonable")
            else:
                sanity_checks.append("⚠️ Orbital period unusual")
            
            if df_features['temperature_sanity_check'].iloc[0] == 1:
                sanity_checks.append("✅ Stellar temperature reasonable")
            else:
                sanity_checks.append("⚠️ Stellar temperature unusual")
            
            for check in sanity_checks:
                st.markdown(f'<div class="info-card">{check}</div>', unsafe_allow_html=True)
        
        else:
            st.info("👆 Enter parameters in the sidebar and click 'Analyze Exoplanet' to get started!")
    
    with col2:
        st.header("📊 Model Information")
        
        st.markdown('<div class="metric-card"><h3>🚀 Pure Transit Model</h3><p>92.97% Accuracy</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><h3>⚡ Ultra-Fast</h3><p>306,825 predictions/sec</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><h3>🔬 No Mass Required</h3><p>Only transit observables</p></div>', unsafe_allow_html=True)
        
        st.subheader("🎯 Model Advantages")
        st.markdown("""
        - ✅ **Highest Accuracy**: 92.97%
        - ✅ **No Mass Required**: Uses only observable parameters
        - ✅ **Ultra-Fast**: 306K+ predictions/second
        - ✅ **Scientifically Rigorous**: Pure transit physics
        - ✅ **Perfect for Surveys**: Kepler, K2, TESS compatible
        """)
        
        st.subheader("📈 Performance Metrics")
        st.metric("Accuracy", "92.97%")
        st.metric("Precision", "90.46%")
        st.metric("Recall", "98.27%")
        st.metric("F1-Score", "94.20%")
        st.metric("ROC-AUC", "96.91%")
        
        # Preset Information Display
        if st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom":
            preset_info = preset_data[st.session_state.selected_preset_name]
            st.subheader("📖 Preset Information")
            st.markdown(f'<div class="info-card"><strong>Status:</strong> {preset_info["status"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card"><strong>Description:</strong> {preset_info["description"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card"><strong>Discovery:</strong> {preset_info["discovery"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
