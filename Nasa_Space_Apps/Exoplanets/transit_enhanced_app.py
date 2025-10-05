import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Custom CSS for better styling
st.set_page_config(
    page_title="Team Grizzlies - Transit Enhanced Exoplanet Hunter",
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
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .feature-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    
    .transit-feature {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and feature information
@st.cache_data
def load_model():
    """Load the enhanced transit model and related files."""
    try:
        model_path = 'Nasa_Space_Apps/Exoplanets/transit_enhanced_model.pkl'
        features_path = 'Nasa_Space_Apps/Exoplanets/transit_enhanced_features.pkl'
        imputer_path = 'Nasa_Space_Apps/Exoplanets/transit_enhanced_imputer.pkl'
        
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        imputer = joblib.load(imputer_path)
        
        return model, features, imputer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

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

def get_preset_data():
    """Get preset exoplanet data for testing."""
    return {
        'K2-18 b': {
            'pl_orbper': 32.94,
            'pl_rade': 2.61,
            'pl_bmasse': 8.63,
            'st_teff': 3457,
            'st_rad': 0.402,
            'st_mass': 0.36,
            'sy_dist': 124,
            'status': 'CONFIRMED',
            'description': 'Super-Earth in habitable zone of M dwarf',
            'discovery': 'K2 mission (2015)'
        },
        'HD 209458 b': {
            'pl_orbper': 3.52,
            'pl_rade': 15.4,
            'pl_bmasse': 220.0,
            'st_teff': 6075,
            'st_rad': 1.2,
            'st_mass': 1.15,
            'sy_dist': 159,
            'status': 'CONFIRMED',
            'description': 'First exoplanet discovered by transit method',
            'discovery': 'Ground-based transit (1999)'
        },
        'Kepler-452 b': {
            'pl_orbper': 384.8,
            'pl_rade': 5.2,
            'pl_bmasse': 5.0,
            'st_teff': 5757,
            'st_rad': 1.05,
            'st_mass': 1.04,
            'sy_dist': 1402,
            'status': 'CONFIRMED',
            'description': 'Earth-like planet in habitable zone',
            'discovery': 'Kepler mission (2015)'
        }
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🌌 Team Grizzlies - Transit Enhanced Exoplanet Hunter</h1>', unsafe_allow_html=True)
    
    # Load model
    model, features, imputer = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check that the model files exist.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("🎯 Exoplanet Parameters")
    
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
            'st_teff': 5778,
            'st_rad': 1.0,
            'st_mass': 1.0,
            'pl_orbper': 365.25,
            'pl_rade': 1.0,
            'pl_bmasse': 1.0,
            'sy_dist': 100
        }
    
    # Load preset values if button was clicked
    if st.session_state.preset_loaded and st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom":
        preset_values = preset_data[st.session_state.selected_preset_name]
        st.session_state.form_values = {
            'st_teff': preset_values['st_teff'],
            'st_rad': preset_values['st_rad'],
            'st_mass': preset_values['st_mass'],
            'pl_orbper': preset_values['pl_orbper'],
            'pl_rade': preset_values['pl_rade'],
            'pl_bmasse': preset_values['pl_bmasse'],
            'sy_dist': preset_values['sy_dist']
        }
        st.session_state.preset_loaded = False
        st.rerun()  # Force rerun to update the form
    
    # Input form
    with st.sidebar.form("exoplanet_form"):
        st.subheader("📊 Stellar Properties")
        st_teff = st.number_input('Star Temperature (K)', min_value=2000, max_value=10000, value=st.session_state.form_values['st_teff'], step=50)
        st_rad = st.number_input('Star Radius (Solar radii)', min_value=0.1, max_value=10.0, value=st.session_state.form_values['st_rad'], step=0.1, format="%.2f")
        st_mass = st.number_input('Star Mass (Solar masses)', min_value=0.1, max_value=5.0, value=st.session_state.form_values['st_mass'], step=0.1, format="%.2f")
        
        st.subheader("🪐 Planetary Properties")
        pl_orbper = st.number_input('Orbital Period (days)', min_value=0.1, max_value=1000.0, value=st.session_state.form_values['pl_orbper'], step=1.0, format="%.2f")
        pl_rade = st.number_input('Planet Radius (Earth radii)', min_value=0.1, max_value=50.0, value=st.session_state.form_values['pl_rade'], step=0.1, format="%.2f")
        pl_bmasse = st.number_input('Planet Mass (Earth masses)', min_value=0.1, max_value=1000.0, value=st.session_state.form_values['pl_bmasse'], step=1.0, format="%.1f")
        
        st.subheader("🌍 System Properties")
        sy_dist = st.number_input('System Distance (pc)', min_value=1, max_value=10000, value=st.session_state.form_values['sy_dist'], step=10)
        
        submitted = st.form_submit_button("🔍 Analyze Exoplanet")
    
    # Update session state with current form values
    st.session_state.form_values = {
        'st_teff': st_teff,
        'st_rad': st_rad,
        'st_mass': st_mass,
        'pl_orbper': pl_orbper,
        'pl_rade': pl_rade,
        'pl_bmasse': pl_bmasse,
        'sy_dist': sy_dist
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Transit Analysis")
        
        if submitted or (st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom"):
            # Create features
            df_features = create_transit_features(pl_orbper, pl_rade, pl_bmasse, st_teff, st_rad, st_mass, sy_dist)
            
            # Prepare data for model (exclude original columns)
            base_features_to_exclude = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist']
            X = df_features.drop(columns=base_features_to_exclude, errors='ignore')
            
            # Ensure feature order matches training
            X_ordered = X[features]
            
            # Impute missing values
            X_imputed = imputer.transform(X_ordered)
            
            # Make prediction
            prediction = model.predict(X_imputed)[0]
            probability = model.predict_proba(X_imputed)[0]
            
            # Display results
            if prediction == 1:
                st.markdown('<div class="prediction-card"><h2>✅ PLANET DETECTED!</h2><p>This object is likely a real exoplanet.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-card"><h2>❌ FALSE POSITIVE</h2><p>This object is likely not a real exoplanet.</p></div>', unsafe_allow_html=True)
            
            # Probability display
            prob_percent = probability[1] * 100
            st.metric("Confidence", f"{prob_percent:.1f}%", f"{prob_percent-50:.1f}%")
            
            # Show key transit features
            st.subheader("🔬 Key Transit Features")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Rp/Rs Ratio", f"{df_features['rp_rs_ratio'].iloc[0]:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Transit Depth", f"{df_features['transit_depth'].iloc[0]:.6f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Transit Duration Proxy", f"{df_features['transit_duration_proxy'].iloc[0]:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional metrics
            col_d, col_e, col_f = st.columns(3)
            
            with col_d:
                st.metric("SNR Proxy", f"{df_features['snr_proxy'].iloc[0]:.2f}")
            
            with col_e:
                st.metric("Transit Probability", f"{df_features['transit_probability'].iloc[0]:.4f}")
            
            with col_f:
                st.metric("Planetary Density", f"{df_features['pl_density'].iloc[0]:.2f} g/cm³")
    
    with col2:
        st.header("📚 Model Information")
        
        st.markdown("""
        <div class="feature-info">
        <h4>🌟 Enhanced Transit Features:</h4>
        <span class="transit-feature">Rp/Rs Ratio</span>
        <span class="transit-feature">Transit Depth</span>
        <span class="transit-feature">Transit Duration</span>
        <span class="transit-feature">SNR Proxy</span>
        <span class="transit-feature">Transit Probability</span>
        <span class="transit-feature">Impact Parameter</span>
        <span class="transit-feature">Observability</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Model Performance:**
        - Test Accuracy: 78.04%
        - Cross-validation: 78.30% ± 0.75%
        - Features: 35 enhanced transit features
        """)
        
        st.success("""
        **Key Features:**
        - Rp/Rs ratio: Planet-to-star size ratio
        - Transit depth: Fractional flux decrease
        - Transit duration: Duration proxy from orbital mechanics
        - SNR proxy: Signal-to-noise estimation
        """)
        
        if st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom":
            preset_info = preset_data[st.session_state.selected_preset_name]
            st.subheader("📖 Preset Information")
            st.write(f"**Status:** {preset_info['status']}")
            st.write(f"**Description:** {preset_info['description']}")
            st.write(f"**Discovery:** {preset_info['discovery']}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Team Grizzlies** - NASA Space Apps Challenge 2024 | Enhanced Transit Detection Model")

if __name__ == "__main__":
    main()
