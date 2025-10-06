import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# Custom CSS for better styling
st.set_page_config(
    page_title="ExoScope AI - Exoplanet Detector",
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
    
    .physics-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .animation-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .spacer {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and feature information
@st.cache_data
def load_corrected_model():
    """Load the scientifically corrected transit model and related files."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(script_dir, 'corrected_transit_model.pkl')
        features_path = os.path.join(script_dir, 'corrected_transit_features.pkl')
        imputer_path = os.path.join(script_dir, 'corrected_transit_imputer.pkl')
        
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        imputer = joblib.load(imputer_path)
        
        return model, features, imputer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def create_corrected_transit_features(pl_orbper, pl_rade, st_teff, st_rad, sy_dist):
    """Create scientifically accurate transit features from input parameters."""
    
    # Create a DataFrame with the input data
    df = pd.DataFrame({
        'pl_orbper': [pl_orbper],
        'pl_rade': [pl_rade],
        'st_teff': [st_teff],
        'st_rad': [st_rad],
        'sy_dist': [sy_dist]
    })
    
    # === CORE TRANSIT OBSERVABLES (PHYSICALLY ACCURATE) ===
    
    # 1. Rp/Rs - Planet to Star radius ratio (THE fundamental transit observable)
    # Convert Earth radii to Solar radii: 1 R_earth = 0.009167 R_sun
    df['rp_rs_ratio'] = df['pl_rade'] * 0.009167 / df['st_rad']
    df['rp_rs_ratio_log'] = np.log1p(df['rp_rs_ratio'])

    # 2. Transit Depth - (Rp/Rs)² (the actual brightness decrease)
    df['transit_depth'] = df['rp_rs_ratio'] ** 2
    df['transit_depth_log'] = np.log1p(df['transit_depth'])

    # 3. Transit Duration (physically correct)
    # For circular orbits: T_dur ∝ (R* * P^(1/3)) / π
    df['transit_duration'] = (df['st_rad'] * (df['pl_orbper'] ** (1/3))) / np.pi
    df['transit_duration_log'] = np.log1p(df['transit_duration'])

    # 4. Transit Probability (geometric probability)
    # P_transit ∝ R* / P^(2/3)
    df['transit_probability'] = df['st_rad'] / (df['pl_orbper'] ** (2/3))
    df['transit_probability_log'] = np.log1p(df['transit_probability'])

    # 5. Signal-to-Noise Ratio proxy
    # SNR ∝ (transit_depth) * sqrt(stellar_flux) / sqrt(noise)
    stellar_flux_proxy = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    noise_proxy = np.sqrt(df['pl_orbper'])
    df['snr_proxy'] = df['transit_depth'] * np.sqrt(stellar_flux_proxy) / noise_proxy
    df['snr_proxy_log'] = np.log1p(df['snr_proxy'])

    # 6. Stellar Luminosity (Stefan-Boltzmann law)
    # L ∝ R*² * T_eff⁴
    df['stellar_luminosity'] = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
    df['stellar_luminosity_log'] = np.log1p(df['stellar_luminosity'])

    # 7. Transit Observability (distance-dependent detectability)
    df['transit_observability'] = df['transit_depth'] / df['sy_dist']
    df['transit_observability_log'] = np.log1p(df['transit_observability'])

    # 8. Transit Frequency (how often transits occur)
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
    df['distance_snr_degradation'] = df['snr_proxy'] / np.sqrt(df['sy_dist'])
    df['distance_snr_degradation_log'] = np.log1p(df['distance_snr_degradation'])

    # 13. Transit Impact Parameter Proxy
    df['impact_parameter_proxy'] = np.sqrt(df['pl_orbper']) / (df['st_rad'] * 10)
    df['impact_parameter_proxy_log'] = np.log1p(df['impact_parameter_proxy'])

    return df

def create_light_curve(pl_orbper, pl_rade, st_teff, st_rad, sy_dist, df_features):
    """Create a simulated transit light curve."""
    
    # Get transit parameters
    rp_rs_ratio = df_features['rp_rs_ratio'].iloc[0]
    transit_depth = df_features['transit_depth'].iloc[0]
    transit_duration = df_features['transit_duration'].iloc[0]  # in hours
    
    # Create time array (3 orbital periods)
    t_total = pl_orbper * 3  # 3 orbital periods
    t = np.linspace(0, t_total, 1000)  # 1000 time points
    
    # Normalize time to 0-1 for one orbital period
    t_norm = t % pl_orbper / pl_orbper
    
    # Calculate transit center (assume mid-transit at t=0.5)
    transit_center = 0.5
    transit_width = (transit_duration / 24) / pl_orbper  # Convert hours to fraction of orbital period
    
    # Create light curve
    flux = np.ones_like(t_norm)  # Normal flux = 1
    
    # Add transit (decrease in brightness)
    for i in range(len(t_norm)):
        # Check if we're in transit
        dist_from_center = min(abs(t_norm[i] - transit_center), abs(t_norm[i] - transit_center + 1), abs(t_norm[i] - transit_center - 1))
        
        if dist_from_center <= transit_width / 2:
            # In transit - apply transit depth
            flux[i] = 1 - transit_depth
        
        # Add some realistic noise
        flux[i] += np.random.normal(0, 0.001)
    
    return t, flux, transit_depth, transit_duration

def plot_light_curve(t, flux, transit_depth, transit_duration, pl_orbper, pl_rade):
    """Create an interactive light curve plot."""
    
    # Create the plot
    fig = go.Figure()
    
    # Add the light curve
    fig.add_trace(go.Scatter(
        x=t,
        y=flux,
        mode='lines',
        name='Light Curve',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='Time: %{x:.2f} days<br>Flux: %{y:.6f}<extra></extra>'
    ))
    
    # Add transit region highlighting
    transit_center = pl_orbper / 2
    transit_start = transit_center - (transit_duration / 24) / 2
    transit_end = transit_center + (transit_duration / 24) / 2
    
    fig.add_vrect(
        x0=transit_start,
        x1=transit_end,
        fillcolor="red",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="Transit",
        annotation_position="top left"
    )
    
    # Add horizontal line at normal flux
    fig.add_hline(y=1, line_dash="dash", line_color="gray", 
                   annotation_text="Normal Flux", annotation_position="top right")
    
    # Update layout
    fig.update_layout(
        title=f'Transit Light Curve - Planet Radius: {pl_rade:.2f} R⊕',
        xaxis_title='Time (days)',
        yaxis_title='Relative Flux',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Add annotations
    fig.add_annotation(
        x=transit_center,
        y=1-transit_depth,
        text=f'Transit Depth: {transit_depth:.6f}\nDuration: {transit_duration:.2f} hours',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ff6b6b",
        ax=0,
        ay=-40
    )
    
    return fig

def get_preset_data():
    """Get preset exoplanet data for testing."""
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
        },
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">ExoScope AI - State of the Art Exoplanet Detector</h1>', unsafe_allow_html=True)
    
    # Load model
    model, features, imputer = load_corrected_model()
    
    if model is None:
        st.error("Failed to load model. Please check that the model files exist.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("🎯 Transit Parameters (Scientifically Accurate)")
    
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
    
    # Initialize prediction result for animation tab
    prediction_result = None
    confidence_result = None
    df_features_result = None
    
    with col1:
        st.header("Exoplanet or not?")
        
        if submitted or (st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom"):
            # Create features
            df_features = create_corrected_transit_features(pl_orbper, pl_rade, st_teff, st_rad, sy_dist)
            
            # Prepare data for model (exclude original columns)
            base_features_to_exclude = ['pl_orbper', 'pl_rade', 'st_teff', 'st_rad', 'sy_dist']
            X = df_features.drop(columns=base_features_to_exclude, errors='ignore')
            X = X[features]  # Ensure correct feature order
            
            # Impute and predict
            X_imputed = imputer.transform(X)
            prediction = model.predict(X_imputed)[0]
            probability = model.predict_proba(X_imputed)[0]
            
            # Store results for animation tab
            prediction_result = prediction
            confidence_result = max(probability) * 100
            df_features_result = df_features
            
            # Display results
            if prediction == 1:
                st.markdown('<div class="prediction-card"><h2>🎉 EXOPLANET DETECTED!</h2><h3>This appears to be a real exoplanet</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-card"><h2>❌ Not an Exoplanet</h2><h3>This appears to be a false positive or stellar variability</h3></div>', unsafe_allow_html=True)
            
            # Confidence
            confidence = max(probability) * 100
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Add animation and light curve tabs if exoplanet detected
            if prediction == 1:
                tab1, tab2 = st.tabs(["🎬 View Animation", "📈 Light Curve"])
                
                with tab1:
                    st.markdown('<div class="animation-card"><h2>🎬 Exoplanet Transit Animation</h2><h3>Watch your exoplanet transit its host star!</h3></div>', unsafe_allow_html=True)
                    
                    # Load and display the Three.js animation
                    try:
                        # Read the animation template
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        animation_path = os.path.join(script_dir, 'simple_animation.html')
                        
                        with open(animation_path, 'r', encoding='utf-8') as f:
                            animation_html = f.read()
                        
                        # Calculate transit duration for animation
                        transit_duration_hours = df_features['transit_duration'].iloc[0]
                        
                        # Create JavaScript to update animation parameters
                        js_params = f"""
                        <script>
                        // Update animation parameters when the page loads
                        window.addEventListener('load', function() {{
                            if (window.updateAnimationParameters) {{
                                window.updateAnimationParameters({{
                                    orbitalPeriod: {pl_orbper},
                                    planetRadius: {pl_rade},
                                    starRadius: {st_rad},
                                    starTemperature: {st_teff},
                                    transitDuration: {transit_duration_hours}
                                }});
                            }}
                        }});
                        </script>
                        """
                        
                        # Combine the animation HTML with parameter updates
                        full_animation_html = animation_html + js_params
                        
                        # Display the animation in an iframe
                        components.html(full_animation_html, height=600, scrolling=False)
                        
                        # Display animation controls
                        st.markdown("### 🎮 Animation Controls")
                        st.markdown("""
                        - **Mouse**: Drag to rotate camera view
                        - **Scroll**: Zoom in/out
                        - **Controls**: Use the buttons in the animation to play, pause, or reset
                        """)
                        
                        # Display transit parameters for animation
                        st.markdown("### 🌟 Animation Parameters")
                        anim_col1, anim_col2 = st.columns(2)
                        with anim_col1:
                            st.metric("Transit Duration", f"{df_features['transit_duration'].iloc[0]:.2f} hours")
                            st.metric("Transit Depth", f"{df_features['transit_depth'].iloc[0]:.6f}")
                        with anim_col2:
                            st.metric("Orbital Period", f"{pl_orbper:.2f} days")
                            st.metric("Planet Radius", f"{pl_rade:.2f} R⊕")
                        
                        # Additional animation information
                        st.markdown("### 📊 Animation Details")
                        info_col1, info_col2 = st.columns(2)
                        with info_col1:
                            st.metric("Star Temperature", f"{st_teff:.0f} K")
                            st.metric("Star Radius", f"{st_rad:.2f} R☉")
                        with info_col2:
                            st.metric("Planet-to-Star Ratio", f"{df_features['rp_rs_ratio'].iloc[0]:.6f}")
                            st.metric("System Distance", f"{sy_dist:.0f} pc")
                        
                    except Exception as e:
                        st.error(f"Error loading animation: {e}")
                        st.info("Animation will be available once the template file is properly set up.")
                
                with tab2:
                    st.markdown('<div class="animation-card"><h2>📈 Transit Light Curve</h2><h3>Observe the brightness variation during transit</h3></div>', unsafe_allow_html=True)
                    
                    # Generate and display light curve
                    try:
                        t, flux, transit_depth_curve, transit_duration_curve = create_light_curve(pl_orbper, pl_rade, st_teff, st_rad, sy_dist, df_features)
                        fig = plot_light_curve(t, flux, transit_depth_curve, transit_duration_curve, pl_orbper, pl_rade)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Light curve information
                        st.markdown("### 📊 Light Curve Analysis")
                        lc_col1, lc_col2 = st.columns(2)
                        with lc_col1:
                            st.metric("Maximum Transit Depth", f"{transit_depth_curve:.6f}")
                            st.metric("Transit Duration", f"{transit_duration_curve:.2f} hours")
                        with lc_col2:
                            st.metric("Orbital Period", f"{pl_orbper:.2f} days")
                            st.metric("Planet-to-Star Ratio", f"{df_features['rp_rs_ratio'].iloc[0]:.6f}")
                    except Exception as e:
                        st.error(f"Error generating light curve: {e}")
            
            # Transit features and sanity checks below animation section
            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
            
            # Create tabs for detailed analysis
            tab3, tab4 = st.tabs(["🔬 Transit Features", "✅ Sanity Checks"])
            
            with tab3:
                st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'<div class="feature-card"><strong>Rp/Rs Ratio:</strong><br>{df_features["rp_rs_ratio"].iloc[0]:.6f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="feature-card"><strong>Transit Depth:</strong><br>{df_features["transit_depth"].iloc[0]:.8f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="feature-card"><strong>Transit Duration:</strong><br>{df_features["transit_duration"].iloc[0]:.3f} hours</div>', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f'<div class="feature-card"><strong>Transit Probability:</strong><br>{df_features["transit_probability"].iloc[0]:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="feature-card"><strong>Signal-to-Noise:</strong><br>{df_features["snr_proxy"].iloc[0]:.3f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="feature-card"><strong>Stellar Luminosity:</strong><br>{df_features["stellar_luminosity"].iloc[0]:.3f} L☉</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
                sanity_checks = []
                if df_features['size_sanity_check'].iloc[0] == 1:
                    sanity_checks.append("✅ Planet radius reasonable (Rp/Rs < 1)")
                else:
                    sanity_checks.append("⚠️ Planet may be too large relative to star")
                
                if df_features['period_sanity_check'].iloc[0] == 1:
                    sanity_checks.append("✅ Orbital period reasonable")
                else:
                    sanity_checks.append("⚠️ Orbital period unusual")
                
                if df_features['temperature_sanity_check'].iloc[0] == 1:
                    sanity_checks.append("✅ Stellar temperature reasonable")
                else:
                    sanity_checks.append("⚠️ Stellar temperature unusual")
                    
                if df_features['transit_depth_sanity'].iloc[0] == 1:
                    sanity_checks.append("✅ Transit depth reasonable")
                else:
                    sanity_checks.append("⚠️ Transit depth unusual")
                
                for check in sanity_checks:
                    st.markdown(f'<div class="info-card">{check}</div>', unsafe_allow_html=True)
        
        else:
            st.info("👆 Enter parameters in the sidebar and click 'Analyze Exoplanet' to get started!")
    
    with col2:
        st.header("📊 Model Information")
        
        st.markdown('<div class="metric-card"><h3>🎯Crazy Accurate!</h3><p>An accuracy of 96.74% while being trained on tens of thousands of exoplanets</p></div>', unsafe_allow_html=True)
        # st.markdown('<div class="metric-card"><h3>⚡ Ultra-Fast</h3><p>306,825 predictions/sec</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><h3>🔬 Transit Method</h3><p>Kepler, K2, TESS compatible</p></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
        
        # st.subheader("Scientific Advantages")
        # st.markdown("""
        # - ✅ **High Accuracy**: 96.74%
        # - ✅ **Physically Accurate**: Correct Earth/Solar radii conversion
        # - ✅ **No Mass Required**: Uses only observable parameters
        # - ✅ **Proper Transit Physics**: Accurate duration & probability
        # - ✅ **Realistic SNR**: Distance-dependent calculations
        # - ✅ **Perfect for Surveys**: Kepler, K2, TESS compatible
        # """)
        
        st.subheader("📈 Performance Metrics")
        st.metric("Test Accuracy", "96.74%")
        st.metric("Cross-Validation", "96.28%")
        st.metric("Precision", "97.0%")
        st.metric("Recall", "99.0%")
        st.metric("F1-Score", "98.0%")
        
        
        # Preset Information Display
        if st.session_state.selected_preset_name and st.session_state.selected_preset_name != "Custom":
            preset_info = preset_data[st.session_state.selected_preset_name]
            st.subheader("📖 Preset Information")
            st.markdown(f'<div class="info-card"><strong>Status:</strong> {preset_info["status"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card"><strong>Description:</strong> {preset_info["description"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-card"><strong>Discovery:</strong> {preset_info["discovery"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
