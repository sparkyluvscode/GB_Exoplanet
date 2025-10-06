# ExoScope AI - Exoplanet Detection System

## Prerequisites
- Python 3.8+ 
- pip package manager
- Modern web browser

## 1. Import/Install Everything

```bash
# Clone the repository
git clone https://github.com/sparkyluvscode/GB_Exoplanet.git
cd GB_Exoplanet

# Navigate to the Exoplanets directory
cd Nasa_Space_Apps/Exoplanets

# Install all required packages
pip install -r requirements.txt
```

## 2. Train the Model

```bash
# Train the corrected physics model
python corrected_transit_model.py
```

This will:
- Load exoplanet data from NASA archives
- Create scientifically accurate features
- Train the XGBoost model
- Save trained model files (.pkl)

## 3. Run the Streamlit Application

```bash
# Start the web application
streamlit run corrected_transit_app.py
```

The app will open at: exoscope-ai.streamlit.app

## Usage
1. Enter exoplanet parameters in the sidebar
2. Click "Analyze Exoplanet" 
3. View AI prediction and 3D animation
4. Use animation controls (play/pause/reset)

## Files Required
- `corrected_transit_app.py` - Main Streamlit app
- `corrected_transit_model.py` - Model training script  
- `simple_animation.html` - 3D visualization
- `sun.webp`, `terrestrial.jpg` - Textures
- `requirements.txt` - Dependencies
