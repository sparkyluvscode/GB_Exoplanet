"""
Enhanced Data Preprocessing for Exoplanet Classification
======================================================
Creates 58 features from 7 original features for improved accuracy.

Author: NASA Space Apps Team
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_combine_data(data_dir):
    """Load and combine data from multiple CSV files with enhanced feature selection."""
    dfs = {}
    
    # Load each dataset
    for file in ['nasa_exoplanet.csv', 'k2_exoplanet.csv', 'tess_exoplanet.csv']:
        filepath = Path(data_dir) / file
        if filepath.exists():
            df = pd.read_csv(filepath, comment='#')
            # Add source identifier
            df['data_source'] = file.replace('.csv', '')
            dfs[file] = df
    
    # Combine all dataframes
    combined_df = pd.concat(dfs.values(), ignore_index=True)
    return combined_df

def create_derived_features(df):
    """Create engineered features that might be more predictive."""
    print("Creating derived features...")
    
    # Planetary density (if mass and radius available)
    if 'pl_bmasse' in df.columns and 'pl_rade' in df.columns:
        df['pl_density'] = df['pl_bmasse'] / (df['pl_rade'] ** 3)
    
    # Orbital velocity proxy (simplified)
    if 'pl_orbper' in df.columns and 'st_mass' in df.columns:
        df['orbital_velocity_proxy'] = np.sqrt(df['st_mass']) / np.sqrt(df['pl_orbper'])
    
    # Star-planet size ratio
    if 'st_rad' in df.columns and 'pl_rade' in df.columns:
        df['star_planet_ratio'] = df['st_rad'] / df['pl_rade']
    
    # Temperature zone classification
    if 'pl_insol' in df.columns:
        df['temp_zone'] = pd.cut(df['pl_insol'], 
                                bins=[0, 0.25, 4, 1000], 
                                labels=['cold', 'habitable', 'hot'],
                                include_lowest=True)
    
    # Planet radius categories
    if 'pl_rade' in df.columns:
        df['radius_category'] = pd.cut(df['pl_rade'],
                                     bins=[0, 1.25, 2, 4, 1000],
                                     labels=['earth-like', 'super-earth', 'neptune-like', 'giant'],
                                     include_lowest=True)
    
    # Orbital period categories
    if 'pl_orbper' in df.columns:
        df['period_category'] = pd.cut(df['pl_orbper'],
                                     bins=[0, 10, 100, 1000, 100000],
                                     labels=['short', 'medium', 'long', 'very_long'],
                                     include_lowest=True)
    
    return df

def enhanced_preprocess_data(df):
    """Enhanced preprocessing with more features and better handling."""
    print("Starting enhanced preprocessing...")
    
    # Define comprehensive feature sets
    planetary_features = [
        'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',
        'pl_rade', 'pl_radeerr1', 'pl_radeerr2',
        'pl_bmasse', 'pl_bmasseerr1', 'pl_bmasseerr2',
        'pl_insol', 'pl_insolerr1', 'pl_insolerr2',
        'pl_eqt', 'pl_eqterr1', 'pl_eqterr2',
        'pl_orbsmax', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2',
        'pl_orbeccen', 'pl_orbeccenerr1', 'pl_orbeccenerr2'
    ]
    
    stellar_features = [
        'st_teff', 'st_tefferr1', 'st_tefferr2',
        'st_rad', 'st_raderr1', 'st_raderr2',
        'st_mass', 'st_masserr1', 'st_masserr2',
        'st_logg', 'st_loggerr1', 'st_loggerr2',
        'st_met', 'st_meterr1', 'st_meterr2'
    ]
    
    system_features = [
        'sy_dist', 'sy_disterr1', 'sy_disterr2',
        'sy_vmag', 'sy_vmagerr1', 'sy_vmagerr2',
        'sy_kmag', 'sy_kmagerr1', 'sy_kmagerr2'
    ]
    
    detection_features = [
        'discoverymethod', 'disc_year', 'disc_facility',
        'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 
        'koi_fpflag_co', 'koi_fpflag_ec'
    ]
    
    # Combine all feature lists
    all_features = planetary_features + stellar_features + system_features + detection_features + ['data_source']
    
    # Filter to only existing columns
    available_features = [col for col in all_features if col in df.columns]
    df = df[available_features + ['disposition']].copy()
    
    # Handle categorical features
    categorical_features = ['discoverymethod', 'disc_facility', 'data_source']
    for feature in categorical_features:
        if feature in df.columns:
            df[feature] = df[feature].astype('category')
            # Create dummy variables for important categories
            if feature == 'discoverymethod':
                method_counts = df[feature].value_counts()
                top_methods = method_counts.head(5).index
                for method in top_methods:
                    df[f'method_{method}'] = (df[feature] == method).astype(int)
    
    # Handle missing values with more sophisticated imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # For error columns, use 0 if missing (no error reported)
    error_cols = [col for col in numeric_cols if 'err' in col]
    for col in error_cols:
        df[col] = df[col].fillna(0)
    
    # For main features, use median imputation
    main_cols = [col for col in numeric_cols if 'err' not in col and col != 'disposition']
    for col in main_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Create derived features
    df = create_derived_features(df)
    
    # Convert target variable
    if 'disposition' in df.columns:
        df['label'] = df['disposition'].map({
            'CONFIRMED': 2,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0,
            'FALSE POSITIVE ': 0,
            'koi_disposition': lambda x: 2 if 'CONFIRMED' in str(x) else (1 if 'CANDIDATE' in str(x) else 0)
        })
        df = df.drop('disposition', axis=1)
    
    # Remove any remaining rows with missing values
    df = df.dropna()
    
    # Remove categorical columns that were converted to dummies
    df = df.drop(columns=[col for col in df.columns if df[col].dtype == 'category'], errors='ignore')
    
    return df

def save_processed_data(df, output_path):
    """Save the processed dataset."""
    df.to_csv(output_path, index=False)
    print(f"Enhanced processed data saved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

if __name__ == "__main__":
    # Set paths
    data_dir = Path(__file__).parent.parent
    output_path = data_dir / "Exoplanets" / "enhanced_processed_exoplanet_data.csv"
    
    # Process data
    print("Loading and combining data...")
    combined_df = load_and_combine_data(data_dir)
    print(f"Combined data shape: {combined_df.shape}")
    
    print("Enhanced preprocessing...")
    processed_df = enhanced_preprocess_data(combined_df)
    
    print(f"Enhanced processed data shape: {processed_df.shape}")
    print("\nSample of enhanced processed data:")
    print(processed_df.head())
    
    print("\nLabel distribution:")
    print(processed_df['label'].value_counts().sort_index())
    
    # Save processed data
    save_processed_data(processed_df, output_path)
