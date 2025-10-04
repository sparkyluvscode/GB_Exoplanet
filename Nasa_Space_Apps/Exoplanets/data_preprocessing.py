import pandas as pd
import numpy as np
from pathlib import Path

def load_and_combine_data(data_dir):
    """Load and combine data from multiple CSV files."""
    # Dictionary to store dataframes
    dfs = {}
    
    # Load each dataset
    for file in ['nasa_exoplanet.csv', 'k2_exoplanet.csv', 'tess_exoplanet.csv']:
        filepath = Path(data_dir) / file
        if filepath.exists():
            # Skip comment lines and load data
            df = pd.read_csv(filepath, comment='#')
            dfs[file] = df
    
    # Combine all dataframes
    combined_df = pd.concat(dfs.values(), ignore_index=True)
    return combined_df

def preprocess_data(df):
    """Preprocess the combined dataset."""
    # Select relevant columns (example - adjust based on actual data)
    relevant_columns = [
        'pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass',
        'sy_dist', 'disposition'  # disposition is our target variable
    ]
    
    # Filter only the columns that exist in the dataframe
    available_columns = [col for col in relevant_columns if col in df.columns]
    df = df[available_columns].copy()
    
    # Convert target variable to numerical values
    if 'disposition' in df.columns:
        df['label'] = df['disposition'].map({
            'CONFIRMED': 2,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0,
            'FALSE POSITIVE ': 0  # Handle potential whitespace
        })
        df = df.drop('disposition', axis=1)
    
    # Handle missing values
    # For simplicity, we'll fill with median for numerical columns
    for col in df.select_dtypes(include=np.number).columns:
        if col != 'label':  # Don't fill label column
            df[col] = df[col].fillna(df[col].median())
    
    # Drop any remaining rows with missing values
    df = df.dropna()
    
    return df

def save_processed_data(df, output_path):
    """Save the processed dataset."""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Set paths
    data_dir = Path(__file__).parent.parent  # Go up one level from Exoplanets directory
    output_path = data_dir / "Exoplanets" / "processed_exoplanet_data.csv"
    
    # Process data
    print("Loading and combining data...")
    combined_df = load_and_combine_data(data_dir)
    
    print("Preprocessing data...")
    processed_df = preprocess_data(combined_df)
    
    print(f"Processed data shape: {processed_df.shape}")
    print("\nSample of processed data:")
    print(processed_df.head())
    
    # Save processed data
    save_processed_data(processed_df, output_path)
