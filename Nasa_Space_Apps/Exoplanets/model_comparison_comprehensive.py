import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_evaluate_model(model_path, features_path, imputer_path, model_name, X_test, y_test):
    """Load a model and evaluate its performance."""
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        imputer = joblib.load(imputer_path)
        
        # Prepare test data
        X_test_ordered = X_test[features]
        X_test_imputed = imputer.transform(X_test_ordered)
        
        # Make predictions
        y_pred = model.predict(X_test_imputed)
        y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'features': features,
            'feature_count': len(features)
        }
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def compare_models():
    """Compare all available models."""
    print("=== COMPREHENSIVE MODEL COMPARISON ===\n")
    
    # Load test data
    data = pd.read_csv('Nasa_Space_Apps/Exoplanets/processed_exoplanet_data.csv')
    data['label'] = data['label'].apply(lambda x: 1 if x == 2.0 else 0)
    
    # Create features for different models
    from transit_enhanced_model import create_transit_enhanced_features
    from simple_transit_model import create_simple_features
    
    # Create enhanced features
    df_enhanced = create_transit_enhanced_features(data.copy())
    df_simple = create_simple_features(data.copy())
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # For enhanced model
    base_features_to_exclude = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist']
    X_enhanced = df_enhanced.drop(columns=base_features_to_exclude + ['label'], errors='ignore')
    y = df_enhanced['label']
    
    # For simple model
    X_simple = df_simple.drop(columns=base_features_to_exclude + ['label'], errors='ignore')
    
    # Split enhanced data
    X_enhanced_train, X_enhanced_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split simple data
    X_simple_train, X_simple_test, _, _ = train_test_split(
        X_simple, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models to compare
    models_to_compare = [
        {
            'name': 'Enhanced Transit Model',
            'model_path': 'Nasa_Space_Apps/Exoplanets/transit_enhanced_model.pkl',
            'features_path': 'Nasa_Space_Apps/Exoplanets/transit_enhanced_features.pkl',
            'imputer_path': 'Nasa_Space_Apps/Exoplanets/transit_enhanced_imputer.pkl',
            'test_data': X_enhanced_test
        },
        {
            'name': 'Simple Transit Model',
            'model_path': 'Nasa_Space_Apps/Exoplanets/simple_transit_model.pkl',
            'features_path': 'Nasa_Space_Apps/Exoplanets/simple_transit_features.pkl',
            'imputer_path': 'Nasa_Space_Apps/Exoplanets/simple_transit_imputer.pkl',
            'test_data': X_simple_test
        }
    ]
    
    # Check if corrected physics model exists
    if os.path.exists('Nasa_Space_Apps/Exoplanets/corrected_physics_model.pkl'):
        models_to_compare.append({
            'name': 'Corrected Physics Model',
            'model_path': 'Nasa_Space_Apps/Exoplanets/corrected_physics_model.pkl',
            'features_path': 'Nasa_Space_Apps/Exoplanets/corrected_physics_features.pkl',
            'imputer_path': 'Nasa_Space_Apps/Exoplanets/corrected_physics_imputer.pkl',
            'test_data': X_enhanced_test  # Use enhanced features for now
        })
    
    # Evaluate each model
    results = []
    for model_info in models_to_compare:
        if all(os.path.exists(path) for path in [model_info['model_path'], model_info['features_path'], model_info['imputer_path']]):
            result = load_and_evaluate_model(
                model_info['model_path'],
                model_info['features_path'], 
                model_info['imputer_path'],
                model_info['name'],
                model_info['test_data'],
                y_test
            )
            if result:
                results.append(result)
        else:
            print(f"⚠️  {model_info['name']} files not found, skipping...")
    
    # Display comparison results
    if results:
        print("📊 MODEL COMPARISON RESULTS")
        print("=" * 50)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Accuracy': f"{r['accuracy']:.4f}",
                'Features': r['feature_count'],
                'Accuracy_Num': r['accuracy']
            }
            for r in results
        ])
        
        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy_Num', ascending=False)
        
        print(comparison_df[['Model', 'Accuracy', 'Features']].to_string(index=False))
        
        # Find best model
        best_model = max(results, key=lambda x: x['accuracy'])
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")
        print(f"   Features: {best_model['feature_count']}")
        
        # Feature importance comparison
        print(f"\n🔍 TOP FEATURES BY MODEL:")
        print("=" * 50)
        
        for result in results:
            print(f"\n{result['model_name']}:")
            try:
                # Find the model path from the original models list
                model_path = None
                for model_info in models_to_compare:
                    if model_info['name'] == result['model_name']:
                        model_path = model_info['model_path']
                        break
                
                if model_path and os.path.exists(model_path):
                    model = joblib.load(model_path)
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': result['features'],
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        print("Top 5 features:")
                        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
                            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            except Exception as e:
                print(f"  Error loading feature importance: {e}")
        
        # Detailed classification reports
        print(f"\n📋 DETAILED CLASSIFICATION REPORTS")
        print("=" * 50)
        
        for result in results:
            print(f"\n{result['model_name']}:")
            print(classification_report(y_test, result['predictions']))
        
        return results
    else:
        print("❌ No models could be loaded for comparison.")
        return None

def analyze_feature_overlap():
    """Analyze feature overlap between models."""
    print("\n🔗 FEATURE OVERLAP ANALYSIS")
    print("=" * 50)
    
    try:
        # Load feature sets
        enhanced_features = joblib.load('Nasa_Space_Apps/Exoplanets/transit_enhanced_features.pkl')
        simple_features = joblib.load('Nasa_Space_Apps/Exoplanets/simple_transit_features.pkl')
        
        enhanced_set = set(enhanced_features)
        simple_set = set(simple_features)
        
        common_features = enhanced_set.intersection(simple_set)
        enhanced_only = enhanced_set - simple_set
        simple_only = simple_set - enhanced_set
        
        print(f"Enhanced Transit Model Features: {len(enhanced_set)}")
        print(f"Simple Transit Model Features: {len(simple_set)}")
        print(f"Common Features: {len(common_features)}")
        print(f"Enhanced Only: {len(enhanced_only)}")
        print(f"Simple Only: {len(simple_only)}")
        
        print(f"\n🔄 Common Features ({len(common_features)}):")
        for feature in sorted(common_features):
            print(f"  - {feature}")
        
        print(f"\n➕ Enhanced Model Only ({len(enhanced_only)}):")
        for feature in sorted(enhanced_only):
            print(f"  - {feature}")
        
        print(f"\n➕ Simple Model Only ({len(simple_only)}):")
        for feature in sorted(simple_only):
            print(f"  - {feature}")
            
    except Exception as e:
        print(f"Error analyzing feature overlap: {e}")

if __name__ == "__main__":
    results = compare_models()
    analyze_feature_overlap()
    
    if results:
        print(f"\n✅ Comparison complete! Found {len(results)} models to compare.")
        print("\n💡 Key Insights:")
        print("- Enhanced Transit Model includes specific Rp/Rs, Transit Depth, and Transit Duration features")
        print("- Simple Transit Model focuses on basic transit observables")
        print("- Feature count and complexity varies between models")
        print("- Accuracy differences reflect the trade-off between feature richness and model complexity")
    else:
        print("\n❌ No models available for comparison.")
