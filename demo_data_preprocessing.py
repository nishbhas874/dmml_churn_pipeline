#!/usr/bin/env python3
"""
Real-time demonstration of the Data Preprocessing System.
Shows live output of data cleaning, preprocessing, and EDA.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing.data_preprocessor import create_data_preprocessor
from loguru import logger

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("📊 Creating sample dataset for preprocessing demonstration...")
    
    # Create base data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic churn data
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'tenure': np.random.exponential(30, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues for demonstration
    print("  🔍 Adding realistic data quality issues...")
    
    # 1. Missing values
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'monthly_charges'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, 'contract_type'] = np.nan
    
    # 2. Some outliers
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'monthly_charges'] = np.random.uniform(150, 300, 20)
    
    # 3. Some inconsistent data types
    type_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[type_indices, 'tenure'] = df.loc[type_indices, 'tenure'].astype(str) + ' months'
    
    print(f"  ✅ Created dataset with {len(df)} rows and {len(df.columns)} columns")
    return df

def demonstrate_preprocessing():
    """Demonstrate the data preprocessing system."""
    print("\n" + "="*80)
    print("🔧 DATA PREPROCESSING SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize preprocessor
    print("\n🔧 Initializing Data Preprocessor...")
    try:
        preprocessor = create_data_preprocessor()
        print("  ✅ Data preprocessor initialized successfully")
    except Exception as e:
        print(f"  ❌ Failed to initialize preprocessor: {e}")
        return
    
    # Create sample dataset
    print("\n📊 Creating Sample Dataset...")
    sample_data = create_sample_dataset()
    
    print(f"\n📋 Dataset Overview:")
    print(f"  • Shape: {sample_data.shape}")
    print(f"  • Memory usage: {sample_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"  • Data types: {sample_data.dtypes.value_counts().to_dict()}")
    
    # Perform EDA
    print("\n🔍 PERFORMING EXPLORATORY DATA ANALYSIS")
    print("-" * 50)
    
    try:
        eda_results = preprocessor.explore_data(sample_data, save_plots=True)
        
        print(f"\n✅ EDA completed!")
        print(f"  • Dataset shape: {eda_results['dataset_info']['shape']}")
        print(f"  • Missing data columns: {len(eda_results['missing_data']['columns_with_missing'])}")
        print(f"  • Numerical columns: {len(eda_results['numerical_analysis'].get('columns', []))}")
        print(f"  • Categorical columns: {len(eda_results['categorical_analysis'].get('columns', []))}")
        
        # Show recommendations
        if eda_results.get('recommendations'):
            print(f"\n💡 EDA Recommendations:")
            for rec in eda_results['recommendations']:
                print(f"  • {rec}")
        
    except Exception as e:
        print(f"  ❌ EDA failed: {e}")
        return
    
    # Clean data
    print("\n🧹 CLEANING DATA")
    print("-" * 50)
    
    try:
        cleaned_data = preprocessor.clean_data(sample_data)
        
        print(f"\n✅ Data cleaning completed!")
        print(f"  • Initial shape: {sample_data.shape}")
        print(f"  • Final shape: {cleaned_data.shape}")
        print(f"  • Rows removed: {len(sample_data) - len(cleaned_data)}")
        
        # Show cleaning statistics
        cleaning_stats = preprocessor.preprocessing_info['statistics'].get('cleaning', {})
        if cleaning_stats:
            print(f"  • Duplicates removed: {cleaning_stats.get('duplicates_removed', 0)}")
            print(f"  • Cleaning steps: {len(cleaning_stats.get('steps', []))}")
        
    except Exception as e:
        print(f"  ❌ Data cleaning failed: {e}")
        return
    
    # Preprocess data
    print("\n⚙️  PREPROCESSING DATA")
    print("-" * 50)
    
    try:
        preprocessed_data, preprocessing_info = preprocessor.preprocess_data(cleaned_data)
        
        print(f"\n✅ Data preprocessing completed!")
        print(f"  • Input shape: {cleaned_data.shape}")
        print(f"  • Output shape: {preprocessed_data.shape}")
        print(f"  • Features added: {len(preprocessed_data.columns) - len(cleaned_data.columns)}")
        
        # Show preprocessing transformations
        transformations = preprocessing_info.get('transformations', {})
        if transformations.get('encoding'):
            print(f"  • Categorical encoding: {len(transformations['encoding'])} columns encoded")
        
        if transformations.get('scaling'):
            print(f"  • Numerical scaling: {len(transformations['scaling'].get('columns', []))} columns scaled")
        
    except Exception as e:
        print(f"  ❌ Data preprocessing failed: {e}")
        return
    
    # Save preprocessed data
    print("\n💾 SAVING PREPROCESSED DATA")
    print("-" * 50)
    
    try:
        output_path = preprocessor.save_preprocessed_data(preprocessed_data)
        print(f"  ✅ Preprocessed data saved to: {output_path}")
        
        # Show file info
        file_size = Path(output_path).stat().st_size / 1024  # KB
        print(f"  • File size: {file_size:.2f} KB")
        
    except Exception as e:
        print(f"  ❌ Failed to save data: {e}")
    
    # Show final dataset info
    print("\n📊 FINAL DATASET INFORMATION")
    print("-" * 50)
    
    print(f"  • Final shape: {preprocessed_data.shape}")
    print(f"  • Memory usage: {preprocessed_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"  • Data types: {preprocessed_data.dtypes.value_counts().to_dict()}")
    
    # Show sample of preprocessed data
    print(f"\n📋 Sample of preprocessed data:")
    print(preprocessed_data.head().to_string())
    
    # Show preprocessing steps
    print(f"\n📝 PREPROCESSING STEPS PERFORMED")
    print("-" * 50)
    
    steps = preprocessor.preprocessing_info.get('steps_performed', [])
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    
    print("\n✅ DEMONSTRATION COMPLETED!")
    print("="*80)
    print("\n📝 Summary:")
    print(f"  • Performed comprehensive EDA with visualizations")
    print(f"  • Cleaned dataset with {len(sample_data) - len(cleaned_data)} rows removed")
    print(f"  • Preprocessed data with encoding and scaling")
    print(f"  • Generated {len(steps)} preprocessing steps")
    print(f"  • Saved clean dataset ready for modeling")

def show_preprocessing_configuration():
    """Show the current preprocessing configuration."""
    print("\n⚙️  PREPROCESSING CONFIGURATION")
    print("-" * 50)
    
    config_path = "config/config.yaml"
    if Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        preprocessing_config = config.get('preprocessing', {})
        print("Current preprocessing configuration:")
        print(json.dumps(preprocessing_config, indent=2))
    else:
        print("❌ Configuration file not found. Using defaults.")

if __name__ == "__main__":
    print("🎯 Data Preprocessing System Real-time Demonstration")
    print("This script will show you exactly how data preprocessing works.")
    
    # Show configuration
    show_preprocessing_configuration()
    
    # Run demonstration
    demonstrate_preprocessing()
    
    print("\n🎉 All done! Check the plots/eda and data/processed directories to see the results.")
