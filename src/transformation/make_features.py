#!/usr/bin/env python3
"""
Simple Feature Engineering - Stage 6
Create new features from cleaned data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_cleaned_data():
    """Load the most recent cleaned data"""
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("No processed data found! Please run data preparation first.")
        return None, None
    
    # Find the most recent cleaned file
    csv_files = list(processed_dir.glob("cleaned_*_data_*.csv"))
    if not csv_files:
        print("No cleaned data files found!")
        return None, None
    
    # Get the most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    
    # Determine data source from filename
    if "kaggle" in str(latest_file):
        data_source = "kaggle"
    elif "huggingface" in str(latest_file):
        data_source = "huggingface"
    else:
        data_source = "unknown"
    
    print(f"Loading cleaned data: {latest_file}")
    return pd.read_csv(latest_file), data_source

def create_kaggle_features(df):
    """Create features for Kaggle telco data"""
    print("Creating features for Kaggle telco data...")
    
    # Create tenure groups
    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 36, float('inf')], 
                               labels=['0-12', '13-24', '25-36', '37+'])
    
    # Create charges per tenure ratio
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
    
    # Create total charges categories
    df['total_charges_category'] = pd.cut(df['TotalCharges'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Create service usage score
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Count services used (binary columns already converted to 1/0)
    df['services_count'] = 0
    for col in service_cols:
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df['services_count'] += df[col]
        else:
                # For non-binary service columns, count non-'No' values
                df['services_count'] += (df[col] != 'No').astype(int)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    print("Created Kaggle-specific features:")
    print("- tenure_group: Customer tenure categories")
    print("- charges_per_tenure: Monthly charges per tenure ratio")
    print("- total_charges_category: Total charges categories")
    print("- services_count: Number of services used")
    print("- Encoded categorical variables")
    
    return df, label_encoders

def create_huggingface_features(df):
    """Create features for HuggingFace census data"""
    print("Creating features for HuggingFace census data...")
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'])
    
    # Create work hours categories
    if 'hours.per.week' in df.columns:
        df['work_intensity'] = pd.cut(df['hours.per.week'], bins=[0, 20, 40, 60, 100], 
                                     labels=['Part-time', 'Full-time', 'Overtime', 'Extreme'])
    
    # Create education level score
    if 'education.num' in df.columns:
        df['education_level'] = pd.cut(df['education.num'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                       'relationship', 'race', 'sex', 'native.country']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    print("Created HuggingFace-specific features:")
    print("- age_group: Age categories")
    print("- work_intensity: Work hours categories")
    print("- education_level: Education level categories")
    print("- Encoded categorical variables")
    
    return df, label_encoders

def scale_numerical_features(df, data_source):
    """Scale numerical features"""
    print("Scaling numerical features...")
    
    # Select numerical columns for scaling
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns and target variables
    exclude_cols = ['customerID', 'Churn', 'income', 'SeniorCitizen', 'Partner', 'Dependents']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df_scaled = df.copy()
        
        # Scale the numerical columns
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        print(f"Scaled {len(numerical_cols)} numerical columns")
        return df_scaled, scaler
    else:
        print("No numerical columns found for scaling")
        return df, None

def save_transformed_data(df, data_source, transformation_info):
    """Save transformed data and transformation info"""
    Path("data/transformed").mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transformed data
    output_file = f"data/transformed/transformed_{data_source}_data_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    
    # Save transformation info
    info_file = f"data/transformed/transformation_info_{timestamp}.json"
    with open(info_file, 'w') as f:
        json.dump(transformation_info, f, indent=2, default=str)
    
    print(f"Transformed data saved: {output_file}")
    print(f"Transformation info saved: {info_file}")
    
    return output_file, info_file

def main():
    """Main feature engineering function"""
    print("Feature Engineering - Stage 6")
    print("=" * 40)
    
    # Load cleaned data
    df, data_source = load_cleaned_data()
    if df is None:
        return
    
    print(f"Loaded {data_source} data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create features based on data source
    if data_source == "kaggle":
        df_features, encoders = create_kaggle_features(df)
    else:
        df_features, encoders = create_huggingface_features(df)
    
    # Scale numerical features
    df_scaled, scaler = scale_numerical_features(df_features, data_source)
    
    # Prepare transformation info
    transformation_info = {
        "timestamp": datetime.now().isoformat(),
        "data_source": data_source,
        "original_shape": df.shape,
        "transformed_shape": df_scaled.shape,
        "features_added": len(df_scaled.columns) - len(df.columns),
        "new_columns": [col for col in df_scaled.columns if col not in df.columns],
        "encoders_used": list(encoders.keys()) if encoders else [],
        "scaling_applied": scaler is not None,
        "numerical_columns_scaled": len(df_scaled.select_dtypes(include=[np.number]).columns) if scaler else 0
    }
    
    # Save transformed data
    output_file, info_file = save_transformed_data(df_scaled, data_source, transformation_info)
    
    # Print summary
    print(f"\nFeature engineering completed!")
    print(f"Original features: {df.shape[1]}")
    print(f"New features: {df_scaled.shape[1]}")
    print(f"Features added: {transformation_info['features_added']}")
    
    print("\nNew features created:")
    for col in transformation_info['new_columns']:
        print(f"  - {col}")
    
    print("=" * 40)
    print("Feature engineering completed!")

if __name__ == "__main__":
    main()