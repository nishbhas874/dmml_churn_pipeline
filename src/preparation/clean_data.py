#!/usr/bin/env python3
"""
Simple Data Preparation - Stage 5
Clean and preprocess the raw data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os

def load_raw_data():
    """Load raw data for cleaning"""
    # Try to load Kaggle data first
    kaggle_file = "data/raw/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    hf_file = "data/raw/huggingface/customer_data.csv"
    
    if os.path.exists(kaggle_file):
        print(f"Loading Kaggle data: {kaggle_file}")
        return pd.read_csv(kaggle_file), "kaggle"
    elif os.path.exists(hf_file):
        print(f"Loading HuggingFace data: {hf_file}")
        return pd.read_csv(hf_file), "huggingface"
    else:
        print("No data found! Please run data ingestion first.")
        return None, None

def clean_kaggle_data(df):
    """Clean Kaggle telco customer churn data"""
    print("Cleaning Kaggle data...")
    
    # Convert TotalCharges to numeric (it's stored as string with spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    
    # Fill missing TotalCharges with 0 (for new customers)
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Convert Yes/No to 1/0 for binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Convert Churn to 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Convert SeniorCitizen to proper binary
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    return df

def clean_huggingface_data(df):
    """Clean HuggingFace adult census data"""
    print("Cleaning HuggingFace data...")
    
    # Handle missing values (marked as '?')
    df = df.replace('?', np.nan)
    
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_value, inplace=True)
    
    # Convert income to binary (target variable)
    if 'income' in df.columns:
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    return df

def perform_basic_eda(df, data_source):
    """Perform basic exploratory data analysis"""
    print(f"\nBasic EDA for {data_source} data:")
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes.value_counts()}")
    
    # Show basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric columns statistics:")
        print(df[numeric_cols].describe())
    
    # Show value counts for categorical columns (first few)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nCategorical columns (first 3):")
        for col in categorical_cols[:3]:
            print(f"{col}: {df[col].nunique()} unique values")
            print(f"Top values: {df[col].value_counts().head(3).to_dict()}")

def save_cleaned_data(df, data_source):
    """Save cleaned data"""
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/processed/cleaned_{data_source}_data_{timestamp}.csv"
    
    df.to_csv(output_file, index=False)
    
    # Create info file
    info = {
        "timestamp": datetime.now().isoformat(),
        "source": data_source,
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "file_path": output_file
    }
    
    info_file = f"data/processed/cleaned_{data_source}_info_{timestamp}.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Cleaned data saved: {output_file}")
    print(f"Data info saved: {info_file}")
    
    return output_file, info_file

def main():
    """Main data cleaning function"""
    print("Data Preparation - Stage 5")
    print("=" * 40)
    
    # Load data
    df, data_source = load_raw_data()
    if df is None:
        return
    
    print(f"Loaded {data_source} data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Clean data based on source
    if data_source == "kaggle":
        df_cleaned = clean_kaggle_data(df)
    else:
        df_cleaned = clean_huggingface_data(df)
    
    # Perform basic EDA
    perform_basic_eda(df_cleaned, data_source)
    
    # Save cleaned data
    output_file, info_file = save_cleaned_data(df_cleaned, data_source)
    
    print("\n" + "=" * 40)
    print("Data preparation completed!")
    print(f"Clean dataset ready: {output_file}")
    print(f"Dataset info: {info_file}")

if __name__ == "__main__":
    main()