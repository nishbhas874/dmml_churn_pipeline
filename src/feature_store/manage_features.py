#!/usr/bin/env python3
"""
Feature Store Management - Stage 7
Manages engineered features with metadata and versioning.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def create_folders():
    """Create necessary folders"""
    Path("data/features").mkdir(parents=True, exist_ok=True)
    Path("data/features/metadata").mkdir(parents=True, exist_ok=True)

def load_transformed_data():
    """Load the latest transformed data"""
    transformed_dir = Path("data/transformed")
    
    if not transformed_dir.exists():
        print("No transformed data found! Please run transformation first.")
        return None
    
    # Find the latest transformed file
    csv_files = list(transformed_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in transformed data!")
        return None
    
    # Get the most recent file
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"Loading transformed data from: {latest_file}")
    
    return pd.read_csv(latest_file)

def create_feature_metadata(df):
    """Create metadata for each feature"""
    print("Creating feature metadata...")
    
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_features": len(df.columns),
        "total_records": len(df),
        "features": {}
    }
    
    for column in df.columns:
        feature_info = {
            "name": column,
            "data_type": str(df[column].dtype),
            "null_count": int(df[column].isnull().sum()),
            "unique_values": int(df[column].nunique()),
            "description": get_feature_description(column)
        }
        
        # Add statistics based on data type
        if df[column].dtype in ['int64', 'float64']:
            feature_info.update({
                "min_value": float(df[column].min()),
                "max_value": float(df[column].max()),
                "mean_value": float(df[column].mean()),
                "std_value": float(df[column].std())
            })
        
        metadata["features"][column] = feature_info
    
    return metadata

def get_feature_description(column_name):
    """Get human-readable description for features"""
    descriptions = {
        "customer_id": "Unique identifier for each customer",
        "tenure": "Number of months customer has been with company",
        "monthly_charges": "Monthly charges for the customer",
        "total_charges": "Total charges for the customer",
        "churn": "Whether customer churned (target variable)",
        "tenure_group": "Grouped tenure categories",
        "charges_per_tenure": "Average charges per month of tenure",
        "contract_encoded": "Encoded contract type",
        "payment_method_encoded": "Encoded payment method",
        "gender_encoded": "Encoded gender",
        "senior_citizen": "Whether customer is senior citizen",
        "partner": "Whether customer has partner",
        "dependents": "Whether customer has dependents"
    }
    
    return descriptions.get(column_name, f"Feature: {column_name}")

def save_features_to_store(df, metadata):
    """Save features and metadata to feature store"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save features
    features_file = f"data/features/features_{timestamp}.csv"
    df.to_csv(features_file, index=False)
    print(f"Features saved to: {features_file}")
    
    # Save metadata
    metadata_file = f"data/features/metadata/features_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")
    
    # Create feature registry
    registry = {
        "latest_version": timestamp,
        "features_file": features_file,
        "metadata_file": metadata_file,
        "total_features": metadata["total_features"],
        "created_at": metadata["created_at"]
    }
    
    with open("data/features/feature_registry.json", 'w') as f:
        json.dump(registry, f, indent=2)
    print("Feature registry updated")
    
    return features_file, metadata_file

def retrieve_latest_features():
    """Retrieve the latest features from feature store"""
    registry_file = "data/features/feature_registry.json"
    
    if not os.path.exists(registry_file):
        print("No feature registry found!")
        return None, None
    
    with open(registry_file, 'r') as f:
        registry = json.load(f)
    
    features_file = registry["features_file"]
    metadata_file = registry["metadata_file"]
    
    print(f"Latest features: {features_file}")
    print(f"Latest metadata: {metadata_file}")
    
    # Load features
    features_df = pd.read_csv(features_file)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return features_df, metadata

def print_feature_summary(metadata):
    """Print summary of features in the store"""
    print("\n" + "="*50)
    print("FEATURE STORE SUMMARY")
    print("="*50)
    print(f"Total Features: {metadata['total_features']}")
    print(f"Total Records: {metadata['total_records']}")
    print(f"Created At: {metadata['created_at']}")
    print("\nFeature Details:")
    print("-" * 30)
    
    for feature_name, info in metadata["features"].items():
        print(f"• {feature_name}")
        print(f"  Type: {info['data_type']}")
        print(f"  Description: {info['description']}")
        print(f"  Null Values: {info['null_count']}")
        print()

def main():
    """Main feature store management function"""
    print("Feature Store Management - Stage 7")
    print("=" * 40)
    
    # Create necessary folders
    create_folders()
    
    # Load transformed data
    df = load_transformed_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    print()
    
    # Create feature metadata
    metadata = create_feature_metadata(df)
    
    # Save to feature store
    features_file, metadata_file = save_features_to_store(df, metadata)
    
    # Print summary
    print_feature_summary(metadata)
    
    print("=" * 40)
    print("Feature store management completed!")
    print(f"Features available at: {features_file}")
    print(f"Metadata available at: {metadata_file}")

if __name__ == "__main__":
    main()
