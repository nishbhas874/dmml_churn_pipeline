#!/usr/bin/env python3
"""
Real-time demonstration of Feature Engineering and Database Storage Systems.
Shows live output of feature creation, transformation, and database operations.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from transformation.feature_engineer import create_feature_engineer
from storage.database_storage import create_database_storage
from loguru import logger

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("📊 Creating sample dataset for feature engineering demonstration...")
    
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
    print(f"  ✅ Created dataset with {len(df)} rows and {len(df.columns)} columns")
    return df

def demonstrate_feature_engineering():
    """Demonstrate the feature engineering system."""
    print("\n" + "="*80)
    print("🔧 FEATURE ENGINEERING SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize feature engineer
    print("\n🔧 Initializing Feature Engineer...")
    try:
        feature_engineer = create_feature_engineer()
        print("  ✅ Feature engineer initialized successfully")
    except Exception as e:
        print(f"  ❌ Failed to initialize feature engineer: {e}")
        return None, None
    
    # Create sample dataset
    print("\n📊 Creating Sample Dataset...")
    sample_data = create_sample_dataset()
    
    print(f"\n📋 Original Dataset Overview:")
    print(f"  • Shape: {sample_data.shape}")
    print(f"  • Memory usage: {sample_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"  • Data types: {sample_data.dtypes.value_counts().to_dict()}")
    
    # Apply feature engineering pipeline
    print("\n⚙️  APPLYING FEATURE ENGINEERING PIPELINE")
    print("-" * 50)
    
    try:
        transformed_data, transformation_info = feature_engineer.transform_data(sample_data)
        
        print(f"\n✅ Feature engineering completed!")
        print(f"  • Original shape: {sample_data.shape}")
        print(f"  • Final shape: {transformed_data.shape}")
        print(f"  • Features added: {len(transformed_data.columns) - len(sample_data.columns)}")
        
        # Show transformation details
        print(f"\n📊 TRANSFORMATION DETAILS")
        print("-" * 30)
        
        # Features created
        features_created = transformation_info.get('features_created', [])
        print(f"  • Total features created: {len(features_created)}")
        
        # Show some example features
        if features_created:
            print(f"  • Example features: {features_created[:10]}")
            if len(features_created) > 10:
                print(f"    ... and {len(features_created) - 10} more")
        
        # Transformations applied
        transformations = transformation_info.get('transformations_applied', {})
        if transformations.get('scaling'):
            scaling_info = transformations['scaling']
            print(f"  • Scaling method: {scaling_info.get('method', 'N/A')}")
            print(f"  • Columns scaled: {len(scaling_info.get('columns', []))}")
        
        if transformations.get('feature_selection'):
            selection_info = transformations['feature_selection']
            print(f"  • Feature selection method: {selection_info.get('method', 'N/A')}")
            print(f"  • Features selected: {len(selection_info.get('selected_features', []))}")
        
        # Final statistics
        stats = transformation_info.get('statistics', {})
        print(f"\n📈 FINAL STATISTICS")
        print("-" * 30)
        print(f"  • Numerical features: {stats.get('numerical_features', 0)}")
        print(f"  • Categorical features: {stats.get('categorical_features', 0)}")
        print(f"  • Total features created: {stats.get('total_features_created', 0)}")
        
        # Show sample of transformed data
        print(f"\n📋 Sample of transformed data:")
        print(transformed_data.head().to_string())
        
        return transformed_data, transformation_info
        
    except Exception as e:
        print(f"  ❌ Feature engineering failed: {e}")
        return None, None

def demonstrate_database_storage(transformed_data, transformation_info):
    """Demonstrate the database storage system."""
    print("\n" + "="*80)
    print("🗄️  DATABASE STORAGE SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize database storage
    print("\n🗄️  Initializing Database Storage...")
    try:
        db_storage = create_database_storage()
        print("  ✅ Database storage initialized successfully")
    except Exception as e:
        print(f"  ❌ Failed to initialize database storage: {e}")
        return None
    
    # Connect to database
    print("\n🔗 Connecting to Database...")
    try:
        if db_storage.connect_database():
            print("  ✅ Database connection established")
        else:
            print("  ❌ Failed to connect to database")
            return None
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return None
    
    # Create schema
    print("\n📋 Creating Database Schema...")
    try:
        if db_storage.create_schema():
            print("  ✅ Database schema created successfully")
        else:
            print("  ❌ Failed to create database schema")
            return None
    except Exception as e:
        print(f"  ❌ Schema creation failed: {e}")
        return None
    
    # Store customer data
    print("\n💾 Storing Customer Data...")
    try:
        if db_storage.store_customer_data(transformed_data):
            print("  ✅ Customer data stored successfully")
        else:
            print("  ❌ Failed to store customer data")
    except Exception as e:
        print(f"  ❌ Customer data storage failed: {e}")
    
    # Store engineered features
    print("\n🔧 Storing Engineered Features...")
    try:
        if db_storage.store_engineered_features(transformed_data):
            print("  ✅ Engineered features stored successfully")
        else:
            print("  ❌ Failed to store engineered features")
    except Exception as e:
        print(f"  ❌ Feature storage failed: {e}")
    
    # Execute sample queries
    print("\n🔍 EXECUTING SAMPLE QUERIES")
    print("-" * 50)
    
    sample_queries = db_storage.get_sample_queries()
    
    for query_name, query in sample_queries.items():
        print(f"\n📊 Running: {query_name}")
        try:
            result = db_storage.execute_query(query)
            print(f"  ✅ Query executed successfully")
            print(f"  • Rows returned: {len(result)}")
            if len(result) > 0:
                print(f"  • Sample result:")
                print(result.head(3).to_string())
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
    
    # Generate schema documentation
    print("\n📄 Generating Schema Documentation...")
    try:
        doc_path = db_storage.save_schema_documentation()
        print(f"  ✅ Schema documentation saved to: {doc_path}")
    except Exception as e:
        print(f"  ❌ Documentation generation failed: {e}")
    
    # Show storage statistics
    print(f"\n📊 STORAGE STATISTICS")
    print("-" * 30)
    
    storage_info = db_storage.storage_info
    print(f"  • Tables created: {len(storage_info.get('tables_created', []))}")
    print(f"  • Data loaded: {len(storage_info.get('data_loaded', {}))} tables")
    print(f"  • Queries executed: {len(storage_info.get('queries_executed', []))}")
    
    # Close database connection
    print("\n🔒 Closing Database Connection...")
    db_storage.close_connection()
    print("  ✅ Database connection closed")
    
    return db_storage

def save_transformation_report(transformation_info):
    """Save transformation report."""
    print("\n📄 SAVING TRANSFORMATION REPORT")
    print("-" * 50)
    
    try:
        # Save transformation info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/transformation_report_{timestamp}.json"
        
        # Ensure directory exists
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(transformation_info, f, indent=2, default=str)
        
        print(f"  ✅ Transformation report saved to: {report_path}")
        
        # Create summary report
        summary_path = f"reports/transformation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("FEATURE ENGINEERING TRANSFORMATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {transformation_info.get('timestamp', 'N/A')}\n")
            f.write(f"Total features created: {len(transformation_info.get('features_created', []))}\n")
            f.write(f"Transformations applied: {len(transformation_info.get('transformations_applied', {}))}\n\n")
            
            f.write("FEATURES CREATED:\n")
            f.write("-" * 20 + "\n")
            for feature in transformation_info.get('features_created', []):
                f.write(f"• {feature}\n")
            
            f.write("\nTRANSFORMATIONS APPLIED:\n")
            f.write("-" * 25 + "\n")
            for transform_name, transform_info in transformation_info.get('transformations_applied', {}).items():
                f.write(f"• {transform_name}: {transform_info}\n")
        
        print(f"  ✅ Transformation summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"  ❌ Failed to save transformation report: {e}")

def main():
    """Main demonstration function."""
    print("🎯 Feature Engineering and Database Storage System Demonstration")
    print("This script demonstrates comprehensive feature engineering and database storage capabilities.")
    
    # Step 1: Feature Engineering
    transformed_data, transformation_info = demonstrate_feature_engineering()
    
    if transformed_data is None:
        print("\n❌ Feature engineering failed. Cannot proceed with database storage.")
        return
    
    # Step 2: Database Storage
    db_storage = demonstrate_database_storage(transformed_data, transformation_info)
    
    # Step 3: Save Reports
    if transformation_info:
        save_transformation_report(transformation_info)
    
    print("\n✅ DEMONSTRATION COMPLETED!")
    print("="*80)
    print("\n📝 Summary:")
    print(f"  • Performed comprehensive feature engineering")
    print(f"  • Created {len(transformation_info.get('features_created', []))} new features")
    print(f"  • Applied {len(transformation_info.get('transformations_applied', {}))} transformations")
    print(f"  • Stored data in database with {len(db_storage.storage_info.get('tables_created', []))} tables")
    print(f"  • Executed {len(db_storage.storage_info.get('queries_executed', []))} sample queries")
    print(f"  • Generated comprehensive documentation and reports")

if __name__ == "__main__":
    main()
    print("\n🎉 All done! Check the reports and data directories to see the results.")
