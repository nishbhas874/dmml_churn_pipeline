#!/usr/bin/env python3
"""
Real-time demonstration of the Data Validation System.
Shows live output of data quality checks and generates comprehensive reports.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from validation.data_validator import create_data_validator
from loguru import logger

def create_sample_dataset_with_issues():
    """Create a sample dataset with various data quality issues for demonstration."""
    print("📊 Creating sample dataset with data quality issues...")
    
    # Create base data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate clean data first
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
    
    # Introduce data quality issues for demonstration
    
    # 1. Missing data
    print("  🔍 Adding missing data...")
    missing_indices = np.random.choice(df.index, size=int(n_samples * 0.1), replace=False)
    df.loc[missing_indices, 'monthly_charges'] = np.nan
    
    # 2. Duplicate records
    print("  🔍 Adding duplicate records...")
    duplicate_indices = np.random.choice(df.index, size=50, replace=False)
    df = pd.concat([df, df.loc[duplicate_indices]], ignore_index=True)
    
    # 3. Outliers
    print("  🔍 Adding outliers...")
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'monthly_charges'] = np.random.uniform(200, 500, 20)
    
    # 4. Inconsistent data types
    print("  🔍 Adding inconsistent data types...")
    type_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[type_indices, 'tenure'] = df.loc[type_indices, 'tenure'].astype(str) + ' months'
    
    # 5. Invalid values
    print("  🔍 Adding invalid values...")
    invalid_indices = np.random.choice(df.index, size=25, replace=False)
    df.loc[invalid_indices, 'age'] = -5  # Negative age
    
    # 6. High cardinality
    print("  🔍 Adding high cardinality data...")
    high_card_indices = np.random.choice(df.index, size=100, replace=False)
    df.loc[high_card_indices, 'customer_id'] = [f'CUST_{i:06d}' for i in range(100)]
    
    # 7. Rare categories
    print("  🔍 Adding rare categories...")
    rare_indices = np.random.choice(df.index, size=5, replace=False)
    df.loc[rare_indices, 'contract_type'] = 'Lifetime'
    
    # 8. Zero variance column
    print("  🔍 Adding zero variance column...")
    df['constant_column'] = 'same_value'
    
    # 9. Highly skewed data
    print("  🔍 Adding highly skewed data...")
    skewed_data = np.random.exponential(1, n_samples + 50 + 20 + 30 + 25 + 100 + 5)
    df['skewed_column'] = skewed_data[:len(df)]
    
    # 10. Target variable issues
    print("  🔍 Adding target variable issues...")
    target_missing = np.random.choice(df.index, size=15, replace=False)
    df.loc[target_missing, 'churn'] = np.nan
    
    print(f"  ✅ Created dataset with {len(df)} rows and {len(df.columns)} columns")
    return df

def demonstrate_validation():
    """Demonstrate the data validation system."""
    print("\n" + "="*80)
    print("🔍 DATA VALIDATION SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Initialize validator
    print("\n🔧 Initializing Data Validator...")
    try:
        validator = create_data_validator()
        print("  ✅ Data validator initialized successfully")
    except Exception as e:
        print(f"  ❌ Failed to initialize validator: {e}")
        return
    
    # Create sample dataset with issues
    print("\n📊 Creating Sample Dataset...")
    sample_data = create_sample_dataset_with_issues()
    
    print(f"\n📋 Dataset Overview:")
    print(f"  • Shape: {sample_data.shape}")
    print(f"  • Memory usage: {sample_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"  • Data types: {sample_data.dtypes.value_counts().to_dict()}")
    
    # Perform validation
    print("\n🔍 PERFORMING DATA VALIDATION")
    print("-" * 50)
    
    try:
        validation_results = validator.validate_dataset(sample_data, "Sample Churn Dataset")
        
        print(f"\n✅ Validation completed!")
        print(f"  • Total checks performed: {validation_results['summary']['total_checks']}")
        print(f"  • Checks passed: {validation_results['summary']['passed_checks']}")
        print(f"  • Checks failed: {validation_results['summary']['failed_checks']}")
        print(f"  • Overall quality score: {validation_results['summary']['overall_quality_score']}/100")
        
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return
    
    # Display detailed results
    print("\n📋 DETAILED VALIDATION RESULTS")
    print("-" * 50)
    
    for check in validation_results['checks_performed']:
        status_icon = "✅" if check['status'] == 'PASSED' else "❌"
        print(f"\n{status_icon} {check['check_name']}")
        print(f"   Status: {check['status']}")
        print(f"   Issues found: {check['issues_count']}")
        
        if check['issues']:
            print("   Issues:")
            for issue in check['issues']:
                print(f"     • {issue}")
    
    # Display issues by severity
    print("\n🚨 ISSUES BY SEVERITY")
    print("-" * 50)
    
    critical_issues = [i for i in validation_results['issues_found'] if i['severity'] == 'CRITICAL']
    warning_issues = [i for i in validation_results['issues_found'] if i['severity'] == 'WARNING']
    info_issues = [i for i in validation_results['issues_found'] if i['severity'] == 'INFO']
    
    print(f"\n🔴 Critical Issues ({len(critical_issues)}):")
    for issue in critical_issues:
        print(f"  • [{issue['check']}] {issue['issue']}")
    
    print(f"\n🟡 Warning Issues ({len(warning_issues)}):")
    for issue in warning_issues:
        print(f"  • [{issue['check']}] {issue['issue']}")
    
    print(f"\n🔵 Info Issues ({len(info_issues)}):")
    for issue in info_issues:
        print(f"  • [{issue['check']}] {issue['issue']}")
    
    # Generate reports
    print("\n📄 GENERATING REPORTS")
    print("-" * 50)
    
    try:
        report_path = validator.generate_report()
        print(f"  ✅ JSON Report: {report_path}")
        print(f"  ✅ CSV Summary: {report_path.replace('.json', '_summary.csv')}")
        print(f"  ✅ HTML Report: {report_path.replace('.json', '.html')}")
        
        # Show report summary
        print(f"\n📊 Report Summary:")
        print(f"  • Quality Score: {validation_results['summary']['overall_quality_score']}/100")
        print(f"  • Critical Issues: {validation_results['summary']['critical_issues']}")
        print(f"  • Warning Issues: {validation_results['summary']['warning_issues']}")
        print(f"  • Total Issues: {validation_results['summary']['total_issues']}")
        
        if validation_results.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in validation_results['recommendations']:
                print(f"  • {rec}")
        
    except Exception as e:
        print(f"  ❌ Report generation failed: {e}")
    
    print("\n✅ DEMONSTRATION COMPLETED!")
    print("="*80)
    print("\n📝 Summary:")
    print(f"  • Validated dataset with {len(sample_data)} rows")
    print(f"  • Performed {validation_results['summary']['total_checks']} validation checks")
    print(f"  • Found {validation_results['summary']['total_issues']} data quality issues")
    print(f"  • Generated comprehensive quality reports")
    print(f"  • Overall quality score: {validation_results['summary']['overall_quality_score']}/100")

def show_validation_configuration():
    """Show the current validation configuration."""
    print("\n⚙️  VALIDATION CONFIGURATION")
    print("-" * 50)
    
    config_path = "config/config.yaml"
    if Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        validation_config = config.get('validation', {})
        print("Current validation configuration:")
        print(json.dumps(validation_config, indent=2))
    else:
        print("❌ Configuration file not found. Using defaults.")

if __name__ == "__main__":
    print("🎯 Data Validation System Real-time Demonstration")
    print("This script will show you exactly how data validation works and what issues it can detect.")
    
    # Show configuration
    show_validation_configuration()
    
    # Run demonstration
    demonstrate_validation()
    
    print("\n🎉 All done! Check the reports directory to see the generated quality reports.")
