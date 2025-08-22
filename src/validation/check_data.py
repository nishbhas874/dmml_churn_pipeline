#!/usr/bin/env python3
"""
Simple Data Validation - Stage 4
Check data quality and generate validation reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os

def load_raw_data():
    """Load raw data for validation"""
    # Try to load Kaggle data first
    kaggle_file = "data/raw/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    hf_file = "data/raw/huggingface/customer_data.csv"
    
    if os.path.exists(kaggle_file):
        print(f"Loading Kaggle data: {kaggle_file}")
        return pd.read_csv(kaggle_file)
    elif os.path.exists(hf_file):
        print(f"Loading HuggingFace data: {hf_file}")
        return pd.read_csv(hf_file)
    else:
        print("No data found! Please run data ingestion first.")
        return None

def check_missing_values(df):
    """Check for missing values"""
    print("Checking for missing values...")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    issues = []
    for col in missing[missing > 0].index:
        issues.append(f"{col}: {missing[col]} missing ({missing_percent[col]:.1f}%)")
    
    return len(issues), issues

def check_duplicates(df):
    """Check for duplicate records"""
    print("Checking for duplicates...")
    duplicates = df.duplicated().sum()
    issues = []
    
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate records")
    
    return len(issues), issues

def check_data_types(df):
    """Check data types"""
    print("Checking data types...")
    issues = []
    
    # Check for mixed types in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            issues.append(f"{col}: Has missing values in numeric column")
    
    return len(issues), issues

def generate_validation_report(df, validation_results):
    """Generate validation report"""
    Path("data/validated").mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create validation summary
    summary = {
        "validation_timestamp": datetime.now().isoformat(),
        "dataset_shape": df.shape,
        "total_checks": len(validation_results),
        "total_issues": sum(len(issues) for _, issues in validation_results.values()),
        "checks_performed": list(validation_results.keys()),
        "validation_results": {
            check: {"issue_count": count, "issues": issues}
            for check, (count, issues) in validation_results.items()
        }
    }
    
    # Save validation report
    report_file = f"data/validated/validation_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Validation report saved: {report_file}")
    return report_file

def main():
    """Main validation function"""
    print("Data Validation - Stage 4")
    print("=" * 40)
    
    # Load data
    df = load_raw_data()
    if df is None:
        return
    
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Perform validation checks
    validation_results = {}
    
    # Check missing values
    count, issues = check_missing_values(df)
    validation_results["missing_values"] = (count, issues)
    
    # Check duplicates
    count, issues = check_duplicates(df)
    validation_results["duplicates"] = (count, issues)
    
    # Check data types
    count, issues = check_data_types(df)
    validation_results["data_types"] = (count, issues)
    
    # Generate report
    report_file = generate_validation_report(df, validation_results)
    
    # Print summary
    total_issues = sum(len(issues) for _, issues in validation_results.values())
    print(f"Validation completed!")
    print(f"Total issues found: {total_issues}")
    
    for check, (count, issues) in validation_results.items():
        print(f"\n{check.replace('_', ' ').title()}:")
        if count == 0:
            print("  ✅ No issues found")
        else:
            for issue in issues[:3]:  # Show first 3 issues
                print(f"  ❌ {issue}")
            if len(issues) > 3:
                print(f"  ... and {len(issues) - 3} more issues")
    
    print("=" * 40)
    print("Data validation completed!")

if __name__ == "__main__":
    main()