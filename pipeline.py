#!/usr/bin/env python3
"""
Main pipeline orchestrator.
Runs the complete churn prediction pipeline.
"""

import os

def main():
    """Run the complete pipeline step by step"""
    print("Customer Churn Prediction Pipeline")
    print("=" * 40)
    
    # Check if we have data
    kaggle_file = "data/raw/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    hf_file = "data/raw/huggingface/customer_data.csv"
    
    if os.path.exists(kaggle_file):
        print(f"Found Kaggle data: {kaggle_file}")
    elif os.path.exists(hf_file):
        print(f"Found HuggingFace data: {hf_file}")
    else:
        print("No data found! Please run: python get_data.py first")
        return
    
    print()
    
    # Run each pipeline step
    steps = [
        ("1. Raw Data Storage", "python src/storage/store_data.py"),
        ("2. Data Validation", "python src/validation/check_data.py"),
        ("3. Generate Quality Report", "python src/validation/generate_quality_report.py"),
        ("4. Data Cleaning", "python src/preparation/clean_data.py"),
        ("5. Feature Engineering", "python src/transformation/make_features.py"),
        ("6. Database Setup", "python src/transformation/database_setup.py"),
        ("7. Feature Store Management", "python src/feature_store/manage_features.py"),
        ("8. Data Versioning", "python src/versioning/version_data.py"),
        ("9. Model Training", "python src/modeling/train_model.py")
    ]
    
    for step_name, command in steps:
        print(f"{step_name}...")
        result = os.system(command)
        if result != 0:
            print(f"Error in {step_name}")
            return
        print()
    
    print("=" * 40)
    print("Pipeline completed successfully!")
    print("Check data/processed/ for cleaned data")
    print("Check models/ for trained models")

if __name__ == "__main__":
    main()
