#!/usr/bin/env python3
"""
Main script to run our complete churn prediction pipeline.
This runs all the steps from data download to model training.
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
        ("1. Data Validation", "python check_data.py"),
        ("2. Data Cleaning", "python clean_data.py"),
        ("3. Feature Engineering", "python make_features.py"),
        ("4. Model Training", "python train_model.py")
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