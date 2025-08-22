#!/usr/bin/env python3
"""
Simple script to download data for our churn prediction project.
"""

import os
import subprocess
import pandas as pd
import yaml
from pathlib import Path
from datasets import load_dataset

def load_config():
    """Load our simple configuration"""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_folder(path):
    """Create a folder if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_kaggle_data():
    """Download data from Kaggle"""
    print("Downloading data from Kaggle...")
    
    config = load_config()
    dataset = config['kaggle_dataset']
    filename = config['kaggle_file']
    
    # Create kaggle folder
    kaggle_folder = "data/raw/kaggle"
    create_folder(kaggle_folder)
    
    # Download using kaggle CLI
    cmd = f"kaggle datasets download -d {dataset} -p {kaggle_folder} --unzip"
    subprocess.run(cmd, shell=True)
    
    file_path = f"{kaggle_folder}/{filename}"
    if os.path.exists(file_path):
        print(f"Successfully downloaded Kaggle data: {file_path}")
        return file_path
    else:
        print(f"Could not find {filename} in downloaded files")
        return None

def get_huggingface_data():
    """Download data from Hugging Face"""
    print("Downloading data from Hugging Face...")
    
    config = load_config()
    repo_id = config['huggingface_dataset']
    filename = config['huggingface_file']
    
    # Create huggingface folder
    hf_folder = "data/raw/huggingface"
    create_folder(hf_folder)
    
    # Load dataset from huggingface
    dataset = load_dataset(repo_id, split="train")
    df = dataset.to_pandas()
    
    # Save to CSV
    file_path = f"{hf_folder}/{filename}"
    df.to_csv(file_path, index=False)
    
    print(f"Successfully downloaded Hugging Face data: {file_path}")
    return file_path

def main():
    """Download all data"""
    print("Starting Data Download...")
    print("This will get data from Kaggle and Hugging Face")
    print()
    
    downloaded_files = []
    
    # Try Kaggle first
    kaggle_file = get_kaggle_data()
    if kaggle_file:
        downloaded_files.append(kaggle_file)
    
    # Try Hugging Face
    hf_file = get_huggingface_data()
    if hf_file:
        downloaded_files.append(hf_file)
    
    print()
    print("=" * 50)
    if downloaded_files:
        print(f"Successfully downloaded {len(downloaded_files)} datasets:")
        for file in downloaded_files:
            print(f"  - {file}")
        print()
        print("SUCCESS! Data download completed.")
        print("You can now run the pipeline with: python pipeline.py")
    else:
        print("No data was downloaded.")
        print("Please check your Kaggle API credentials.")

if __name__ == "__main__":
    main()