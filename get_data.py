# get_data.py - Data Ingestion Script
import pandas as pd
import os
import subprocess
from datasets import load_dataset

print("Starting Data Download...")
print("This will get data from Kaggle and Hugging Face")
print()

# Step 1: Get data from Kaggle 
print("Downloading data from Kaggle...")
os.makedirs("data/raw/kaggle", exist_ok=True)

# We use kaggle API to download the dataset
kaggle_cmd = "kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/kaggle --unzip"
subprocess.run(kaggle_cmd, shell=True)
print("Successfully downloaded Kaggle data: data/raw/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print()

# Step 2: Get data from Hugging Face
print("Downloading data from Hugging Face...")
os.makedirs("data/raw/huggingface", exist_ok=True)

# Load dataset and save as CSV using pandas
dataset = load_dataset("scikit-learn/adult-census-income", split="train")
df = dataset.to_pandas()
df.to_csv("data/raw/huggingface/customer_data.csv", index=False)
print("Successfully downloaded Hugging Face data: data/raw/huggingface/customer_data.csv")
print()

# Summary - show what we got
print()
print("=" * 50)
print("Successfully downloaded 2 datasets:")
print("  - data/raw/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("  - data/raw/huggingface/customer_data.csv")
print()
print("SUCCESS! Data download completed.")
print("You can now run the pipeline with: python pipeline.py")
