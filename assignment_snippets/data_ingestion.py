"""
Data Ingestion Code for Assignment
This code demonstrates data ingestion from Hugging Face and Kaggle sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import shutil
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataIngestion:
    """Unified data ingestion system for Hugging Face and Kaggle datasets."""
    
    def __init__(self):
        """Initialize data ingestion system."""
        self.ingestion_info = {
            'timestamp': datetime.now().isoformat(),
            'sources_processed': [],
            'files_downloaded': [],
            'errors': []
        }
    
    def ingest_huggingface(self, repo_id: str, filename: str = None, output_dir: str = "data/raw/huggingface"):
        """Ingest data from Hugging Face datasets."""
        print(f"Ingesting data from Hugging Face: {repo_id}")
        
        try:
            # Import datasets library
            from datasets import load_dataset
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Load dataset
            dataset = load_dataset(repo_id, split="train")
            df = dataset.to_pandas()
            
            # Save to file
            if filename is None:
                filename = f"{repo_id.replace('/', '_')}.csv"
            
            output_path = Path(output_dir) / filename
            df.to_csv(output_path, index=False)
            
            # Update ingestion info
            self.ingestion_info['sources_processed'].append({
                'source': 'huggingface',
                'repo_id': repo_id,
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'output_path': str(output_path),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Successfully ingested {len(df)} rows from Hugging Face")
            print(f"   Saved to: {output_path}")
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to ingest from Hugging Face: {e}"
            print(f"❌ {error_msg}")
            self.ingestion_info['errors'].append({
                'source': 'huggingface',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def ingest_kaggle(self, dataset: str, file: str = None, output_dir: str = "data/raw/kaggle"):
        """Ingest data from Kaggle datasets."""
        print(f"Ingesting data from Kaggle: {dataset}")
        
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Download dataset using Kaggle CLI
            subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", dataset, 
                "-p", output_dir, 
                "--force"
            ], check=True)
            
            # Unzip downloaded files
            for zip_file in Path(output_dir).glob("*.zip"):
                shutil.unpack_archive(zip_file, output_dir)
                zip_file.unlink()  # Remove zip file
            
            # Find the target file
            if file:
                target_file = Path(output_dir) / file
                if target_file.exists():
                    df = pd.read_csv(target_file)
                else:
                    raise FileNotFoundError(f"Target file {file} not found")
            else:
                # Find first CSV file
                csv_files = list(Path(output_dir).glob("*.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    file = csv_files[0].name
                else:
                    raise FileNotFoundError("No CSV files found in downloaded dataset")
            
            # Update ingestion info
            self.ingestion_info['sources_processed'].append({
                'source': 'kaggle',
                'dataset': dataset,
                'filename': file,
                'rows': len(df),
                'columns': len(df.columns),
                'output_path': str(Path(output_dir) / file),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Successfully ingested {len(df)} rows from Kaggle")
            print(f"   Saved to: {Path(output_dir) / file}")
            
            return df
            
        except Exception as e:
            error_msg = f"Failed to ingest from Kaggle: {e}"
            print(f"❌ {error_msg}")
            self.ingestion_info['errors'].append({
                'source': 'kaggle',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def select_primary_dataset(self, datasets: list) -> pd.DataFrame:
        """Select the primary dataset from multiple ingested datasets."""
        print("Selecting primary dataset...")
        
        if not datasets:
            print("❌ No datasets provided for selection")
            return None
        
        # Select the largest dataset as primary
        largest_dataset = max(datasets, key=lambda x: len(x) if x is not None else 0)
        
        if largest_dataset is not None:
            print(f"✅ Selected primary dataset with {len(largest_dataset)} rows")
            return largest_dataset
        else:
            print("❌ No valid datasets found")
            return None
    
    def get_ingestion_summary(self):
        """Get summary of ingestion process."""
        return {
            'total_sources': len(self.ingestion_info['sources_processed']),
            'successful_ingestions': len([s for s in self.ingestion_info['sources_processed'] if 'error' not in s]),
            'errors': len(self.ingestion_info['errors']),
            'total_rows': sum([s.get('rows', 0) for s in self.ingestion_info['sources_processed']]),
            'sources': self.ingestion_info['sources_processed'],
            'errors': self.ingestion_info['errors']
        }

# Example usage
if __name__ == "__main__":
    # Initialize data ingestion
    ingestion = DataIngestion()
    
    # Example: Ingest from Hugging Face
    try:
        hf_data = ingestion.ingest_huggingface(
            repo_id="mkechinov/ecommerce-behavior-data-from-multi-category-store",
            filename="ecommerce_churn.csv"
        )
    except Exception as e:
        print(f"Hugging Face ingestion failed: {e}")
        hf_data = None
    
    # Example: Ingest from Kaggle
    try:
        kaggle_data = ingestion.ingest_kaggle(
            dataset="blastchar/telco-customer-churn",
            file="WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    except Exception as e:
        print(f"Kaggle ingestion failed: {e}")
        kaggle_data = None
    
    # Select primary dataset
    datasets = [hf_data, kaggle_data]
    primary_data = ingestion.select_primary_dataset(datasets)
    
    # Get ingestion summary
    summary = ingestion.get_ingestion_summary()
    print(f"\n📊 Ingestion Summary:")
    print(f"   Total sources processed: {summary['total_sources']}")
    print(f"   Successful ingestions: {summary['successful_ingestions']}")
    print(f"   Errors: {summary['errors']}")
    print(f"   Total rows ingested: {summary['total_rows']}")
    
    if primary_data is not None:
        print(f"\n✅ Primary dataset selected:")
        print(f"   Shape: {primary_data.shape}")
        print(f"   Columns: {list(primary_data.columns)}")
        print(f"   Sample data:")
        print(primary_data.head())
