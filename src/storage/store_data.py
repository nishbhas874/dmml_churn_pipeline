#!/usr/bin/env python3
"""
Simple Raw Data Storage - Stage 3
Store ingested data in organized folder structure.
"""

import os
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path

def create_storage_folders():
    """Create organized storage structure"""
    folders = [
        "data/storage/raw/kaggle",
        "data/storage/raw/huggingface",
        "data/storage/backup"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder}")

def copy_raw_data_to_storage():
    """Copy raw data to organized storage"""
    print("Organizing raw data into storage structure...")
    
    # Copy Kaggle data
    kaggle_source = "data/raw/kaggle"
    kaggle_dest = "data/storage/raw/kaggle"
    
    if os.path.exists(kaggle_source):
        for file in Path(kaggle_source).glob("*.csv"):
            dest_file = Path(kaggle_dest) / file.name
            shutil.copy2(file, dest_file)
            print(f"Stored Kaggle data: {dest_file}")
    
    # Copy HuggingFace data
    hf_source = "data/raw/huggingface"
    hf_dest = "data/storage/raw/huggingface"
    
    if os.path.exists(hf_source):
        for file in Path(hf_source).glob("*.csv"):
            dest_file = Path(hf_dest) / file.name
            shutil.copy2(file, dest_file)
            print(f"Stored HuggingFace data: {dest_file}")

def create_storage_manifest():
    """Create manifest file listing all stored data"""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "storage_structure": "Organized by source and timestamp",
        "files": []
    }
    
    # List all stored files
    storage_root = Path("data/storage")
    for file_path in storage_root.rglob("*.csv"):
        file_info = {
            "file": str(file_path),
            "size_mb": round(file_path.stat().st_size / (1024*1024), 2),
            "source": "kaggle" if "kaggle" in str(file_path) else "huggingface"
        }
        manifest["files"].append(file_info)
    
    # Save manifest
    import json
    manifest_file = "data/storage/storage_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Storage manifest created: {manifest_file}")
    return manifest

def main():
    """Main storage function"""
    print("Raw Data Storage - Stage 3")
    print("=" * 40)
    
    # Create storage structure
    create_storage_folders()
    
    # Copy data to organized storage
    copy_raw_data_to_storage()
    
    # Create manifest
    manifest = create_storage_manifest()
    
    print(f"\nStorage completed!")
    print(f"Organized {len(manifest['files'])} files")
    print("Data is now stored in organized structure for easy access")
    
    print("=" * 40)
    print("Raw data storage completed!")

if __name__ == "__main__":
    main()
