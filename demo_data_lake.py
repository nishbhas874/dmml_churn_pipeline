#!/usr/bin/env python3
"""
Real-time demonstration of the Data Lake Storage System.
Shows live output of file uploads, partitioning, and metadata creation.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from storage.data_lake import create_data_lake_storage
from loguru import logger

def create_sample_data():
    """Create sample data files for demonstration."""
    print("📊 Creating sample data files...")
    
    # Create sample CSV data
    sample_data = {
        'customer_id': range(1, 101),
        'age': [25 + i % 50 for i in range(100)],
        'gender': ['M' if i % 2 == 0 else 'F' for i in range(100)],
        'tenure': [i % 60 for i in range(100)],
        'monthly_charges': [50 + i * 2 for i in range(100)],
        'churn': [1 if i % 10 == 0 else 0 for i in range(100)]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create different sample files
    files = {
        'kaggle_sample.csv': df,
        'huggingface_sample.csv': df.sample(frac=0.8),
        'external_sample.csv': df.sample(frac=0.6)
    }
    
    # Save files
    for filename, data in files.items():
        filepath = Path(f"temp_{filename}")
        data.to_csv(filepath, index=False)
        print(f"  ✅ Created: {filepath}")
    
    return files

def demonstrate_storage_operations():
    """Demonstrate real-time storage operations."""
    print("\n" + "="*80)
    print("🚀 DATA LAKE STORAGE DEMONSTRATION")
    print("="*80)
    
    # Initialize storage
    print("\n🔧 Initializing Data Lake Storage...")
    try:
        storage = create_data_lake_storage()
        storage_info = storage.get_storage_info()
        print(f"  ✅ Storage Type: {storage_info['storage_type']}")
        print(f"  ✅ Base Path: {storage_info['base_path']}")
        print(f"  ✅ Available: {storage_info['available']}")
    except Exception as e:
        print(f"  ❌ Failed to initialize storage: {e}")
        return
    
    # Create sample data
    sample_files = create_sample_data()
    
    print("\n📤 UPLOADING FILES TO DATA LAKE")
    print("-" * 50)
    
    uploaded_files = []
    
    # Upload files with different sources and timestamps
    sources = ['kaggle', 'huggingface', 'external']
    
    for i, (filename, df) in enumerate(sample_files.items()):
        source = sources[i]
        filepath = Path(f"temp_{filename}")
        
        print(f"\n📁 Uploading {filename} from {source}...")
        
        # Add some delay to show different timestamps
        time.sleep(1)
        
        try:
            # Upload with metadata
            metadata = {
                'dataset_name': filename.replace('_sample.csv', ''),
                'rows': len(df),
                'columns': len(df.columns),
                'description': f'Sample churn data from {source}',
                'version': '1.0'
            }
            
            storage_path = storage.upload_file(
                file_path=filepath,
                source=source,
                data_type='raw',
                metadata=metadata
            )
            
            uploaded_files.append({
                'original_file': filename,
                'storage_path': storage_path,
                'source': source
            })
            
            print(f"  ✅ Uploaded to: {storage_path}")
            print(f"  📊 Metadata: {json.dumps(metadata, indent=4)}")
            
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
    
    print("\n📋 LISTING FILES IN DATA LAKE")
    print("-" * 50)
    
    # List all files
    all_files = storage.list_files()
    print(f"📁 Total files in data lake: {len(all_files)}")
    
    for file_info in all_files:
        print(f"\n  📄 File: {file_info['name']}")
        print(f"     Path: {file_info['path']}")
        print(f"     Size: {file_info['size']} bytes")
        print(f"     Modified: {file_info['modified']}")
        if file_info['metadata']:
            print(f"     Source: {file_info['metadata'].get('source', 'N/A')}")
            print(f"     Data Type: {file_info['metadata'].get('data_type', 'N/A')}")
    
    print("\n🔍 FILTERING BY SOURCE")
    print("-" * 50)
    
    # Filter by source
    for source in sources:
        source_files = storage.list_files(source=source)
        print(f"\n📂 Files from {source}: {len(source_files)}")
        for file_info in source_files:
            print(f"  - {file_info['name']} ({file_info['path']})")
    
    print("\n📊 STORAGE STRUCTURE")
    print("-" * 50)
    
    # Show directory structure
    if storage.storage_type == 'local':
        print("📁 Local Data Lake Structure:")
        storage.base_path.mkdir(parents=True, exist_ok=True)
        
        def print_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return
            
            items = sorted(path.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                if item.is_dir():
                    print(f"{prefix}{current_prefix}{item.name}/")
                    print_tree(item, prefix + next_prefix, max_depth, current_depth + 1)
                else:
                    size = item.stat().st_size
                    print(f"{prefix}{current_prefix}{item.name} ({size} bytes)")
        
        print_tree(storage.base_path)
    
    print("\n🧹 CLEANING UP")
    print("-" * 50)
    
    # Clean up temporary files
    for filename in sample_files.keys():
        filepath = Path(f"temp_{filename}")
        if filepath.exists():
            filepath.unlink()
            print(f"  🗑️  Deleted: {filepath}")
    
    print("\n✅ DEMONSTRATION COMPLETED!")
    print("="*80)
    print("\n📝 Summary:")
    print(f"  • Uploaded {len(uploaded_files)} files")
    print(f"  • Used {storage.storage_type} storage")
    print(f"  • Created partitioned structure")
    print(f"  • Generated metadata for each file")
    print(f"  • Demonstrated filtering capabilities")

def show_configuration():
    """Show the current storage configuration."""
    print("\n⚙️  STORAGE CONFIGURATION")
    print("-" * 50)
    
    config_path = "config/config.yaml"
    if Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        storage_config = config.get('storage', {})
        print("Current storage configuration:")
        print(json.dumps(storage_config, indent=2))
    else:
        print("❌ Configuration file not found. Using defaults.")

if __name__ == "__main__":
    print("🎯 Data Lake Storage Real-time Demonstration")
    print("This script will show you exactly what happens when files are uploaded to the data lake.")
    
    # Show configuration
    show_configuration()
    
    # Run demonstration
    demonstrate_storage_operations()
    
    print("\n🎉 All done! Check the data_lake directory to see the results.")
