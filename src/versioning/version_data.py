#!/usr/bin/env python3
"""
Simple Data Versioning - Stage 8
Track changes in our data files to ensure reproducibility.
"""

import os
import json
import hashlib
import pandas as pd
import subprocess
from datetime import datetime
from pathlib import Path

def create_folders():
    """Create necessary folders"""
    Path("data/versioned").mkdir(parents=True, exist_ok=True)
    Path("data/versioned/metadata").mkdir(parents=True, exist_ok=True)

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_data_files():
    """Get all data files from different stages"""
    data_files = {}
    
    # Raw data
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        for file_path in raw_dir.rglob("*.csv"):
            data_files[f"raw/{file_path.name}"] = str(file_path)
    
    # Processed data
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for file_path in processed_dir.glob("*.csv"):
            data_files[f"processed/{file_path.name}"] = str(file_path)
    
    # Transformed data
    transformed_dir = Path("data/transformed")
    if transformed_dir.exists():
        for file_path in transformed_dir.glob("*.csv"):
            data_files[f"transformed/{file_path.name}"] = str(file_path)
    
    # Feature store data
    features_dir = Path("data/features")
    if features_dir.exists():
        for file_path in features_dir.glob("*.csv"):
            data_files[f"features/{file_path.name}"] = str(file_path)
    
    return data_files

def create_data_version():
    """Create a new data version"""
    print("Creating new data version...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_files = get_data_files()
    
    if not data_files:
        print("No data files found to version!")
        return None
    
    # Create version metadata
    version_info = {
        "version_id": f"v_{timestamp}",
        "created_at": datetime.now().isoformat(),
        "total_files": len(data_files),
        "files": {}
    }
    
    print(f"Found {len(data_files)} files to version:")
    
    for file_key, file_path in data_files.items():
        print(f"  - {file_key}")
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        
        # Get file stats
        file_stats = os.stat(file_path)
        
        # Load data to get basic info
        try:
            df = pd.read_csv(file_path)
            rows, cols = df.shape
        except:
            rows, cols = 0, 0
        
        # Store file metadata
        version_info["files"][file_key] = {
            "path": file_path,
            "hash": file_hash,
            "size_bytes": file_stats.st_size,
            "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "rows": rows,
            "columns": cols
        }
    
    return version_info, timestamp

def save_version_metadata(version_info, timestamp):
    """Save version metadata"""
    # Save detailed metadata
    metadata_file = f"data/versioned/metadata/version_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(version_info, f, indent=2)
    print(f"Version metadata saved: {metadata_file}")
    
    # Update version registry
    registry_file = "data/versioned/version_registry.json"
    
    # Load existing registry or create new
    if os.path.exists(registry_file):
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"versions": [], "latest_version": None}
    
    # Add new version
    version_entry = {
        "version_id": version_info["version_id"],
        "created_at": version_info["created_at"],
        "total_files": version_info["total_files"],
        "metadata_file": metadata_file
    }
    
    registry["versions"].append(version_entry)
    registry["latest_version"] = version_info["version_id"]
    registry["total_versions"] = len(registry["versions"])
    
    # Save updated registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"Version registry updated: {registry_file}")
    
    return metadata_file

def compare_versions():
    """Compare current data with previous version"""
    registry_file = "data/versioned/version_registry.json"
    
    if not os.path.exists(registry_file):
        print("No previous versions found for comparison")
        return
    
    with open(registry_file, 'r') as f:
        registry = json.load(f)
    
    if len(registry["versions"]) < 2:
        print("Need at least 2 versions for comparison")
        return
    
    # Get last two versions
    current_version = registry["versions"][-1]
    previous_version = registry["versions"][-2]
    
    print(f"\nComparing versions:")
    print(f"Previous: {previous_version['version_id']} ({previous_version['created_at']})")
    print(f"Current:  {current_version['version_id']} ({current_version['created_at']})")
    print("-" * 50)
    
    # Load metadata for both versions
    with open(current_version["metadata_file"], 'r') as f:
        current_meta = json.load(f)
    
    with open(previous_version["metadata_file"], 'r') as f:
        previous_meta = json.load(f)
    
    # Compare files
    current_files = set(current_meta["files"].keys())
    previous_files = set(previous_meta["files"].keys())
    
    # New files
    new_files = current_files - previous_files
    if new_files:
        print(f"New files ({len(new_files)}):")
        for file in new_files:
            print(f"  + {file}")
    
    # Removed files
    removed_files = previous_files - current_files
    if removed_files:
        print(f"Removed files ({len(removed_files)}):")
        for file in removed_files:
            print(f"  - {file}")
    
    # Modified files
    common_files = current_files & previous_files
    modified_files = []
    
    for file in common_files:
        current_hash = current_meta["files"][file]["hash"]
        previous_hash = previous_meta["files"][file]["hash"]
        
        if current_hash != previous_hash:
            modified_files.append(file)
    
    if modified_files:
        print(f"Modified files ({len(modified_files)}):")
        for file in modified_files:
            current_info = current_meta["files"][file]
            previous_info = previous_meta["files"][file]
            print(f"  ~ {file}")
            print(f"    Rows: {previous_info['rows']} → {current_info['rows']}")
            print(f"    Size: {previous_info['size_bytes']} → {current_info['size_bytes']} bytes")
    
    if not new_files and not removed_files and not modified_files:
        print("No changes detected between versions")

def list_versions():
    """List all available versions"""
    registry_file = "data/versioned/version_registry.json"
    
    if not os.path.exists(registry_file):
        print("No versions found")
        return
    
    with open(registry_file, 'r') as f:
        registry = json.load(f)
    
    print(f"\nAvailable Versions ({registry['total_versions']}):")
    print("-" * 50)
    
    for i, version in enumerate(registry["versions"], 1):
        marker = " (latest)" if version["version_id"] == registry["latest_version"] else ""
        print(f"{i}. {version['version_id']}{marker}")
        print(f"   Created: {version['created_at']}")
        print(f"   Files: {version['total_files']}")
        print()

def create_simple_git_version():
    """Create simple git version for data tracking"""
    try:
        # Initialize git if needed
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            print("Git repository initialized")
        
        # Add all files
        subprocess.run(['git', 'add', '.'], check=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Commit with message
        commit_msg = f"Data version {timestamp} - pipeline run"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        
        # Create tag
        tag_name = f"data-v{timestamp}"
        subprocess.run(['git', 'tag', tag_name], check=True)
        
        print(f"Created Git version: {tag_name}")
        return tag_name
        
    except subprocess.CalledProcessError:
        print("Git versioning failed - continuing with custom versioning")
        return None

def main():
    """Simple data versioning function"""
    print("Data Versioning - Stage 8")
    print("=" * 40)
    
    # Create necessary folders
    create_folders()
    
    # Try Git versioning first (meets DVC/Git requirement)
    git_version = create_simple_git_version()
    
    # Create custom version (backup method)
    version_info, timestamp = create_data_version()
    if version_info is None:
        return
    
    # Save version metadata
    metadata_file = save_version_metadata(version_info, timestamp)
    
    print(f"\nVersion {version_info['version_id']} created!")
    print(f"Tracked {version_info['total_files']} data files")
    
    if git_version:
        print(f"Git version: {git_version}")
    
    # Show simple comparison with previous version
    compare_versions()
    
    print("=" * 40)
    print("Data versioning completed!")
    print("All data changes are now tracked for reproducibility")

if __name__ == "__main__":
    main()
