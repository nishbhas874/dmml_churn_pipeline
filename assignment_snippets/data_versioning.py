"""
Data Versioning Code for Assignment
This code demonstrates comprehensive data versioning using DVC for churn prediction datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml
import subprocess
import os
import hashlib
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class DataVersioning:
    """Comprehensive data versioning system using DVC for churn prediction datasets."""
    
    def __init__(self, project_root: str = ".", dvc_remote: str = "local"):
        """Initialize data versioning system."""
        self.project_root = Path(project_root)
        self.dvc_remote = dvc_remote
        self.version_metadata = {}
        
        # Ensure DVC is initialized
        self._initialize_dvc()
        
        # Create data directories
        self.data_dirs = {
            'raw': self.project_root / 'data' / 'raw',
            'processed': self.project_root / 'data' / 'processed',
            'transformed': self.project_root / 'data' / 'transformed',
            'features': self.project_root / 'data' / 'features'
        }
        
        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _initialize_dvc(self):
        """Initialize DVC in the project if not already initialized."""
        if not (self.project_root / '.dvc').exists():
            print("🔧 Initializing DVC...")
            subprocess.run(['dvc', 'init'], cwd=self.project_root, check=True)
            
            # Add .dvc to .gitignore if not exists
            gitignore_path = self.project_root / '.gitignore'
            if not gitignore_path.exists():
                gitignore_path.touch()
            
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            if '.dvc' not in content:
                with open(gitignore_path, 'a') as f:
                    f.write('\n# DVC\n.dvc/\n')
            
            print("✅ DVC initialized successfully")
    
    def _run_dvc_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run DVC command and return result."""
        try:
            result = subprocess.run(['dvc'] + command, 
                                  cwd=self.project_root, 
                                  capture_output=True, 
                                  text=True, 
                                  check=check)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ DVC command failed: {' '.join(command)}")
            print(f"Error: {e.stderr}")
            raise
    
    def create_dataset_version(self, dataset_path: str, version_tag: str, 
                              metadata: Dict[str, Any] = None) -> str:
        """Create a new version of a dataset with DVC."""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Generate version metadata
        timestamp = datetime.now().isoformat()
        file_hash = self._calculate_file_hash(dataset_path)
        
        version_info = {
            'version_tag': version_tag,
            'timestamp': timestamp,
            'file_hash': file_hash,
            'file_size': dataset_path.stat().st_size,
            'dataset_path': str(dataset_path),
            'metadata': metadata or {}
        }
        
        # Add dataset to DVC
        print(f"📦 Adding dataset to DVC: {dataset_path}")
        self._run_dvc_command(['add', str(dataset_path)])
        
        # Commit to DVC
        self._run_dvc_command(['commit', '-m', f'Add dataset version {version_tag}'])
        
        # Create Git tag for this version
        tag_name = f"dataset-{version_tag}"
        self._run_dvc_command(['tag', tag_name, '-m', f'Dataset version {version_tag}'])
        
        # Store version metadata
        self.version_metadata[version_tag] = version_info
        
        # Save metadata to file
        self._save_version_metadata()
        
        print(f"✅ Created dataset version: {version_tag}")
        return version_tag
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _save_version_metadata(self):
        """Save version metadata to JSON file."""
        metadata_path = self.project_root / 'data' / 'version_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.version_metadata, f, indent=2, default=str)
    
    def list_dataset_versions(self) -> Dict[str, Any]:
        """List all dataset versions."""
        try:
            # Get DVC tags
            result = self._run_dvc_command(['tag', 'list'], check=False)
            tags = []
            if result.returncode == 0:
                tags = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
            # Load metadata
            metadata_path = self.project_root / 'data' / 'version_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.version_metadata = json.load(f)
            
            return {
                'versions': self.version_metadata,
                'tags': tags,
                'total_versions': len(self.version_metadata)
            }
        except Exception as e:
            print(f"⚠️  Error listing versions: {e}")
            return {'versions': {}, 'tags': [], 'total_versions': 0}
    
    def checkout_dataset_version(self, version_tag: str) -> bool:
        """Checkout a specific dataset version."""
        try:
            tag_name = f"dataset-{version_tag}"
            print(f"🔄 Checking out dataset version: {version_tag}")
            
            # Checkout the tag
            self._run_dvc_command(['checkout', tag_name])
            
            print(f"✅ Successfully checked out version: {version_tag}")
            return True
        except Exception as e:
            print(f"❌ Error checking out version {version_tag}: {e}")
            return False
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two dataset versions."""
        try:
            # Get metadata for both versions
            v1_meta = self.version_metadata.get(version1, {})
            v2_meta = self.version_metadata.get(version2, {})
            
            if not v1_meta or not v2_meta:
                raise ValueError(f"One or both versions not found: {version1}, {version2}")
            
            comparison = {
                'version1': {
                    'tag': version1,
                    'timestamp': v1_meta.get('timestamp'),
                    'file_hash': v1_meta.get('file_hash'),
                    'file_size': v1_meta.get('file_size')
                },
                'version2': {
                    'tag': version2,
                    'timestamp': v2_meta.get('timestamp'),
                    'file_hash': v2_meta.get('file_hash'),
                    'file_size': v2_meta.get('file_size')
                },
                'changes': {
                    'file_size_diff': v2_meta.get('file_size', 0) - v1_meta.get('file_size', 0),
                    'hash_changed': v1_meta.get('file_hash') != v2_meta.get('file_hash'),
                    'time_diff_hours': self._calculate_time_diff(
                        v1_meta.get('timestamp'), v2_meta.get('timestamp')
                    )
                }
            }
            
            return comparison
        except Exception as e:
            print(f"❌ Error comparing versions: {e}")
            return {}
    
    def _calculate_time_diff(self, timestamp1: str, timestamp2: str) -> float:
        """Calculate time difference between two timestamps in hours."""
        try:
            dt1 = datetime.fromisoformat(timestamp1)
            dt2 = datetime.fromisoformat(timestamp2)
            diff = abs((dt2 - dt1).total_seconds() / 3600)
            return round(diff, 2)
        except:
            return 0.0
    
    def create_version_report(self, output_path: str = None) -> str:
        """Create a comprehensive version report."""
        if output_path is None:
            output_path = self.project_root / 'data' / 'version_report.md'
        
        versions_info = self.list_dataset_versions()
        
        report = "# Dataset Version Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"Total Versions: {versions_info['total_versions']}\n\n"
        
        # Version details
        report += "## Version Details\n\n"
        for version_tag, metadata in versions_info['versions'].items():
            report += f"### Version {version_tag}\n\n"
            report += f"- **Timestamp:** {metadata.get('timestamp', 'N/A')}\n"
            report += f"- **File Hash:** {metadata.get('file_hash', 'N/A')[:16]}...\n"
            report += f"- **File Size:** {metadata.get('file_size', 0):,} bytes\n"
            report += f"- **Dataset Path:** {metadata.get('dataset_path', 'N/A')}\n"
            
            # Additional metadata
            if metadata.get('metadata'):
                report += f"- **Additional Info:** {metadata['metadata']}\n"
            
            report += "\n"
        
        # DVC tags
        report += "## DVC Tags\n\n"
        for tag in versions_info['tags']:
            report += f"- `{tag}`\n"
        
        report += "\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Version report saved to: {output_path}")
        return str(output_path)
    
    def setup_remote_storage(self, remote_url: str = None, remote_name: str = "origin"):
        """Setup remote storage for DVC."""
        if remote_url is None:
            # Use local storage
            remote_url = str(self.project_root / 'dvc_remote')
            Path(remote_url).mkdir(exist_ok=True)
        
        try:
            # Add remote
            self._run_dvc_command(['remote', 'add', remote_name, remote_url])
            
            # Set default remote
            self._run_dvc_command(['remote', 'default', remote_name])
            
            print(f"✅ Remote storage configured: {remote_name} -> {remote_url}")
        except Exception as e:
            print(f"⚠️  Remote already configured or error: {e}")
    
    def push_to_remote(self, remote_name: str = "origin"):
        """Push dataset versions to remote storage."""
        try:
            print(f"📤 Pushing to remote: {remote_name}")
            self._run_dvc_command(['push', '-r', remote_name])
            print(f"✅ Successfully pushed to remote: {remote_name}")
        except Exception as e:
            print(f"❌ Error pushing to remote: {e}")
    
    def pull_from_remote(self, remote_name: str = "origin"):
        """Pull dataset versions from remote storage."""
        try:
            print(f"📥 Pulling from remote: {remote_name}")
            self._run_dvc_command(['pull', '-r', remote_name])
            print(f"✅ Successfully pulled from remote: {remote_name}")
        except Exception as e:
            print(f"❌ Error pulling from remote: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize data versioning
    versioning = DataVersioning()
    
    # Create sample datasets
    np.random.seed(42)
    n_samples = 1000
    
    # Sample raw data
    raw_data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'tenure': np.random.exponential(30, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Save raw data
    raw_path = versioning.data_dirs['raw'] / 'customer_data_v1.csv'
    raw_data.to_csv(raw_path, index=False)
    
    # Create version 1.0
    print("🔧 Creating dataset version 1.0...")
    versioning.create_dataset_version(
        dataset_path=str(raw_path),
        version_tag="1.0",
        metadata={
            'description': 'Initial customer dataset',
            'source': 'synthetic_data',
            'rows': len(raw_data),
            'columns': len(raw_data.columns)
        }
    )
    
    # Create processed data (version 2.0)
    processed_data = raw_data.copy()
    processed_data['age_group'] = pd.cut(processed_data['age'], 
                                        bins=[0, 30, 50, 70, 100], 
                                        labels=['Young', 'Adult', 'Senior', 'Elderly'])
    processed_data['high_value'] = (processed_data['monthly_charges'] > 80).astype(int)
    
    processed_path = versioning.data_dirs['processed'] / 'customer_data_v2.csv'
    processed_data.to_csv(processed_path, index=False)
    
    # Create version 2.0
    print("\n🔧 Creating dataset version 2.0...")
    versioning.create_dataset_version(
        dataset_path=str(processed_path),
        version_tag="2.0",
        metadata={
            'description': 'Processed customer dataset with derived features',
            'source': 'processed_from_v1',
            'rows': len(processed_data),
            'columns': len(processed_data.columns),
            'new_features': ['age_group', 'high_value']
        }
    )
    
    # Create transformed data (version 3.0)
    from sklearn.preprocessing import StandardScaler
    
    # Select numeric features for scaling
    numeric_features = ['age', 'tenure', 'monthly_charges', 'total_charges']
    scaler = StandardScaler()
    
    transformed_data = processed_data.copy()
    transformed_data[numeric_features] = scaler.fit_transform(processed_data[numeric_features])
    
    transformed_path = versioning.data_dirs['transformed'] / 'customer_data_v3.csv'
    transformed_data.to_csv(transformed_path, index=False)
    
    # Create version 3.0
    print("\n🔧 Creating dataset version 3.0...")
    versioning.create_dataset_version(
        dataset_path=str(transformed_path),
        version_tag="3.0",
        metadata={
            'description': 'Transformed customer dataset with scaled features',
            'source': 'transformed_from_v2',
            'rows': len(transformed_data),
            'columns': len(transformed_data.columns),
            'scaled_features': numeric_features,
            'scaler_type': 'StandardScaler'
        }
    )
    
    # List all versions
    print("\n📋 Listing all dataset versions...")
    versions = versioning.list_dataset_versions()
    print(f"Total versions: {versions['total_versions']}")
    for version_tag, metadata in versions['versions'].items():
        print(f"  • Version {version_tag}: {metadata.get('timestamp', 'N/A')}")
    
    # Compare versions
    print("\n🔍 Comparing versions 1.0 and 3.0...")
    comparison = versioning.compare_versions("1.0", "3.0")
    if comparison:
        print(f"File size difference: {comparison['changes']['file_size_diff']:,} bytes")
        print(f"Hash changed: {comparison['changes']['hash_changed']}")
        print(f"Time difference: {comparison['changes']['time_diff_hours']} hours")
    
    # Create version report
    print("\n📄 Creating version report...")
    report_path = versioning.create_version_report()
    
    # Setup remote storage (local for demo)
    print("\n☁️  Setting up remote storage...")
    versioning.setup_remote_storage()
    
    # Push to remote
    print("\n📤 Pushing to remote storage...")
    versioning.push_to_remote()
    
    print(f"\n✅ Data versioning demonstration completed!")
    print(f"   Versions created: {versions['total_versions']}")
    print(f"   Version report: {report_path}")
    print(f"   DVC tags: {len(versions['tags'])}")
