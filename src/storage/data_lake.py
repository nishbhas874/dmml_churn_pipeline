"""
Data Lake Storage System for Churn Prediction Pipeline.
Supports local filesystem, AWS S3, and Google Cloud Storage with partitioning.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import yaml
from typing import Optional, Dict, List, Union, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Optional imports for cloud storage
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.warning("AWS SDK not available. Install with: pip install boto3")

try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("Google Cloud Storage not available. Install with: pip install google-cloud-storage")


class DataLakeStorage:
    """
    Data Lake Storage System with support for multiple storage backends.
    Implements efficient folder/bucket structure with partitioning.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data lake storage system."""
        self.config = self._load_config(config_path)
        self.storage_config = self.config.get('storage', {})
        self.storage_type = self.storage_config.get('type', 'local')
        
        # Initialize storage backend
        if self.storage_type == 'aws_s3':
            self._init_aws_s3()
        elif self.storage_type == 'gcs':
            self._init_gcs()
        else:
            self._init_local()
        
        # Create base directory structure
        self._create_directory_structure()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def _init_local(self):
        """Initialize local filesystem storage."""
        self.base_path = Path(self.storage_config.get('local', {}).get('base_path', 'data_lake'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized local storage at: {self.base_path}")
    
    def _init_aws_s3(self):
        """Initialize AWS S3 storage."""
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK not available. Install with: pip install boto3")
        
        s3_config = self.storage_config.get('aws_s3', {})
        self.bucket_name = s3_config.get('bucket_name')
        self.aws_region = s3_config.get('region', 'us-east-1')
        
        if not self.bucket_name:
            raise ValueError("AWS S3 bucket name not configured")
        
        try:
            self.s3_client = boto3.client('s3', region_name=self.aws_region)
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Initialized AWS S3 storage with bucket: {self.bucket_name}")
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to initialize AWS S3: {e}")
            raise
    
    def _init_gcs(self):
        """Initialize Google Cloud Storage."""
        if not GCS_AVAILABLE:
            raise ImportError("Google Cloud Storage not available. Install with: pip install google-cloud-storage")
        
        gcs_config = self.storage_config.get('gcs', {})
        self.bucket_name = gcs_config.get('bucket_name')
        
        if not self.bucket_name:
            raise ValueError("GCS bucket name not configured")
        
        try:
            self.gcs_client = storage.Client()
            self.bucket = self.gcs_client.bucket(self.bucket_name)
            # Test connection
            self.bucket.reload()
            logger.info(f"Initialized GCS storage with bucket: {self.bucket_name}")
        except DefaultCredentialsError as e:
            logger.error(f"Failed to initialize GCS: {e}")
            raise
    
    def _create_directory_structure(self):
        """Create the base directory structure for data lake."""
        if self.storage_type == 'local':
            # Create local directory structure
            directories = [
                'raw',
                'raw/kaggle',
                'raw/huggingface',
                'raw/external',
                'processed',
                'processed/features',
                'processed/models',
                'logs',
                'metadata'
            ]
            
            for directory in directories:
                (self.base_path / directory).mkdir(parents=True, exist_ok=True)
            
            logger.info("Created local directory structure")
    
    def _get_partition_path(self, source: str, data_type: str, timestamp: datetime = None) -> str:
        """
        Generate partitioned path based on source, type, and timestamp.
        
        Args:
            source (str): Data source (kaggle, huggingface, external)
            data_type (str): Type of data (raw, processed, features, models)
            timestamp (datetime): Timestamp for partitioning
            
        Returns:
            str: Partitioned path
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create partition structure: year/month/day/hour
        partition = timestamp.strftime("%Y/%m/%d/%H")
        
        # Build path: raw/source/partition/
        path = f"{data_type}/{source}/{partition}"
        
        return path
    
    def _get_metadata_path(self, source: str, data_type: str, timestamp: datetime = None) -> str:
        """Generate metadata file path."""
        if timestamp is None:
            timestamp = datetime.now()
        
        partition = timestamp.strftime("%Y/%m/%d/%H")
        return f"metadata/{data_type}/{source}/{partition}"
    
    def upload_file(self, file_path: Union[str, Path], source: str, 
                   data_type: str = 'raw', timestamp: datetime = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a file to the data lake with proper partitioning.
        
        Args:
            file_path: Path to the file to upload
            source: Data source (kaggle, huggingface, external)
            data_type: Type of data (raw, processed, features, models)
            timestamp: Timestamp for partitioning
            metadata: Additional metadata to store
            
        Returns:
            str: Storage path of the uploaded file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate partitioned path
        partition_path = self._get_partition_path(source, data_type, timestamp)
        storage_path = f"{partition_path}/{file_path.name}"
        
        # Prepare metadata
        file_metadata = {
            'source': source,
            'data_type': data_type,
            'original_filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'upload_timestamp': timestamp.isoformat(),
            'partition_path': partition_path,
            'storage_path': storage_path,
            'file_extension': file_path.suffix,
            'checksum': self._calculate_checksum(file_path)
        }
        
        if metadata:
            file_metadata.update(metadata)
        
        try:
            if self.storage_type == 'local':
                self._upload_to_local(file_path, storage_path, file_metadata)
            elif self.storage_type == 'aws_s3':
                self._upload_to_s3(file_path, storage_path, file_metadata)
            elif self.storage_type == 'gcs':
                self._upload_to_gcs(file_path, storage_path, file_metadata)
            
            logger.info(f"Successfully uploaded {file_path.name} to {storage_path}")
            return storage_path
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path.name}: {e}")
            raise
    
    def _upload_to_local(self, file_path: Path, storage_path: str, metadata: Dict[str, Any]):
        """Upload file to local filesystem."""
        target_path = self.base_path / storage_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, target_path)
        
        # Save metadata
        metadata_path = self.base_path / self._get_metadata_path(
            metadata['source'], metadata['data_type']
        ) / f"{file_path.stem}_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _upload_to_s3(self, file_path: Path, storage_path: str, metadata: Dict[str, Any]):
        """Upload file to AWS S3."""
        # Upload file
        self.s3_client.upload_file(
            str(file_path),
            self.bucket_name,
            storage_path,
            ExtraArgs={'Metadata': {k: str(v) for k, v in metadata.items()}}
        )
        
        # Upload metadata
        metadata_path = f"{self._get_metadata_path(metadata['source'], metadata['data_type'])}/{file_path.stem}_metadata.json"
        metadata_json = json.dumps(metadata, indent=2)
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=metadata_path,
            Body=metadata_json,
            ContentType='application/json'
        )
    
    def _upload_to_gcs(self, file_path: Path, storage_path: str, metadata: Dict[str, Any]):
        """Upload file to Google Cloud Storage."""
        # Upload file
        blob = self.bucket.blob(storage_path)
        blob.upload_from_filename(str(file_path))
        
        # Set metadata
        blob.metadata = {k: str(v) for k, v in metadata.items()}
        blob.patch()
        
        # Upload metadata
        metadata_path = f"{self._get_metadata_path(metadata['source'], metadata['data_type'])}/{file_path.stem}_metadata.json"
        metadata_blob = self.bucket.blob(metadata_path)
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type='application/json'
        )
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum for integrity verification."""
        import hashlib
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def list_files(self, source: Optional[str] = None, data_type: str = 'raw', 
                  partition: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List files in the data lake with optional filtering.
        
        Args:
            source: Filter by data source
            data_type: Filter by data type
            partition: Filter by partition (YYYY/MM/DD/HH format)
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        try:
            if self.storage_type == 'local':
                files = self._list_local_files(source, data_type, partition)
            elif self.storage_type == 'aws_s3':
                files = self._list_s3_files(source, data_type, partition)
            elif self.storage_type == 'gcs':
                files = self._list_gcs_files(source, data_type, partition)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def _list_local_files(self, source: Optional[str], data_type: str, partition: Optional[str]) -> List[Dict[str, Any]]:
        """List files in local storage."""
        files = []
        base_dir = self.base_path / data_type
        
        if not base_dir.exists():
            return files
        
        # Build search pattern
        if source:
            base_dir = base_dir / source
        if partition:
            base_dir = base_dir / partition
        
        if not base_dir.exists():
            return files
        
        # Recursively find all files
        for file_path in base_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith('_metadata.json'):
                # Load metadata if available
                metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                files.append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(self.base_path)),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'metadata': metadata
                })
        
        return files
    
    def _list_s3_files(self, source: Optional[str], data_type: str, partition: Optional[str]) -> List[Dict[str, Any]]:
        """List files in S3 storage."""
        files = []
        prefix = f"{data_type}/"
        
        if source:
            prefix += f"{source}/"
        if partition:
            prefix += f"{partition}/"
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if not obj['Key'].endswith('_metadata.json'):
                            # Get metadata
                            try:
                                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=obj['Key'])
                                metadata = response.get('Metadata', {})
                            except:
                                metadata = {}
                            
                            files.append({
                                'name': Path(obj['Key']).name,
                                'path': obj['Key'],
                                'size': obj['Size'],
                                'modified': obj['LastModified'].isoformat(),
                                'metadata': metadata
                            })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            return []
    
    def _list_gcs_files(self, source: Optional[str], data_type: str, partition: Optional[str]) -> List[Dict[str, Any]]:
        """List files in GCS storage."""
        files = []
        prefix = f"{data_type}/"
        
        if source:
            prefix += f"{source}/"
        if partition:
            prefix += f"{partition}/"
        
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if not blob.name.endswith('_metadata.json'):
                    files.append({
                        'name': Path(blob.name).name,
                        'path': blob.name,
                        'size': blob.size,
                        'modified': blob.updated.isoformat(),
                        'metadata': blob.metadata or {}
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list GCS files: {e}")
            return []
    
    def download_file(self, storage_path: str, local_path: Union[str, Path]) -> bool:
        """
        Download a file from the data lake.
        
        Args:
            storage_path: Path in the data lake
            local_path: Local path to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.storage_type == 'local':
                source_path = self.base_path / storage_path
                if source_path.exists():
                    shutil.copy2(source_path, local_path)
                    return True
            
            elif self.storage_type == 'aws_s3':
                self.s3_client.download_file(self.bucket_name, storage_path, str(local_path))
                return True
            
            elif self.storage_type == 'gcs':
                blob = self.bucket.blob(storage_path)
                blob.download_to_filename(str(local_path))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to download {storage_path}: {e}")
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage system."""
        info = {
            'storage_type': self.storage_type,
            'base_path': str(self.base_path) if self.storage_type == 'local' else None,
            'bucket_name': getattr(self, 'bucket_name', None),
            'region': getattr(self, 'aws_region', None) if self.storage_type == 'aws_s3' else None,
            'available': True
        }
        
        # Test connectivity
        try:
            if self.storage_type == 'aws_s3':
                self.s3_client.head_bucket(Bucket=self.bucket_name)
            elif self.storage_type == 'gcs':
                self.bucket.reload()
        except Exception as e:
            info['available'] = False
            info['error'] = str(e)
        
        return info


def create_data_lake_storage(config_path: str = "config/config.yaml") -> DataLakeStorage:
    """Factory function to create data lake storage instance."""
    return DataLakeStorage(config_path)


if __name__ == "__main__":
    # Example usage
    storage = create_data_lake_storage()
    print("Storage info:", storage.get_storage_info())
