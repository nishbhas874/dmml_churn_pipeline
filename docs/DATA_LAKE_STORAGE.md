# Data Lake Storage System Documentation

## Overview

The Data Lake Storage System provides a comprehensive solution for storing and managing raw data in a churn prediction pipeline. It supports multiple storage backends (local filesystem, AWS S3, Google Cloud Storage) with efficient partitioning and metadata management.

## Features

- **Multi-backend Support**: Local filesystem, AWS S3, Google Cloud Storage
- **Intelligent Partitioning**: Time-based partitioning (year/month/day/hour)
- **Metadata Management**: Automatic metadata generation and storage
- **Data Integrity**: Checksum verification for file integrity
- **Flexible Filtering**: Query files by source, type, and partition
- **Scalable Architecture**: Designed for large-scale data operations

## Folder/Bucket Structure

### Local Storage Structure

```
data_lake/
├── raw/                          # Raw data storage
│   ├── kaggle/                   # Kaggle datasets
│   │   └── YYYY/MM/DD/HH/       # Time-based partitioning
│   │       └── dataset.csv
│   ├── huggingface/              # Hugging Face datasets
│   │   └── YYYY/MM/DD/HH/
│   │       └── dataset.csv
│   └── external/                 # External data sources
│       └── YYYY/MM/DD/HH/
│           └── dataset.csv
├── processed/                    # Processed data
│   ├── features/                 # Feature engineering outputs
│   └── models/                   # Model artifacts
├── metadata/                     # Metadata storage
│   └── raw/
│       ├── kaggle/
│       ├── huggingface/
│       └── external/
│           └── YYYY/MM/DD/HH/
│               └── dataset_metadata.json
└── logs/                         # System logs
```

### Cloud Storage Structure (AWS S3 / GCS)

```
bucket-name/
├── raw/
│   ├── kaggle/YYYY/MM/DD/HH/dataset.csv
│   ├── huggingface/YYYY/MM/DD/HH/dataset.csv
│   └── external/YYYY/MM/DD/HH/dataset.csv
├── processed/
│   ├── features/
│   └── models/
├── metadata/
│   └── raw/
│       ├── kaggle/YYYY/MM/DD/HH/dataset_metadata.json
│       ├── huggingface/YYYY/MM/DD/HH/dataset_metadata.json
│       └── external/YYYY/MM/DD/HH/dataset_metadata.json
└── logs/
```

## Partitioning Strategy

### Time-based Partitioning

The system uses a hierarchical time-based partitioning strategy:

- **Year**: YYYY (e.g., 2025)
- **Month**: MM (e.g., 08)
- **Day**: DD (e.g., 21)
- **Hour**: HH (e.g., 23)

**Example Path**: `raw/kaggle/2025/08/21/23/dataset.csv`

### Benefits of Partitioning

1. **Query Performance**: Efficient filtering by time ranges
2. **Data Organization**: Logical grouping of data by ingestion time
3. **Scalability**: Easy to manage large datasets
4. **Retention Policies**: Simple implementation of data retention rules

## Configuration

### Local Storage Configuration

```yaml
storage:
  type: "local"
  local:
    base_path: "data_lake"
```

### AWS S3 Configuration

```yaml
storage:
  type: "aws_s3"
  aws_s3:
    bucket_name: "your-churn-data-lake"
    region: "us-east-1"
```

**Prerequisites**:
- AWS CLI configured with appropriate credentials
- S3 bucket created with proper permissions

### Google Cloud Storage Configuration

```yaml
storage:
  type: "gcs"
  gcs:
    bucket_name: "your-churn-data-lake"
```

**Prerequisites**:
- Google Cloud SDK installed and configured
- GCS bucket created with proper permissions

## Usage Examples

### Basic Usage

```python
from src.storage.data_lake import create_data_lake_storage

# Initialize storage
storage = create_data_lake_storage()

# Upload a file
storage_path = storage.upload_file(
    file_path="data.csv",
    source="kaggle",
    data_type="raw",
    metadata={
        "dataset_name": "telco_churn",
        "rows": 1000,
        "columns": 20,
        "description": "Telecom customer churn dataset"
    }
)

print(f"File uploaded to: {storage_path}")
```

### Listing Files

```python
# List all files
all_files = storage.list_files()

# Filter by source
kaggle_files = storage.list_files(source="kaggle")

# Filter by partition
today_files = storage.list_files(partition="2025/08/21")

# Filter by source and partition
kaggle_today = storage.list_files(
    source="kaggle", 
    partition="2025/08/21"
)
```

### Downloading Files

```python
# Download a file from storage
success = storage.download_file(
    storage_path="raw/kaggle/2025/08/21/23/dataset.csv",
    local_path="downloaded_data.csv"
)

if success:
    print("File downloaded successfully")
```

### Integration with Data Ingestion

```python
from src.ingestion.unified_ingestion import ingest_all_data
from src.storage.data_lake import create_data_lake_storage

# Run data ingestion
ingested_files = ingest_all_data()

# Initialize storage
storage = create_data_lake_storage()

# Upload ingested files to data lake
for source, file_path in ingested_files.items():
    if file_path:
        storage_path = storage.upload_file(
            file_path=file_path,
            source=source,
            data_type="raw",
            metadata={
                "ingestion_source": source,
                "ingestion_timestamp": datetime.now().isoformat()
            }
        )
        print(f"Uploaded {source} data to: {storage_path}")
```

## Metadata Schema

Each file in the data lake has associated metadata stored in JSON format:

```json
{
  "source": "kaggle",
  "data_type": "raw",
  "original_filename": "dataset.csv",
  "file_size": 1800,
  "upload_timestamp": "2025-08-21T23:14:08.545961",
  "partition_path": "raw/kaggle/2025/08/21/23",
  "storage_path": "raw/kaggle/2025/08/21/23/dataset.csv",
  "file_extension": ".csv",
  "checksum": "62b29bea89222663afc1a5fb2ddee6cb",
  "dataset_name": "telco_churn",
  "rows": 1000,
  "columns": 20,
  "description": "Telecom customer churn dataset",
  "version": "1.0"
}
```

### Metadata Fields

- **source**: Data source (kaggle, huggingface, external)
- **data_type**: Type of data (raw, processed, features, models)
- **original_filename**: Original file name
- **file_size**: File size in bytes
- **upload_timestamp**: ISO timestamp of upload
- **partition_path**: Partition path without filename
- **storage_path**: Complete storage path including filename
- **file_extension**: File extension
- **checksum**: MD5 checksum for integrity verification
- **dataset_name**: Custom dataset identifier
- **rows**: Number of rows in the dataset
- **columns**: Number of columns in the dataset
- **description**: Human-readable description
- **version**: Dataset version

## Best Practices

### 1. Consistent Naming

- Use descriptive file names
- Include version information in metadata
- Follow consistent naming conventions

### 2. Metadata Management

- Always provide meaningful metadata
- Include data lineage information
- Document data quality metrics

### 3. Partitioning Strategy

- Choose appropriate partition granularity
- Consider query patterns when designing partitions
- Balance partition size with query performance

### 4. Data Integrity

- Verify checksums after upload
- Implement data validation procedures
- Monitor storage health regularly

### 5. Security

- Use appropriate access controls
- Encrypt sensitive data
- Implement audit logging

## Monitoring and Maintenance

### Storage Health Checks

```python
# Check storage connectivity
storage_info = storage.get_storage_info()
print(f"Storage available: {storage_info['available']}")

# List storage statistics
files = storage.list_files()
total_size = sum(f['size'] for f in files)
print(f"Total files: {len(files)}")
print(f"Total size: {total_size} bytes")
```

### Data Retention Policies

```python
# Example: Delete files older than 30 days
from datetime import datetime, timedelta

cutoff_date = datetime.now() - timedelta(days=30)
old_files = storage.list_files()

for file_info in old_files:
    file_date = datetime.fromisoformat(file_info['modified'])
    if file_date < cutoff_date:
        # Implement deletion logic
        print(f"Marking for deletion: {file_info['path']}")
```

## Troubleshooting

### Common Issues

1. **Storage Connection Failed**
   - Check credentials and permissions
   - Verify network connectivity
   - Ensure storage backend is available

2. **File Upload Failed**
   - Check file permissions
   - Verify sufficient storage space
   - Review error logs for details

3. **Metadata Issues**
   - Ensure metadata is valid JSON
   - Check file paths are correct
   - Verify metadata file permissions

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Initialize storage with debug output
storage = create_data_lake_storage()
```

## Performance Considerations

### Local Storage

- Use SSD storage for better performance
- Monitor disk space usage
- Implement regular cleanup procedures

### Cloud Storage

- Use appropriate storage classes
- Implement caching strategies
- Monitor bandwidth usage

### General Optimization

- Batch operations when possible
- Use appropriate chunk sizes for large files
- Implement parallel processing for bulk operations

## Future Enhancements

1. **Compression Support**: Automatic file compression
2. **Data Versioning**: Version control for datasets
3. **Advanced Queries**: SQL-like query interface
4. **Data Catalog**: Centralized metadata management
5. **Workflow Integration**: Integration with data pipelines
6. **Real-time Streaming**: Support for streaming data ingestion

## Conclusion

The Data Lake Storage System provides a robust foundation for managing data in the churn prediction pipeline. Its flexible architecture supports multiple storage backends while maintaining data integrity and providing efficient querying capabilities.

For more information, refer to the source code in `src/storage/data_lake.py` and the demonstration script `demo_data_lake.py`.
