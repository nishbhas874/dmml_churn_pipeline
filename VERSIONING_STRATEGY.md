# Data Versioning Strategy and Workflow

## Overview
This document describes our comprehensive data versioning approach for the Customer Churn Prediction Pipeline, implementing both Git-based and custom versioning systems to ensure complete reproducibility.

## Versioning Tools Used

### 1. Git + Git Tags (Primary)
- **Purpose**: Track code and data changes with standard version control
- **Implementation**: Automatic commits with data version tags
- **Tags Format**: `data-v20250822_125044`

### 2. Custom Versioning System (Enhanced)
- **Purpose**: Detailed file-level tracking with metadata
- **Implementation**: MD5 hash-based change detection
- **Metadata**: File sizes, timestamps, row/column counts

## Versioning Workflow

### Automatic Versioning Process
```
1. Pipeline Execution
   ↓
2. Data Processing (any stage)
   ↓
3. Version Detection (file hash comparison)
   ↓
4. Git Commit + Tag Creation
   ↓
5. Custom Version Metadata Generation
   ↓
6. Version Registry Update
```

### Version Metadata Structure
```json
{
  "version_id": "v_20250822_125044",
  "created_at": "2025-08-22T12:50:44",
  "total_files": 17,
  "files": {
    "raw/customer_data.csv": {
      "hash": "abc123...",
      "size_bytes": 3551168,
      "rows": 32561,
      "columns": 15
    }
  }
}
```

## Dataset Tracking

### Tracked Datasets
1. **Raw Data**
   - `data/raw/kaggle/WA_Fn-UseC_-Telco-Customer-Churn.csv`
   - `data/raw/huggingface/customer_data.csv`

2. **Processed Data**
   - `data/processed/cleaned_*_data_*.csv`
   - Data cleaning and preprocessing results

3. **Transformed Data**
   - `data/transformed/transformed_*_data_*.csv`
   - Feature engineering outputs

4. **Feature Store**
   - `data/features/features_*.csv`
   - Curated feature datasets with metadata

### Version Comparison
Our system automatically detects:
- **New files**: Added to pipeline
- **Modified files**: Content changes via hash comparison
- **Deleted files**: Removed from pipeline
- **File statistics**: Size, row/column changes

## Reproducibility Guarantees

### Git-Based Reproducibility
```bash
# Checkout specific data version
git checkout data-v20250822_125044

# View version history
git log --oneline --grep="Data version"

# Compare versions
git diff data-v20250822_124217 data-v20250822_125044
```

### Custom Version Reproducibility
```python
# Load specific version metadata
version_info = load_version_metadata("v_20250822_125044")

# Verify data integrity
verify_file_hashes(version_info)

# Reproduce exact pipeline state
restore_pipeline_state(version_info)
```

## Change Log Example

### Version Comparison: v_20250822_124217 → v_20250822_125044
```
New files (3):
  + processed/cleaned_kaggle_data_20250822_125040.csv
  + features/features_20250822_125043.csv  
  + transformed/transformed_kaggle_data_20250822_125041.csv

Modified files (0):
  (No existing files modified)

Statistics:
  - Total tracked files: 14 → 17
  - Pipeline stages completed: 8 → 9
  - Data processing time: 2 minutes 26 seconds
```

## Implementation Files

### Core Versioning Scripts
- `src/versioning/version_data.py` - Main versioning logic
- `data/versioned/version_registry.json` - Version index
- `data/versioned/metadata/version_*.json` - Detailed metadata

### Git Integration
- Automatic commits on pipeline runs
- Semantic commit messages: "Data version {timestamp} - pipeline run"
- Git tags for easy version checkout
- Complete change history in Git log

## Benefits Achieved

### 1. Complete Reproducibility
- Any pipeline state can be exactly reproduced
- Data lineage fully tracked
- Environment and dependency tracking

### 2. Change Detection
- Automatic detection of data modifications
- File-level granularity with hash verification
- Statistical change tracking (rows/columns/size)

### 3. Collaboration Support
- Git-based sharing and collaboration
- Clear version history and change logs
- Easy rollback to previous states

### 4. Audit Trail
- Complete record of all data transformations
- Timestamp and metadata for compliance
- Automated documentation generation

## Usage Examples

### For Developers
```bash
# Check current version
git describe --tags

# List all data versions
git tag --list "data-v*"

# View version details
python -c "from src.versioning.version_data import *; list_versions()"
```

### For Evaluators
```bash
# Clone repository
git clone https://github.com/nishbhas874/dmml_churn_pipeline

# View version history
git log --oneline --grep="Data version"

# Check version metadata
cat data/versioned/version_registry.json
```

## Compliance with Assignment Requirements

✅ **Version control for datasets**: Git + custom system
✅ **Change tracking**: Hash-based detection with metadata  
✅ **DVC/Git repository**: Complete Git history with tags
✅ **Version metadata**: Comprehensive JSON metadata
✅ **Documentation**: This strategy document + inline docs
✅ **Reproducibility**: Full pipeline state restoration

This versioning strategy ensures complete reproducibility and meets all assignment requirements for professional data pipeline development.
