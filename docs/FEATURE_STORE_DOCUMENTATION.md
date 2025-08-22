# Feature Store Documentation

## Overview
Our feature store manages engineered features with comprehensive metadata, versioning, and automated retrieval capabilities for both training and inference.

## Feature Store Architecture

### Storage Structure
```
data/features/
├── features_YYYYMMDD_HHMMSS.csv          # Feature data
├── metadata/
│   └── features_metadata_YYYYMMDD_HHMMSS.json  # Feature metadata
└── feature_registry.json                  # Master registry
```

## Feature Metadata Schema

Each feature includes comprehensive metadata:

```json
{
  "name": "feature_name",
  "data_type": "float64",
  "null_count": 0,
  "unique_values": 1234,
  "description": "Human-readable description",
  "min_value": 0.0,
  "max_value": 100.0,
  "mean_value": 45.2,
  "std_value": 12.8
}
```

## Available Features

### Customer Demographics
- **customer_id**: Unique identifier for each customer
- **tenure**: Number of months customer has been with company
- **senior_citizen**: Whether customer is senior citizen (0/1)
- **partner**: Whether customer has partner (0/1)
- **dependents**: Whether customer has dependents (0/1)

### Financial Features
- **monthly_charges**: Monthly charges for the customer
- **total_charges**: Total charges for the customer
- **charges_per_tenure**: Average charges per month of tenure

### Engineered Features
- **tenure_group**: Grouped tenure categories (Short/Medium/Long)
- **contract_encoded**: Encoded contract type (0/1/2)
- **payment_method_encoded**: Encoded payment method (0/1/2/3)
- **gender_encoded**: Encoded gender (0/1)

### Target Variable
- **churn**: Whether customer churned (0/1)

## Feature Versioning

Features are versioned using timestamps:
- **Version Format**: `YYYYMMDD_HHMMSS`
- **Registry Tracking**: Latest version tracked in `feature_registry.json`
- **Metadata History**: All versions preserved with full lineage

## Feature Retrieval API

### Get Latest Features
```python
from src.feature_store.manage_features import retrieve_latest_features

# Get latest feature set
features_df, metadata = retrieve_latest_features()
```

### Feature Registry Access
```python
import json

# Load feature registry
with open("data/features/feature_registry.json", 'r') as f:
    registry = json.load(f)

latest_version = registry["latest_version"]
features_file = registry["features_file"]
```

## Automated Feature Management

### Feature Creation Process
1. **Data Input**: Transformed data from Stage 6
2. **Metadata Generation**: Automatic statistical profiling
3. **Quality Validation**: Data type and range checks
4. **Storage**: Versioned storage with registry update
5. **Documentation**: Auto-generated feature descriptions

### Feature Update Workflow
1. **Change Detection**: Compare with previous version
2. **Version Creation**: New timestamp-based version
3. **Metadata Update**: Refresh statistical profiles
4. **Registry Update**: Update latest version pointer
5. **Backward Compatibility**: Maintain previous versions

## Feature Quality Monitoring

### Automated Checks
- **Data Type Validation**: Ensure consistent types
- **Range Validation**: Check for unexpected values
- **Null Value Monitoring**: Track missing data patterns
- **Distribution Monitoring**: Detect data drift

### Quality Metrics
- **Completeness**: Percentage of non-null values
- **Uniqueness**: Cardinality of categorical features
- **Consistency**: Data type and format consistency
- **Freshness**: Time since last update

## Integration Points

### Training Pipeline Integration
```python
# Features automatically loaded in training
features_df, metadata = retrieve_latest_features()
X_train = features_df.drop(['customer_id', 'churn'], axis=1)
y_train = features_df['churn']
```

### Inference Integration
```python
# Same feature retrieval for consistent inference
features_df, metadata = retrieve_latest_features()
# Apply same preprocessing pipeline
```

## Feature Store Benefits

### For Data Scientists
- **Consistent Features**: Same features across training/inference
- **Feature Discovery**: Browse available features with metadata
- **Version Control**: Track feature evolution over time
- **Quality Assurance**: Automated validation and monitoring

### For ML Engineers
- **Automated Retrieval**: Programmatic feature access
- **Metadata Access**: Feature descriptions and statistics
- **Version Management**: Controlled feature updates
- **Pipeline Integration**: Seamless workflow integration

### For Business Users
- **Feature Documentation**: Clear descriptions and business meaning
- **Impact Tracking**: Understand feature importance
- **Data Lineage**: Track feature creation and updates
- **Quality Reports**: Monitor feature health

## Best Practices

### Feature Naming
- Use descriptive, business-friendly names
- Include units where applicable (e.g., `monthly_charges_usd`)
- Use consistent naming conventions

### Feature Documentation
- Provide clear business descriptions
- Document calculation logic
- Include data quality expectations
- Maintain update history

### Version Management
- Create new versions for schema changes
- Maintain backward compatibility when possible
- Archive old versions after validation period
- Document breaking changes

This feature store provides a robust foundation for consistent, high-quality feature management across our churn prediction pipeline.
