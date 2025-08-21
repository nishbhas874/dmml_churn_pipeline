# Data Validation System Documentation

## Overview

The Data Validation System provides comprehensive data quality assessment for the churn prediction pipeline. It automatically detects and reports various data quality issues, ensuring that only high-quality data proceeds to the modeling stage.

## Features

- **Comprehensive Validation Checks**: 9 different validation categories
- **Automated Issue Detection**: Identifies missing data, duplicates, anomalies, and more
- **Severity Classification**: Categorizes issues as Critical, Warning, or Info
- **Quality Scoring**: Provides overall data quality score (0-100)
- **Multiple Report Formats**: JSON, CSV, and HTML reports
- **Configurable Thresholds**: Customizable validation parameters
- **Real-time Feedback**: Immediate validation results and recommendations

## Validation Checks

### 1. Basic Structure Validation
- **Purpose**: Ensures dataset meets basic requirements
- **Checks**:
  - Dataset is not empty
  - Minimum required rows (configurable)
  - Required columns are present
- **Severity**: Critical

### 2. Missing Data Validation
- **Purpose**: Identifies missing data patterns
- **Checks**:
  - Percentage of missing values per column
  - Completely empty columns
  - High missing data thresholds
- **Severity**: Warning/Critical

### 3. Data Type Validation
- **Purpose**: Ensures correct data types
- **Checks**:
  - Categorical columns have appropriate cardinality
  - Numerical columns contain numeric data
  - Mixed data types in columns
- **Severity**: Info/Warning

### 4. Data Range Validation
- **Purpose**: Validates data ranges and bounds
- **Checks**:
  - Negative values where inappropriate
  - Zero values where not expected
  - Out-of-range values
- **Severity**: Warning

### 5. Duplicate Validation
- **Purpose**: Identifies duplicate records
- **Checks**:
  - Exact duplicate rows
  - Duplicates based on key columns
  - Near-duplicate detection
- **Severity**: Info

### 6. Anomaly Detection
- **Purpose**: Identifies statistical anomalies
- **Checks**:
  - Z-score based outliers
  - IQR based outliers
  - Statistical anomalies
- **Severity**: Warning

### 7. Categorical Data Validation
- **Purpose**: Validates categorical data quality
- **Checks**:
  - High cardinality detection
  - Rare categories
  - Inconsistent text casing
- **Severity**: Warning

### 8. Numerical Data Validation
- **Purpose**: Validates numerical data quality
- **Checks**:
  - Zero variance columns
  - Highly skewed distributions
  - Kurtosis analysis
- **Severity**: Warning

### 9. Target Variable Validation
- **Purpose**: Ensures target variable quality
- **Checks**:
  - Class imbalance detection
  - Missing target values
  - Target variable distribution
- **Severity**: Critical/Warning

## Configuration

### Validation Settings

```yaml
validation:
  # Basic validation thresholds
  min_rows: 100
  high_missing_threshold: 50  # Percentage
  rare_category_threshold: 0.01  # 1%
  
  # Key columns for duplicate detection
  key_columns: ["customer_id"]
  
  # Data quality thresholds
  similarity_threshold: 0.95
  max_missing_per_row: 0.5  # 50% of columns
  
  # Report settings
  generate_reports: true
  report_formats: ["json", "csv", "html"]
```

### Feature Configuration

```yaml
features:
  categorical_columns:
    - "gender"
    - "contract_type"
    - "payment_method"
    # ... other categorical columns
  
  numerical_columns:
    - "tenure"
    - "monthly_charges"
    - "total_charges"
    # ... other numerical columns
  
  target_column: "churn"
```

## Usage Examples

### Basic Usage

```python
from src.validation.data_validator import create_data_validator
import pandas as pd

# Initialize validator
validator = create_data_validator()

# Load your data
data = pd.read_csv("your_data.csv")

# Perform validation
results = validator.validate_dataset(data, "My Dataset")

# Generate reports
report_path = validator.generate_report()

print(f"Quality Score: {results['summary']['overall_quality_score']}/100")
print(f"Total Issues: {results['summary']['total_issues']}")
```

### Advanced Usage

```python
# Custom validation with specific checks
validator = create_data_validator("custom_config.yaml")

# Validate specific aspects
results = validator.validate_dataset(data, "Custom Validation")

# Access detailed results
for check in results['checks_performed']:
    print(f"{check['check_name']}: {check['status']}")
    if check['issues']:
        for issue in check['issues']:
            print(f"  - {issue}")

# Filter issues by severity
critical_issues = [i for i in results['issues_found'] if i['severity'] == 'CRITICAL']
warning_issues = [i for i in results['issues_found'] if i['severity'] == 'WARNING']
```

### Integration with Pipeline

```python
from src.ingestion.unified_ingestion import ingest_all_data
from src.validation.data_validator import create_data_validator

# Run data ingestion
ingested_files = ingest_all_data()

# Validate ingested data
validator = create_data_validator()

for source, file_path in ingested_files.items():
    if file_path:
        data = pd.read_csv(file_path)
        results = validator.validate_dataset(data, f"{source}_data")
        
        # Check if data quality is acceptable
        if results['summary']['overall_quality_score'] < 70:
            print(f"Warning: {source} data has low quality score")
        
        # Generate reports
        validator.generate_report(f"reports/{source}_validation_report.json")
```

## Report Formats

### JSON Report
Comprehensive report with all validation details:

```json
{
  "timestamp": "2025-08-21T23:29:18.803",
  "dataset_name": "Sample Churn Dataset",
  "dataset_shape": [1050, 21],
  "checks_performed": [...],
  "issues_found": [...],
  "summary": {
    "total_checks": 9,
    "passed_checks": 2,
    "failed_checks": 7,
    "overall_quality_score": 2.22,
    "total_issues": 9,
    "critical_issues": 0,
    "warning_issues": 4
  },
  "recommendations": [...]
}
```

### CSV Summary
Concise summary for quick review:

```csv
Check Name,Status,Issues Count,Issues
Basic Structure Validation,PASSED,0,None
Missing Data Validation,PASSED,0,None
Data Type Validation,FAILED,1,Column 'tenure' is object type but should be numerical
...
```

### HTML Report
Visual report with styling and formatting for easy reading.

## Quality Scoring

The system calculates an overall quality score (0-100) based on:

1. **Base Score**: Percentage of passed checks
2. **Penalties**:
   - Critical issues: -10 points each
   - Warning issues: -5 points each
   - Info issues: -1 point each

**Score Interpretation**:
- **90-100**: Excellent quality
- **70-89**: Good quality
- **50-69**: Acceptable quality
- **30-49**: Poor quality
- **0-29**: Very poor quality

## Issue Severity Levels

### Critical
- Dataset completely empty
- Missing required columns
- Zero variance columns
- Highly imbalanced target variable

### Warning
- High missing data percentage
- Statistical anomalies
- Highly skewed distributions
- High cardinality categorical data

### Info
- Data type inconsistencies
- Duplicate records
- Rare categories
- Minor data range issues

## Best Practices

### 1. Pre-validation Setup
- Configure appropriate thresholds for your data
- Define key columns for duplicate detection
- Set minimum quality score requirements

### 2. Validation Workflow
- Run validation after data ingestion
- Review critical issues first
- Address warning issues for better quality
- Document validation results

### 3. Quality Thresholds
- Set minimum quality score for pipeline continuation
- Implement automated quality gates
- Monitor quality trends over time

### 4. Report Management
- Store validation reports with timestamps
- Track quality metrics over time
- Share reports with stakeholders

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   - Check YAML syntax in config file
   - Verify column names match your data
   - Ensure all required sections are present

2. **Performance Issues**
   - For large datasets, consider sampling for validation
   - Adjust validation thresholds for better performance
   - Use appropriate data types

3. **False Positives**
   - Review and adjust validation thresholds
   - Consider domain-specific validation rules
   - Customize validation logic for your use case

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

validator = create_data_validator()
results = validator.validate_dataset(data)
```

## Integration with Other Systems

### Data Lake Integration
```python
from src.storage.data_lake import create_data_lake_storage
from src.validation.data_validator import create_data_validator

# Validate data before storage
validator = create_data_validator()
results = validator.validate_dataset(data)

if results['summary']['overall_quality_score'] >= 70:
    # Store in data lake
    storage = create_data_lake_storage()
    storage.upload_file(data_path, source="validated", metadata=results)
```

### Pipeline Integration
```python
# In your main pipeline
def run_pipeline():
    # Data ingestion
    ingested_files = ingest_all_data()
    
    # Data validation
    validator = create_data_validator()
    for source, file_path in ingested_files.items():
        if file_path:
            data = pd.read_csv(file_path)
            validation_results = validator.validate_dataset(data, source)
            
            # Quality gate
            if validation_results['summary']['overall_quality_score'] < 70:
                raise ValueError(f"Data quality too low for {source}")
    
    # Continue with processing...
```

## Future Enhancements

1. **Advanced Validation Rules**
   - Domain-specific validation rules
   - Custom validation functions
   - Machine learning-based anomaly detection

2. **Real-time Validation**
   - Streaming data validation
   - Real-time quality monitoring
   - Automated quality alerts

3. **Enhanced Reporting**
   - Interactive dashboards
   - Trend analysis
   - Quality forecasting

4. **Integration Capabilities**
   - Database validation
   - API data validation
   - Cloud storage validation

## Conclusion

The Data Validation System provides a robust foundation for ensuring data quality in the churn prediction pipeline. Its comprehensive checks, configurable thresholds, and detailed reporting capabilities help maintain high data quality standards throughout the data lifecycle.

For more information, refer to the source code in `src/validation/data_validator.py` and the demonstration script `demo_data_validation.py`.
