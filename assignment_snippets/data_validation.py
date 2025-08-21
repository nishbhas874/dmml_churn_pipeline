"""
Data Validation Code for Assignment
This code demonstrates comprehensive data quality validation for churn prediction datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """Comprehensive data validation system for churn prediction data."""
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0,
            'checks_performed': [],
            'issues_found': {
                'critical': [],
                'warning': [],
                'info': []
            },
            'statistics': {}
        }
    
    def validate_basic_structure(self, data: pd.DataFrame) -> dict:
        """Validate basic data structure."""
        print("Validating basic data structure...")
        
        results = {
            'check_name': 'basic_structure',
            'status': 'passed',
            'issues': [],
            'statistics': {}
        }
        
        # Check minimum rows
        if len(data) < 100:
            results['issues'].append({
                'type': 'critical',
                'message': f"Dataset has only {len(data)} rows, minimum required is 100"
            })
            results['status'] = 'failed'
        
        # Check minimum columns
        if len(data.columns) < 5:
            results['issues'].append({
                'type': 'warning',
                'message': f"Dataset has only {len(data.columns)} columns, expected more features"
            })
        
        # Check for empty dataset
        if data.empty:
            results['issues'].append({
                'type': 'critical',
                'message': "Dataset is completely empty"
            })
            results['status'] = 'failed'
        
        # Store statistics
        results['statistics'] = {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        return results
    
    def validate_missing_data(self, data: pd.DataFrame) -> dict:
        """Validate missing data patterns."""
        print("Validating missing data...")
        
        results = {
            'check_name': 'missing_data',
            'status': 'passed',
            'issues': [],
            'statistics': {}
        }
        
        # Calculate missing data statistics
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        # Check for high missing percentages
        high_missing_cols = missing_percentages[missing_percentages > 50]
        if len(high_missing_cols) > 0:
            for col, pct in high_missing_cols.items():
                results['issues'].append({
                    'type': 'critical',
                    'message': f"Column '{col}' has {pct:.1f}% missing values"
                })
                results['status'] = 'failed'
        
        # Check for moderate missing percentages
        moderate_missing_cols = missing_percentages[(missing_percentages > 10) & (missing_percentages <= 50)]
        if len(moderate_missing_cols) > 0:
            for col, pct in moderate_missing_cols.items():
                results['issues'].append({
                    'type': 'warning',
                    'message': f"Column '{col}' has {pct:.1f}% missing values"
                })
        
        # Check for rows with too many missing values
        missing_per_row = data.isnull().sum(axis=1)
        high_missing_rows = missing_per_row[missing_per_row > len(data.columns) * 0.5]
        if len(high_missing_rows) > 0:
            results['issues'].append({
                'type': 'warning',
                'message': f"{len(high_missing_rows)} rows have more than 50% missing values"
            })
        
        # Store statistics
        results['statistics'] = {
            'total_missing_values': data.isnull().sum().sum(),
            'columns_with_missing': len(missing_counts[missing_counts > 0]),
            'max_missing_percentage': missing_percentages.max(),
            'avg_missing_percentage': missing_percentages.mean()
        }
        
        return results
    
    def validate_data_types(self, data: pd.DataFrame) -> dict:
        """Validate data types and formats."""
        print("Validating data types...")
        
        results = {
            'check_name': 'data_types',
            'status': 'passed',
            'issues': [],
            'statistics': {}
        }
        
        # Check for expected data types
        expected_numeric_cols = ['tenure', 'monthly_charges', 'total_charges', 'age']
        expected_categorical_cols = ['gender', 'contract_type', 'payment_method', 'churn']
        
        for col in expected_numeric_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    results['issues'].append({
                        'type': 'warning',
                        'message': f"Column '{col}' is not numeric, expected numeric type"
                    })
        
        for col in expected_categorical_cols:
            if col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]) and col != 'churn':
                    results['issues'].append({
                        'type': 'warning',
                        'message': f"Column '{col}' is numeric, expected categorical type"
                    })
        
        # Check for mixed data types
        mixed_type_cols = []
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(data[col], errors='raise')
                    mixed_type_cols.append(col)
                except:
                    pass
        
        if mixed_type_cols:
            results['issues'].append({
                'type': 'warning',
                'message': f"Columns with mixed types detected: {mixed_type_cols}"
            })
        
        # Store statistics
        results['statistics'] = {
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'mixed_type_columns': len(mixed_type_cols)
        }
        
        return results
    
    def validate_data_ranges(self, data: pd.DataFrame) -> dict:
        """Validate data ranges and outliers."""
        print("Validating data ranges...")
        
        results = {
            'check_name': 'data_ranges',
            'status': 'passed',
            'issues': [],
            'statistics': {}
        }
        
        # Check numeric columns for outliers
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in data.columns:
                # Check for negative values where inappropriate
                if col in ['tenure', 'monthly_charges', 'total_charges', 'age']:
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        results['issues'].append({
                            'type': 'critical',
                            'message': f"Column '{col}' has {negative_count} negative values"
                        })
                        results['status'] = 'failed'
                
                # Check for zero values where inappropriate
                if col in ['tenure', 'age']:
                    zero_count = (data[col] == 0).sum()
                    if zero_count > 0:
                        results['issues'].append({
                            'type': 'warning',
                            'message': f"Column '{col}' has {zero_count} zero values"
                        })
                
                # Check for extreme outliers using IQR method
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > len(data) * 0.05:  # More than 5% outliers
                    results['issues'].append({
                        'type': 'warning',
                        'message': f"Column '{col}' has {outliers} outliers ({outliers/len(data)*100:.1f}%)"
                    })
        
        # Check categorical columns for unexpected values
        if 'churn' in data.columns:
            unique_churn_values = data['churn'].unique()
            if not all(val in [0, 1] for val in unique_churn_values):
                results['issues'].append({
                    'type': 'critical',
                    'message': f"Churn column has unexpected values: {unique_churn_values}"
                })
                results['status'] = 'failed'
        
        # Store statistics
        results['statistics'] = {
            'columns_checked': len(numeric_cols),
            'outlier_columns': len([col for col in numeric_cols if col in data.columns])
        }
        
        return results
    
    def validate_duplicates(self, data: pd.DataFrame) -> dict:
        """Validate for duplicate records."""
        print("Validating duplicates...")
        
        results = {
            'check_name': 'duplicates',
            'status': 'passed',
            'issues': [],
            'statistics': {}
        }
        
        # Check for exact duplicates
        exact_duplicates = data.duplicated().sum()
        if exact_duplicates > 0:
            results['issues'].append({
                'type': 'warning',
                'message': f"Found {exact_duplicates} exact duplicate rows"
            })
        
        # Check for duplicates based on key columns
        key_columns = ['customer_id'] if 'customer_id' in data.columns else []
        if key_columns:
            key_duplicates = data.duplicated(subset=key_columns).sum()
            if key_duplicates > 0:
                results['issues'].append({
                    'type': 'critical',
                    'message': f"Found {key_duplicates} duplicate customer IDs"
                })
                results['status'] = 'failed'
        
        # Store statistics
        results['statistics'] = {
            'exact_duplicates': exact_duplicates,
            'key_duplicates': key_duplicates if key_columns else 0
        }
        
        return results
    
    def validate_data_quality(self, data: pd.DataFrame) -> dict:
        """Perform comprehensive data quality validation."""
        print("Starting comprehensive data quality validation...")
        
        # Reset validation results
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0,
            'checks_performed': [],
            'issues_found': {
                'critical': [],
                'warning': [],
                'info': []
            },
            'statistics': {}
        }
        
        # Perform all validation checks
        checks = [
            self.validate_basic_structure(data),
            self.validate_missing_data(data),
            self.validate_data_types(data),
            self.validate_data_ranges(data),
            self.validate_duplicates(data)
        ]
        
        # Collect results
        for check in checks:
            self.validation_results['checks_performed'].append(check)
            
            # Collect issues by severity
            for issue in check.get('issues', []):
                severity = issue['type']
                self.validation_results['issues_found'][severity].append({
                    'check': check['check_name'],
                    'message': issue['message']
                })
        
        # Calculate overall quality score
        total_checks = len(checks)
        passed_checks = len([c for c in checks if c['status'] == 'passed'])
        self.validation_results['overall_score'] = (passed_checks / total_checks) * 100
        
        # Generate summary
        critical_issues = len(self.validation_results['issues_found']['critical'])
        warning_issues = len(self.validation_results['issues_found']['warning'])
        
        print(f"\n📊 Data Quality Validation Summary:")
        print(f"   Overall Quality Score: {self.validation_results['overall_score']:.1f}%")
        print(f"   Checks Performed: {total_checks}")
        print(f"   Checks Passed: {passed_checks}")
        print(f"   Critical Issues: {critical_issues}")
        print(f"   Warning Issues: {warning_issues}")
        
        if critical_issues > 0:
            print(f"\n❌ Critical Issues Found:")
            for issue in self.validation_results['issues_found']['critical']:
                print(f"   • {issue['check']}: {issue['message']}")
        
        if warning_issues > 0:
            print(f"\n⚠️  Warning Issues Found:")
            for issue in self.validation_results['issues_found']['warning']:
                print(f"   • {issue['check']}: {issue['message']}")
        
        return self.validation_results
    
    def generate_quality_report(self, output_path: str = None) -> str:
        """Generate comprehensive data quality report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/data_quality_report_{timestamp}.json"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        print(f"✅ Data quality report saved to: {output_path}")
        return output_path

# Example usage
if __name__ == "__main__":
    # Create sample data with some quality issues
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'tenure': np.random.exponential(30, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Introduce some quality issues for testing
    sample_data.loc[0, 'age'] = -5  # Negative age
    sample_data.loc[1, 'tenure'] = 0  # Zero tenure
    sample_data.loc[2:5, 'monthly_charges'] = np.nan  # Missing values
    sample_data.loc[10:15] = sample_data.loc[0:5].values  # Duplicates
    
    # Initialize and run validation
    validator = DataValidator()
    validation_results = validator.validate_data_quality(sample_data)
    
    # Generate report
    report_path = validator.generate_quality_report()
    
    print(f"\n📋 Validation completed!")
    print(f"   Quality Score: {validation_results['overall_score']:.1f}%")
    print(f"   Report saved to: {report_path}")
