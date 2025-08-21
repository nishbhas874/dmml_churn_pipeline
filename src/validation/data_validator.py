"""
Data Validation System for Churn Prediction Pipeline.
Implements comprehensive data quality checks and generates detailed reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml
from typing import Dict, List, Any, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """Comprehensive data validation system for churn prediction data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data validator with configuration."""
        self.config = self._load_config(config_path)
        self.validation_config = self.config.get('validation', {})
        self.features_config = self.config.get('features', {})
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'issues_found': [],
            'summary': {},
            'recommendations': []
        }
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def validate_dataset(self, data: pd.DataFrame, dataset_name: str = "unknown") -> Dict[str, Any]:
        """Perform comprehensive data validation on a dataset."""
        logger.info(f"Starting data validation for dataset: {dataset_name}")
        
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_name': dataset_name,
            'dataset_shape': data.shape,
            'checks_performed': [],
            'issues_found': [],
            'summary': {},
            'recommendations': []
        }
        
        # Perform all validation checks
        self._validate_basic_structure(data)
        self._validate_missing_data(data)
        self._validate_data_types(data)
        self._validate_data_ranges(data)
        self._validate_duplicates(data)
        self._validate_anomalies(data)
        self._validate_categorical_data(data)
        self._validate_numerical_data(data)
        self._validate_target_variable(data)
        
        self._generate_summary()
        logger.info(f"Data validation completed for {dataset_name}")
        return self.validation_results
    
    def _validate_basic_structure(self, data: pd.DataFrame):
        """Validate basic dataset structure."""
        issues = []
        required_columns = self.features_config.get('categorical_columns', []) + \
                         self.features_config.get('numerical_columns', [])
        
        if self.features_config.get('target_column'):
            required_columns.append(self.features_config['target_column'])
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            issues.append("Dataset is completely empty")
        
        min_rows = self.validation_config.get('min_rows', 100)
        if len(data) < min_rows:
            issues.append(f"Dataset has only {len(data)} rows, minimum required: {min_rows}")
        
        self._record_check("Basic Structure Validation", issues, {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_columns': missing_columns
        })
    
    def _validate_missing_data(self, data: pd.DataFrame):
        """Validate missing data patterns."""
        issues = []
        missing_stats = data.isnull().sum()
        missing_percentage = (missing_stats / len(data)) * 100
        
        high_missing_threshold = self.validation_config.get('high_missing_threshold', 50)
        high_missing_cols = missing_percentage[missing_percentage > high_missing_threshold]
        
        if not high_missing_cols.empty:
            issues.append(f"Columns with >{high_missing_threshold}% missing data: {list(high_missing_cols.index)}")
        
        empty_cols = missing_stats[missing_stats == len(data)]
        if not empty_cols.empty:
            issues.append(f"Completely empty columns: {list(empty_cols.index)}")
        
        self._record_check("Missing Data Validation", issues, {
            'total_missing_values': missing_stats.sum(),
            'missing_percentage_by_column': missing_percentage.to_dict(),
            'high_missing_columns': list(high_missing_cols.index)
        })
    
    def _validate_data_types(self, data: pd.DataFrame):
        """Validate data types and formats."""
        issues = []
        categorical_cols = self.features_config.get('categorical_columns', [])
        numerical_cols = self.features_config.get('numerical_columns', [])
        
        for col in categorical_cols:
            if col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    unique_ratio = data[col].nunique() / len(data)
                    if unique_ratio < 0.1:
                        issues.append(f"Column '{col}' is numerical but has low cardinality")
        
        for col in numerical_cols:
            if col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        pd.to_numeric(data[col], errors='raise')
                        issues.append(f"Column '{col}' is object type but contains numeric data")
                    except:
                        issues.append(f"Column '{col}' is object type but should be numerical")
        
        self._record_check("Data Type Validation", issues, {
            'categorical_columns_checked': len(categorical_cols),
            'numerical_columns_checked': len(numerical_cols)
        })
    
    def _validate_data_ranges(self, data: pd.DataFrame):
        """Validate data ranges and bounds."""
        issues = []
        numerical_cols = self.features_config.get('numerical_columns', [])
        
        for col in numerical_cols:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    if 'charges' in col.lower() and (col_data < 0).any():
                        issues.append(f"Column '{col}' contains negative values")
                    
                    if 'tenure' in col.lower() and (col_data == 0).any():
                        issues.append(f"Column '{col}' contains zero values")
        
        self._record_check("Data Range Validation", issues, {
            'numerical_columns_checked': len(numerical_cols)
        })
    
    def _validate_duplicates(self, data: pd.DataFrame):
        """Validate duplicate records."""
        issues = []
        exact_duplicates = data.duplicated().sum()
        
        if exact_duplicates > 0:
            issues.append(f"Found {exact_duplicates} exact duplicate rows")
        
        key_columns = self.validation_config.get('key_columns', [])
        if key_columns:
            key_cols_present = [col for col in key_columns if col in data.columns]
            if key_cols_present:
                key_duplicates = data.duplicated(subset=key_cols_present).sum()
                if key_duplicates > 0:
                    issues.append(f"Found {key_duplicates} duplicate rows based on key columns")
        
        self._record_check("Duplicate Validation", issues, {
            'exact_duplicates': exact_duplicates
        })
    
    def _validate_anomalies(self, data: pd.DataFrame):
        """Validate for data anomalies."""
        issues = []
        numerical_cols = self.features_config.get('numerical_columns', [])
        
        for col in numerical_cols:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                col_data = data[col].dropna()
                if len(col_data) > 10:
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    anomalies = z_scores > 3
                    
                    if anomalies.sum() > 0:
                        issues.append(f"Column '{col}' has {anomalies.sum()} statistical anomalies")
        
        self._record_check("Anomaly Detection", issues, {
            'numerical_columns_checked': len(numerical_cols)
        })
    
    def _validate_categorical_data(self, data: pd.DataFrame):
        """Validate categorical data quality."""
        issues = []
        categorical_cols = self.features_config.get('categorical_columns', [])
        
        for col in categorical_cols:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    unique_count = col_data.nunique()
                    cardinality_ratio = unique_count / len(col_data)
                    
                    if cardinality_ratio > 0.5:
                        issues.append(f"Column '{col}' has high cardinality")
                    
                    rare_threshold = self.validation_config.get('rare_category_threshold', 0.01)
                    value_counts = col_data.value_counts()
                    rare_categories = value_counts[value_counts / len(col_data) < rare_threshold]
                    
                    if len(rare_categories) > 0:
                        issues.append(f"Column '{col}' has {len(rare_categories)} rare categories")
        
        self._record_check("Categorical Data Validation", issues, {
            'categorical_columns_checked': len(categorical_cols)
        })
    
    def _validate_numerical_data(self, data: pd.DataFrame):
        """Validate numerical data quality."""
        issues = []
        numerical_cols = self.features_config.get('numerical_columns', [])
        
        for col in numerical_cols:
            if col in data.columns and data[col].dtype in ['int64', 'float64']:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    if col_data.std() == 0:
                        issues.append(f"Column '{col}' has zero variance")
                    
                    skewness = col_data.skew()
                    if abs(skewness) > 2:
                        issues.append(f"Column '{col}' is highly skewed")
        
        self._record_check("Numerical Data Validation", issues, {
            'numerical_columns_checked': len(numerical_cols)
        })
    
    def _validate_target_variable(self, data: pd.DataFrame):
        """Validate target variable quality."""
        issues = []
        target_col = self.features_config.get('target_column')
        
        if target_col and target_col in data.columns:
            target_data = data[target_col].dropna()
            if len(target_data) > 0:
                if target_data.dtype in ['int64', 'float64']:
                    value_counts = target_data.value_counts()
                    if len(value_counts) == 2:
                        min_class_ratio = value_counts.min() / value_counts.max()
                        if min_class_ratio < 0.1:
                            issues.append(f"Target variable is highly imbalanced")
                
                missing_target = data[target_col].isnull().sum()
                if missing_target > 0:
                    issues.append(f"Target variable has {missing_target} missing values")
        
        self._record_check("Target Variable Validation", issues, {
            'target_column': target_col,
            'target_present': target_col in data.columns if target_col else False
        })
    
    def _record_check(self, check_name: str, issues: List[str], metadata: Dict[str, Any]):
        """Record validation check results."""
        self.validation_results['checks_performed'].append({
            'check_name': check_name,
            'status': 'FAILED' if issues else 'PASSED',
            'issues_count': len(issues),
            'issues': issues,
            'metadata': metadata
        })
        
        if issues:
            self.validation_results['issues_found'].extend([
                {
                    'check': check_name,
                    'issue': issue,
                    'severity': self._assess_severity(issue)
                }
                for issue in issues
            ])
    
    def _assess_severity(self, issue: str) -> str:
        """Assess the severity of a validation issue."""
        critical_keywords = ['empty', 'missing required', 'completely empty', 'zero variance']
        warning_keywords = ['high cardinality', 'skewed', 'imbalanced', 'outliers', 'anomalies']
        
        issue_lower = issue.lower()
        
        if any(keyword in issue_lower for keyword in critical_keywords):
            return 'CRITICAL'
        elif any(keyword in issue_lower for keyword in warning_keywords):
            return 'WARNING'
        else:
            return 'INFO'
    
    def _generate_summary(self):
        """Generate validation summary."""
        total_checks = len(self.validation_results['checks_performed'])
        passed_checks = sum(1 for check in self.validation_results['checks_performed'] if check['status'] == 'PASSED')
        failed_checks = total_checks - passed_checks
        
        total_issues = len(self.validation_results['issues_found'])
        critical_issues = sum(1 for issue in self.validation_results['issues_found'] if issue['severity'] == 'CRITICAL')
        warning_issues = sum(1 for issue in self.validation_results['issues_found'] if issue['severity'] == 'WARNING')
        
        recommendations = []
        if critical_issues > 0:
            recommendations.append("Address critical issues before proceeding with analysis")
        if warning_issues > 0:
            recommendations.append("Review and address warning issues for better data quality")
        
        self.validation_results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'warning_issues': warning_issues,
            'overall_quality_score': self._calculate_quality_score()
        }
        
        self.validation_results['recommendations'] = recommendations
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        if not self.validation_results['checks_performed']:
            return 0.0
        
        total_checks = len(self.validation_results['checks_performed'])
        passed_checks = sum(1 for check in self.validation_results['checks_performed'] if check['status'] == 'PASSED')
        base_score = (passed_checks / total_checks) * 100
        
        critical_penalty = len([i for i in self.validation_results['issues_found'] if i['severity'] == 'CRITICAL']) * 10
        warning_penalty = len([i for i in self.validation_results['issues_found'] if i['severity'] == 'WARNING']) * 5
        
        final_score = max(0, base_score - critical_penalty - warning_penalty)
        return round(final_score, 2)
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive data quality report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/data_quality_report_{timestamp}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        csv_path = output_path.replace('.json', '_summary.csv')
        self._generate_csv_summary(csv_path)
        
        html_path = output_path.replace('.json', '.html')
        self._generate_html_report(html_path)
        
        logger.info(f"Data quality report generated: {output_path}")
        return output_path
    
    def _generate_csv_summary(self, csv_path: str):
        """Generate CSV summary of validation results."""
        summary_data = []
        
        for check in self.validation_results['checks_performed']:
            summary_data.append({
                'Check Name': check['check_name'],
                'Status': check['status'],
                'Issues Count': check['issues_count'],
                'Issues': '; '.join(check['issues']) if check['issues'] else 'None'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(csv_path, index=False)
    
    def _generate_html_report(self, html_path: str):
        """Generate HTML report of validation results."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {self.validation_results.get('dataset_name', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .check {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .passed {{ border-left-color: #4CAF50; }}
                .failed {{ border-left-color: #f44336; }}
                .score {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p><strong>Dataset:</strong> {self.validation_results.get('dataset_name', 'Unknown')}</p>
                <p><strong>Generated:</strong> {self.validation_results.get('timestamp', 'Unknown')}</p>
                <p><strong>Dataset Shape:</strong> {self.validation_results.get('dataset_shape', 'Unknown')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p class="score">Quality Score: {self.validation_results.get('summary', {}).get('overall_quality_score', 0)}/100</p>
                <p><strong>Total Checks:</strong> {self.validation_results.get('summary', {}).get('total_checks', 0)}</p>
                <p><strong>Passed:</strong> {self.validation_results.get('summary', {}).get('passed_checks', 0)}</p>
                <p><strong>Failed:</strong> {self.validation_results.get('summary', {}).get('failed_checks', 0)}</p>
                <p><strong>Total Issues:</strong> {self.validation_results.get('summary', {}).get('total_issues', 0)}</p>
                <p><strong>Critical Issues:</strong> {self.validation_results.get('summary', {}).get('critical_issues', 0)}</p>
                <p><strong>Warning Issues:</strong> {self.validation_results.get('summary', {}).get('warning_issues', 0)}</p>
            </div>
            
            <h2>Validation Checks</h2>
        """
        
        for check in self.validation_results['checks_performed']:
            status_class = 'passed' if check['status'] == 'PASSED' else 'failed'
            html_content += f"""
            <div class="check {status_class}">
                <h3>{check['check_name']} - {check['status']}</h3>
                <p><strong>Issues Found:</strong> {check['issues_count']}</p>
            """
            
            if check['issues']:
                html_content += "<h4>Issues:</h4><ul>"
                for issue in check['issues']:
                    html_content += f"<li>{issue}</li>"
                html_content += "</ul>"
            
            html_content += "</div>"
        
        if self.validation_results.get('recommendations'):
            html_content += """
            <h2>Recommendations</h2>
            <ul>
            """
            for rec in self.validation_results['recommendations']:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)


def create_data_validator(config_path: str = "config/config.yaml") -> DataValidator:
    """Factory function to create data validator instance."""
    return DataValidator(config_path)
