"""
Data Preprocessing System for Churn Prediction Pipeline.
Handles data cleaning, preprocessing, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import visualization and analysis tools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class DataPreprocessor:
    """Comprehensive data preprocessing system for churn prediction data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.features_config = self.config.get('features', {})
        self.preprocessing_config = self.config.get('preprocessing', {})
        
        # Initialize preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        # Store preprocessing metadata
        self.preprocessing_info = {
            'timestamp': datetime.now().isoformat(),
            'steps_performed': [],
            'transformations': {},
            'statistics': {}
        }
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats."""
        logger.info(f"Loading data from: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_extension == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    
    def explore_data(self, data: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
        """Perform comprehensive Exploratory Data Analysis (EDA)."""
        logger.info("Starting Exploratory Data Analysis...")
        
        eda_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {},
            'missing_data': {},
            'numerical_analysis': {},
            'categorical_analysis': {},
            'correlation_analysis': {},
            'target_analysis': {},
            'recommendations': []
        }
        
        # Basic dataset information
        eda_results['dataset_info'] = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'data_types': data.dtypes.value_counts().to_dict(),
            'duplicates': data.duplicated().sum()
        }
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        missing_percentage = (missing_data / len(data)) * 100
        
        eda_results['missing_data'] = {
            'missing_counts': missing_data.to_dict(),
            'missing_percentage': missing_percentage.to_dict(),
            'columns_with_missing': missing_data[missing_data > 0].index.tolist()
        }
        
        # Numerical data analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            numerical_stats = data[numerical_cols].describe()
            eda_results['numerical_analysis'] = {
                'columns': numerical_cols.tolist(),
                'statistics': numerical_stats.to_dict(),
                'skewness': data[numerical_cols].skew().to_dict()
            }
        
        # Categorical data analysis
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            categorical_stats = {}
            for col in categorical_cols:
                categorical_stats[col] = {
                    'unique_count': data[col].nunique(),
                    'most_common': data[col].value_counts().head(5).to_dict(),
                    'missing_count': data[col].isnull().sum()
                }
            
            eda_results['categorical_analysis'] = {
                'columns': categorical_cols.tolist(),
                'statistics': categorical_stats
            }
        
        # Correlation analysis
        if len(numerical_cols) > 1:
            correlation_matrix = data[numerical_cols].corr()
            eda_results['correlation_analysis'] = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlations': self._find_high_correlations(correlation_matrix)
            }
        
        # Target variable analysis
        target_col = self.features_config.get('target_column')
        if target_col and target_col in data.columns:
            target_analysis = self._analyze_target_variable(data, target_col)
            eda_results['target_analysis'] = target_analysis
        
        # Generate recommendations
        eda_results['recommendations'] = self._generate_eda_recommendations(eda_results)
        
        # Create visualizations
        if save_plots:
            self._create_eda_visualizations(data, eda_results)
        
        logger.info("EDA completed successfully")
        return eda_results
    
    def _analyze_target_variable(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze target variable distribution and characteristics."""
        target_data = data[target_col].dropna()
        
        analysis = {
            'distribution': target_data.value_counts().to_dict(),
            'missing_count': data[target_col].isnull().sum(),
            'missing_percentage': (data[target_col].isnull().sum() / len(data)) * 100
        }
        
        if target_data.dtype in ['int64', 'float64']:
            analysis['statistics'] = {
                'mean': target_data.mean(),
                'std': target_data.std(),
                'min': target_data.min(),
                'max': target_data.max()
            }
            
            # Class imbalance analysis
            if target_data.nunique() == 2:
                class_counts = target_data.value_counts()
                analysis['class_imbalance'] = {
                    'ratio': class_counts.min() / class_counts.max(),
                    'is_imbalanced': (class_counts.min() / class_counts.max()) < 0.3
                }
        
        return analysis
    
    def _find_high_correlations(self, correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Find highly correlated feature pairs."""
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return high_correlations
    
    def _generate_eda_recommendations(self, eda_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on EDA results."""
        recommendations = []
        
        # Missing data recommendations
        missing_cols = eda_results['missing_data']['columns_with_missing']
        if missing_cols:
            recommendations.append(f"Handle missing values in {len(missing_cols)} columns")
        
        # High correlation recommendations
        high_corrs = eda_results.get('correlation_analysis', {}).get('high_correlations', [])
        if high_corrs:
            recommendations.append(f"Consider removing one of {len(high_corrs)} highly correlated feature pairs")
        
        # Class imbalance recommendations
        target_analysis = eda_results.get('target_analysis', {})
        if target_analysis.get('class_imbalance', {}).get('is_imbalanced', False):
            recommendations.append("Address class imbalance using techniques like SMOTE or class weights")
        
        return recommendations
    
    def _create_eda_visualizations(self, data: pd.DataFrame, eda_results: Dict[str, Any]):
        """Create comprehensive EDA visualizations."""
        logger.info("Creating EDA visualizations...")
        
        # Ensure plots directory exists
        plots_dir = Path("plots/eda")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Missing data visualization
        self._plot_missing_data(data, plots_dir)
        
        # 2. Numerical features distribution
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            self._plot_numerical_distributions(data, numerical_cols, plots_dir)
        
        # 3. Categorical features analysis
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            self._plot_categorical_analysis(data, categorical_cols, plots_dir)
        
        # 4. Correlation heatmap
        if len(numerical_cols) > 1:
            self._plot_correlation_heatmap(data, numerical_cols, plots_dir)
        
        # 5. Target variable analysis
        target_col = self.features_config.get('target_column')
        if target_col and target_col in data.columns:
            self._plot_target_analysis(data, target_col, plots_dir)
        
        logger.info(f"EDA visualizations saved to: {plots_dir}")
    
    def _plot_missing_data(self, data: pd.DataFrame, plots_dir: Path):
        """Plot missing data patterns."""
        missing_data = data.isnull().sum()
        missing_percentage = (missing_data / len(data)) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing counts
        missing_data[missing_data > 0].plot(kind='bar', ax=ax1, color='salmon')
        ax1.set_title('Missing Values Count')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Missing percentage
        missing_percentage[missing_percentage > 0].plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Missing Values Percentage')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Percentage (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'missing_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_numerical_distributions(self, data: pd.DataFrame, numerical_cols: pd.Index, plots_dir: Path):
        """Plot distributions of numerical features."""
        n_cols = len(numerical_cols)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            
            # Histogram
            ax.hist(data[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
            # Add statistics
            mean_val = data[col].mean()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_categorical_analysis(self, data: pd.DataFrame, categorical_cols: pd.Index, plots_dir: Path):
        """Plot categorical features analysis."""
        n_cols = len(categorical_cols)
        n_rows = (n_cols + 1) // 2  # 2 columns per row
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 6 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(categorical_cols):
            ax = axes[i]
            
            # Value counts plot
            value_counts = data[col].value_counts()
            value_counts.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, data: pd.DataFrame, numerical_cols: pd.Index, plots_dir: Path):
        """Plot correlation heatmap."""
        correlation_matrix = data[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_analysis(self, data: pd.DataFrame, target_col: str, plots_dir: Path):
        """Plot target variable analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        target_counts = data[target_col].value_counts()
        axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title(f'Distribution of {target_col}')
        
        # Target bar plot
        target_counts.plot(kind='bar', ax=axes[0, 1], color=['lightblue', 'lightcoral'])
        axes[0, 1].set_title(f'Count of {target_col}')
        axes[0, 1].set_xlabel(target_col)
        axes[0, 1].set_ylabel('Count')
        
        # Target vs numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            first_num_col = numerical_cols[0]
            data.boxplot(column=first_num_col, by=target_col, ax=axes[1, 0])
            axes[1, 0].set_title(f'{first_num_col} by {target_col}')
            axes[1, 0].set_xlabel(target_col)
            axes[1, 0].set_ylabel(first_num_col)
        
        # Target vs categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            first_cat_col = categorical_cols[0]
            cross_tab = pd.crosstab(data[first_cat_col], data[target_col])
            cross_tab.plot(kind='bar', ax=axes[1, 1], stacked=True)
            axes[1, 1].set_title(f'{first_cat_col} vs {target_col}')
            axes[1, 1].set_xlabel(first_cat_col)
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values, duplicates, and data type issues."""
        logger.info("Starting data cleaning process...")
        
        cleaned_data = data.copy()
        cleaning_steps = []
        
        # 1. Remove duplicates
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        duplicates_removed = initial_rows - len(cleaned_data)
        if duplicates_removed > 0:
            cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows")
        
        # 2. Handle missing values
        missing_strategy = self.preprocessing_config.get('missing_values', {})
        
        # Numerical columns
        numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if cleaned_data[col].isnull().sum() > 0:
                strategy = missing_strategy.get('numerical', 'mean')
                if strategy == 'mean':
                    cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
                elif strategy == 'median':
                    cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                elif strategy == 'mode':
                    cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
                
                cleaning_steps.append(f"Handled missing values in {col} using {strategy}")
        
        # Categorical columns
        categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_data[col].isnull().sum() > 0:
                strategy = missing_strategy.get('categorical', 'mode')
                if strategy == 'mode':
                    cleaned_data[col].fillna(cleaned_data[col].mode()[0], inplace=True)
                elif strategy == 'unknown':
                    cleaned_data[col].fillna('Unknown', inplace=True)
                
                cleaning_steps.append(f"Handled missing values in {col} using {strategy}")
        
        # 3. Data type conversions
        cleaned_data = self._convert_data_types(cleaned_data)
        cleaning_steps.append("Converted data types")
        
        # Store cleaning information
        self.preprocessing_info['steps_performed'].extend(cleaning_steps)
        self.preprocessing_info['statistics']['cleaning'] = {
            'initial_rows': initial_rows,
            'final_rows': len(cleaned_data),
            'duplicates_removed': duplicates_removed,
            'steps': cleaning_steps
        }
        
        logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
        return cleaned_data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types based on configuration."""
        converted_data = data.copy()
        
        # Convert categorical columns
        categorical_cols = self.features_config.get('categorical_columns', [])
        for col in categorical_cols:
            if col in converted_data.columns:
                converted_data[col] = converted_data[col].astype('category')
        
        # Convert numerical columns
        numerical_cols = self.features_config.get('numerical_columns', [])
        for col in numerical_cols:
            if col in converted_data.columns:
                converted_data[col] = pd.to_numeric(converted_data[col], errors='coerce')
        
        return converted_data
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess the cleaned data for modeling."""
        logger.info("Starting data preprocessing...")
        
        preprocessed_data = data.copy()
        preprocessing_steps = []
        
        # 1. Feature encoding
        preprocessed_data, encoding_info = self._encode_features(preprocessed_data)
        preprocessing_steps.append("Encoded categorical features")
        
        # 2. Feature scaling
        preprocessed_data, scaling_info = self._scale_features(preprocessed_data)
        preprocessing_steps.append("Scaled numerical features")
        
        # 3. Handle target variable
        target_col = self.features_config.get('target_column')
        if target_col and target_col in preprocessed_data.columns:
            preprocessed_data = self._prepare_target_variable(preprocessed_data, target_col)
            preprocessing_steps.append("Prepared target variable")
        
        # Store preprocessing information
        self.preprocessing_info['steps_performed'].extend(preprocessing_steps)
        self.preprocessing_info['transformations'] = {
            'encoding': encoding_info,
            'scaling': scaling_info
        }
        
        logger.info(f"Data preprocessing completed. Final shape: {preprocessed_data.shape}")
        return preprocessed_data, self.preprocessing_info
    
    def _encode_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Encode categorical features."""
        encoding_info = {}
        encoded_data = data.copy()
        
        categorical_cols = self.features_config.get('categorical_columns', [])
        encoding_method = self.preprocessing_config.get('encoding', 'onehot')
        
        for col in categorical_cols:
            if col in encoded_data.columns:
                if encoding_method == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(encoded_data[col], prefix=col, drop_first=True)
                    encoded_data = pd.concat([encoded_data, dummies], axis=1)
                    encoded_data.drop(col, axis=1, inplace=True)
                    
                    encoding_info[col] = {
                        'method': 'onehot',
                        'new_columns': dummies.columns.tolist()
                    }
                
                elif encoding_method == 'label':
                    # Label encoding
                    le = LabelEncoder()
                    encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
                    
                    encoding_info[col] = {
                        'method': 'label',
                        'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
                    }
                    
                    self.encoders[col] = le
        
        return encoded_data, encoding_info
    
    def _scale_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Scale numerical features."""
        scaling_info = {}
        scaled_data = data.copy()
        
        numerical_cols = self.features_config.get('numerical_columns', [])
        scaling_method = self.preprocessing_config.get('scaling', 'standard')
        
        if len(numerical_cols) > 0:
            available_cols = [col for col in numerical_cols if col in scaled_data.columns]
            
            if available_cols:
                if scaling_method == 'standard':
                    scaler = StandardScaler()
                elif scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                
                scaled_data[available_cols] = scaler.fit_transform(scaled_data[available_cols])
                
                scaling_info = {
                    'method': scaling_method,
                    'columns': available_cols
                }
                
                self.scalers['numerical'] = scaler
        
        return scaled_data, scaling_info
    
    def _prepare_target_variable(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare target variable for modeling."""
        prepared_data = data.copy()
        
        # Handle missing target values
        if prepared_data[target_col].isnull().sum() > 0:
            prepared_data = prepared_data.dropna(subset=[target_col])
        
        # Ensure target is binary for classification
        if prepared_data[target_col].dtype == 'object':
            # Convert to binary if needed
            unique_values = prepared_data[target_col].unique()
            if len(unique_values) == 2:
                le = LabelEncoder()
                prepared_data[target_col] = le.fit_transform(prepared_data[target_col])
                self.encoders[target_col] = le
        
        return prepared_data
    
    def save_preprocessed_data(self, data: pd.DataFrame, output_path: str = None) -> str:
        """Save preprocessed data to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/processed/preprocessed_data_{timestamp}.csv"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data.to_csv(output_path, index=False)
        
        # Save preprocessing info
        info_path = output_path.replace('.csv', '_info.json')
        import json
        with open(info_path, 'w') as f:
            json.dump(self.preprocessing_info, f, indent=2, default=str)
        
        logger.info(f"Preprocessed data saved to: {output_path}")
        return output_path


def create_data_preprocessor(config_path: str = "config/config.yaml") -> DataPreprocessor:
    """Factory function to create data preprocessor instance."""
    return DataPreprocessor(config_path)
