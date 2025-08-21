"""
Data processing module for churn prediction pipeline.
Handles data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from loguru import logger
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Data processor for churn prediction pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data processor with configuration."""
        self.config = self._load_config(config_path)
        self.raw_data_path = Path(self.config['data']['raw_data_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.categorical_columns = self.config['features']['categorical_columns']
        self.numerical_columns = self.config['features']['numerical_columns']
        self.target_column = self.config['features']['target_column']
        
        # Create directories if they don't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
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
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded data with shape: {data.shape}")
        return data
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        logger.info("Starting data cleaning process...")
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # Handle duplicates
        cleaned_data = self._handle_duplicates(cleaned_data)
        
        # Handle outliers
        cleaned_data = self._handle_outliers(cleaned_data)
        
        # Data type conversions
        cleaned_data = self._convert_data_types(cleaned_data)
        
        logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # For categorical columns, fill with mode
        for col in self.categorical_columns:
            if col in data.columns and data[col].isnull().sum() > 0:
                mode_value = data[col].mode()[0]
                data[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        # For numerical columns, fill with median
        for col in self.numerical_columns:
            if col in data.columns and data[col].isnull().sum() > 0:
                median_value = data[col].median()
                data[col].fillna(median_value, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        return data
    
    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate records."""
        initial_count = len(data)
        data = data.drop_duplicates()
        final_count = len(data)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate records")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical columns using IQR method."""
        logger.info("Handling outliers...")
        
        for col in self.numerical_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                
                outliers_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers_count > 0:
                    logger.info(f"Capped {outliers_count} outliers in {col}")
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for optimal performance."""
        logger.info("Converting data types...")
        
        # Convert categorical columns to category type
        for col in self.categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category')
        
        # Convert numerical columns to appropriate types
        for col in self.numerical_columns:
            if col in data.columns:
                if data[col].dtype == 'object':
                    # Try to convert to numeric
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    # Fill any resulting NaN values
                    data[col].fillna(data[col].median(), inplace=True)
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        encoded_data = data.copy()
        
        # One-hot encoding for categorical variables
        for col in self.categorical_columns:
            if col in encoded_data.columns:
                # Create dummy variables
                dummies = pd.get_dummies(encoded_data[col], prefix=col, drop_first=True)
                encoded_data = pd.concat([encoded_data, dummies], axis=1)
                encoded_data.drop(col, axis=1, inplace=True)
                logger.info(f"Encoded {col} with {len(dummies.columns)} dummy variables")
        
        return encoded_data
    
    def split_data(self, data: pd.DataFrame, test_size: float = 0.2, 
                   val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        from sklearn.model_selection import train_test_split
        
        logger.info("Splitting data into train, validation, and test sets...")
        
        # First split: separate test set
        train_val, test = train_test_split(
            data, test_size=test_size, random_state=random_state, stratify=data[self.target_column]
        )
        
        # Second split: separate validation set from train
        train, val = train_test_split(
            train_val, test_size=val_size/(1-test_size), random_state=random_state, 
            stratify=train_val[self.target_column]
        )
        
        logger.info(f"Train set: {train.shape}")
        logger.info(f"Validation set: {val.shape}")
        logger.info(f"Test set: {test.shape}")
        
        return train, val, test
    
    def save_processed_data(self, train: pd.DataFrame, val: pd.DataFrame, 
                           test: pd.DataFrame, filename_prefix: str = "processed"):
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        train.to_csv(self.processed_data_path / f"{filename_prefix}_train.csv", index=False)
        val.to_csv(self.processed_data_path / f"{filename_prefix}_val.csv", index=False)
        test.to_csv(self.processed_data_path / f"{filename_prefix}_test.csv", index=False)
        
        logger.info(f"Processed data saved to {self.processed_data_path}")
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """Generate data summary statistics."""
        summary = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'numerical_summary': data[self.numerical_columns].describe().to_dict() if self.numerical_columns else {},
            'categorical_summary': {col: data[col].value_counts().to_dict() for col in self.categorical_columns if col in data.columns},
            'target_distribution': data[self.target_column].value_counts().to_dict() if self.target_column in data.columns else {}
        }
        
        return summary


def main():
    """Main function to run data processing pipeline."""
    processor = DataProcessor()
    
    # Example usage - replace with your actual data file
    try:
        # Load data (replace with your actual file path)
        data_file = processor.raw_data_path / "churn_data.csv"
        if data_file.exists():
            data = processor.load_data(str(data_file))
            
            # Clean data
            cleaned_data = processor.clean_data(data)
            
            # Encode categorical features
            encoded_data = processor.encode_categorical_features(cleaned_data)
            
            # Split data
            train, val, test = processor.split_data(encoded_data)
            
            # Save processed data
            processor.save_processed_data(train, val, test)
            
            # Generate summary
            summary = processor.get_data_summary(cleaned_data)
            logger.info("Data processing completed successfully!")
            
        else:
            logger.warning(f"Data file not found: {data_file}")
            logger.info("Please place your data file in the data/raw/ directory")
            
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
