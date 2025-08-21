"""
Feature engineering module for churn prediction pipeline.
Handles feature creation, selection, and transformation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from loguru import logger
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineer for churn prediction pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer with configuration."""
        self.config = self._load_config(config_path)
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.feature_config = self.config['features']
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical variables."""
        logger.info("Creating interaction features...")
        
        numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
        
        # Create ratio features
        if 'tenure' in data.columns and 'monthly_charges' in data.columns:
            data['tenure_monthly_ratio'] = data['tenure'] / (data['monthly_charges'] + 1e-8)
            data['monthly_tenure_ratio'] = data['monthly_charges'] / (data['tenure'] + 1e-8)
        
        if 'total_charges' in data.columns and 'tenure' in data.columns:
            data['avg_monthly_charges'] = data['total_charges'] / (data['tenure'] + 1e-8)
        
        if 'total_charges' in data.columns and 'monthly_charges' in data.columns:
            data['charges_ratio'] = data['total_charges'] / (data['monthly_charges'] + 1e-8)
        
        # Create polynomial features for important numerical variables
        important_numerical = ['tenure', 'monthly_charges', 'total_charges']
        for col in important_numerical:
            if col in data.columns:
                data[f'{col}_squared'] = data[col] ** 2
                data[f'{col}_cubed'] = data[col] ** 3
        
        logger.info(f"Created interaction features. New shape: {data.shape}")
        return data
    
    def create_time_based_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from tenure."""
        logger.info("Creating time-based features...")
        
        if 'tenure' in data.columns:
            # Convert tenure to years
            data['tenure_years'] = data['tenure'] / 12
            
            # Create tenure categories
            data['tenure_category'] = pd.cut(
                data['tenure'], 
                bins=[0, 12, 24, 36, 48, 60, float('inf')],
                labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5+yr']
            )
            
            # Create tenure segments
            data['tenure_segment'] = pd.cut(
                data['tenure'],
                bins=[0, 6, 12, 24, 48, float('inf')],
                labels=['New', 'Recent', 'Established', 'Long-term', 'Loyal']
            )
            
            # Is new customer (less than 6 months)
            data['is_new_customer'] = (data['tenure'] <= 6).astype(int)
            
            # Is long-term customer (more than 2 years)
            data['is_long_term_customer'] = (data['tenure'] > 24).astype(int)
        
        logger.info(f"Created time-based features. New shape: {data.shape}")
        return data
    
    def create_charge_based_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on charges."""
        logger.info("Creating charge-based features...")
        
        if 'monthly_charges' in data.columns:
            # Monthly charges categories
            data['monthly_charges_category'] = pd.cut(
                data['monthly_charges'],
                bins=[0, 30, 60, 90, 120, float('inf')],
                labels=['Low', 'Medium', 'High', 'Premium', 'Ultra']
            )
            
            # Is high spender
            data['is_high_spender'] = (data['monthly_charges'] > data['monthly_charges'].quantile(0.75)).astype(int)
            
            # Is low spender
            data['is_low_spender'] = (data['monthly_charges'] < data['monthly_charges'].quantile(0.25)).astype(int)
        
        if 'total_charges' in data.columns:
            # Total charges categories
            data['total_charges_category'] = pd.cut(
                data['total_charges'],
                bins=[0, 500, 1000, 2000, 5000, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        logger.info(f"Created charge-based features. New shape: {data.shape}")
        return data
    
    def create_service_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on service usage."""
        logger.info("Creating service-based features...")
        
        # Count of additional services
        service_columns = [
            'online_security', 'online_backup', 'device_protection', 
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        available_services = [col for col in service_columns if col in data.columns]
        
        if available_services:
            # Count of Yes responses for services
            data['total_services'] = data[available_services].apply(
                lambda x: (x == 'Yes').sum(), axis=1
            )
            
            # Service usage categories
            data['service_usage_category'] = pd.cut(
                data['total_services'],
                bins=[0, 1, 2, 3, 4, float('inf')],
                labels=['None', 'Low', 'Medium', 'High', 'Very High']
            )
            
            # Has any additional services
            data['has_additional_services'] = (data['total_services'] > 0).astype(int)
            
            # Is heavy service user
            data['is_heavy_service_user'] = (data['total_services'] >= 3).astype(int)
        
        logger.info(f"Created service-based features. New shape: {data.shape}")
        return data
    
    def create_contract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on contract information."""
        logger.info("Creating contract-based features...")
        
        if 'contract_type' in data.columns:
            # Contract type categories
            data['is_monthly_contract'] = (data['contract_type'] == 'Month-to-month').astype(int)
            data['is_yearly_contract'] = (data['contract_type'] == 'One year').astype(int)
            data['is_two_year_contract'] = (data['contract_type'] == 'Two year').astype(int)
            
            # Contract stability (longer contracts are more stable)
            contract_stability = {
                'Month-to-month': 1,
                'One year': 2,
                'Two year': 3
            }
            data['contract_stability'] = data['contract_type'].map(contract_stability)
        
        if 'payment_method' in data.columns:
            # Payment method categories
            data['is_electronic_payment'] = data['payment_method'].str.contains('Electronic').astype(int)
            data['is_mailed_payment'] = data['payment_method'].str.contains('Mailed').astype(int)
            data['is_bank_transfer'] = data['payment_method'].str.contains('Bank transfer').astype(int)
            data['is_credit_card'] = data['payment_method'].str.contains('Credit card').astype(int)
        
        logger.info(f"Created contract-based features. New shape: {data.shape}")
        return data
    
    def create_demographic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create demographic-based features."""
        logger.info("Creating demographic features...")
        
        if 'age' in data.columns:
            # Age categories
            data['age_category'] = pd.cut(
                data['age'],
                bins=[0, 30, 45, 60, 75, float('inf')],
                labels=['Young', 'Young Adult', 'Middle Age', 'Senior', 'Elderly']
            )
            
            # Is young customer
            data['is_young_customer'] = (data['age'] <= 30).astype(int)
            
            # Is senior customer
            data['is_senior_customer'] = (data['age'] >= 60).astype(int)
        
        if 'gender' in data.columns:
            # Gender encoding
            data['is_male'] = (data['gender'] == 'Male').astype(int)
            data['is_female'] = (data['gender'] == 'Female').astype(int)
        
        if 'partner' in data.columns:
            data['has_partner'] = (data['partner'] == 'Yes').astype(int)
        
        if 'dependents' in data.columns:
            data['has_dependents'] = (data['dependents'] == 'Yes').astype(int)
        
        logger.info(f"Created demographic features. New shape: {data.shape}")
        return data
    
    def scale_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features."""
        logger.info(f"Scaling features using {method} method...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        scaled_data = data.copy()
        scaled_data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        
        logger.info(f"Scaled {len(numerical_cols)} numerical features")
        return scaled_data
    
    def select_features(self, data: pd.DataFrame, target_col: str, 
                       method: str = 'mutual_info', k: int = 20) -> pd.DataFrame:
        """Select the best features using various methods."""
        logger.info(f"Selecting features using {method} method...")
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols]
        y = data[target_col]
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'chi2':
            self.feature_selector = SelectKBest(score_func=chi2, k=k)
        elif method == 'f_classif':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'recursive':
            # Use Random Forest for recursive feature elimination
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_selector = RFE(estimator=rf, n_features_to_select=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit and transform
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
        else:
            # For RFE
            self.selected_features = X.columns[self.feature_selector.support_].tolist()
        
        # Create new dataframe with selected features
        selected_data = data[self.selected_features + [target_col]]
        
        logger.info(f"Selected {len(self.selected_features)} features out of {len(feature_cols)}")
        logger.info(f"Selected features: {self.selected_features}")
        
        return selected_data
    
    def engineer_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")
        
        # Create all feature types
        data = self.create_interaction_features(data)
        data = self.create_time_based_features(data)
        data = self.create_charge_based_features(data)
        data = self.create_service_features(data)
        data = self.create_contract_features(data)
        data = self.create_demographic_features(data)
        
        # Handle categorical features (convert to numeric for scaling)
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns
        for col in categorical_cols:
            if col != target_col:
                data[col] = pd.Categorical(data[col]).codes
        
        # Scale features
        scaling_method = self.feature_config['scaling']['method']
        data = self.scale_features(data, method=scaling_method)
        
        # Select features
        selection_method = self.feature_config['feature_selection']['method']
        k_best = self.feature_config['feature_selection']['k_best']
        data = self.select_features(data, target_col, method=selection_method, k=k_best)
        
        logger.info(f"Feature engineering completed. Final shape: {data.shape}")
        return data
    
    def save_feature_importance(self, output_path: str = "results/"):
        """Save feature importance scores."""
        if self.feature_selector and hasattr(self.feature_selector, 'scores_'):
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance_score': self.feature_selector.scores_
            }).sort_values('importance_score', ascending=False)
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            importance_df.to_csv(output_path / "feature_importance.csv", index=False)
            logger.info(f"Feature importance saved to {output_path / 'feature_importance.csv'}")


def main():
    """Main function to run feature engineering pipeline."""
    engineer = FeatureEngineer()
    
    try:
        # Load processed data
        train_file = engineer.processed_data_path / "processed_train.csv"
        val_file = engineer.processed_data_path / "processed_val.csv"
        test_file = engineer.processed_data_path / "processed_test.csv"
        
        if train_file.exists():
            train = pd.read_csv(train_file)
            val = pd.read_csv(val_file)
            test = pd.read_csv(test_file)
            
            target_col = engineer.config['features']['target_column']
            
            # Engineer features for all datasets
            train_engineered = engineer.engineer_features(train, target_col)
            val_engineered = engineer.engineer_features(val, target_col)
            test_engineered = engineer.engineer_features(test, target_col)
            
            # Save engineered data
            train_engineered.to_csv(engineer.processed_data_path / "engineered_train.csv", index=False)
            val_engineered.to_csv(engineer.processed_data_path / "engineered_val.csv", index=False)
            test_engineered.to_csv(engineer.processed_data_path / "engineered_test.csv", index=False)
            
            # Save feature importance
            engineer.save_feature_importance()
            
            logger.info("Feature engineering completed successfully!")
            
        else:
            logger.warning("Processed data files not found. Please run data processing first.")
            
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


if __name__ == "__main__":
    main()
