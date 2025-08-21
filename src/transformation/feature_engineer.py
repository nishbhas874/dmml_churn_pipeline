"""
Feature Engineering System for Churn Prediction Pipeline.
Handles feature creation, aggregation, and transformation.
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA


class FeatureEngineer:
    """Comprehensive feature engineering system for churn prediction data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer with configuration."""
        self.config = self._load_config(config_path)
        self.features_config = self.config.get('features', {})
        self.transformation_config = self.config.get('transformation', {})
        
        # Initialize transformation components
        self.scalers = {}
        self.feature_selectors = {}
        self.pca_models = {}
        
        # Store transformation metadata
        self.transformation_info = {
            'timestamp': datetime.now().isoformat(),
            'features_created': [],
            'transformations_applied': {},
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
    
    def create_aggregated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from existing data."""
        logger.info("Creating aggregated features...")
        
        transformed_data = data.copy()
        features_created = []
        
        # 1. Customer spending features
        if 'monthly_charges' in data.columns and 'total_charges' in data.columns:
            # Average monthly spending
            transformed_data['avg_monthly_spending'] = data['total_charges'] / data['tenure'].replace(0, 1)
            features_created.append('avg_monthly_spending')
            
            # Spending trend (positive if monthly charges > average)
            transformed_data['spending_trend'] = data['monthly_charges'] - transformed_data['avg_monthly_spending']
            features_created.append('spending_trend')
            
            # Spending volatility (if we have historical data)
            if 'tenure' in data.columns:
                transformed_data['spending_volatility'] = data['monthly_charges'] / transformed_data['avg_monthly_spending']
                features_created.append('spending_volatility')
        
        # 2. Service usage features
        service_columns = [col for col in data.columns if any(service in col.lower() 
                        for service in ['online_security', 'online_backup', 'device_protection', 
                                      'tech_support', 'streaming_tv', 'streaming_movies'])]
        
        if service_columns:
            # Count of services used
            transformed_data['services_count'] = data[service_columns].apply(
                lambda x: (x == 'Yes').sum(), axis=1
            )
            features_created.append('services_count')
            
            # Service usage ratio
            transformed_data['service_usage_ratio'] = transformed_data['services_count'] / len(service_columns)
            features_created.append('service_usage_ratio')
        
        # 3. Contract and payment features
        if 'contract_type' in data.columns:
            # Contract length (numeric)
            contract_mapping = {
                'Month-to-month': 1,
                'One year': 12,
                'Two year': 24
            }
            transformed_data['contract_length_months'] = data['contract_type'].map(contract_mapping)
            features_created.append('contract_length_months')
            
            # Contract commitment level
            transformed_data['contract_commitment'] = transformed_data['contract_length_months'] / 24.0
            features_created.append('contract_commitment')
        
        # 4. Customer lifecycle features
        if 'tenure' in data.columns:
            # Customer lifecycle stage
            transformed_data['lifecycle_stage'] = pd.cut(
                data['tenure'], 
                bins=[0, 12, 24, 48, float('inf')], 
                labels=['New', 'Established', 'Mature', 'Long-term']
            )
            features_created.append('lifecycle_stage')
            
            # Tenure categories
            transformed_data['tenure_category'] = pd.cut(
                data['tenure'],
                bins=[0, 6, 12, 24, 60, float('inf')],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
            )
            features_created.append('tenure_category')
        
        # 5. Demographic features
        if 'age' in data.columns:
            # Age groups
            transformed_data['age_group'] = pd.cut(
                data['age'],
                bins=[0, 30, 45, 60, float('inf')],
                labels=['Young', 'Middle-aged', 'Senior', 'Elderly']
            )
            features_created.append('age_group')
        
        # 6. Interaction features
        if all(col in data.columns for col in ['monthly_charges', 'tenure']):
            # Value per month
            transformed_data['value_per_month'] = data['monthly_charges'] * transformed_data['contract_commitment']
            features_created.append('value_per_month')
        
        # 7. Risk features
        if 'contract_type' in data.columns and 'payment_method' in data.columns:
            # High-risk customer (month-to-month + electronic check)
            transformed_data['high_risk_customer'] = (
                (data['contract_type'] == 'Month-to-month') & 
                (data['payment_method'] == 'Electronic check')
            ).astype(int)
            features_created.append('high_risk_customer')
        
        # 8. Service bundle features
        if all(col in data.columns for col in ['internet_service', 'streaming_tv', 'streaming_movies']):
            # Premium service bundle
            transformed_data['premium_bundle'] = (
                (data['internet_service'] == 'Fiber optic') & 
                (data['streaming_tv'] == 'Yes') & 
                (data['streaming_movies'] == 'Yes')
            ).astype(int)
            features_created.append('premium_bundle')
        
        # 9. Customer satisfaction indicators
        if 'tech_support' in data.columns and 'online_security' in data.columns:
            # Support usage indicator
            transformed_data['support_usage'] = (
                (data['tech_support'] == 'Yes') | 
                (data['online_security'] == 'Yes')
            ).astype(int)
            features_created.append('support_usage')
        
        # 10. Financial features
        if 'monthly_charges' in data.columns:
            # Spending percentile
            transformed_data['spending_percentile'] = data['monthly_charges'].rank(pct=True)
            features_created.append('spending_percentile')
            
            # High spender flag
            transformed_data['high_spender'] = (transformed_data['spending_percentile'] > 0.8).astype(int)
            features_created.append('high_spender')
        
        logger.info(f"Created {len(features_created)} aggregated features")
        self.transformation_info['features_created'].extend(features_created)
        
        return transformed_data
    
    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data."""
        logger.info("Creating derived features...")
        
        transformed_data = data.copy()
        features_created = []
        
        # 1. Time-based features
        if 'tenure' in data.columns:
            # Tenure squared (for non-linear relationships)
            transformed_data['tenure_squared'] = data['tenure'] ** 2
            features_created.append('tenure_squared')
            
            # Tenure log (for diminishing returns)
            transformed_data['tenure_log'] = np.log1p(data['tenure'])
            features_created.append('tenure_log')
            
            # Tenure inverse (for early churn patterns)
            transformed_data['tenure_inverse'] = 1 / (data['tenure'] + 1)
            features_created.append('tenure_inverse')
        
        # 2. Financial ratio features
        if all(col in data.columns for col in ['monthly_charges', 'total_charges', 'tenure']):
            # Monthly to total charges ratio
            transformed_data['monthly_total_ratio'] = data['monthly_charges'] / (data['total_charges'] + 1)
            features_created.append('monthly_total_ratio')
            
            # Average daily spending
            transformed_data['daily_spending'] = data['total_charges'] / (data['tenure'] * 30 + 1)
            features_created.append('daily_spending')
        
        # 3. Interaction features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 2:
            # Create interaction features between numerical columns
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    if col1 != col2:
                        # Multiplication interaction
                        interaction_name = f"{col1}_{col2}_interaction"
                        transformed_data[interaction_name] = data[col1] * data[col2]
                        features_created.append(interaction_name)
                        
                        # Ratio interaction
                        ratio_name = f"{col1}_{col2}_ratio"
                        transformed_data[ratio_name] = data[col1] / (data[col2] + 1)
                        features_created.append(ratio_name)
        
        # 4. Polynomial features for key numerical variables
        key_features = ['tenure', 'monthly_charges', 'total_charges']
        for feature in key_features:
            if feature in data.columns:
                # Square
                transformed_data[f'{feature}_squared'] = data[feature] ** 2
                features_created.append(f'{feature}_squared')
                
                # Cube
                transformed_data[f'{feature}_cubed'] = data[feature] ** 3
                features_created.append(f'{feature}_cubed')
                
                # Square root
                transformed_data[f'{feature}_sqrt'] = np.sqrt(data[feature])
                features_created.append(f'{feature}_sqrt')
        
        # 5. Binning features
        if 'age' in data.columns:
            # Age bins
            age_bins = [0, 25, 35, 45, 55, 65, 100]
            age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            transformed_data['age_bin'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)
            features_created.append('age_bin')
        
        if 'monthly_charges' in data.columns:
            # Spending bins
            spending_bins = [0, 30, 50, 70, 100, 200]
            spending_labels = ['Low', 'Medium', 'High', 'Very High', 'Premium']
            transformed_data['spending_bin'] = pd.cut(data['monthly_charges'], bins=spending_bins, labels=spending_labels)
            features_created.append('spending_bin')
        
        # 6. Categorical encoding features
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in data.columns:
                # Frequency encoding
                freq_encoding = data[col].value_counts(normalize=True)
                transformed_data[f'{col}_freq_encoded'] = data[col].map(freq_encoding)
                features_created.append(f'{col}_freq_encoded')
                
                # Target encoding (if target is available)
                target_col = self.features_config.get('target_column')
                if target_col and target_col in data.columns:
                    target_encoding = data.groupby(col)[target_col].mean()
                    transformed_data[f'{col}_target_encoded'] = data[col].map(target_encoding)
                    features_created.append(f'{col}_target_encoded')
        
        # 7. Statistical features
        if len(numerical_cols) > 0:
            # Z-score normalization for key features
            for col in ['tenure', 'monthly_charges', 'total_charges']:
                if col in data.columns:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    transformed_data[f'{col}_zscore'] = (data[col] - mean_val) / (std_val + 1e-8)
                    features_created.append(f'{col}_zscore')
        
        logger.info(f"Created {len(features_created)} derived features")
        self.transformation_info['features_created'].extend(features_created)
        
        return transformed_data
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale and normalize features."""
        logger.info("Scaling and normalizing features...")
        
        scaled_data = data.copy()
        scaling_info = {}
        
        # Get numerical features (excluding target)
        target_col = self.features_config.get('target_column')
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if target_col and target_col in numerical_cols:
            numerical_cols = numerical_cols.drop(target_col)
        
        if len(numerical_cols) > 0:
            scaling_method = self.transformation_config.get('scaling', 'standard')
            
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            # Scale numerical features
            scaled_data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
            
            # Store scaler for later use
            self.scalers['numerical'] = scaler
            
            scaling_info = {
                'method': scaling_method,
                'columns': numerical_cols.tolist(),
                'scaler_params': scaler.get_params()
            }
        
        self.transformation_info['transformations_applied']['scaling'] = scaling_info
        logger.info(f"Scaled {len(numerical_cols)} numerical features using {scaling_method} scaling")
        
        return scaled_data
    
    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select the most important features."""
        logger.info("Selecting important features...")
        
        target_col = self.features_config.get('target_column')
        if not target_col or target_col not in data.columns:
            logger.warning("Target column not found, skipping feature selection")
            return data
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols]
        y = data[target_col]
        
        # Feature selection method
        selection_method = self.transformation_config.get('feature_selection', 'mutual_info')
        k_best = self.transformation_config.get('k_best', 20)
        
        if selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k_best, len(feature_cols)))
        elif selection_method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k_best, len(feature_cols)))
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k_best, len(feature_cols)))
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Update dataset
        selected_data = data[selected_features + [target_col]]
        
        # Store selector for later use
        self.feature_selectors['main'] = selector
        
        selection_info = {
            'method': selection_method,
            'k_best': k_best,
            'selected_features': selected_features,
            'feature_scores': dict(zip(feature_cols, selector.scores_))
        }
        
        self.transformation_info['transformations_applied']['feature_selection'] = selection_info
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_cols)}")
        
        return selected_data
    
    def apply_pca(self, data: pd.DataFrame, n_components: int = None) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction."""
        logger.info("Applying PCA for dimensionality reduction...")
        
        target_col = self.features_config.get('target_column')
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols]
        
        if n_components is None:
            n_components = min(len(feature_cols), 10)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create new dataframe with PCA components
        pca_columns = [f'pca_component_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=data.index)
        
        # Add target column back
        if target_col in data.columns:
            pca_df[target_col] = data[target_col]
        
        # Store PCA model
        self.pca_models['main'] = pca
        
        pca_info = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist()
        }
        
        self.transformation_info['transformations_applied']['pca'] = pca_info
        logger.info(f"Applied PCA with {n_components} components, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return pca_df
    
    def transform_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply complete feature engineering pipeline."""
        logger.info("Starting complete feature engineering pipeline...")
        
        # Reset transformation info
        self.transformation_info = {
            'timestamp': datetime.now().isoformat(),
            'features_created': [],
            'transformations_applied': {},
            'statistics': {}
        }
        
        # Step 1: Create aggregated features
        data = self.create_aggregated_features(data)
        
        # Step 2: Create derived features
        data = self.create_derived_features(data)
        
        # Step 3: Scale features
        data = self.scale_features(data)
        
        # Step 4: Select features (optional)
        if self.transformation_config.get('apply_feature_selection', True):
            data = self.select_features(data)
        
        # Step 5: Apply PCA (optional)
        if self.transformation_config.get('apply_pca', False):
            data = self.apply_pca(data)
        
        # Store final statistics
        self.transformation_info['statistics'] = {
            'final_shape': data.shape,
            'total_features_created': len(self.transformation_info['features_created']),
            'numerical_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        logger.info(f"Feature engineering completed. Final shape: {data.shape}")
        return data, self.transformation_info
    
    def save_transformation_info(self, output_path: str = None) -> str:
        """Save transformation information to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/transformation_info_{timestamp}.json"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save transformation info
        import json
        with open(output_path, 'w') as f:
            json.dump(self.transformation_info, f, indent=2, default=str)
        
        logger.info(f"Transformation info saved to: {output_path}")
        return output_path


def create_feature_engineer(config_path: str = "config/config.yaml") -> FeatureEngineer:
    """Factory function to create feature engineer instance."""
    return FeatureEngineer(config_path)
