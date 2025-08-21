"""
Feature Engineering Code Snippet for Assignment
This code demonstrates comprehensive feature engineering for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Comprehensive feature engineering system for churn prediction data."""
    
    def __init__(self):
        """Initialize feature engineer."""
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
    
    def create_aggregated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features from existing data."""
        print("Creating aggregated features...")
        
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
        
        # 5. Risk features
        if 'contract_type' in data.columns and 'payment_method' in data.columns:
            # High-risk customer (month-to-month + electronic check)
            transformed_data['high_risk_customer'] = (
                (data['contract_type'] == 'Month-to-month') & 
                (data['payment_method'] == 'Electronic check')
            ).astype(int)
            features_created.append('high_risk_customer')
        
        print(f"Created {len(features_created)} aggregated features")
        self.transformation_info['features_created'].extend(features_created)
        
        return transformed_data
    
    def create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data."""
        print("Creating derived features...")
        
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
        
        # 3. Polynomial features for key numerical variables
        key_features = ['tenure', 'monthly_charges', 'total_charges']
        for feature in key_features:
            if feature in data.columns:
                # Square
                transformed_data[f'{feature}_squared'] = data[feature] ** 2
                features_created.append(f'{feature}_squared')
                
                # Square root
                transformed_data[f'{feature}_sqrt'] = np.sqrt(data[feature])
                features_created.append(f'{feature}_sqrt')
        
        print(f"Created {len(features_created)} derived features")
        self.transformation_info['features_created'].extend(features_created)
        
        return transformed_data
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale and normalize features."""
        print("Scaling and normalizing features...")
        
        scaled_data = data.copy()
        
        # Get numerical features (excluding target)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if 'churn' in numerical_cols:
            numerical_cols = numerical_cols.drop('churn')
        
        if len(numerical_cols) > 0:
            # Use StandardScaler
            scaler = StandardScaler()
            
            # Scale numerical features
            scaled_data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
            
            # Store scaler for later use
            self.scalers['numerical'] = scaler
            
            scaling_info = {
                'method': 'standard',
                'columns': numerical_cols.tolist(),
                'scaler_params': scaler.get_params()
            }
            
            self.transformation_info['transformations_applied']['scaling'] = scaling_info
            print(f"Scaled {len(numerical_cols)} numerical features using standard scaling")
        
        return scaled_data
    
    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select the most important features."""
        print("Selecting important features...")
        
        if 'churn' not in data.columns:
            print("Target column 'churn' not found, skipping feature selection")
            return data
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col != 'churn']
        X = data[feature_cols]
        y = data['churn']
        
        # Feature selection using mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=min(20, len(feature_cols)))
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Update dataset
        selected_data = data[selected_features + ['churn']]
        
        # Store selector for later use
        self.feature_selectors['main'] = selector
        
        selection_info = {
            'method': 'mutual_info',
            'k_best': 20,
            'selected_features': selected_features,
            'feature_scores': dict(zip(feature_cols, selector.scores_))
        }
        
        self.transformation_info['transformations_applied']['feature_selection'] = selection_info
        print(f"Selected {len(selected_features)} features out of {len(feature_cols)}")
        
        return selected_data
    
    def transform_data(self, data: pd.DataFrame):
        """Apply complete feature engineering pipeline."""
        print("Starting complete feature engineering pipeline...")
        
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
        
        # Step 4: Select features
        data = self.select_features(data)
        
        # Store final statistics
        self.transformation_info['statistics'] = {
            'final_shape': data.shape,
            'total_features_created': len(self.transformation_info['features_created']),
            'numerical_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        print(f"Feature engineering completed. Final shape: {data.shape}")
        return data, self.transformation_info

# Example usage
if __name__ == "__main__":
    # Create sample data
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
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Initialize and run feature engineering
    feature_engineer = FeatureEngineer()
    transformed_data, transformation_info = feature_engineer.transform_data(sample_data)
    
    print(f"\nOriginal shape: {sample_data.shape}")
    print(f"Transformed shape: {transformed_data.shape}")
    print(f"Features created: {len(transformation_info['features_created'])}")
    print(f"Sample features: {transformation_info['features_created'][:10]}")
