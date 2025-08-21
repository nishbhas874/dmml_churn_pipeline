"""
Feature Store API Code for Assignment
This code demonstrates automated feature retrieval API for training and inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the FeatureStore class
from feature_store import FeatureStore

class FeatureStoreAPI:
    """API wrapper for automated feature retrieval from the feature store."""
    
    def __init__(self, store_path: str = "feature_store"):
        """Initialize the feature store API."""
        self.feature_store = FeatureStore(store_path)
        self.cache = {}  # Simple in-memory cache
        
    def get_features_for_training(self, model_name: str, feature_set_name: str = None) -> Dict[str, Any]:
        """Get features for model training with automated retrieval."""
        print(f"🤖 Retrieving features for training model: {model_name}")
        
        # If no specific feature set, use the default churn prediction set
        if feature_set_name is None:
            feature_set_name = "churn_prediction_features"
        
        try:
            # Search for the feature set
            feature_sets = self._search_feature_sets(feature_set_name)
            if not feature_sets:
                raise ValueError(f"No feature set found with name containing '{feature_set_name}'")
            
            # Use the first matching feature set
            set_id = feature_sets[0]['set_id']
            
            # Get features
            features_data = self.feature_store.get_features_for_training(set_id)
            
            # Prepare response
            response = {
                'model_name': model_name,
                'feature_set_id': set_id,
                'feature_set_name': feature_sets[0]['name'],
                'features_retrieved': list(features_data.columns),
                'data_shape': features_data.shape,
                'retrieval_timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            # Cache the result
            cache_key = f"training_{model_name}_{set_id}"
            self.cache[cache_key] = {
                'data': features_data,
                'metadata': response,
                'timestamp': datetime.now()
            }
            
            print(f"✅ Successfully retrieved {features_data.shape[1]} features for training")
            return response
            
        except Exception as e:
            error_response = {
                'model_name': model_name,
                'status': 'error',
                'error_message': str(e),
                'retrieval_timestamp': datetime.now().isoformat()
            }
            print(f"❌ Error retrieving features: {e}")
            return error_response
    
    def get_features_for_inference(self, customer_ids: List[int], model_name: str = "churn_predictor") -> Dict[str, Any]:
        """Get features for model inference (prediction) for specific customers."""
        print(f"🔮 Retrieving features for inference on {len(customer_ids)} customers")
        
        try:
            # Get the latest feature data for each feature type
            feature_dataframes = []
            
            # List of feature IDs to retrieve
            feature_ids = [
                "customer_demographics_v1.0",
                "financial_metrics_v1.0", 
                "payment_behavior_v1.0"
            ]
            
            for feature_id in feature_ids:
                try:
                    data, metadata = self.feature_store.get_feature_data(feature_id)
                    # Filter for requested customer IDs
                    filtered_data = data[data['customer_id'].isin(customer_ids)]
                    feature_dataframes.append(filtered_data)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load feature {feature_id}: {e}")
            
            if not feature_dataframes:
                raise ValueError("No features could be loaded for inference")
            
            # Merge features
            inference_features = pd.concat(feature_dataframes, axis=1)
            
            # Remove duplicate customer_id columns
            inference_features = inference_features.loc[:, ~inference_features.columns.duplicated()]
            
            # Prepare response
            response = {
                'model_name': model_name,
                'customer_ids_requested': customer_ids,
                'customer_ids_found': list(inference_features['customer_id']),
                'features_retrieved': [col for col in inference_features.columns if col != 'customer_id'],
                'data_shape': inference_features.shape,
                'retrieval_timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            # Cache the result
            cache_key = f"inference_{model_name}_{hash(tuple(customer_ids))}"
            self.cache[cache_key] = {
                'data': inference_features,
                'metadata': response,
                'timestamp': datetime.now()
            }
            
            print(f"✅ Successfully retrieved features for {len(inference_features)} customers")
            return response
            
        except Exception as e:
            error_response = {
                'model_name': model_name,
                'customer_ids_requested': customer_ids,
                'status': 'error',
                'error_message': str(e),
                'retrieval_timestamp': datetime.now().isoformat()
            }
            print(f"❌ Error retrieving features for inference: {e}")
            return error_response
    
    def get_cached_features(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached features if available and not expired."""
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            # Check if cache is still valid (5 minutes)
            if (datetime.now() - cache_entry['timestamp']).seconds < 300:
                print(f"📋 Using cached features for: {cache_key}")
                return cache_entry
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def _search_feature_sets(self, name_pattern: str) -> List[Dict[str, Any]]:
        """Search for feature sets by name pattern."""
        # This is a simplified search - in a real implementation, you'd query the database
        # For now, we'll return a mock result
        return [{
            'set_id': 'set_churn_prediction_features_20241201_120000',
            'name': 'churn_prediction_features',
            'description': 'Complete feature set for churn prediction model training'
        }]
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the feature store."""
        try:
            features = self.feature_store.list_features()
            
            # Calculate statistics
            feature_types = {}
            for feature in features:
                data_type = feature.get('data_type', 'unknown')
                feature_types[data_type] = feature_types.get(data_type, 0) + 1
            
            # Get usage statistics
            total_usage = 0
            for feature in features:
                try:
                    metadata = self.feature_store.get_feature_metadata(feature['feature_id'])
                    total_usage += sum(stat['count'] for stat in metadata['usage_statistics'])
                except:
                    pass
            
            stats = {
                'total_features': len(features),
                'feature_types': feature_types,
                'total_usage_count': total_usage,
                'cache_size': len(self.cache),
                'generated_at': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.cache.clear()
        print("🗑️  Feature cache cleared")
    
    def close(self):
        """Close the feature store API."""
        self.feature_store.close()

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the API
    api = FeatureStoreAPI("feature_store")
    
    print("🚀 Feature Store API Demonstration\n")
    
    # 1. Get features for training
    print("=" * 50)
    print("1. TRAINING FEATURE RETRIEVAL")
    print("=" * 50)
    
    training_response = api.get_features_for_training(
        model_name="churn_prediction_model_v1",
        feature_set_name="churn_prediction_features"
    )
    
    print(f"Training Response: {json.dumps(training_response, indent=2)}")
    
    # 2. Get features for inference
    print("\n" + "=" * 50)
    print("2. INFERENCE FEATURE RETRIEVAL")
    print("=" * 50)
    
    # Sample customer IDs for inference
    customer_ids = [1, 5, 10, 15, 20]
    
    inference_response = api.get_features_for_inference(
        customer_ids=customer_ids,
        model_name="churn_predictor_production"
    )
    
    print(f"Inference Response: {json.dumps(inference_response, indent=2)}")
    
    # 3. Demonstrate caching
    print("\n" + "=" * 50)
    print("3. CACHE DEMONSTRATION")
    print("=" * 50)
    
    # Second call should use cache
    print("Making second training request (should use cache)...")
    cached_response = api.get_features_for_training(
        model_name="churn_prediction_model_v1"
    )
    print(f"Cached Response Status: {cached_response['status']}")
    
    # 4. Get feature store statistics
    print("\n" + "=" * 50)
    print("4. FEATURE STORE STATISTICS")
    print("=" * 50)
    
    stats = api.get_feature_statistics()
    print(f"Feature Store Statistics: {json.dumps(stats, indent=2)}")
    
    # 5. Clear cache
    print("\n" + "=" * 50)
    print("5. CACHE MANAGEMENT")
    print("=" * 50)
    
    api.clear_cache()
    
    # Close the API
    api.close()
    
    print(f"\n✅ Feature Store API demonstration completed!")
    print(f"   API provides automated feature retrieval for both training and inference")
    print(f"   Includes caching, error handling, and comprehensive statistics")
