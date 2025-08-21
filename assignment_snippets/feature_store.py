"""
Feature Store Code for Assignment
This code demonstrates a comprehensive feature store system for managing engineered features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import pickle
import warnings
warnings.filterwarnings('ignore')

class FeatureStore:
    """Comprehensive feature store system for managing engineered features."""
    
    def __init__(self, store_path: str = "feature_store"):
        """Initialize feature store."""
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Database for metadata
        self.db_path = self.store_path / "feature_store.db"
        self.conn = None
        
        # Initialize store
        self._initialize_store()
        
        # Feature registry
        self.feature_registry = {}
        self._load_feature_registry()
    
    def _initialize_store(self):
        """Initialize the feature store database."""
        self.conn = sqlite3.connect(self.db_path)
        
        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                source_table TEXT,
                data_type TEXT,
                version TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_versions (
                version_id TEXT PRIMARY KEY,
                feature_id TEXT,
                version TEXT,
                schema_hash TEXT,
                data_hash TEXT,
                created_at TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (feature_id) REFERENCES features (feature_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_sets (
                set_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                features TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_id TEXT,
                model_name TEXT,
                usage_type TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (feature_id) REFERENCES features (feature_id)
            )
        """)
        
        self.conn.commit()
    
    def _load_feature_registry(self):
        """Load feature registry from file."""
        registry_path = self.store_path / "feature_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.feature_registry = json.load(f)
    
    def _save_feature_registry(self):
        """Save feature registry to file."""
        registry_path = self.store_path / "feature_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.feature_registry, f, indent=2)
    
    def register_feature(self, name: str, description: str, source_table: str = None, 
                        data_type: str = None, version: str = "1.0") -> str:
        """Register a new feature in the feature store."""
        feature_id = f"{name}_v{version}"
        timestamp = datetime.now().isoformat()
        
        # Insert into database
        self.conn.execute("""
            INSERT OR REPLACE INTO features 
            (feature_id, name, description, source_table, data_type, version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (feature_id, name, description, source_table, data_type, version, timestamp, timestamp))
        
        # Update registry
        self.feature_registry[feature_id] = {
            'name': name,
            'description': description,
            'source_table': source_table,
            'data_type': data_type,
            'version': version,
            'created_at': timestamp,
            'updated_at': timestamp,
            'is_active': True
        }
        
        self._save_feature_registry()
        self.conn.commit()
        
        print(f"✅ Registered feature: {feature_id}")
        return feature_id
    
    def store_feature_data(self, feature_id: str, data: pd.DataFrame, 
                          metadata: Dict[str, Any] = None) -> str:
        """Store feature data with versioning."""
        timestamp = datetime.now().isoformat()
        
        # Generate hashes
        schema_hash = hashlib.md5(str(data.dtypes.to_dict()).encode()).hexdigest()
        data_hash = hashlib.md5(data.to_string().encode()).hexdigest()
        
        # Create version ID
        version_id = f"{feature_id}_{timestamp.replace(':', '-')}"
        
        # Store data
        data_path = self.store_path / f"{version_id}.parquet"
        data.to_parquet(data_path, index=False)
        
        # Store metadata
        if metadata is None:
            metadata = {}
        
        metadata['data_shape'] = data.shape
        metadata['columns'] = list(data.columns)
        metadata['dtypes'] = data.dtypes.to_dict()
        metadata['storage_path'] = str(data_path)
        
        # Insert version record
        self.conn.execute("""
            INSERT INTO feature_versions 
            (version_id, feature_id, version, schema_hash, data_hash, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (version_id, feature_id, self.feature_registry[feature_id]['version'], 
              schema_hash, data_hash, timestamp, json.dumps(metadata)))
        
        self.conn.commit()
        
        print(f"✅ Stored feature data: {version_id}")
        return version_id
    
    def get_feature_data(self, feature_id: str, version: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Retrieve feature data."""
        if version is None:
            # Get latest version
            cursor = self.conn.execute("""
                SELECT version_id, metadata FROM feature_versions 
                WHERE feature_id = ? ORDER BY created_at DESC LIMIT 1
            """, (feature_id,))
        else:
            cursor = self.conn.execute("""
                SELECT version_id, metadata FROM feature_versions 
                WHERE feature_id = ? AND version = ? ORDER BY created_at DESC LIMIT 1
            """, (feature_id, version))
        
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Feature {feature_id} not found")
        
        version_id, metadata_str = result
        metadata = json.loads(metadata_str)
        
        # Load data
        data_path = metadata['storage_path']
        data = pd.read_parquet(data_path)
        
        return data, metadata
    
    def create_feature_set(self, name: str, feature_ids: List[str], description: str = "") -> str:
        """Create a feature set for model training."""
        set_id = f"set_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()
        
        # Insert feature set
        self.conn.execute("""
            INSERT INTO feature_sets 
            (set_id, name, description, features, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (set_id, name, description, json.dumps(feature_ids), timestamp, timestamp))
        
        self.conn.commit()
        
        print(f"✅ Created feature set: {set_id}")
        return set_id
    
    def get_feature_set(self, set_id: str) -> Dict[str, Any]:
        """Retrieve feature set information."""
        cursor = self.conn.execute("""
            SELECT * FROM feature_sets WHERE set_id = ?
        """, (set_id,))
        
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Feature set {set_id} not found")
        
        columns = [desc[0] for desc in cursor.description]
        feature_set = dict(zip(columns, result))
        feature_set['features'] = json.loads(feature_set['features'])
        
        return feature_set
    
    def get_features_for_training(self, set_id: str) -> pd.DataFrame:
        """Get all features for a feature set for model training."""
        feature_set = self.get_feature_set(set_id)
        feature_ids = feature_set['features']
        
        # Load all features
        feature_dataframes = []
        for feature_id in feature_ids:
            try:
                data, metadata = self.get_feature_data(feature_id)
                feature_dataframes.append(data)
            except Exception as e:
                print(f"⚠️  Warning: Could not load feature {feature_id}: {e}")
        
        if not feature_dataframes:
            raise ValueError("No features could be loaded")
        
        # Merge features (assuming they have the same index)
        combined_data = pd.concat(feature_dataframes, axis=1)
        
        # Log usage
        self._log_feature_usage(feature_ids, "training", "feature_set_retrieval")
        
        return combined_data
    
    def _log_feature_usage(self, feature_ids: List[str], usage_type: str, model_name: str = "unknown"):
        """Log feature usage for monitoring."""
        timestamp = datetime.now().isoformat()
        
        for feature_id in feature_ids:
            self.conn.execute("""
                INSERT INTO feature_usage 
                (feature_id, model_name, usage_type, timestamp)
                VALUES (?, ?, ?, ?)
            """, (feature_id, model_name, usage_type, timestamp))
        
        self.conn.commit()
    
    def get_feature_metadata(self, feature_id: str) -> Dict[str, Any]:
        """Get comprehensive feature metadata."""
        # Get feature info
        cursor = self.conn.execute("""
            SELECT * FROM features WHERE feature_id = ?
        """, (feature_id,))
        
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"Feature {feature_id} not found")
        
        columns = [desc[0] for desc in cursor.description]
        feature_info = dict(zip(columns, result))
        
        # Get version history
        cursor = self.conn.execute("""
            SELECT * FROM feature_versions WHERE feature_id = ? ORDER BY created_at DESC
        """, (feature_id,))
        
        versions = []
        for row in cursor.fetchall():
            columns = [desc[0] for desc in cursor.description]
            version_info = dict(zip(columns, row))
            version_info['metadata'] = json.loads(version_info['metadata'])
            versions.append(version_info)
        
        # Get usage statistics
        cursor = self.conn.execute("""
            SELECT usage_type, COUNT(*) as count, 
                   MIN(timestamp) as first_used, 
                   MAX(timestamp) as last_used
            FROM feature_usage WHERE feature_id = ?
            GROUP BY usage_type
        """, (feature_id,))
        
        usage_stats = []
        for row in cursor.fetchall():
            usage_stats.append({
                'usage_type': row[0],
                'count': row[1],
                'first_used': row[2],
                'last_used': row[3]
            })
        
        return {
            'feature_info': feature_info,
            'versions': versions,
            'usage_statistics': usage_stats
        }
    
    def list_features(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all features in the store."""
        query = "SELECT * FROM features"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY updated_at DESC"
        
        cursor = self.conn.execute(query)
        features = []
        
        for row in cursor.fetchall():
            columns = [desc[0] for desc in cursor.description]
            feature = dict(zip(columns, row))
            features.append(feature)
        
        return features
    
    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """Search features by name or description."""
        cursor = self.conn.execute("""
            SELECT * FROM features 
            WHERE (name LIKE ? OR description LIKE ?) AND is_active = 1
            ORDER BY updated_at DESC
        """, (f"%{query}%", f"%{query}%"))
        
        features = []
        for row in cursor.fetchall():
            columns = [desc[0] for desc in cursor.description]
            feature = dict(zip(columns, row))
            features.append(feature)
        
        return features
    
    def generate_feature_documentation(self) -> str:
        """Generate comprehensive feature documentation."""
        features = self.list_features()
        
        doc = "# Feature Store Documentation\n\n"
        doc += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        doc += f"Total Features: {len(features)}\n\n"
        
        for feature in features:
            doc += f"## {feature['name']} (v{feature['version']})\n\n"
            doc += f"**Feature ID:** {feature['feature_id']}\n\n"
            doc += f"**Description:** {feature['description']}\n\n"
            doc += f"**Source Table:** {feature['source_table'] or 'N/A'}\n\n"
            doc += f"**Data Type:** {feature['data_type'] or 'N/A'}\n\n"
            doc += f"**Created:** {feature['created_at']}\n\n"
            doc += f"**Last Updated:** {feature['updated_at']}\n\n"
            
            # Get metadata
            try:
                metadata = self.get_feature_metadata(feature['feature_id'])
                doc += f"**Versions:** {len(metadata['versions'])}\n\n"
                doc += f"**Usage Count:** {sum(stat['count'] for stat in metadata['usage_statistics'])}\n\n"
            except:
                doc += "**Metadata:** Unable to retrieve\n\n"
            
            doc += "---\n\n"
        
        return doc
    
    def save_documentation(self, output_path: str = None) -> str:
        """Save feature documentation to file."""
        if output_path is None:
            output_path = self.store_path / "feature_documentation.md"
        
        doc = self.generate_feature_documentation()
        with open(output_path, 'w') as f:
            f.write(doc)
        
        print(f"✅ Feature documentation saved to: {output_path}")
        return str(output_path)
    
    def close(self):
        """Close the feature store connection."""
        if self.conn:
            self.conn.close()

# Example usage
if __name__ == "__main__":
    # Initialize feature store
    feature_store = FeatureStore("feature_store")
    
    # Create sample feature data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'tenure': np.random.exponential(30, n_samples).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Register and store features
    print("🔧 Registering and storing features...")
    
    # 1. Basic customer features
    feature_store.register_feature(
        name="customer_demographics",
        description="Basic customer demographic information including age, tenure, and contract details",
        source_table="customers",
        data_type="demographic"
    )
    
    customer_features = customer_data[['customer_id', 'age', 'tenure', 'contract_type']].copy()
    feature_store.store_feature_data("customer_demographics_v1.0", customer_features)
    
    # 2. Financial features
    feature_store.register_feature(
        name="financial_metrics",
        description="Customer financial metrics including monthly and total charges",
        source_table="customers",
        data_type="financial"
    )
    
    financial_features = customer_data[['customer_id', 'monthly_charges', 'total_charges']].copy()
    feature_store.store_feature_data("financial_metrics_v1.0", financial_features)
    
    # 3. Payment behavior features
    feature_store.register_feature(
        name="payment_behavior",
        description="Customer payment method and behavior patterns",
        source_table="customers",
        data_type="behavioral"
    )
    
    payment_features = customer_data[['customer_id', 'payment_method']].copy()
    feature_store.store_feature_data("payment_behavior_v1.0", payment_features)
    
    # 4. Target variable
    feature_store.register_feature(
        name="churn_target",
        description="Customer churn status (target variable for prediction)",
        source_table="customers",
        data_type="target"
    )
    
    target_features = customer_data[['customer_id', 'churn']].copy()
    feature_store.store_feature_data("churn_target_v1.0", target_features)
    
    # Create feature set for training
    print("\n📦 Creating feature set for training...")
    feature_set_id = feature_store.create_feature_set(
        name="churn_prediction_features",
        feature_ids=["customer_demographics_v1.0", "financial_metrics_v1.0", "payment_behavior_v1.0"],
        description="Complete feature set for churn prediction model training"
    )
    
    # Retrieve features for training
    print("\n📊 Retrieving features for training...")
    training_features = feature_store.get_features_for_training(feature_set_id)
    print(f"Training features shape: {training_features.shape}")
    print(f"Training features columns: {list(training_features.columns)}")
    
    # Get feature metadata
    print("\n📋 Feature metadata example:")
    metadata = feature_store.get_feature_metadata("customer_demographics_v1.0")
    print(f"Feature: {metadata['feature_info']['name']}")
    print(f"Description: {metadata['feature_info']['description']}")
    print(f"Versions: {len(metadata['versions'])}")
    print(f"Usage count: {sum(stat['count'] for stat in metadata['usage_statistics'])}")
    
    # List all features
    print("\n📝 All features in store:")
    features = feature_store.list_features()
    for feature in features:
        print(f"  • {feature['name']} (v{feature['version']}) - {feature['description']}")
    
    # Search features
    print("\n🔍 Searching for 'financial' features:")
    search_results = feature_store.search_features("financial")
    for feature in search_results:
        print(f"  • {feature['name']} - {feature['description']}")
    
    # Generate and save documentation
    print("\n📄 Generating documentation...")
    doc_path = feature_store.save_documentation()
    
    # Close feature store
    feature_store.close()
    
    print(f"\n✅ Feature store demonstration completed!")
    print(f"   Features registered: {len(features)}")
    print(f"   Feature set created: {feature_set_id}")
    print(f"   Documentation saved: {doc_path}")
