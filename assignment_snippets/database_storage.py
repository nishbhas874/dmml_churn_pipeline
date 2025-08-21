"""
Database Storage Code Snippet for Assignment
This code demonstrates SQL schema design and database operations for churn prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

class DatabaseStorage:
    """Database storage system for churn prediction data."""
    
    def __init__(self, db_path: str = "churn_prediction.db"):
        """Initialize database storage."""
        self.db_path = db_path
        self.engine = None
        self.connection = None
        
        # Schema metadata
        self.metadata = MetaData()
        self.Base = declarative_base()
        
        # Store storage metadata
        self.storage_info = {
            'timestamp': datetime.now().isoformat(),
            'tables_created': [],
            'data_loaded': {},
            'queries_executed': []
        }
    
    def connect_database(self) -> bool:
        """Connect to SQLite database."""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create SQLite engine
            self.engine = create_engine(f'sqlite:///{self.db_path}')
            self.connection = self.engine.connect()
            
            print(f"Connected to SQLite database: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            return False
    
    def create_schema(self) -> bool:
        """Create database schema for churn prediction data."""
        print("Creating database schema...")
        
        try:
            # Define tables
            self._define_customers_table()
            self._define_features_table()
            self._define_predictions_table()
            
            # Create tables
            self.Base.metadata.create_all(self.engine)
            
            print("Database schema created successfully")
            return True
            
        except Exception as e:
            print(f"Failed to create schema: {e}")
            return False
    
    def _define_customers_table(self):
        """Define customers table."""
        class Customer(self.Base):
            __tablename__ = 'customers'
            
            customer_id = Column(Integer, primary_key=True)
            age = Column(Integer)
            gender = Column(String(10))
            tenure = Column(Integer)
            monthly_charges = Column(Float)
            total_charges = Column(Float)
            contract_type = Column(String(20))
            payment_method = Column(String(30))
            internet_service = Column(String(20))
            online_security = Column(String(20))
            tech_support = Column(String(20))
            churn = Column(Integer)
            created_at = Column(DateTime, default=datetime.now)
        
        self.Customer = Customer
        self.storage_info['tables_created'].append('customers')
    
    def _define_features_table(self):
        """Define engineered features table."""
        class Feature(self.Base):
            __tablename__ = 'engineered_features'
            
            feature_id = Column(Integer, primary_key=True, autoincrement=True)
            customer_id = Column(Integer)
            avg_monthly_spending = Column(Float)
            spending_trend = Column(Float)
            services_count = Column(Integer)
            service_usage_ratio = Column(Float)
            contract_length_months = Column(Integer)
            contract_commitment = Column(Float)
            lifecycle_stage = Column(String(20))
            high_risk_customer = Column(Integer)
            tenure_squared = Column(Float)
            tenure_log = Column(Float)
            tenure_inverse = Column(Float)
            monthly_total_ratio = Column(Float)
            daily_spending = Column(Float)
            created_at = Column(DateTime, default=datetime.now)
        
        self.Feature = Feature
        self.storage_info['tables_created'].append('engineered_features')
    
    def _define_predictions_table(self):
        """Define predictions table."""
        class Prediction(self.Base):
            __tablename__ = 'predictions'
            
            prediction_id = Column(Integer, primary_key=True, autoincrement=True)
            customer_id = Column(Integer)
            model_name = Column(String(50))
            prediction_probability = Column(Float)
            prediction_class = Column(Integer)
            confidence_score = Column(Float)
            prediction_date = Column(DateTime, default=datetime.now)
            created_at = Column(DateTime, default=datetime.now)
        
        self.Prediction = Prediction
        self.storage_info['tables_created'].append('predictions')
    
    def store_customer_data(self, data: pd.DataFrame) -> bool:
        """Store customer data in database."""
        print("Storing customer data in database...")
        
        try:
            # Prepare data for storage
            customer_data = data.copy()
            
            # Select only columns that exist in the table
            table_columns = [col.name for col in self.Customer.__table__.columns 
                           if col.name not in ['created_at']]
            
            available_columns = [col for col in table_columns if col in customer_data.columns]
            customer_data = customer_data[available_columns]
            
            # Store data
            customer_data.to_sql('customers', self.engine, if_exists='replace', index=False)
            
            self.storage_info['data_loaded']['customers'] = {
                'rows': len(customer_data),
                'columns': len(customer_data.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Stored {len(customer_data)} customer records")
            return True
            
        except Exception as e:
            print(f"Failed to store customer data: {e}")
            return False
    
    def store_engineered_features(self, data: pd.DataFrame) -> bool:
        """Store engineered features in database."""
        print("Storing engineered features in database...")
        
        try:
            # Prepare data for storage
            feature_data = data.copy()
            
            # Select only feature columns
            feature_columns = [col.name for col in self.Feature.__table__.columns 
                             if col.name not in ['feature_id', 'created_at']]
            
            available_columns = [col for col in feature_columns if col in feature_data.columns]
            feature_data = feature_data[available_columns]
            
            # Store data
            feature_data.to_sql('engineered_features', self.engine, if_exists='replace', index=False)
            
            self.storage_info['data_loaded']['engineered_features'] = {
                'rows': len(feature_data),
                'columns': len(feature_data.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Stored {len(feature_data)} feature records")
            return True
            
        except Exception as e:
            print(f"Failed to store engineered features: {e}")
            return False
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results."""
        try:
            result = pd.read_sql_query(query, self.engine)
            self.storage_info['queries_executed'].append({
                'query': query,
                'rows_returned': len(result),
                'timestamp': datetime.now().isoformat()
            })
            return result
        except Exception as e:
            print(f"Failed to execute query: {e}")
            return pd.DataFrame()
    
    def get_sample_queries(self):
        """Get sample queries for data retrieval."""
        return {
            "customer_overview": """
                SELECT 
                    COUNT(*) as total_customers,
                    AVG(tenure) as avg_tenure,
                    AVG(monthly_charges) as avg_monthly_charges,
                    SUM(churn) as churned_customers,
                    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) as churn_rate
                FROM customers
            """,
            
            "churn_by_contract_type": """
                SELECT 
                    contract_type,
                    COUNT(*) as total_customers,
                    SUM(churn) as churned_customers,
                    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) as churn_rate
                FROM customers
                GROUP BY contract_type
                ORDER BY churn_rate DESC
            """,
            
            "high_risk_customers": """
                SELECT 
                    c.customer_id,
                    c.tenure,
                    c.monthly_charges,
                    c.contract_type,
                    c.payment_method,
                    f.high_risk_customer,
                    f.spending_trend
                FROM customers c
                LEFT JOIN engineered_features f ON c.customer_id = f.customer_id
                WHERE f.high_risk_customer = 1
                ORDER BY f.spending_trend DESC
                LIMIT 20
            """,
            
            "feature_importance_analysis": """
                SELECT 
                    f.customer_id,
                    f.avg_monthly_spending,
                    f.services_count,
                    f.contract_commitment,
                    c.churn
                FROM engineered_features f
                JOIN customers c ON f.customer_id = c.customer_id
                ORDER BY f.avg_monthly_spending DESC
            """
        }
    
    def generate_schema_documentation(self) -> str:
        """Generate SQL schema documentation."""
        schema_doc = """
# Database Schema Documentation

## Overview
This database stores churn prediction data including customer information, engineered features, and model predictions.

## Tables

### 1. customers
Stores basic customer information and demographic data.

**Columns:**
- `customer_id` (INTEGER, PRIMARY KEY): Unique customer identifier
- `age` (INTEGER): Customer age
- `gender` (VARCHAR(10)): Customer gender
- `tenure` (INTEGER): Customer tenure in months
- `monthly_charges` (FLOAT): Monthly charges
- `total_charges` (FLOAT): Total charges
- `contract_type` (VARCHAR(20)): Contract type
- `payment_method` (VARCHAR(30)): Payment method
- `internet_service` (VARCHAR(20)): Internet service type
- `online_security` (VARCHAR(20)): Online security service
- `tech_support` (VARCHAR(20)): Tech support service
- `churn` (INTEGER): Churn status (0/1)
- `created_at` (DATETIME): Record creation timestamp

### 2. engineered_features
Stores feature-engineered data for machine learning models.

**Columns:**
- `feature_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): Unique feature record identifier
- `customer_id` (INTEGER): Customer identifier (foreign key)
- `avg_monthly_spending` (FLOAT): Average monthly spending
- `spending_trend` (FLOAT): Spending trend indicator
- `services_count` (INTEGER): Number of services used
- `service_usage_ratio` (FLOAT): Ratio of services used
- `contract_length_months` (INTEGER): Contract length in months
- `contract_commitment` (FLOAT): Contract commitment level
- `lifecycle_stage` (VARCHAR(20)): Customer lifecycle stage
- `high_risk_customer` (INTEGER): High-risk customer flag
- `tenure_squared` (FLOAT): Tenure squared
- `tenure_log` (FLOAT): Log of tenure
- `tenure_inverse` (FLOAT): Inverse of tenure
- `monthly_total_ratio` (FLOAT): Monthly to total charges ratio
- `daily_spending` (FLOAT): Daily spending
- `created_at` (DATETIME): Record creation timestamp

### 3. predictions
Stores model predictions for customers.

**Columns:**
- `prediction_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): Unique prediction identifier
- `customer_id` (INTEGER): Customer identifier (foreign key)
- `model_name` (VARCHAR(50)): Name of the model used
- `prediction_probability` (FLOAT): Prediction probability
- `prediction_class` (INTEGER): Predicted class (0/1)
- `confidence_score` (FLOAT): Confidence score
- `prediction_date` (DATETIME): Prediction timestamp
- `created_at` (DATETIME): Record creation timestamp

## Relationships
- `customers.customer_id` → `engineered_features.customer_id`
- `customers.customer_id` → `predictions.customer_id`

## Sample Queries
See the `get_sample_queries()` method for comprehensive query examples.
"""
        return schema_doc
    
    def close_connection(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        print("Database connection closed")

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
    
    # Add some engineered features
    sample_data['avg_monthly_spending'] = sample_data['total_charges'] / sample_data['tenure'].replace(0, 1)
    sample_data['services_count'] = np.random.randint(0, 4, n_samples)
    sample_data['high_risk_customer'] = ((sample_data['contract_type'] == 'Month-to-month') & 
                                        (sample_data['payment_method'] == 'Electronic check')).astype(int)
    
    # Initialize database storage
    db_storage = DatabaseStorage("churn_prediction.db")
    
    # Connect and create schema
    if db_storage.connect_database():
        if db_storage.create_schema():
            # Store data
            db_storage.store_customer_data(sample_data)
            db_storage.store_engineered_features(sample_data)
            
            # Execute sample queries
            sample_queries = db_storage.get_sample_queries()
            for query_name, query in sample_queries.items():
                print(f"\nRunning: {query_name}")
                result = db_storage.execute_query(query)
                print(f"Rows returned: {len(result)}")
                if len(result) > 0:
                    print(result.head())
            
            # Generate documentation
            schema_doc = db_storage.generate_schema_documentation()
            print("\nSchema Documentation:")
            print(schema_doc)
            
            # Close connection
            db_storage.close_connection()
