"""
Database Storage System for Churn Prediction Pipeline.
Handles SQL schema design, data storage, and querying capabilities.
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

# Database imports
import sqlite3
import psycopg2
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


class DatabaseStorage:
    """Database storage system for churn prediction data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize database storage with configuration."""
        self.config = self._load_config(config_path)
        self.db_config = self.config.get('database', {})
        
        # Database connection
        self.engine = None
        self.connection = None
        self.session = None
        
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
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def connect_database(self, db_type: str = None) -> bool:
        """Connect to database."""
        if db_type is None:
            db_type = self.db_config.get('type', 'sqlite')
        
        try:
            if db_type == 'sqlite':
                db_path = self.db_config.get('sqlite', {}).get('path', 'data/churn_prediction.db')
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                
                self.engine = create_engine(f'sqlite:///{db_path}')
                self.connection = self.engine.connect()
                
            elif db_type == 'postgresql':
                pg_config = self.db_config.get('postgresql', {})
                connection_string = (
                    f"postgresql://{pg_config.get('username', 'postgres')}:"
                    f"{pg_config.get('password', '')}@"
                    f"{pg_config.get('host', 'localhost')}:"
                    f"{pg_config.get('port', '5432')}/"
                    f"{pg_config.get('database', 'churn_prediction')}"
                )
                self.engine = create_engine(connection_string)
                self.connection = self.engine.connect()
            
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            logger.info(f"Connected to {db_type} database successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_schema(self) -> bool:
        """Create database schema for churn prediction data."""
        logger.info("Creating database schema...")
        
        try:
            # Define tables
            self._define_customers_table()
            self._define_transactions_table()
            self._define_features_table()
            self._define_predictions_table()
            self._define_model_performance_table()
            
            # Create tables
            self.Base.metadata.create_all(self.engine)
            
            logger.info("Database schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
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
            online_backup = Column(String(20))
            device_protection = Column(String(20))
            tech_support = Column(String(20))
            streaming_tv = Column(String(20))
            streaming_movies = Column(String(20))
            paperless_billing = Column(String(5))
            partner = Column(String(5))
            dependents = Column(String(5))
            churn = Column(Integer)
            created_at = Column(DateTime, default=datetime.now)
            updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
        
        self.Customer = Customer
        self.storage_info['tables_created'].append('customers')
    
    def _define_transactions_table(self):
        """Define transactions table."""
        class Transaction(self.Base):
            __tablename__ = 'transactions'
            
            transaction_id = Column(Integer, primary_key=True, autoincrement=True)
            customer_id = Column(Integer)
            transaction_date = Column(DateTime)
            amount = Column(Float)
            transaction_type = Column(String(30))
            description = Column(String(100))
            created_at = Column(DateTime, default=datetime.now)
        
        self.Transaction = Transaction
        self.storage_info['tables_created'].append('transactions')
    
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
            tenure_category = Column(String(20))
            age_group = Column(String(20))
            value_per_month = Column(Float)
            high_risk_customer = Column(Integer)
            premium_bundle = Column(Integer)
            support_usage = Column(Integer)
            spending_percentile = Column(Float)
            high_spender = Column(Integer)
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
    
    def _define_model_performance_table(self):
        """Define model performance table."""
        class ModelPerformance(self.Base):
            __tablename__ = 'model_performance'
            
            performance_id = Column(Integer, primary_key=True, autoincrement=True)
            model_name = Column(String(50))
            accuracy = Column(Float)
            precision = Column(Float)
            recall = Column(Float)
            f1_score = Column(Float)
            roc_auc = Column(Float)
            training_date = Column(DateTime)
            test_date = Column(DateTime, default=datetime.now)
            created_at = Column(DateTime, default=datetime.now)
        
        self.ModelPerformance = ModelPerformance
        self.storage_info['tables_created'].append('model_performance')
    
    def store_customer_data(self, data: pd.DataFrame) -> bool:
        """Store customer data in database."""
        logger.info("Storing customer data in database...")
        
        try:
            # Prepare data for storage
            customer_data = data.copy()
            
            # Ensure required columns exist
            required_columns = ['customer_id', 'age', 'gender', 'tenure', 'monthly_charges', 
                              'total_charges', 'contract_type', 'payment_method', 'churn']
            
            missing_columns = [col for col in required_columns if col not in customer_data.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            
            # Select only columns that exist in the table
            table_columns = [col.name for col in self.Customer.__table__.columns 
                           if col.name not in ['created_at', 'updated_at']]
            
            available_columns = [col for col in table_columns if col in customer_data.columns]
            customer_data = customer_data[available_columns]
            
            # Store data
            customer_data.to_sql('customers', self.engine, if_exists='replace', index=False)
            
            self.storage_info['data_loaded']['customers'] = {
                'rows': len(customer_data),
                'columns': len(customer_data.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Stored {len(customer_data)} customer records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store customer data: {e}")
            return False
    
    def store_engineered_features(self, data: pd.DataFrame) -> bool:
        """Store engineered features in database."""
        logger.info("Storing engineered features in database...")
        
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
            
            logger.info(f"Stored {len(feature_data)} feature records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store engineered features: {e}")
            return False
    
    def store_predictions(self, predictions: pd.DataFrame) -> bool:
        """Store model predictions in database."""
        logger.info("Storing predictions in database...")
        
        try:
            # Prepare data for storage
            prediction_data = predictions.copy()
            
            # Ensure required columns
            required_columns = ['customer_id', 'model_name', 'prediction_probability', 'prediction_class']
            missing_columns = [col for col in required_columns if col not in prediction_data.columns]
            
            if missing_columns:
                logger.warning(f"Missing prediction columns: {missing_columns}")
                return False
            
            # Store data
            prediction_data.to_sql('predictions', self.engine, if_exists='append', index=False)
            
            self.storage_info['data_loaded']['predictions'] = {
                'rows': len(prediction_data),
                'columns': len(prediction_data.columns),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Stored {len(prediction_data)} prediction records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
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
            logger.error(f"Failed to execute query: {e}")
            return pd.DataFrame()
    
    def get_sample_queries(self) -> Dict[str, str]:
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
                    f.spending_percentile,
                    c.churn
                FROM engineered_features f
                JOIN customers c ON f.customer_id = c.customer_id
                WHERE f.spending_percentile > 0.8
                ORDER BY f.avg_monthly_spending DESC
            """,
            
            "model_performance_comparison": """
                SELECT 
                    model_name,
                    AVG(accuracy) as avg_accuracy,
                    AVG(precision) as avg_precision,
                    AVG(recall) as avg_recall,
                    AVG(f1_score) as avg_f1_score,
                    AVG(roc_auc) as avg_roc_auc,
                    COUNT(*) as evaluation_count
                FROM model_performance
                GROUP BY model_name
                ORDER BY avg_f1_score DESC
            """,
            
            "recent_predictions": """
                SELECT 
                    p.customer_id,
                    p.model_name,
                    p.prediction_probability,
                    p.prediction_class,
                    p.confidence_score,
                    p.prediction_date,
                    c.tenure,
                    c.monthly_charges
                FROM predictions p
                JOIN customers c ON p.customer_id = c.customer_id
                WHERE p.prediction_date >= datetime('now', '-7 days')
                ORDER BY p.prediction_date DESC
                LIMIT 50
            """,
            
            "customer_segmentation": """
                SELECT 
                    f.lifecycle_stage,
                    f.age_group,
                    f.spending_bin,
                    COUNT(*) as customer_count,
                    AVG(c.monthly_charges) as avg_monthly_charges,
                    SUM(c.churn) as churned_count,
                    ROUND(SUM(c.churn) * 100.0 / COUNT(*), 2) as churn_rate
                FROM engineered_features f
                JOIN customers c ON f.customer_id = c.customer_id
                GROUP BY f.lifecycle_stage, f.age_group, f.spending_bin
                ORDER BY churn_rate DESC
            """,
            
            "service_usage_analysis": """
                SELECT 
                    internet_service,
                    COUNT(*) as total_customers,
                    SUM(churn) as churned_customers,
                    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) as churn_rate,
                    AVG(monthly_charges) as avg_monthly_charges
                FROM customers
                GROUP BY internet_service
                ORDER BY churn_rate DESC
            """
        }
    
    def generate_schema_documentation(self) -> str:
        """Generate SQL schema documentation."""
        schema_doc = """
# Database Schema Documentation

## Overview
This database stores churn prediction data including customer information, engineered features, model predictions, and performance metrics.

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
- `contract_type` (VARCHAR(20)): Contract type (Month-to-month, One year, Two year)
- `payment_method` (VARCHAR(30)): Payment method
- `internet_service` (VARCHAR(20)): Internet service type
- `online_security` (VARCHAR(20)): Online security service
- `online_backup` (VARCHAR(20)): Online backup service
- `device_protection` (VARCHAR(20)): Device protection service
- `tech_support` (VARCHAR(20)): Tech support service
- `streaming_tv` (VARCHAR(20)): Streaming TV service
- `streaming_movies` (VARCHAR(20)): Streaming movies service
- `paperless_billing` (VARCHAR(5)): Paperless billing (Yes/No)
- `partner` (VARCHAR(5)): Partner status (Yes/No)
- `dependents` (VARCHAR(5)): Dependents status (Yes/No)
- `churn` (INTEGER): Churn status (0/1)
- `created_at` (DATETIME): Record creation timestamp
- `updated_at` (DATETIME): Record update timestamp

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
- `tenure_category` (VARCHAR(20)): Tenure category
- `age_group` (VARCHAR(20)): Age group
- `value_per_month` (FLOAT): Value per month
- `high_risk_customer` (INTEGER): High-risk customer flag
- `premium_bundle` (INTEGER): Premium bundle flag
- `support_usage` (INTEGER): Support usage indicator
- `spending_percentile` (FLOAT): Spending percentile
- `high_spender` (INTEGER): High spender flag
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

### 4. model_performance
Stores model performance metrics.

**Columns:**
- `performance_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): Unique performance record identifier
- `model_name` (VARCHAR(50)): Name of the model
- `accuracy` (FLOAT): Model accuracy
- `precision` (FLOAT): Model precision
- `recall` (FLOAT): Model recall
- `f1_score` (FLOAT): F1 score
- `roc_auc` (FLOAT): ROC AUC score
- `training_date` (DATETIME): Model training date
- `test_date` (DATETIME): Model testing date
- `created_at` (DATETIME): Record creation timestamp

### 5. transactions
Stores customer transaction history.

**Columns:**
- `transaction_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): Unique transaction identifier
- `customer_id` (INTEGER): Customer identifier (foreign key)
- `transaction_date` (DATETIME): Transaction date
- `amount` (FLOAT): Transaction amount
- `transaction_type` (VARCHAR(30)): Type of transaction
- `description` (VARCHAR(100)): Transaction description
- `created_at` (DATETIME): Record creation timestamp

## Relationships
- `customers.customer_id` → `engineered_features.customer_id`
- `customers.customer_id` → `predictions.customer_id`
- `customers.customer_id` → `transactions.customer_id`

## Indexes
- Primary keys on all tables
- Foreign key indexes on customer_id columns
- Index on prediction_date for time-based queries
- Index on model_name for model-specific queries

## Sample Queries
See the `get_sample_queries()` method for comprehensive query examples.
"""
        return schema_doc
    
    def save_schema_documentation(self, output_path: str = None) -> str:
        """Save schema documentation to file."""
        if output_path is None:
            output_path = "docs/database_schema.md"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save documentation
        schema_doc = self.generate_schema_documentation()
        with open(output_path, 'w') as f:
            f.write(schema_doc)
        
        logger.info(f"Schema documentation saved to: {output_path}")
        return output_path
    
    def close_connection(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")


def create_database_storage(config_path: str = "config/config.yaml") -> DatabaseStorage:
    """Factory function to create database storage instance."""
    return DatabaseStorage(config_path)
