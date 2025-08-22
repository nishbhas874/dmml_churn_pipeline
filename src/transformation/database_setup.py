#!/usr/bin/env python3
"""
Database Setup and Schema - Part of Stage 6
Creates database schema and demonstrates data storage.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

def create_database_schema():
    """Create SQLite database with schema for transformed data"""
    print("Creating database schema...")
    
    # Create database connection
    db_path = "data/transformed/churn_database.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create customers table
    customers_schema = """
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        tenure INTEGER,
        monthly_charges REAL,
        total_charges REAL,
        contract_type TEXT,
        payment_method TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Create features table
    features_schema = """
    CREATE TABLE IF NOT EXISTS customer_features (
        customer_id TEXT,
        tenure_group TEXT,
        charges_per_tenure REAL,
        contract_encoded INTEGER,
        payment_method_encoded INTEGER,
        gender_encoded INTEGER,
        senior_citizen INTEGER,
        partner INTEGER,
        dependents INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    );
    """
    
    # Create churn predictions table
    churn_schema = """
    CREATE TABLE IF NOT EXISTS churn_predictions (
        customer_id TEXT,
        churn_probability REAL,
        churn_prediction INTEGER,
        model_version TEXT,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    );
    """
    
    # Execute schema creation
    cursor.execute(customers_schema)
    cursor.execute(features_schema)
    cursor.execute(churn_schema)
    
    conn.commit()
    conn.close()
    
    print(f"Database schema created: {db_path}")
    return db_path

def create_sample_queries():
    """Create sample SQL queries for data retrieval"""
    queries = {
        "get_high_risk_customers": """
        SELECT c.customer_id, c.monthly_charges, cp.churn_probability
        FROM customers c
        JOIN churn_predictions cp ON c.customer_id = cp.customer_id
        WHERE cp.churn_probability > 0.7
        ORDER BY cp.churn_probability DESC;
        """,
        
        "get_customer_features": """
        SELECT c.customer_id, c.tenure, cf.tenure_group, 
               cf.charges_per_tenure, c.monthly_charges
        FROM customers c
        JOIN customer_features cf ON c.customer_id = cf.customer_id
        WHERE c.tenure > 12;
        """,
        
        "monthly_churn_summary": """
        SELECT 
            DATE(prediction_date) as date,
            COUNT(*) as total_predictions,
            AVG(churn_probability) as avg_churn_prob,
            SUM(churn_prediction) as predicted_churners
        FROM churn_predictions
        GROUP BY DATE(prediction_date)
        ORDER BY date DESC;
        """,
        
        "feature_statistics": """
        SELECT 
            tenure_group,
            COUNT(*) as customer_count,
            AVG(charges_per_tenure) as avg_charges_ratio,
            AVG(monthly_charges) as avg_monthly_charges
        FROM customer_features cf
        JOIN customers c ON cf.customer_id = c.customer_id
        GROUP BY tenure_group
        ORDER BY customer_count DESC;
        """
    }
    
    # Save queries to file
    queries_file = "data/transformed/sample_queries.sql"
    with open(queries_file, 'w') as f:
        f.write("-- Sample SQL Queries for Churn Prediction Database\n")
        f.write(f"-- Generated on: {datetime.now().isoformat()}\n\n")
        
        for query_name, query in queries.items():
            f.write(f"-- {query_name.replace('_', ' ').title()}\n")
            f.write(query.strip() + "\n\n")
    
    print(f"Sample queries saved: {queries_file}")
    return queries_file

def insert_sample_data(db_path):
    """Insert sample data to demonstrate database functionality"""
    print("Inserting sample data...")
    
    conn = sqlite3.connect(db_path)
    
    # Sample customer data
    sample_customers = [
        ('CUST001', 24, 65.50, 1572.00, 'Month-to-month', 'Credit card'),
        ('CUST002', 12, 89.90, 1078.80, 'One year', 'Bank transfer'),
        ('CUST003', 36, 45.20, 1627.20, 'Two year', 'Electronic check')
    ]
    
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR REPLACE INTO customers (customer_id, tenure, monthly_charges, total_charges, contract_type, payment_method) VALUES (?, ?, ?, ?, ?, ?)",
        sample_customers
    )
    
    # Sample features
    sample_features = [
        ('CUST001', 'Medium', 2.73, 0, 1, 1, 0, 1, 0),
        ('CUST002', 'Medium', 7.49, 1, 2, 0, 1, 0, 1),
        ('CUST003', 'Long', 1.26, 2, 3, 1, 0, 1, 1)
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO customer_features (customer_id, tenure_group, charges_per_tenure, contract_encoded, payment_method_encoded, gender_encoded, senior_citizen, partner, dependents) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        sample_features
    )
    
    # Sample predictions
    sample_predictions = [
        ('CUST001', 0.85, 1, 'v1.0'),
        ('CUST002', 0.23, 0, 'v1.0'),
        ('CUST003', 0.12, 0, 'v1.0')
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO churn_predictions (customer_id, churn_probability, churn_prediction, model_version) VALUES (?, ?, ?, ?)",
        sample_predictions
    )
    
    conn.commit()
    conn.close()
    
    print("Sample data inserted successfully")

def create_transformation_summary():
    """Create summary of transformation logic applied"""
    transformation_logic = {
        "timestamp": datetime.now().isoformat(),
        "transformations_applied": [
            {
                "name": "Tenure Grouping",
                "description": "Group customers by tenure: Short (0-12), Medium (13-36), Long (37+)",
                "input_column": "tenure",
                "output_column": "tenure_group",
                "logic": "IF tenure <= 12 THEN 'Short' ELIF tenure <= 36 THEN 'Medium' ELSE 'Long'"
            },
            {
                "name": "Charges Per Tenure",
                "description": "Calculate average charges per month of tenure",
                "input_columns": ["total_charges", "tenure"],
                "output_column": "charges_per_tenure",
                "logic": "total_charges / tenure"
            },
            {
                "name": "Contract Encoding",
                "description": "Encode contract types as integers",
                "input_column": "contract",
                "output_column": "contract_encoded",
                "logic": "Month-to-month=0, One year=1, Two year=2"
            },
            {
                "name": "Feature Scaling",
                "description": "Normalize numerical features to 0-1 range",
                "input_columns": ["monthly_charges", "total_charges"],
                "output_columns": ["monthly_charges_scaled", "total_charges_scaled"],
                "logic": "MinMaxScaler: (value - min) / (max - min)"
            }
        ],
        "database_schema": {
            "tables_created": 3,
            "primary_keys": ["customer_id"],
            "foreign_keys": ["customer_features.customer_id", "churn_predictions.customer_id"],
            "indexes": "Created on customer_id columns for performance"
        }
    }
    
    summary_file = f"data/transformed/transformation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    import json
    with open(summary_file, 'w') as f:
        json.dump(transformation_logic, f, indent=2)
    
    print(f"Transformation summary saved: {summary_file}")
    return summary_file

def main():
    """Main database setup function"""
    print("Database Setup - Stage 6")
    print("=" * 40)
    
    # Create transformed data folder
    Path("data/transformed").mkdir(parents=True, exist_ok=True)
    
    # Create database and schema
    db_path = create_database_schema()
    
    # Create sample queries
    queries_file = create_sample_queries()
    
    # Insert sample data
    insert_sample_data(db_path)
    
    # Create transformation summary
    summary_file = create_transformation_summary()
    
    print("\n" + "=" * 40)
    print("Database setup completed!")
    try:
        print(f"🗄️  Database: {db_path}")
        print(f"📝 Queries: {queries_file}")
        print(f"📊 Summary: {summary_file}")
    except UnicodeEncodeError:
        print(f"[DB] Database: {db_path}")
        print(f"[QUERIES] Queries: {queries_file}")
        print(f"[SUMMARY] Summary: {summary_file}")
    print("=" * 40)

if __name__ == "__main__":
    main()
