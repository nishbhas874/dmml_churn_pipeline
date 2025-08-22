# run_airflow.py - Simple Airflow Pipeline Runner
# Student: [Your Name]
# Course: Data Mining & Machine Learning

import os
import subprocess

print("🚀 Customer Churn Pipeline - Using Apache Airflow")
print("=" * 50)
print("This shows our DAG structure and task dependencies")
print()

# Show the DAG structure
print("📊 Our Pipeline DAG (Directed Acyclic Graph):")
print("=" * 50)
print("data_ingestion")
print("    ↓")
print("data_storage") 
print("    ↓")
print("data_validation")
print("    ↓")
print("quality_report")
print("    ↓") 
print("data_preparation")
print("    ↓")
print("feature_engineering")
print("    ↓")
print("database_setup")
print("    ↓")
print("feature_store")
print("    ↓")
print("data_versioning")
print()

print("✅ Key Airflow Features Implemented:")
print("- DAG with proper task dependencies")
print("- Automatic retries on failure")
print("- Task scheduling (daily)")
print("- Failure handling")
print()

# For demonstration, we'll just run our regular pipeline
print("🔄 Running pipeline tasks in correct order...")
print("(In real Airflow, this would be managed automatically)")
print()

# Run the pipeline in the correct dependency order
result = os.system("python pipeline.py")

if result == 0:
    print()
    print("✅ Pipeline completed successfully!")
    print("📈 In real Airflow, you would see this in the web UI")
    print("🗄️ DAG file: dags/churn_pipeline_dag.py")
else:
    print("❌ Pipeline failed - Airflow would retry automatically")
