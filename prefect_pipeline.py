# prefect_pipeline.py - Simple Prefect Orchestration
# Student: [Your Name]  
# Course: Data Mining & Machine Learning

from prefect import flow, task
import subprocess
import sys
from datetime import datetime

@task(name="Data Ingestion")
def data_ingestion():
    """Download data from Kaggle and HuggingFace"""
    print("🔄 Starting data ingestion...")
    result = subprocess.run([sys.executable, "get_data.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Data ingestion completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Data ingestion failed: {result.stderr}")
        return "FAILED"

@task(name="Data Storage")  
def data_storage():
    """Organize data in storage structure"""
    print("🔄 Starting data storage...")
    result = subprocess.run([sys.executable, "src/storage/store_data.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Data storage completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Data storage failed: {result.stderr}")
        return "FAILED"

@task(name="Data Validation")
def data_validation():
    """Validate data quality and integrity"""
    print("🔄 Starting data validation...")
    result = subprocess.run([sys.executable, "src/validation/check_data.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Data validation completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Data validation failed: {result.stderr}")
        return "FAILED"

@task(name="Quality Report")
def quality_report():
    """Generate quality reports and charts"""
    print("🔄 Starting quality report generation...")
    result = subprocess.run([sys.executable, "src/validation/generate_quality_report.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Quality report completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Quality report failed: {result.stderr}")
        return "FAILED"

@task(name="Data Preparation")
def data_preparation():
    """Clean and preprocess data"""
    print("🔄 Starting data preparation...")
    result = subprocess.run([sys.executable, "src/preparation/clean_data.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Data preparation completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Data preparation failed: {result.stderr}")
        return "FAILED"

@task(name="Feature Engineering")
def feature_engineering():
    """Create and scale features"""
    print("🔄 Starting feature engineering...")
    result = subprocess.run([sys.executable, "src/transformation/make_features.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Feature engineering completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Feature engineering failed: {result.stderr}")
        return "FAILED"

@task(name="Database Setup")
def database_setup():
    """Set up SQLite database"""
    print("🔄 Starting database setup...")
    result = subprocess.run([sys.executable, "src/transformation/database_setup.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Database setup completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Database setup failed: {result.stderr}")
        return "FAILED"

@task(name="Feature Store")
def feature_store():
    """Manage feature metadata"""
    print("🔄 Starting feature store management...")
    result = subprocess.run([sys.executable, "src/feature_store/manage_features.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Feature store completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Feature store failed: {result.stderr}")
        return "FAILED"

@task(name="Data Versioning")
def data_versioning():
    """Version control with Git"""
    print("🔄 Starting data versioning...")
    result = subprocess.run([sys.executable, "src/versioning/version_data.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Data versioning completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Data versioning failed: {result.stderr}")
        return "FAILED"

@flow(name="Customer Churn Pipeline", description="Complete ML pipeline for customer churn prediction")
def churn_pipeline():
    """
    Complete customer churn prediction pipeline with proper task dependencies
    """
    print("=" * 60)
    print("🚀 PREFECT ORCHESTRATION - Customer Churn Pipeline")
    print("=" * 60)
    print(f"📅 Pipeline Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Execute tasks with dependencies
    task1 = data_ingestion()
    task2 = data_storage(wait_for=[task1])
    task3 = data_validation(wait_for=[task2])
    task4 = quality_report(wait_for=[task3])
    task5 = data_preparation(wait_for=[task4])
    task6 = feature_engineering(wait_for=[task5])
    task7 = database_setup(wait_for=[task6])
    task8 = feature_store(wait_for=[task7])
    task9 = data_versioning(wait_for=[task8])
    
    print()
    print("=" * 60)
    print("🎉 PIPELINE EXECUTION COMPLETED!")
    print("=" * 60)
    
    # Return summary
    return {
        "pipeline_status": "COMPLETED",
        "total_tasks": 9,
        "completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "tasks": [task1, task2, task3, task4, task5, task6, task7, task8, task9]
    }

if __name__ == "__main__":
    # Run the pipeline
    result = churn_pipeline()
    print(f"\n📊 Pipeline Result: {result}")
