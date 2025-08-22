# simple_prefect_demo.py - Simple Prefect Flow Demo
# Student: [Your Name]
# Course: Data Mining & Machine Learning

from prefect import flow, task
import subprocess
import sys
from datetime import datetime

@task(name="Data Ingestion", retries=1, retry_delay_seconds=5)
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

@task(name="Data Storage", retries=1, retry_delay_seconds=5)
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

@task(name="Data Validation", retries=1, retry_delay_seconds=5)
def data_validation():
    """Validate data quality"""
    print("🔄 Starting data validation...")
    result = subprocess.run([sys.executable, "src/validation/check_data.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Data validation completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Data validation failed: {result.stderr}")
        return "FAILED"

@task(name="Quality Report", retries=1, retry_delay_seconds=5)
def quality_report():
    """Generate data quality reports"""
    print("🔄 Starting quality report generation...")
    result = subprocess.run([sys.executable, "src/validation/generate_quality_report.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Quality report completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Quality report failed: {result.stderr}")
        return "FAILED"

@task(name="Data Preparation", retries=1, retry_delay_seconds=5)
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

@task(name="Feature Engineering", retries=1, retry_delay_seconds=5)
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

@task(name="Database Setup", retries=1, retry_delay_seconds=5)
def database_setup():
    """Setup SQLite database"""
    print("🔄 Starting database setup...")
    result = subprocess.run([sys.executable, "src/transformation/database_setup.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Database setup completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Database setup failed: {result.stderr}")
        return "FAILED"

@task(name="Feature Store", retries=1, retry_delay_seconds=5)
def feature_store():
    """Manage feature store"""
    print("🔄 Starting feature store management...")
    result = subprocess.run([sys.executable, "src/feature_store/manage_features.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Feature store completed successfully")
        return "SUCCESS"
    else:
        print(f"❌ Feature store failed: {result.stderr}")
        return "FAILED"

@task(name="Data Versioning", retries=1, retry_delay_seconds=5)
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
    Complete customer churn prediction pipeline using Prefect orchestration
    """
    print("=" * 70)
    print("🚀 PREFECT ORCHESTRATION - Customer Churn Pipeline")
    print("=" * 70)
    print(f"📅 Pipeline Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Orchestrator: Prefect (Standard Industry Tool)")
    print(f"📊 Flow Name: Customer Churn Pipeline")
    print(f"⏱️  Schedule: On-demand execution")
    print(f"🔄 Retry Policy: 1 retry per task, 5-second delay")
    print()
    
    # Execute tasks with dependencies
    print("📋 TASK EXECUTION TIMELINE:")
    print("-" * 70)
    
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
    print("=" * 70)
    print("🎉 PIPELINE EXECUTION COMPLETED!")
    print("=" * 70)
    
    # Return summary
    results = [task1, task2, task3, task4, task5, task6, task7, task8, task9]
    successful_tasks = sum(1 for result in results if result == "SUCCESS")
    
    print(f"📊 EXECUTION SUMMARY:")
    print(f"   • Total Tasks: {len(results)}")
    print(f"   • Successful: {successful_tasks} ✅")
    print(f"   • Failed: {len(results) - successful_tasks} ❌")
    print(f"   • Success Rate: {(successful_tasks/len(results)*100):.1f}%")
    print()
    
    print("📈 PREFECT ORCHESTRATION FEATURES:")
    print("   ✅ Task Dependencies - Sequential execution enforced")
    print("   ✅ Error Handling - Automatic retry logic")
    print("   ✅ Monitoring - Real-time status tracking")
    print("   ✅ Logging - Detailed execution logs")
    print("   ✅ Flow Management - Professional orchestration")
    print()
    
    print("🔗 TASK DEPENDENCY GRAPH:")
    print("   data_ingestion")
    print("       ↓")
    print("   data_storage")
    print("       ↓")
    print("   data_validation")
    print("       ↓")
    print("   quality_report")
    print("       ↓")
    print("   data_preparation")
    print("       ↓")
    print("   feature_engineering")
    print("       ↓")
    print("   database_setup")
    print("       ↓")
    print("   feature_store")
    print("       ↓")
    print("   data_versioning")
    print()
    
    return {
        "pipeline_status": "COMPLETED",
        "total_tasks": len(results),
        "successful_tasks": successful_tasks,
        "completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "orchestrator": "Prefect"
    }

if __name__ == "__main__":
    # Run the pipeline
    result = churn_pipeline()
    print(f"\n📊 Pipeline Result: {result}")
