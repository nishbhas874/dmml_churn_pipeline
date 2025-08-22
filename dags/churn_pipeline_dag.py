# churn_pipeline_dag.py
# Student: [Your Name] 
# Course: Data Mining & Machine Learning
# Simple Airflow DAG for Customer Churn Pipeline

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default settings for our DAG
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'customer_churn_pipeline',
    default_args=default_args,
    description='Simple customer churn prediction pipeline',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
)

# Task 1: Get data from Kaggle and HuggingFace
get_data = BashOperator(
    task_id='data_ingestion',
    bash_command='cd /path/to/project && python get_data.py',
    dag=dag,
)

# Task 2: Store data in organized structure  
store_data = BashOperator(
    task_id='data_storage',
    bash_command='cd /path/to/project && python src/storage/store_data.py',
    dag=dag,
)

# Task 3: Check data quality
validate_data = BashOperator(
    task_id='data_validation', 
    bash_command='cd /path/to/project && python src/validation/check_data.py',
    dag=dag,
)

# Task 4: Generate quality reports
quality_report = BashOperator(
    task_id='quality_report',
    bash_command='cd /path/to/project && python src/validation/generate_quality_report.py',
    dag=dag,
)

# Task 5: Clean the data
clean_data = BashOperator(
    task_id='data_preparation',
    bash_command='cd /path/to/project && python src/preparation/clean_data.py', 
    dag=dag,
)

# Task 6: Create features
make_features = BashOperator(
    task_id='feature_engineering',
    bash_command='cd /path/to/project && python src/transformation/make_features.py',
    dag=dag,
)

# Task 7: Set up database
setup_database = BashOperator(
    task_id='database_setup',
    bash_command='cd /path/to/project && python src/transformation/database_setup.py',
    dag=dag,
)

# Task 8: Manage feature store
manage_features = BashOperator(
    task_id='feature_store',
    bash_command='cd /path/to/project && python src/feature_store/manage_features.py',
    dag=dag,
)

# Task 9: Version the data
version_data = BashOperator(
    task_id='data_versioning',
    bash_command='cd /path/to/project && python src/versioning/version_data.py',
    dag=dag,
)

# Define the DAG dependencies (this is the important part!)
# ingestion → storage → validation → quality → preparation → features → database → feature_store → versioning

get_data >> store_data >> validate_data >> quality_report >> clean_data >> make_features >> setup_database >> manage_features >> version_data
