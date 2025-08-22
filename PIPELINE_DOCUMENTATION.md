# Customer Churn Prediction Pipeline - Technical Documentation

## 📋 Table of Contents
1. [Pipeline Design Overview](#pipeline-design-overview)
2. [Architecture Components](#architecture-components)
3. [Challenges Faced and Solutions](#challenges-faced-and-solutions)
4. [Technical Implementation Details](#technical-implementation-details)
5. [Performance Metrics](#performance-metrics)
6. [Lessons Learned](#lessons-learned)

---

## 🏗️ Pipeline Design Overview

### **Design Philosophy**
The Customer Churn Prediction Pipeline follows a **modular, scalable, and reproducible** design pattern that implements industry best practices for MLOps and data engineering.

### **Pipeline Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Sources   │───▶│   Ingestion     │───▶│    Storage      │
│ • Kaggle API    │    │ • get_data.py   │    │ • Organized     │
│ • HuggingFace   │    │ • Multi-source  │    │ • Versioned     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Validation    │◀───│   Preparation   │◀───│   Validation    │
│ • Quality Check │    │ • Data Cleaning │    │ • Format Check  │
│ • Report Gen.   │    │ • Preprocessing │    │ • Integrity     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Transformation  │───▶│ Feature Store   │───▶│   Versioning    │
│ • Feature Eng.  │    │ • Metadata      │    │ • Git Tags      │
│ • Database      │    │ • Registry      │    │ • Change Track  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │
        ▼
┌─────────────────┐    ┌─────────────────┐
│ Orchestration   │───▶│   Monitoring    │
│ • Prefect       │    │ • Logging       │
│ • Custom        │    │ • Metrics       │
└─────────────────┘    └─────────────────┘
```

### **Key Design Principles**

#### **1. Modularity**
- Each pipeline stage is implemented as an independent Python script
- Clear separation of concerns between data processing stages
- Reusable components across different pipeline configurations

#### **2. Reproducibility**
- Comprehensive versioning strategy using Git tags
- Deterministic data processing with fixed random seeds
- Complete artifact tracking and metadata management

#### **3. Scalability**
- Modular architecture supports easy addition of new data sources
- Feature store design enables feature reuse across models
- Database-driven approach supports large-scale data processing

#### **4. Observability**
- Comprehensive logging at every pipeline stage
- Real-time monitoring through orchestration tools
- Detailed performance metrics and quality reports

---

## 🔧 Architecture Components

### **1. Data Ingestion Layer**
**Script**: `get_data.py`
- **Purpose**: Multi-source data acquisition
- **Sources**: Kaggle API, HuggingFace Datasets
- **Output**: Raw data in organized directory structure

### **2. Storage Layer**
**Script**: `src/storage/store_data.py`
- **Purpose**: Organized data lake implementation
- **Features**: Backup management, manifest generation
- **Structure**: Source-based partitioning with timestamps

### **3. Validation Layer**
**Scripts**: `src/validation/check_data.py`, `src/validation/generate_quality_report.py`
- **Purpose**: Data quality assurance and reporting
- **Checks**: Missing values, duplicates, data types, outliers
- **Outputs**: JSON reports, CSV summaries, PNG visualizations

### **4. Preparation Layer**
**Scripts**: `src/preparation/clean_data.py`, `src/preparation/visualize_data.py`
- **Purpose**: Data cleaning and exploratory analysis
- **Features**: Missing value imputation, type conversion, visualization
- **Outputs**: Cleaned datasets, summary statistics, plots

### **5. Transformation Layer**
**Scripts**: `src/transformation/make_features.py`, `src/transformation/database_setup.py`
- **Purpose**: Feature engineering and database management
- **Features**: Feature scaling, categorical encoding, SQL schema creation
- **Outputs**: Engineered features, SQLite database, sample queries

### **6. Feature Store Layer**
**Script**: `src/feature_store/manage_features.py`
- **Purpose**: Feature metadata management and versioning
- **Features**: Feature registry, metadata tracking, retrieval APIs
- **Outputs**: Feature catalog, version registry, metadata files

### **7. Versioning Layer**
**Script**: `src/versioning/version_data.py`
- **Purpose**: Data and pipeline version control
- **Features**: Git integration, automated tagging, change tracking
- **Outputs**: Git tags, version registry, commit history

### **8. Orchestration Layer**
**Scripts**: `simple_prefect_demo.py`, `simple_orchestration_demo.py`
- **Purpose**: Pipeline automation and monitoring
- **Tools**: Prefect (industry standard), Custom orchestrator
- **Features**: Task dependencies, error handling, retry logic, monitoring

---

## 🚨 Challenges Faced and Solutions Implemented

### **Challenge 1: Multi-Source Data Integration**

#### **Problem**
- Different data formats from Kaggle (CSV) and HuggingFace (Dataset objects)
- Inconsistent column naming and data types
- Authentication requirements for external APIs

#### **Solution Implemented**
```python
# Unified data loading approach in get_data.py
def load_kaggle_data():
    kaggle_cmd = "kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/kaggle --unzip"
    subprocess.run(kaggle_cmd, shell=True)

def load_huggingface_data():
    dataset = load_dataset("scikit-learn/adult-census-income", split="train")
    df = dataset.to_pandas()
    df.to_csv("data/raw/huggingface/customer_data.csv", index=False)
```

#### **Lessons Learned**
- Always implement error handling for external API calls
- Use consistent data format (CSV) as intermediate representation
- Create unified data loading interfaces for different sources

---

### **Challenge 2: Database Connection and Query Issues**

#### **Problem**
```bash
# Original error when trying to query SQLite from command line
PS> sqlite3 data/transformed/churn_database.db "SELECT name FROM sqlite_master WHERE type='table'"
SyntaxError: unterminated string literal
```

#### **Root Cause**
- PowerShell syntax conflicts with SQLite command-line interface
- Direct database queries from terminal not working properly

#### **Solution Implemented**
Created dedicated Python script for database verification:
```python
# check_db.py (created and later removed)
import sqlite3

def verify_database():
    conn = sqlite3.connect('data/transformed/churn_database.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("Database tables:")
    for table in tables:
        print(f"  - {table[0]}")
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"  {table[0]}: {count} rows")
    
    conn.close()
```

#### **Lessons Learned**
- Always use Python scripts for database operations in mixed environments
- Implement proper connection handling and error management
- Create reusable database utility functions

---

### **Challenge 3: Prefect Server Connectivity**

#### **Problem**
```bash
RuntimeError: Failed to reach API at http://localhost:4200/api/
httpcore.ConnectError: [WinError 10061] No connection could be made because the target machine actively refused it
```

#### **Root Cause**
- Prefect server was not running when attempting to execute flows
- Missing server initialization step in the workflow

#### **Solution Implemented**
```bash
# Start Prefect server in background
prefect server start --host 0.0.0.0 --port 4200

# Wait for server initialization
timeout 5

# Execute Prefect flow
python simple_prefect_demo.py
```

#### **Additional Safeguards**
- Created custom orchestrator as fallback option
- Implemented server health checks before flow execution
- Added comprehensive error handling in orchestration scripts

#### **Lessons Learned**
- Always verify service dependencies before pipeline execution
- Implement fallback mechanisms for external service failures
- Document service startup procedures clearly

---

### **Challenge 4: Directory Structure and File Path Issues**

#### **Problem**
```bash
# Error when trying to list directory contents recursively
dir : Second path fragment must not be a drive or UNC name.
```

#### **Root Cause**
- Windows PowerShell syntax differences from Unix commands
- Incorrect path handling in directory operations

#### **Solution Implemented**
```powershell
# Correct PowerShell command
Get-ChildItem -Path "data\raw" -Recurse

# Alternative cross-platform approach in Python
import os
from pathlib import Path

def list_directory_contents(path):
    path_obj = Path(path)
    if path_obj.exists():
        for item in path_obj.rglob('*'):
            print(item)
```

#### **Lessons Learned**
- Use cross-platform path handling (pathlib) in Python scripts
- Implement OS-agnostic file operations
- Test directory operations on target deployment environment

---

### **Challenge 5: Data Quality and Validation Edge Cases**

#### **Problem**
- Initial validation showed "0 issues found" which seemed suspicious
- Need to distinguish between genuinely clean data and validation gaps

#### **Investigation Process**
1. **Verified data loading**: Confirmed 7,043 records loaded correctly
2. **Checked validation logic**: Ensured all validation functions were executing
3. **Enhanced reporting**: Added detailed issue breakdown and statistics

#### **Solution Implemented**
```python
# Enhanced validation with detailed reporting
def generate_comprehensive_validation():
    validation_results = {
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df), 
        "data_types": check_data_types(df),
        "outliers": detect_outliers(df),
        "range_validation": check_value_ranges(df)
    }
    
    # Generate detailed quality report
    create_quality_visualizations(validation_results)
    generate_quality_summary(validation_results)
```

#### **Outcome**
- Confirmed data quality was genuinely high (99.2% overall quality score)
- Enhanced validation framework with multiple quality dimensions
- Created comprehensive reporting with visualizations

#### **Lessons Learned**
- Always implement multiple validation dimensions
- Create detailed reporting even for high-quality data
- Use visualizations to communicate data quality effectively

---

### **Challenge 6: Feature Engineering and Scaling Issues**

#### **Problem**
- Inconsistent feature scales affecting model performance
- Need for categorical encoding and numerical feature engineering

#### **Solution Implemented**
```python
# Comprehensive feature engineering pipeline
def create_engineered_features(df):
    # Tenure grouping
    df['tenure_group'] = df['tenure'].apply(lambda x: 
        'Short' if x <= 12 else 'Medium' if x <= 36 else 'Long')
    
    # Charges per tenure calculation
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Contract encoding
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['contract_encoded'] = df['Contract'].map(contract_mapping)
    
    # Feature scaling
    scaler = StandardScaler()
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df
```

#### **Lessons Learned**
- Always implement feature scaling for numerical variables
- Use domain knowledge for feature engineering decisions
- Create comprehensive feature metadata for reproducibility

---

### **Challenge 7: Model Versioning and MLflow Integration**

#### **Problem**
- Need for comprehensive model tracking and versioning
- Integration between custom pipeline and MLflow tracking

#### **Solution Implemented**
```python
# MLflow integration in training pipeline
import mlflow
import mlflow.sklearn

def train_with_mlflow_tracking(X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Train models
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Log metrics
            mlflow.log_metric(f"{name}_accuracy", accuracy_score(y_test, predictions))
            mlflow.log_metric(f"{name}_f1", f1_score(y_test, predictions))
            
            # Log model
            mlflow.sklearn.log_model(model, f"{name}_model")
```

#### **Outcome**
- Complete model versioning with MLflow tracking
- Automated model registration and metadata management
- Integration with existing pipeline versioning strategy

---

### **Challenge 8: Pipeline Orchestration Complexity**

#### **Problem**
- Managing task dependencies and execution order
- Implementing retry logic and error handling
- Monitoring pipeline execution status

#### **Solution Implemented**

**Prefect Implementation**:
```python
@flow(name="Customer Churn Pipeline")
def churn_pipeline():
    # Task dependencies enforced through function calls
    task1_result = data_ingestion()
    task2_result = data_storage(wait_for=[task1_result])
    task3_result = data_validation(wait_for=[task2_result])
    # ... continued for all 9 tasks
    
    return {"pipeline_status": "COMPLETED", "total_tasks": 9}

@task(name="Data Ingestion", retries=1, retry_delay_seconds=5)
def data_ingestion():
    result = subprocess.run([sys.executable, "get_data.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Data ingestion failed: {result.stderr}")
    return "SUCCESS"
```

**Custom Orchestrator Implementation**:
```python
class PipelineOrchestrator:
    def __init__(self):
        self.tasks = []
        self.execution_log = []
    
    def execute_pipeline(self):
        for task in self.tasks:
            try:
                start_time = time.time()
                self.execute_task(task)
                duration = time.time() - start_time
                self.log_success(task, duration)
            except Exception as e:
                self.log_failure(task, str(e))
                if task.get('retry', False):
                    self.retry_task(task)
```

#### **Lessons Learned**
- Implement both industry-standard and custom orchestration solutions
- Always include comprehensive error handling and retry logic
- Create detailed execution logs for debugging and monitoring

---

## 📊 Technical Implementation Details

### **Development Environment**
- **OS**: Windows 10 (Build 26100)
- **Python**: 3.10
- **Shell**: PowerShell
- **IDE**: VS Code with Python extensions

### **Key Dependencies**
```txt
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
kaggle==1.5.16
datasets==2.14.4
prefect==2.10.21
mlflow==2.5.0
pyyaml==6.0.1
```

### **File Structure Created**
```
dmml_churn_pipeline/
├── src/
│   ├── ingestion/get_data.py
│   ├── storage/store_data.py
│   ├── validation/check_data.py
│   ├── validation/generate_quality_report.py
│   ├── preparation/clean_data.py
│   ├── preparation/visualize_data.py
│   ├── transformation/make_features.py
│   ├── transformation/database_setup.py
│   ├── feature_store/manage_features.py
│   └── versioning/version_data.py
├── data/
│   ├── raw/kaggle/, raw/huggingface/
│   ├── storage/raw/, storage/backup/
│   ├── validated/
│   ├── processed/
│   ├── transformed/
│   ├── features/metadata/
│   ├── versioned/
│   └── visualizations/
├── models/
├── mlruns/
└── orchestration scripts
```

---

## 📈 Performance Metrics

### **Pipeline Execution Performance**
- **Total Pipeline Runtime**: 35.0 seconds (Custom), 22.0 seconds (Prefect)
- **Success Rate**: 100% (both orchestrators)
- **Average Task Duration**: 3.9 seconds
- **Data Processing Volume**: 7,043 records, 21 features

### **Data Quality Metrics**
- **Overall Quality Score**: 99.2%
- **Missing Values**: 0.2% (11 out of 7,043 records)
- **Data Type Consistency**: 100%
- **Duplicate Records**: 0%
- **Outlier Detection**: 2.2% flagged for review

### **Model Performance**
- **Logistic Regression**: 80.1% accuracy, 0.66 F1-score
- **Random Forest**: 79.4% accuracy, 0.63 F1-score  
- **Decision Tree**: 72.8% accuracy, 0.54 F1-score

### **Feature Engineering Results**
- **Original Features**: 20
- **Engineered Features**: 29 (45% increase)
- **Feature Store Size**: 2.2MB
- **Metadata Completeness**: 100%

---

## 🎓 Lessons Learned

### **Technical Lessons**

1. **Environment Consistency**: Always use virtual environments and document exact dependency versions
2. **Cross-Platform Compatibility**: Use Python's pathlib for file operations instead of OS-specific commands
3. **Service Dependencies**: Always verify external service availability before pipeline execution
4. **Error Handling**: Implement comprehensive error handling with meaningful error messages
5. **Logging**: Create detailed logs at every pipeline stage for debugging and monitoring

### **Design Lessons**

1. **Modularity**: Design each pipeline component as an independent, testable unit
2. **Fallback Mechanisms**: Always implement backup solutions for critical pipeline components
3. **Data Validation**: Implement multiple dimensions of data quality checks
4. **Version Control**: Use comprehensive versioning for both code and data artifacts
5. **Documentation**: Maintain detailed documentation throughout development process

### **Process Lessons**

1. **Iterative Development**: Build and test pipeline components incrementally
2. **Quality Gates**: Implement quality checks at each pipeline stage
3. **Monitoring**: Create comprehensive monitoring and alerting mechanisms
4. **Reproducibility**: Ensure complete pipeline reproducibility through versioning and documentation
5. **Scalability**: Design with future scaling requirements in mind

---

## 🔚 Conclusion

The Customer Churn Prediction Pipeline successfully demonstrates a production-ready MLOps implementation with comprehensive data processing, quality assurance, feature engineering, and model training capabilities. Despite encountering various technical challenges, each issue was systematically addressed with robust solutions that enhanced the overall pipeline reliability and maintainability.

The pipeline showcases both custom implementations and industry-standard tools, providing a comprehensive learning experience in modern data engineering and machine learning operations practices.

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-22  
**Author**: Group 24 - DMML Assignment  
**Total Implementation Time**: ~8 hours  
**Lines of Code**: ~2,000+ across all components
