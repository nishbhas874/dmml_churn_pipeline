# Customer Churn Prediction Project

This is our assignment project for predicting customer churn using machine learning.

## What's in this project

```
dmml_churn_pipeline/
├── src/                    # Source code organized by pipeline stage
│   ├── ingestion/         # Stage 2: Data Ingestion
│   ├── storage/           # Stage 3: Raw Data Storage
│   ├── validation/        # Stage 4: Data Validation
│   ├── preparation/       # Stage 5: Data Preparation
│   ├── transformation/    # Stage 6: Data Transformation
│   ├── feature_store/     # Stage 7: Feature Store Management
│   ├── versioning/        # Stage 8: Data Versioning
│   ├── modeling/          # Stage 9: Model Building
│   └── orchestration/     # Stage 10: Pipeline Orchestration
├── data/                  # Data organized by pipeline stage
│   ├── raw/               # Original data files from kaggle/huggingface
│   ├── storage/           # Data lake storage architecture
│   ├── validated/         # Data validation reports and checked data
│   ├── processed/         # Cleaned and prepared data
│   ├── transformed/       # Feature engineered data
│   ├── features/          # Feature store with metadata
│   └── versioned/         # Data versioning information
├── docs/                 # Documentation files
│   ├── PROBLEM_FORMULATION.md
│   ├── FEATURE_STORE_DOCUMENTATION.md
│   └── VERSIONING_STRATEGY.md
├── assignment_docs/      # Assignment documents
├── config/               # Settings file
├── models/               # Trained models
├── requirements.txt      # Python packages needed
├── get_data.py          # Easy access to data ingestion
├── pipeline.py          # Easy access to full pipeline
└── README.md           # This file
```

## What it does

- Gets real customer data from Kaggle and Hugging Face
- Cleans the data and handles missing values
- Creates new features from existing ones
- Trains different ML models (logistic regression, random forest, etc.)
- Evaluates which model works best
- Makes predictions on new data

## How to run it

1. First install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Kaggle API key by putting kaggle.json file in ~/.kaggle/

3. Run the data ingestion:
   ```bash
   python get_data.py
   ```

4. Run the pipeline (choose one option):

   **Option A: Run everything at once:**
   ```bash
   python pipeline.py
   ```

   **Option B: Run step by step:**
   ```bash
   python check_data.py
   python clean_data.py
   python make_features.py
   python train_model.py
   ```

## Getting the data

- Kaggle: Telco Customer Churn dataset (blastchar/telco-customer-churn)
- Hugging Face: E-commerce customer behavior data

## The pipeline steps

1. **Data Ingestion** - Downloads from kaggle or huggingface → `data/raw/`
2. **Raw Data Storage** - Data lake architecture → `data/storage/`
3. **Data Validation** - Check data quality and consistency → `data/validated/`
4. **Data Preparation** - Handle missing values, fix data types → `data/processed/`
5. **Data Transformation** - Create new features → `data/transformed/`
6. **Feature Store** - Manage engineered features → `data/features/`
7. **Data Versioning** - Track data changes → `data/versioned/`
8. **Model Building** - Train different algorithms → `models/`
9. **Orchestration** - Run everything automatically via `pipeline.py`

## Files you might want to look at

**Main Scripts:**
- `get_data.py` - Easy access to data ingestion
- `pipeline.py` - Easy access to full pipeline

**Source Code (organized by stage):**
- `src/ingestion/get_data.py` - Downloads data from kaggle and huggingface
- `src/validation/check_data.py` - Checks data quality
- `src/preparation/clean_data.py` - Cleans the data
- `src/transformation/make_features.py` - Creates new features
- `src/feature_store/manage_features.py` - Manages feature store with metadata
- `src/versioning/version_data.py` - Tracks data changes and versions
- `src/modeling/train_model.py` - Trains the ML models
- `src/orchestration/pipeline.py` - Full pipeline orchestration

## Notes

- The config file `config/config.yaml` has all the settings
- We tried to make it modular so each part can work independently
