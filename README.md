# Churn Prediction Data Pipeline

A comprehensive machine learning pipeline for customer churn prediction using Python.

## Project Structure

```
dmml_churn_pipeline/
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data files
│   └── external/          # External data sources
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   ├── visualization/    # Data visualization
│   └── ingestion/        # Data ingestion from external sources
├── notebooks/            # Jupyter notebooks
├── tests/               # Unit tests
├── config/              # Configuration files
├── logs/                # Log files
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Features

- **Data Ingestion**: Automated data loading from Kaggle and Hugging Face
- **Data Processing**: Automated data cleaning and preprocessing
- **Feature Engineering**: Advanced feature creation and selection
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Model Evaluation**: Comprehensive evaluation metrics
- **Pipeline Automation**: End-to-end pipeline orchestration
- **Monitoring**: Model performance monitoring and logging

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Ingestion

The pipeline supports automatic data ingestion from multiple sources:

### Kaggle Data Ingestion

1. **Setup Kaggle API**:
   - Go to your Kaggle account settings
   - Create a new API token
   - Update `config/config.yaml` with your credentials:
     ```yaml
     sources:
       kaggle:
         dataset: "blastchar/telco-customer-churn"
         file: "WA_Fn-UseC_-Telco-Customer-Churn.csv"
         api_credentials:
           username: "your_kaggle_username"
           key: "your_kaggle_api_key"
     ```

2. **Run Kaggle ingestion**:
   ```bash
   python src/ingestion/unified_ingestion.py
   ```

### Hugging Face Data Ingestion

1. **Configure dataset** in `config/config.yaml`:
   ```yaml
   sources:
     huggingface:
       repo_id: "mkechinov/ecommerce-behavior-data-from-multi-category-store"
       filename: "ecommerce_churn.csv"
       split: "train"
   ```

2. **Run Hugging Face ingestion**:
   ```bash
   python src/ingestion/unified_ingestion.py
   ```

### Unified Data Ingestion

Run both Kaggle and Hugging Face ingestion in a single command:

```bash
python run_ingestion.py
```

This will:
- Try to download data from Kaggle (if credentials are configured)
- Try to download data from Hugging Face
- Select the best available data source
- Provide a summary of ingested files

## Usage

### 1. Data Ingestion Only:
```bash
python run_ingestion.py
```

### 2. Data Preparation:
```bash
python src/data/data_processor.py
```

### 3. Feature Engineering:
```bash
python src/features/feature_engineer.py
```

### 4. Model Training:
```bash
python src/models/train_model.py
```

### 5. Run Complete Pipeline:
```bash
python run_pipeline.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Data sources (Kaggle, Hugging Face)
- API credentials
- Model parameters
- Feature engineering settings
- Evaluation metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
