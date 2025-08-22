# 🚀 Customer Churn Pipeline - Setup Instructions

## For Testing in VS Code

### 📋 **Step 1: Clone the Project**
```bash
git clone <your-github-repo-url>
cd dmml_churn_pipeline
```

### 📋 **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### 📋 **Step 3: Set Up API Keys**

#### **A. Kaggle API Setup**
1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → "Create New Token"
2. Download `kaggle.json` file
3. **Windows**: Place in `C:\Users\<username>\.kaggle\kaggle.json`
4. **Mac/Linux**: Place in `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Mac/Linux only)

#### **B. HuggingFace Access**
- No API key needed! Our script uses public datasets
- The `datasets` library handles authentication automatically

### 📋 **Step 4: Test the Pipeline**

#### **Quick Test (Single Command):**
```bash
python get_data.py
```

#### **Full Pipeline Test:**
```bash
python pipeline.py
```

#### **Orchestrated Pipeline Test:**
```bash
python run_airflow.py
```

#### **Model Training Test:**
```bash
python train_models.py
```

### 📋 **Step 5: Expected Results**
After running, you should see:
- `data/raw/` - Downloaded datasets
- `data/processed/` - Cleaned data
- `data/features/` - Engineered features  
- `models/` - Trained ML models
- Various reports and logs

### 🔧 **Troubleshooting**

#### **Issue: Kaggle API not working**
```bash
# Test Kaggle connection
kaggle datasets list
```

#### **Issue: Missing packages**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### **Issue: Permission errors**
- Make sure `kaggle.json` has correct permissions
- Run VS Code as administrator if needed (Windows)

### 📊 **What You'll Get**
- Complete data pipeline (9 stages)
- Real customer churn data
- Trained ML models (3 algorithms)
- Performance reports
- Airflow DAG for orchestration

### 🎯 **Assignment Deliverables Generated**
- ✅ Data ingestion scripts
- ✅ Pipeline orchestration (Airflow)
- ✅ Model training & evaluation
- ✅ Data quality reports
- ✅ Feature engineering
- ✅ Data versioning

---
**Need help? Check the logs in each `data/` subfolder for detailed execution info!**
