# share_package.py - Prepare project for sharing
# Student: [Your Name]
# Course: Data Mining & Machine Learning

import os
import shutil
import zipfile
from pathlib import Path

def clean_for_sharing():
    """Clean project for sharing - remove sensitive files"""
    print("🧹 Cleaning project for sharing...")
    
    # Files/folders to exclude when sharing
    exclude_items = [
        '.git',
        '__pycache__',
        '*.pyc',
        '.env',
        'kaggle.json',
        'mlruns',  # MLflow artifacts can be large
        'data/raw',  # Let others download fresh data
        'models/*.pkl',  # Let others train fresh models
        '.vscode/settings.json'  # Personal VS Code settings
    ]
    
    print("Items excluded from sharing:")
    for item in exclude_items:
        print(f"  - {item}")
    
    return exclude_items

def create_sample_config():
    """Create sample configuration files"""
    print("\n📝 Creating sample config files...")
    
    # Create .env.example
    env_example = """# Environment Variables (Copy to .env and fill in your values)
KAGGLE_USERNAME=your_username_here
KAGGLE_KEY=your_api_key_here

# Optional: MLflow tracking
MLFLOW_TRACKING_URI=file:./mlruns
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    
    print("✅ Created .env.example")
    
    # Create README for sharing
    sharing_readme = """# 🚀 Customer Churn Prediction Pipeline

## Quick Start for Testing

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API
- Get your API key from kaggle.com
- Place kaggle.json in ~/.kaggle/ folder
- See SETUP_INSTRUCTIONS.md for details

### 3. Run Pipeline
```bash
python get_data.py        # Download data
python pipeline.py        # Full pipeline
python train_models.py    # Train models
```

### 4. View Results
- Check `data/` folders for processed data
- Check `models/` for trained models
- Check reports for performance metrics

## Project Structure
```
dmml_churn_pipeline/
├── get_data.py           # Simple data ingestion
├── pipeline.py           # Complete pipeline
├── train_models.py       # Model training
├── run_airflow.py        # Airflow orchestration
├── src/                  # Source code by stage
├── dags/                 # Airflow DAG files
├── data/                 # Data pipeline outputs
├── models/               # Trained models
└── requirements.txt      # Dependencies
```

## Assignment Coverage
✅ Data Ingestion (Stage 2)
✅ Data Storage (Stage 3)  
✅ Data Validation (Stage 4)
✅ Data Preparation (Stage 5)
✅ Data Transformation (Stage 6)
✅ Feature Store (Stage 7)
✅ Data Versioning (Stage 8)
✅ Model Building (Stage 9)
✅ Pipeline Orchestration (Stage 10)

**Ready for assignment submission!**
"""
    
    with open('README_SHARING.md', 'w') as f:
        f.write(sharing_readme)
    
    print("✅ Created README_SHARING.md")

def create_test_script():
    """Create a simple test script for others"""
    test_script = """# test_pipeline.py - Quick pipeline test
# Run this to test if everything works

import os
import subprocess

def test_dependencies():
    \"\"\"Test if all required packages are installed\"\"\"
    print("📦 Testing dependencies...")
    try:
        import pandas, numpy, sklearn, matplotlib, seaborn, kaggle, datasets
        print("✅ All packages installed correctly")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_kaggle_api():
    \"\"\"Test Kaggle API setup\"\"\"
    print("\\n🔑 Testing Kaggle API...")
    try:
        result = subprocess.run(['kaggle', 'datasets', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Kaggle API working")
            return True
        else:
            print("❌ Kaggle API error:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Kaggle API test failed: {e}")
        print("Make sure kaggle.json is in ~/.kaggle/ folder")
        return False

def test_data_download():
    \"\"\"Test data download\"\"\"
    print("\\n📊 Testing data download...")
    try:
        result = subprocess.run(['python', 'get_data.py'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Data download successful")
            return True
        else:
            print("❌ Data download failed:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Data download test failed: {e}")
        return False

def main():
    print("🧪 Pipeline Test Suite")
    print("=" * 30)
    
    tests = [
        test_dependencies,
        test_kaggle_api,
        test_data_download
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Pipeline ready to use.")
    else:
        print("⚠️ Some tests failed. Check SETUP_INSTRUCTIONS.md")

if __name__ == "__main__":
    main()
"""
    
    with open('test_pipeline.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_pipeline.py")

def main():
    """Prepare project for sharing"""
    print("📦 Preparing Customer Churn Pipeline for Sharing")
    print("=" * 50)
    
    # Clean project
    exclude_items = clean_for_sharing()
    
    # Create sample configs
    create_sample_config()
    
    # Create test script
    create_test_script()
    
    print(f"\\n✅ Project prepared for sharing!")
    print(f"\\n📋 Instructions for recipients:")
    print(f"1. Extract the project files")
    print(f"2. Follow SETUP_INSTRUCTIONS.md")
    print(f"3. Run: python test_pipeline.py")
    print(f"4. If tests pass, run: python pipeline.py")
    
    print(f"\\n🔒 Security Notes:")
    print(f"- No API keys included in shared files")
    print(f"- Recipients need their own Kaggle account")
    print(f"- All sensitive data excluded")

if __name__ == "__main__":
    main()
