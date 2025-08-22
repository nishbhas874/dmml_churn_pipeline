# How to Share Your Pipeline with Others

## Step 1: Prepare Your Project for Sharing

### A. Clean Up Sensitive Files
Before sharing, remove these files:
- `kaggle.json` (contains your API key)
- `.env` files (if any)
- Large model files (optional - let others train fresh)
- Personal VS Code settings

### B. Create Instructions
Include these files in your share:
- `SETUP_INSTRUCTIONS.md` (already created)
- `requirements.txt` (already exists)
- `kaggle_setup_example.json` (already created)

## Step 2: Share via GitHub (Recommended)

### A. Push to GitHub
```bash
git add .
git commit -m "Complete pipeline ready for sharing"
git push origin main
```

### B. Make Repository Public
1. Go to your GitHub repository
2. Settings > General > Danger Zone
3. Change repository visibility to "Public"

### C. Share the GitHub URL
Send this to others:
```
https://github.com/yourusername/dmml_churn_pipeline
```

## Step 3: Alternative - ZIP File Sharing

### A. Create ZIP (Manual)
1. Copy project folder
2. Remove sensitive files:
   - Delete `.git` folder (optional)
   - Delete `kaggle.json`
   - Delete large model files
3. ZIP the folder
4. Share via email/drive

## Step 4: Instructions for Recipients

### They need to:
1. **Clone/Extract** your project
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Get Kaggle API key** from their own Kaggle account
4. **Place kaggle.json** in `~/.kaggle/` folder
5. **Test**: `python get_data.py`
6. **Run pipeline**: `python pipeline.py`

## Step 5: What They'll Get

### Working Pipeline:
- Data ingestion from Kaggle + HuggingFace
- Complete 9-stage pipeline
- Model training with 3 algorithms
- Airflow orchestration
- All assignment deliverables

### No Setup Required for:
- HuggingFace (uses public datasets)
- GitHub (if sharing via public repo)
- MLflow (local tracking)

## Security Notes

### Safe to Share:
- All Python scripts
- Configuration templates
- Documentation
- Pipeline structure

### NEVER Share:
- Your `kaggle.json` file
- Any `.env` files with keys
- Personal credentials

## Testing Instructions for Recipients

### Quick Test:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test data download
python get_data.py

# 3. Run full pipeline
python pipeline.py

# 4. Train models
python train_models.py
```

### Expected Results:
- `data/raw/` - Downloaded datasets
- `data/processed/` - Cleaned data  
- `models/` - Trained ML models
- Reports and logs in various folders

Perfect! Your pipeline is now ready to share securely!
