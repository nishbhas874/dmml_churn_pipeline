# Simple Data Versioning Strategy

## Overview
We track changes in our data files to ensure reproducibility using Git version control and custom metadata tracking.

## What We Track
- Raw data from Kaggle and Hugging Face
- Processed/cleaned data
- Transformed data with new features
- Feature store data

## How It Works

### 1. Git Version Control
- Creates Git commits for each pipeline run
- Tags each version with timestamp (data-v20250822_143052)
- Shows what changed between versions

### 2. Custom Metadata
- Tracks file hashes to detect changes
- Records file sizes and row counts
- Stores timestamps for each version

## Simple Usage

### Run Pipeline with Versioning
```bash
python pipeline.py    # Automatically creates new version
```

### Check Version History
```bash
git tag -l            # See all data versions
git log --oneline     # See commit history
```

### What Gets Tracked
1. **Raw data** - Original downloads from Kaggle/HuggingFace
2. **Processed data** - Cleaned files
3. **Transformed data** - Files with new features
4. **Feature data** - Feature store files

### Version Format
- Git tags: `data-v20250822_143052`
- Metadata files: `version_20250822_143052.json`

## Requirements Met ✅

- ✅ **Git repository** - Shows dataset versions with tags
- ✅ **Track changes** - File hashes detect modifications  
- ✅ **Version metadata** - Source, timestamp, change log
- ✅ **Reproducibility** - Can recreate any version
- ✅ **Documentation** - This strategy file

## Files Created
```
data/versioned/
├── metadata/
│   └── version_20250822_143052.json
└── version_registry.json

.git/                           # Git repository
└── tags/                       # Version tags
```

This simple approach meets all the problem statement requirements while keeping it student-friendly!
