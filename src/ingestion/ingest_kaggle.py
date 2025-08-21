"""
Kaggle data ingestion module for churn prediction pipeline.
"""

from pydantic import BaseModel
from pathlib import Path
import os
import subprocess
import shutil
import pandas as pd
import yaml
from loguru import logger


class Config(BaseModel):
    """Configuration model for data ingestion."""
    project_name: str
    random_state: int
    paths: dict
    sources: dict


def load_config(path: str = "config/config.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def ensure_dir(p: str | Path):
    """Ensure directory exists."""
    Path(p).mkdir(parents=True, exist_ok=True)


def setup_kaggle_credentials(username: str, key: str):
    """Setup Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / "kaggle.json"
    credentials = {
        "username": username,
        "key": key
    }
    
    import json
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)
    
    # Set proper permissions (Unix-like systems)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass  # Windows systems don't support chmod
    
    logger.info("Kaggle credentials configured")


def ingest_kaggle():
    """Ingest data from Kaggle datasets."""
    cfg = load_config()
    raw_dir = Path(cfg.paths["raw"]) / "kaggle"
    ensure_dir(raw_dir)
    
    dataset = cfg.sources["kaggle"]["dataset"]
    file = cfg.sources["kaggle"]["file"]
    
    # Setup credentials if provided
    credentials = cfg.sources["kaggle"]["api_credentials"]
    if credentials.get("username") and credentials.get("key"):
        setup_kaggle_credentials(credentials["username"], credentials["key"])
    
    logger.info("Starting Kaggle data ingestion...")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"File: {file}")
    
    try:
        # Use Kaggle CLI to download dataset
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "--force"], 
            check=True
        )
        
        # Unzip any downloaded zip files
        for zip_file in raw_dir.glob("*.zip"):
            logger.info(f"Extracting {zip_file}")
            shutil.unpack_archive(zip_file, raw_dir)
            zip_file.unlink()  # Remove zip file after extraction
        
        logger.info("Kaggle data successfully downloaded and extracted")
        
        # Check if target file exists
        target_file = raw_dir / file
        if target_file.exists():
            logger.info(f"Target file found: {target_file}")
            return target_file
        else:
            # List available files
            available_files = list(raw_dir.glob("*.csv"))
            logger.info(f"Available CSV files: {[f.name for f in available_files]}")
            
            if available_files:
                # Use the first CSV file found
                return available_files[0]
            else:
                raise FileNotFoundError(f"Target file {file} not found and no CSV files available")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle CLI command failed: {e}")
        logger.info("Please ensure Kaggle CLI is installed and configured")
        logger.info("Install with: pip install kaggle")
        logger.info("Configure with: kaggle config set-credentials")
        raise
    except Exception as e:
        logger.error(f"Kaggle ingestion failed: {e}")
        raise


if __name__ == "__main__":
    ingest_kaggle()
