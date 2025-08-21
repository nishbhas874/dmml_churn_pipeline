"""
Unified data ingestion module for churn prediction pipeline.
Handles both Kaggle and Hugging Face data sources.
"""

from pydantic import BaseModel
from pathlib import Path
import os
import subprocess
import shutil
import pandas as pd
import yaml
from loguru import logger
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional imports for Hugging Face
try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("Hugging Face datasets not available. Install with: pip install datasets")


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


def ingest_kaggle_data(cfg: Config) -> Optional[Path]:
    """Ingest data from Kaggle datasets."""
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
        return None
    except Exception as e:
        logger.error(f"Kaggle ingestion failed: {e}")
        return None


def ingest_huggingface_data(cfg: Config) -> Optional[Path]:
    """Ingest data from Hugging Face datasets."""
    if not HUGGINGFACE_AVAILABLE:
        logger.error("Hugging Face datasets not available. Install with: pip install datasets")
        return None
    
    raw_dir = Path(cfg.paths["raw"]) / "huggingface"
    ensure_dir(raw_dir)
    
    repo_id = cfg.sources["huggingface"]["repo_id"]
    filename = cfg.sources["huggingface"]["filename"]
    split = cfg.sources["huggingface"]["split"]
    
    logger.info("Starting Hugging Face data ingestion...")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Split: {split}")
    
    try:
        # Load dataset from Hugging Face
        ds = load_dataset(repo_id, split=split)
        df = ds.to_pandas()
        
        # Save to CSV
        outpath = raw_dir / filename
        df.to_csv(outpath, index=False)
        
        logger.info(f"Hugging Face data successfully saved to {outpath}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return outpath
        
    except Exception as e:
        logger.error(f"Hugging Face ingestion failed: {e}")
        return None


def ingest_all_data(config_path: str = "config/config.yaml") -> Dict[str, Optional[Path]]:
    """
    Unified data ingestion function that handles both Kaggle and Hugging Face.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Optional[Path]]: Dictionary mapping source names to file paths
    """
    logger.info("Starting unified data ingestion pipeline...")
    
    # Load configuration
    try:
        cfg = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}
    
    ingested_files = {}
    
    # Try Kaggle ingestion
    try:
        logger.info("=" * 50)
        logger.info("KAGGLE DATA INGESTION")
        logger.info("=" * 50)
        kaggle_file = ingest_kaggle_data(cfg)
        if kaggle_file:
            ingested_files["kaggle"] = kaggle_file
            logger.info("Kaggle ingestion completed successfully")
        else:
            logger.warning("Kaggle ingestion failed")
    except Exception as e:
        logger.warning(f"Kaggle ingestion skipped due to error: {e}")
        ingested_files["kaggle"] = None
    
    # Try Hugging Face ingestion
    try:
        logger.info("=" * 50)
        logger.info("HUGGING FACE DATA INGESTION")
        logger.info("=" * 50)
        huggingface_file = ingest_huggingface_data(cfg)
        if huggingface_file:
            ingested_files["huggingface"] = huggingface_file
            logger.info("Hugging Face ingestion completed successfully")
        else:
            logger.warning("Hugging Face ingestion failed")
    except Exception as e:
        logger.warning(f"Hugging Face ingestion skipped due to error: {e}")
        ingested_files["huggingface"] = None
    
    # Summary
    logger.info("=" * 50)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 50)
    
    successful_ingestions = {k: v for k, v in ingested_files.items() if v is not None}
    
    if successful_ingestions:
        logger.info(f"Successfully ingested {len(successful_ingestions)} data sources:")
        for source, file_path in successful_ingestions.items():
            logger.info(f"  - {source}: {file_path}")
    else:
        logger.warning("No data sources were successfully ingested")
        logger.info("Please check your configuration and credentials")
    
    return ingested_files


def get_primary_data_file(ingested_files: Dict[str, Optional[Path]]) -> Optional[Path]:
    """
    Get the primary data file to use for the pipeline.
    Priority: Kaggle > Hugging Face > None
    
    Args:
        ingested_files (Dict[str, Optional[Path]]): Dictionary of ingested files
        
    Returns:
        Optional[Path]: Path to the primary data file
    """
    # Priority order: Kaggle first, then Hugging Face
    for source in ["kaggle", "huggingface"]:
        if source in ingested_files and ingested_files[source] is not None:
            logger.info(f"Using {source} data as primary source: {ingested_files[source]}")
            return ingested_files[source]
    
    logger.warning("No primary data file available")
    return None


def main():
    """Main function to run unified data ingestion."""
    try:
        # Run ingestion
        ingested_files = ingest_all_data()
        
        # Get primary data file
        primary_file = get_primary_data_file(ingested_files)
        
        if primary_file:
            logger.info(f"Primary data file ready: {primary_file}")
            return primary_file
        else:
            logger.error("No data files available for processing")
            return None
            
    except Exception as e:
        logger.error(f"Unified ingestion failed: {e}")
        return None


if __name__ == "__main__":
    main()
