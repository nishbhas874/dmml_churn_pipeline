"""
Hugging Face data ingestion module for churn prediction pipeline.
"""

from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import yaml
from loguru import logger
from datasets import load_dataset


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


def ingest_huggingface():
    """Ingest data from Hugging Face datasets."""
    cfg = load_config()
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
        raise


if __name__ == "__main__":
    ingest_huggingface()
