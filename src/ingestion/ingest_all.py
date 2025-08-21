"""
Main data ingestion orchestrator for churn prediction pipeline.
"""

from src.ingestion.ingest_kaggle import ingest_kaggle
from src.ingestion.ingest_huggingface import ingest_huggingface
from loguru import logger


def ingest_all_data():
    """Run all data ingestion processes."""
    logger.info("Starting data ingestion pipeline...")
    
    ingested_files = []
    
    # Try Kaggle ingestion
    try:
        logger.info("=" * 50)
        logger.info("KAGGLE DATA INGESTION")
        logger.info("=" * 50)
        kaggle_file = ingest_kaggle()
        ingested_files.append(("kaggle", kaggle_file))
        logger.info("Kaggle ingestion completed successfully")
    except Exception as e:
        logger.warning(f"Kaggle ingestion skipped due to error: {e}")
    
    # Try Hugging Face ingestion
    try:
        logger.info("=" * 50)
        logger.info("HUGGING FACE DATA INGESTION")
        logger.info("=" * 50)
        huggingface_file = ingest_huggingface()
        ingested_files.append(("huggingface", huggingface_file))
        logger.info("Hugging Face ingestion completed successfully")
    except Exception as e:
        logger.warning(f"Hugging Face ingestion skipped due to error: {e}")
    
    # Summary
    logger.info("=" * 50)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 50)
    
    if ingested_files:
        logger.info(f"Successfully ingested {len(ingested_files)} data sources:")
        for source, file_path in ingested_files:
            logger.info(f"  - {source}: {file_path}")
    else:
        logger.warning("No data sources were successfully ingested")
        logger.info("Please check your configuration and credentials")
    
    return ingested_files


if __name__ == "__main__":
    ingest_all_data()
