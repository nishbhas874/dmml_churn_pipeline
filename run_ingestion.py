#!/usr/bin/env python3
"""
Simple script to run the unified data ingestion pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from ingestion.unified_ingestion import ingest_all_data, get_primary_data_file


def main():
    """Run the unified data ingestion."""
    print("Starting Unified Data Ingestion Pipeline...")
    print("=" * 60)
    
    try:
        # Run ingestion
        ingested_files = ingest_all_data()
        
        # Get primary data file
        primary_file = get_primary_data_file(ingested_files)
        
        if primary_file:
            print("\n" + "=" * 60)
            print("INGESTION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Primary data file: {primary_file}")
            print("\nYou can now run the main pipeline with:")
            print("python run_pipeline.py")
        else:
            print("\n" + "=" * 60)
            print("INGESTION FAILED!")
            print("=" * 60)
            print("No data files were successfully ingested.")
            print("Please check your configuration and credentials.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nIngestion failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
