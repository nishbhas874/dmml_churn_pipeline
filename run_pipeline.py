#!/usr/bin/env python3
"""
Simple script to run the complete churn prediction pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from main import ChurnPredictionPipeline


def main():
    """Run the complete pipeline."""
    print("Starting Churn Prediction Pipeline...")
    print("=" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = ChurnPredictionPipeline()
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\n" + "=" * 50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print("\nOutput files:")
            print("- Processed data: data/processed/")
            print("- Trained models: models/")
            print("- Results: results/")
            print("- Plots: plots/")
            print("- Logs: logs/")
        else:
            print("\n" + "=" * 50)
            print("PIPELINE FAILED!")
            print("=" * 50)
            print("Check the logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
