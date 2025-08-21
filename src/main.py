"""
Main pipeline orchestrator for churn prediction.
Runs the complete pipeline from data ingestion to model evaluation.
"""

import sys
from pathlib import Path
import yaml
from loguru import logger
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data.data_processor import DataProcessor
from features.feature_engineer import FeatureEngineer
from models.train_model import ModelTrainer
from visualization.visualizer import DataVisualizer
from ingestion.unified_ingestion import ingest_all_data, get_primary_data_file


class ChurnPredictionPipeline:
    """Main pipeline orchestrator for churn prediction."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.start_time = None
        self.end_time = None
        
        # Initialize components
        self.data_processor = DataProcessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.model_trainer = ModelTrainer(config_path)
        self.visualizer = DataVisualizer(config_path)
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config['logging']['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(
            log_file,
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            rotation="10 MB",
            retention="7 days"
        )
        logger.add(
            sys.stdout,
            level=self.config['logging']['level'],
            format=self.config['logging']['format']
        )
    
    def run_data_ingestion(self) -> bool:
        """Run data ingestion pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Run unified data ingestion
            ingested_files = ingest_all_data()
            
            # Get primary data file
            primary_file = get_primary_data_file(ingested_files)
            
            if primary_file:
                logger.info(f"Primary data file ready: {primary_file}")
                return True
            else:
                logger.warning("No data files were successfully ingested")
                logger.info("Please check your configuration and credentials")
                return False
                
        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            return False
    
    def run_data_processing(self) -> bool:
        """Run data processing pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Check if raw data exists (from ingestion or manual placement)
            data_file = self.data_processor.raw_data_path / "churn_data.csv"
            
            # If no churn_data.csv, look for other CSV files
            if not data_file.exists():
                csv_files = list(self.data_processor.raw_data_path.glob("*.csv"))
                if csv_files:
                    data_file = csv_files[0]
                    logger.info(f"Using data file: {data_file}")
                else:
                    logger.warning(f"No data files found in {self.data_processor.raw_data_path}")
                    logger.info("Please run data ingestion first or place your data file in the data/raw/ directory")
                    return False
            
            # Load and process data
            data = self.data_processor.load_data(str(data_file))
            cleaned_data = self.data_processor.clean_data(data)
            encoded_data = self.data_processor.encode_categorical_features(cleaned_data)
            
            # Split data
            train, val, test = self.data_processor.split_data(encoded_data)
            
            # Save processed data
            self.data_processor.save_processed_data(train, val, test)
            
            # Generate data summary
            summary = self.data_processor.get_data_summary(cleaned_data)
            logger.info("Data processing completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            return False
    
    def run_feature_engineering(self) -> bool:
        """Run feature engineering pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Load processed data
            train_file = self.feature_engineer.processed_data_path / "processed_train.csv"
            val_file = self.feature_engineer.processed_data_path / "processed_val.csv"
            test_file = self.feature_engineer.processed_data_path / "processed_test.csv"
            
            if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
                logger.error("Processed data files not found. Please run data processing first.")
                return False
            
            train = self.feature_engineer.load_data(str(train_file))
            val = self.feature_engineer.load_data(str(val_file))
            test = self.feature_engineer.load_data(str(test_file))
            
            target_col = self.config['features']['target_column']
            
            # Engineer features
            train_engineered = self.feature_engineer.engineer_features(train, target_col)
            val_engineered = self.feature_engineer.engineer_features(val, target_col)
            test_engineered = self.feature_engineer.engineer_features(test, target_col)
            
            # Save engineered data
            train_engineered.to_csv(self.feature_engineer.processed_data_path / "engineered_train.csv", index=False)
            val_engineered.to_csv(self.feature_engineer.processed_data_path / "engineered_val.csv", index=False)
            test_engineered.to_csv(self.feature_engineer.processed_data_path / "engineered_test.csv", index=False)
            
            # Save feature importance
            self.feature_engineer.save_feature_importance()
            
            logger.info("Feature engineering completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return False
    
    def run_model_training(self) -> bool:
        """Run model training pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Run complete training pipeline
            self.model_trainer.run_training_pipeline()
            
            logger.info("Model training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return False
    
    def run_visualization(self) -> bool:
        """Run visualization pipeline."""
        logger.info("=" * 50)
        logger.info("STARTING VISUALIZATION PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Load data for visualization
            data_file = self.visualizer.processed_data_path / "processed_train.csv"
            if not data_file.exists():
                logger.error("Processed data file not found. Please run data processing first.")
                return False
            
            data = self.visualizer.load_data(str(data_file))
            
            # Load model results if available
            results_file = self.visualizer.results_path / "model_results.json"
            results = None
            if results_file.exists():
                import json
                with open(results_file, 'r') as f:
                    results = json.load(f)
            
            # Generate all plots
            self.visualizer.generate_all_plots(data, results)
            
            logger.info("Visualization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete churn prediction pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE CHURN PREDICTION PIPELINE")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        try:
            # Step 1: Data Ingestion
            if not self.run_data_ingestion():
                logger.warning("Data ingestion failed, but continuing with existing data...")
            
            # Step 2: Data Processing
            if not self.run_data_processing():
                logger.error("Data processing failed. Stopping pipeline.")
                return False
            
            # Step 3: Feature Engineering
            if not self.run_feature_engineering():
                logger.error("Feature engineering failed. Stopping pipeline.")
                return False
            
            # Step 4: Model Training
            if not self.run_model_training():
                logger.error("Model training failed. Stopping pipeline.")
                return False
            
            # Step 5: Visualization
            if not self.run_visualization():
                logger.warning("Visualization failed, but continuing...")
            
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {execution_time:.2f} seconds")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate a comprehensive report of the pipeline execution."""
        logger.info("Generating pipeline report...")
        
        report = {
            'pipeline_info': {
                'start_time': datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S') if self.end_time else None,
                'execution_time': f"{self.end_time - self.start_time:.2f} seconds" if self.start_time and self.end_time else None
            },
            'configuration': self.config,
            'output_files': {
                'processed_data': str(self.data_processor.processed_data_path),
                'models': str(self.model_trainer.model_path),
                'results': str(self.model_trainer.results_path),
                'plots': str(self.visualizer.plots_path),
                'logs': str(Path(self.config['logging']['file']))
            }
        }
        
        # Save report
        import json
        report_file = Path("results") / "pipeline_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Pipeline report saved to {report_file}")


def main():
    """Main function to run the complete pipeline."""
    try:
        # Initialize pipeline
        pipeline = ChurnPredictionPipeline()
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            # Generate report
            pipeline.generate_report()
            logger.info("Pipeline completed successfully! Check the results directory for outputs.")
        else:
            logger.error("Pipeline failed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
