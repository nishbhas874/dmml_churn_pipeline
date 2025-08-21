"""
Model training module for churn prediction pipeline.
Handles model training, evaluation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from loguru import logger
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


class ModelTrainer:
    """Model trainer for churn prediction pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model trainer with configuration."""
        self.config = self._load_config(config_path)
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.model_path = Path(self.config['output']['model_path'])
        self.results_path = Path(self.config['output']['results_path'])
        self.plots_path = Path(self.config['output']['plots_path'])
        
        # Create directories
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load engineered data."""
        logger.info("Loading engineered data...")
        
        train_file = self.processed_data_path / "engineered_train.csv"
        val_file = self.processed_data_path / "engineered_val.csv"
        test_file = self.processed_data_path / "engineered_test.csv"
        
        if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
            raise FileNotFoundError("Engineered data files not found. Please run feature engineering first.")
        
        train = pd.read_csv(train_file)
        val = pd.read_csv(val_file)
        test = pd.read_csv(test_file)
        
        logger.info(f"Loaded data - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        return train, val, test
    
    def prepare_data(self, train: pd.DataFrame, val: pd.DataFrame, 
                    test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        target_col = self.config['features']['target_column']
        
        # Separate features and target
        X_train = train.drop(target_col, axis=1)
        y_train = train[target_col]
        
        X_val = val.drop(target_col, axis=1)
        y_val = val[target_col]
        
        X_test = test.drop(target_col, axis=1)
        y_test = test[target_col]
        
        logger.info(f"Data prepared - X_train: {X_train.shape}, y_train: {y_train.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with their configurations."""
        logger.info("Initializing models...")
        
        models = {}
        
        # Random Forest
        rf_config = self.config['models']['random_forest']
        models['random_forest'] = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            min_samples_leaf=rf_config['min_samples_leaf'],
            random_state=rf_config['random_state']
        )
        
        # XGBoost
        xgb_config = self.config['models']['xgboost']
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            subsample=xgb_config['subsample'],
            colsample_bytree=xgb_config['colsample_bytree'],
            random_state=xgb_config['random_state']
        )
        
        # Logistic Regression
        lr_config = self.config['models']['logistic_regression']
        models['logistic_regression'] = LogisticRegression(
            C=lr_config['C'],
            penalty=lr_config['penalty'],
            solver=lr_config['solver'],
            random_state=lr_config['random_state']
        )
        
        # SVM
        svm_config = self.config['models']['svm']
        models['svm'] = SVC(
            C=svm_config['C'],
            kernel=svm_config['kernel'],
            gamma=svm_config['gamma'],
            random_state=svm_config['random_state'],
            probability=True
        )
        
        # LightGBM
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.models = models
        logger.info(f"Initialized {len(models)} models")
        return models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train all models and evaluate on validation set."""
        logger.info("Training models...")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        self.results = results
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate various evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def perform_hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                                    model_name: str = 'random_forest') -> Any:
        """Perform hyperparameter tuning for a specific model."""
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=self.models[model_name],
            param_grid=param_grids[model_name],
            cv=self.config['training']['cv_folds'],
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def select_best_model(self) -> Tuple[str, Any]:
        """Select the best model based on validation performance."""
        logger.info("Selecting best model...")
        
        best_model_name = None
        best_score = 0
        
        for name, result in self.results.items():
            score = result['metrics']['f1_score']  # Using F1 score as selection criterion
            if score > best_score:
                best_score = score
                best_model_name = name
        
        self.best_model = self.results[best_model_name]['model']
        self.best_score = best_score
        
        logger.info(f"Best model: {best_model_name} with F1 score: {best_score:.4f}")
        return best_model_name, self.best_model
    
    def evaluate_on_test_set(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the best model on the test set."""
        logger.info("Evaluating best model on test set...")
        
        if self.best_model is None:
            raise ValueError("No best model selected. Please run select_best_model first.")
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        test_results = {
            'metrics': test_metrics,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Test set results - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        
        return test_results
    
    def save_model(self, model_name: str = "best_model"):
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model to save. Please run select_best_model first.")
        
        model_file = self.model_path / f"{model_name}.joblib"
        joblib.dump(self.best_model, model_file)
        logger.info(f"Model saved to {model_file}")
    
    def save_results(self, test_results: Dict[str, Any]):
        """Save training and test results."""
        logger.info("Saving results...")
        
        # Save validation results
        val_results = {}
        for name, result in self.results.items():
            val_results[name] = result['metrics']
        
        # Combine all results
        all_results = {
            'validation_results': val_results,
            'test_results': test_results,
            'best_model_score': self.best_score
        }
        
        # Save to JSON
        import json
        results_file = self.results_path / "model_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4, default=str)
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(val_results).T
        results_df.to_csv(self.results_path / "validation_results.csv")
        
        logger.info(f"Results saved to {self.results_path}")
    
    def generate_feature_importance(self, X_train: np.ndarray, feature_names: List[str]):
        """Generate and save feature importance for tree-based models."""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(self.results_path / "feature_importance_model.csv", index=False)
            logger.info("Feature importance saved")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting model training pipeline...")
        
        try:
            # Load data
            train, val, test = self.load_data()
            
            # Prepare data
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(train, val, test)
            
            # Initialize models
            self.initialize_models()
            
            # Train models
            self.train_models(X_train, y_train, X_val, y_val)
            
            # Select best model
            best_model_name, best_model = self.select_best_model()
            
            # Evaluate on test set
            test_results = self.evaluate_on_test_set(X_test, y_test)
            
            # Save model and results
            self.save_model()
            self.save_results(test_results)
            
            # Generate feature importance
            feature_names = X_train.columns.tolist()
            self.generate_feature_importance(X_train, feature_names)
            
            logger.info("Model training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise


def main():
    """Main function to run model training."""
    trainer = ModelTrainer()
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()
