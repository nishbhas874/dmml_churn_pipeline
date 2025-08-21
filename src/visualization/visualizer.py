"""
Data visualization module for churn prediction pipeline.
Handles plotting and visualization of data, features, and model results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml
from loguru import logger
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DataVisualizer:
    """Data visualizer for churn prediction pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize visualizer with configuration."""
        self.config = self._load_config(config_path)
        self.plots_path = Path(self.config['output']['plots_path'])
        self.results_path = Path(self.config['output']['results_path'])
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        
        # Create plots directory
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        # Set color palette
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def plot_data_distribution(self, data: pd.DataFrame, save_plot: bool = True):
        """Plot distribution of target variable and key features."""
        logger.info("Creating data distribution plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Target distribution
        target_col = self.config['features']['target_column']
        if target_col in data.columns:
            data[target_col].value_counts().plot(kind='bar', ax=axes[0, 0], color=self.colors[:2])
            axes[0, 0].set_title('Target Distribution (Churn)')
            axes[0, 0].set_xlabel('Churn')
            axes[0, 0].set_ylabel('Count')
        
        # Numerical features distribution
        numerical_cols = self.config['features']['numerical_columns']
        for i, col in enumerate(numerical_cols[:3]):
            if col in data.columns:
                row = (i + 1) // 3
                col_idx = (i + 1) % 3
                data[col].hist(ax=axes[row, col_idx], bins=30, color=self.colors[i % len(self.colors)])
                axes[row, col_idx].set_title(f'{col} Distribution')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Frequency')
        
        # Categorical features distribution
        categorical_cols = self.config['features']['categorical_columns'][:3]
        for i, col in enumerate(categorical_cols):
            if col in data.columns:
                row = 1
                col_idx = i
                data[col].value_counts().plot(kind='bar', ax=axes[row, col_idx], color=self.colors[i % len(self.colors)])
                axes[row, col_idx].set_title(f'{col} Distribution')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Count')
                axes[row, col_idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.plots_path / 'data_distribution.png', dpi=300, bbox_inches='tight')
            logger.info("Data distribution plot saved")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, save_plot: bool = True):
        """Plot correlation matrix of numerical features."""
        logger.info("Creating correlation matrix plot...")
        
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.plots_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            logger.info("Correlation matrix plot saved")
        
        plt.show()
    
    def plot_feature_vs_target(self, data: pd.DataFrame, save_plot: bool = True):
        """Plot relationship between features and target variable."""
        logger.info("Creating feature vs target plots...")
        
        target_col = self.config['features']['target_column']
        numerical_cols = self.config['features']['numerical_columns']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature vs Target Analysis', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(numerical_cols[:4]):
            if col in data.columns:
                row = i // 2
                col_idx = i % 2
                
                # Box plot
                data.boxplot(column=col, by=target_col, ax=axes[row, col_idx])
                axes[row, col_idx].set_title(f'{col} vs {target_col}')
                axes[row, col_idx].set_xlabel(target_col)
                axes[row, col_idx].set_ylabel(col)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.plots_path / 'feature_vs_target.png', dpi=300, bbox_inches='tight')
            logger.info("Feature vs target plot saved")
        
        plt.show()
    
    def plot_categorical_analysis(self, data: pd.DataFrame, save_plot: bool = True):
        """Plot analysis of categorical features vs target."""
        logger.info("Creating categorical analysis plots...")
        
        target_col = self.config['features']['target_column']
        categorical_cols = self.config['features']['categorical_columns'][:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Categorical Features vs Target Analysis', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(categorical_cols):
            if col in data.columns:
                row = i // 3
                col_idx = i % 3
                
                # Cross tabulation
                crosstab = pd.crosstab(data[col], data[target_col], normalize='index')
                crosstab.plot(kind='bar', ax=axes[row, col_idx], color=self.colors[:2])
                axes[row, col_idx].set_title(f'{col} vs {target_col}')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Proportion')
                axes[row, col_idx].tick_params(axis='x', rotation=45)
                axes[row, col_idx].legend(title=target_col)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.plots_path / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Categorical analysis plot saved")
        
        plt.show()
    
    def plot_model_performance(self, results: Dict, save_plot: bool = True):
        """Plot model performance comparison."""
        logger.info("Creating model performance plots...")
        
        # Extract metrics
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col_idx = i % 3
            
            values = [results[model]['metrics'][metric] for model in model_names]
            
            bars = axes[row, col_idx].bar(model_names, values, color=self.colors[:len(model_names)])
            axes[row, col_idx].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col_idx].set_ylabel(metric.replace("_", " ").title())
            axes[row, col_idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[row, col_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                      f'{value:.3f}', ha='center', va='bottom')
        
        # Remove the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.plots_path / 'model_performance.png', dpi=300, bbox_inches='tight')
            logger.info("Model performance plot saved")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model", save_plot: bool = True):
        """Plot confusion matrix."""
        logger.info(f"Creating confusion matrix for {model_name}...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_plot:
            plt.savefig(self.plots_path / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix for {model_name} saved")
        
        plt.show()
    
    def plot_roc_curves(self, results: Dict, save_plot: bool = True):
        """Plot ROC curves for all models."""
        logger.info("Creating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for i, (model_name, result) in enumerate(results.items()):
            y_true = result.get('y_true', [])
            y_pred_proba = result.get('probabilities', [])
            
            if len(y_true) > 0 and len(y_pred_proba) > 0:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', 
                        color=self.colors[i % len(self.colors)], linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig(self.plots_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
            logger.info("ROC curves plot saved")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: List[float], 
                              model_name: str = "Model", top_n: int = 20, save_plot: bool = True):
        """Plot feature importance."""
        logger.info(f"Creating feature importance plot for {model_name}...")
        
        # Create dataframe and sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        # Take top N features
        top_features = importance_df.tail(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors[0])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.plots_path / f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot for {model_name} saved")
        
        plt.show()
    
    def create_interactive_dashboard(self, data: pd.DataFrame, results: Dict):
        """Create an interactive dashboard using Plotly."""
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Target Distribution', 'Numerical Features Distribution',
                          'Model Performance', 'Feature Importance'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Target distribution
        target_col = self.config['features']['target_column']
        if target_col in data.columns:
            target_counts = data[target_col].value_counts()
            fig.add_trace(
                go.Pie(labels=target_counts.index, values=target_counts.values, name="Target"),
                row=1, col=1
            )
        
        # Numerical features distribution
        numerical_cols = self.config['features']['numerical_columns']
        if len(numerical_cols) > 0:
            fig.add_trace(
                go.Histogram(x=data[numerical_cols[0]], name=numerical_cols[0]),
                row=1, col=2
            )
        
        # Model performance
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            values = [results[model]['metrics'][metric] for model in model_names]
            fig.add_trace(
                go.Bar(x=model_names, y=values, name=metric.title()),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Churn Prediction Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save dashboard
        fig.write_html(self.plots_path / 'interactive_dashboard.html')
        logger.info("Interactive dashboard saved")
        
        return fig
    
    def generate_all_plots(self, data: pd.DataFrame, results: Dict = None):
        """Generate all visualization plots."""
        logger.info("Generating all visualization plots...")
        
        try:
            # Data distribution plots
            self.plot_data_distribution(data)
            
            # Correlation matrix
            self.plot_correlation_matrix(data)
            
            # Feature vs target analysis
            self.plot_feature_vs_target(data)
            
            # Categorical analysis
            self.plot_categorical_analysis(data)
            
            # Model performance plots (if results provided)
            if results:
                self.plot_model_performance(results)
                self.plot_roc_curves(results)
            
            # Interactive dashboard
            if results:
                self.create_interactive_dashboard(data, results)
            
            logger.info("All visualization plots generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise


def main():
    """Main function to run visualization."""
    visualizer = DataVisualizer()
    
    try:
        # Load data
        data_file = visualizer.processed_data_path / "processed_train.csv"
        if data_file.exists():
            data = pd.read_csv(data_file)
            visualizer.generate_all_plots(data)
        else:
            logger.warning("Processed data file not found. Please run data processing first.")
            
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise


if __name__ == "__main__":
    main()
