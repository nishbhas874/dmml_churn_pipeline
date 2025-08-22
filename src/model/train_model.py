"""
Model Building Demonstration Script
This script demonstrates the complete model building pipeline including training, evaluation, and versioning.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.model_trainer import ModelTrainer
from transformation.feature_engineer import FeatureEngineer
from sklearn.model_selection import train_test_split

def create_sample_data():
    """Create sample customer data for demonstration."""
    np.random.seed(42)
    n_samples = 2000
    
    print("📊 Creating sample customer dataset...")
    
    # Generate base features
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
        'tenure': np.random.exponential(30, n_samples).clip(0, 100).astype(int),
        'monthly_charges': np.random.normal(65, 20, n_samples).clip(20, 150),
        'total_charges': np.random.normal(2000, 1000, n_samples).clip(0, 8000),
        
        # Categorical features
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        
        # Services
        'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        
        # Contract and payment
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'payment_method': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
    })
    
    # Create realistic churn target based on features
    churn_probability = (
        0.3 * (data['contract'] == 'Month-to-month') +
        0.2 * (data['senior_citizen'] == 1) +
        0.15 * (data['monthly_charges'] > 80) +
        0.1 * (data['tenure'] < 12) +
        0.1 * (data['payment_method'] == 'Electronic check') +
        0.05 * (data['partner'] == 'No') +
        0.1 * np.random.random(n_samples)  # Add some randomness
    )
    
    data['churn'] = np.random.binomial(1, churn_probability.clip(0, 1), n_samples)
    
    print(f"✅ Created dataset with {len(data)} samples")
    print(f"   Features: {len(data.columns) - 2}")  # Exclude customer_id and churn
    print(f"   Churn rate: {data['churn'].mean():.2%}")
    
    return data

def main():
    """Main demonstration function."""
    print("🚀 Model Building Pipeline Demonstration")
    print("=" * 60)
    
    # Create sample data
    raw_data = create_sample_data()
    
    # Prepare features and target
    print("\n🔧 Preparing features...")
    
    # Remove customer_id and separate target
    X = raw_data.drop(['customer_id', 'churn'], axis=1)
    y = raw_data['churn']
    
    # Feature engineering
    print("🔧 Applying feature engineering...")
    feature_engineer = FeatureEngineer()
    X_transformed = feature_engineer.transform_data(X)
    
    print(f"✅ Feature engineering completed")
    print(f"   Original features: {len(X.columns)}")
    print(f"   Transformed features: {len(X_transformed.columns)}")
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data split completed")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Training churn rate: {y_train.mean():.2%}")
    print(f"   Testing churn rate: {y_test.mean():.2%}")
    
    # Configure model trainer
    print("\n🤖 Configuring model trainer...")
    
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'scoring': 'f1',
        'models_to_train': [
            'logistic_regression', 
            'random_forest', 
            'gradient_boosting',
            'decision_tree'
        ],
        'hyperparameter_tuning': True,
        'save_models': True,
        'generate_plots': True
    }
    
    trainer = ModelTrainer(config)
    
    # Train models
    print("\n🚀 Training models...")
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Generate performance report
    print("\n📊 Generating performance report...")
    report_path = trainer.generate_performance_report()
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    trainer.generate_visualizations(X_test, y_test)
    
    # Save models
    print("\n💾 Saving models...")
    saved_files = trainer.save_models()
    
    print(f"\n✅ Saved {len(saved_files)} files:")
    for file_path in saved_files[:10]:  # Show first 10 files
        print(f"   • {Path(file_path).name}")
    if len(saved_files) > 10:
        print(f"   ... and {len(saved_files) - 10} more files")
    
    # Display final results
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    if trainer.best_model_name:
        best_result = trainer.results[trainer.best_model_name]
        metrics = best_result['metrics']
        
        print(f"🏆 Best Model: {trainer.best_model_name.replace('_', ' ').title()}")
        print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"   📊 Precision: {metrics['precision']:.4f}")
        print(f"   📊 Recall: {metrics['recall']:.4f}")
        print(f"   📊 F1 Score: {metrics['f1_score']:.4f}")
        print(f"   📊 ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   📊 CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • Performance Report: {Path(report_path).name}")
    print(f"   • Model Files: {len([f for f in saved_files if f.endswith('.pkl')])} models")
    print(f"   • Metadata Files: {len([f for f in saved_files if f.endswith('.json')])} metadata files")
    print(f"   • Visualizations: plots/ directory")
    
    # Model comparison table
    print(f"\n📊 Model Comparison:")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'ROC AUC':<10}")
    print("-" * 50)
    
    for model_name, result in trainer.results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            model_display = model_name.replace('_', ' ').title()[:18]
            print(f"{model_display:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['roc_auc']:<10.4f}")
    
    print(f"\n✅ Model building pipeline completed successfully!")
    print(f"   Total models trained: {len(trainer.models)}")
    print(f"   Best performing model: {trainer.best_model_name.replace('_', ' ').title() if trainer.best_model_name else 'None'}")
    
    # Demonstrate prediction
    if trainer.best_model:
        print(f"\n🔮 Making sample predictions...")
        sample_X = X_test.head(5)
        predictions, probabilities = trainer.predict(sample_X)
        
        print(f"Sample predictions (first 5 test samples):")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities if probabilities is not None else [None]*5)):
            churn_status = "Will Churn" if pred == 1 else "Will Stay"
            prob_text = f" (Probability: {prob:.3f})" if prob is not None else ""
            print(f"   Customer {i+1}: {churn_status}{prob_text}")

if __name__ == "__main__":
    main()
