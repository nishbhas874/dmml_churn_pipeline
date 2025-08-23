# train_models.py - Model Building Script
# Student: [Your Name]
# Course: Data Mining & Machine Learning
# Assignment: Customer Churn Prediction Models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import os
from datetime import datetime

print("*** Customer Churn Model Training ***")
print("=" * 40)
print("We will train different ML models and see which one works best!")
print()

# Step 1: Load our prepared data
print("📊 Loading the cleaned data...")
try:
    # Find the latest cleaned data file
    data_files = [f for f in os.listdir("data/processed/") if f.startswith("cleaned_kaggle_data")]
    latest_file = sorted(data_files)[-1]  # Get the most recent one
    
    df = pd.read_csv(f"data/processed/{latest_file}")
    print(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
    print(f"   File: {latest_file}")
except:
    print("❌ Could not find cleaned data. Please run the pipeline first!")
    exit()

# Step 2: Prepare features and target
print()
print("🔧 Preparing features and target variable...")

# Remove columns we don't need for modeling
columns_to_remove = ['customerID']  # ID columns
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Our target variable is 'Churn' 
target_column = 'Churn'
if target_column not in df.columns:
    print(f"❌ Target column '{target_column}' not found!")
    print(f"Available columns: {list(df.columns)}")
    exit()

# Separate features (X) and target (y)
X = df.drop(target_column, axis=1)
y = df[target_column]

print(f"✅ Features prepared: {len(X.columns)} features")
print(f"✅ Target prepared: {len(y)} samples")
print(f"   Churn rate: {y.mean():.2%}")

# Step 3: Handle categorical variables (simple approach)
print()
print("🔧 Converting categorical variables to numbers...")

# Convert categorical columns to numeric using simple encoding
for column in X.columns:
    if X[column].dtype == 'object':
        # Simple label encoding for categorical variables
        unique_values = X[column].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        X[column] = X[column].map(value_map)
        print(f"   Encoded {column}: {len(unique_values)} categories")

print("✅ All categorical variables converted to numbers")

# Step 4: Split data into training and testing sets
print()
print("📊 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split completed:")
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")
print(f"   Train churn rate: {y_train.mean():.2%}")
print(f"   Test churn rate: {y_test.mean():.2%}")

# Step 5: Train different models
print()
print("🚀 Training different machine learning models...")
print("We'll try 3 different algorithms and see which works best!")

# Create a dictionary to store our models and results
models = {}
results = {}

# Model 1: Logistic Regression
print()
print("1️⃣ Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Calculate metrics
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

models['logistic_regression'] = lr_model
results['logistic_regression'] = {
    'accuracy': lr_accuracy,
    'precision': lr_precision,
    'recall': lr_recall,
    'f1_score': lr_f1
}

print(f"   ✅ Logistic Regression trained!")
print(f"      Accuracy: {lr_accuracy:.3f}")
print(f"      F1-Score: {lr_f1:.3f}")

# Model 2: Random Forest
print()
print("2️⃣ Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

models['random_forest'] = rf_model
results['random_forest'] = {
    'accuracy': rf_accuracy,
    'precision': rf_precision,
    'recall': rf_recall,
    'f1_score': rf_f1
}

print(f"   ✅ Random Forest trained!")
print(f"      Accuracy: {rf_accuracy:.3f}")
print(f"      F1-Score: {rf_f1:.3f}")

# Model 3: Decision Tree
print()
print("3️⃣ Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Calculate metrics
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)

models['decision_tree'] = dt_model
results['decision_tree'] = {
    'accuracy': dt_accuracy,
    'precision': dt_precision,
    'recall': dt_recall,
    'f1_score': dt_f1
}

print(f"   ✅ Decision Tree trained!")
print(f"      Accuracy: {dt_accuracy:.3f}")
print(f"      F1-Score: {dt_f1:.3f}")

# Step 6: Compare all models
print()
print("📊 MODEL PERFORMANCE COMPARISON")
print("=" * 50)
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-" * 60)

for model_name, metrics in results.items():
    model_display = model_name.replace('_', ' ').title()
    print(f"{model_display:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}")

# Find the best model based on F1-score
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_model = models[best_model_name]
best_metrics = results[best_model_name]

print()
print(f"🏆 BEST MODEL: {best_model_name.replace('_', ' ').title()}")
print(f"   📈 Accuracy: {best_metrics['accuracy']:.3f}")
print(f"   📈 Precision: {best_metrics['precision']:.3f}")
print(f"   📈 Recall: {best_metrics['recall']:.3f}")
print(f"   📈 F1-Score: {best_metrics['f1_score']:.3f}")

# Step 7: Save the models with MLflow versioning
print()
print("💾 Saving trained models with MLflow versioning...")
os.makedirs("models", exist_ok=True)

# Simple MLflow setup (student-friendly)
import mlflow
import mlflow.sklearn

mlflow.set_experiment("customer_churn_models")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

saved_models = []
for model_name, model in models.items():
    # Save the model as .pkl file (traditional way)
    model_filename = f"models/{model_name}_{timestamp}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Also save with MLflow (modern versioning)
    with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", len(X.columns))
        
        # Log metrics
        mlflow.log_metric("accuracy", results[model_name]['accuracy'])
        mlflow.log_metric("precision", results[model_name]['precision'])
        mlflow.log_metric("recall", results[model_name]['recall'])
        mlflow.log_metric("f1_score", results[model_name]['f1_score'])
        
        # Log the model
        mlflow.sklearn.log_model(model, f"churn_model_{model_name}")
    
    saved_models.append(model_filename)
    print(f"   ✅ Saved: {model_filename}")
    print(f"   ✅ MLflow: {model_name} versioned")

# Step 8: Create performance report
print()
print("📄 Creating model performance report...")
report_filename = f"model_performance_report_{timestamp}.txt"

with open(report_filename, 'w') as f:
    f.write("CUSTOMER CHURN PREDICTION - MODEL PERFORMANCE REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {len(df)} total samples\n")
    f.write(f"Features: {len(X.columns)}\n")
    f.write(f"Churn rate: {y.mean():.2%}\n\n")
    
    f.write("MODELS TRAINED:\n")
    f.write("-" * 20 + "\n")
    for model_name in models.keys():
        f.write(f"• {model_name.replace('_', ' ').title()}\n")
    
    f.write(f"\nPERFORMANCE COMPARISON:\n")
    f.write("-" * 25 + "\n")
    f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
    f.write("-" * 60 + "\n")
    
    for model_name, metrics in results.items():
        model_display = model_name.replace('_', ' ').title()
        f.write(f"{model_display:<20} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}\n")
    
    f.write(f"\nBEST PERFORMING MODEL:\n")
    f.write("-" * 22 + "\n")
    f.write(f"Model: {best_model_name.replace('_', ' ').title()}\n")
    f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
    f.write(f"Precision: {best_metrics['precision']:.4f}\n")
    f.write(f"Recall: {best_metrics['recall']:.4f}\n")
    f.write(f"F1-Score: {best_metrics['f1_score']:.4f}\n")
    
    f.write(f"\nMODEL FILES SAVED:\n")
    f.write("-" * 18 + "\n")
    for model_file in saved_models:
        f.write(f"• {model_file}\n")

print(f"   ✅ Report saved: {report_filename}")

# Step 9: Test prediction on a few samples
print()
print("🔮 Testing prediction on sample data...")
sample_data = X_test.head(3)
sample_predictions = best_model.predict(sample_data)
sample_probabilities = best_model.predict_proba(sample_data)[:, 1]  # Probability of churn

print("Sample predictions:")
for i in range(len(sample_data)):
    churn_status = "Will Churn" if sample_predictions[i] == 1 else "Will Stay"
    confidence = sample_probabilities[i] if sample_predictions[i] == 1 else (1 - sample_probabilities[i])
    print(f"   Customer {i+1}: {churn_status} (Confidence: {confidence:.2%})")

print()
print("🎉 MODEL TRAINING COMPLETED!")
print("=" * 40)
print(f"✅ Trained {len(models)} different models")
print(f"✅ Best model: {best_model_name.replace('_', ' ').title()}")
print(f"✅ Best F1-Score: {best_metrics['f1_score']:.3f}")
print(f"✅ Models saved in: models/ folder")
print(f"✅ Performance report: {report_filename}")
print()
print("Next steps: Use the best model to predict churn for new customers!")
