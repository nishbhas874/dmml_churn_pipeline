#!/usr/bin/env python3
"""
Data Visualization and Summary Statistics - Part of Stage 5
Generate comprehensive visualizations and summary statistics for data preparation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os
import json

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_cleaned_data():
    """Load the most recent cleaned data"""
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        print("No processed data found! Please run data preparation first.")
        return None, None
    
    # Find the most recent cleaned data file
    csv_files = list(processed_dir.glob("cleaned_*_data_*.csv"))
    if not csv_files:
        print("No cleaned data files found!")
        return None, None
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading cleaned data: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df, latest_file.stem

def create_summary_statistics(df, output_dir):
    """Generate comprehensive summary statistics"""
    print("Generating summary statistics...")
    
    # Basic statistics
    summary_stats = {
        "dataset_info": {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().sum()
        },
        "numeric_summary": df.describe().to_dict(),
        "categorical_summary": {}
    }
    
    # Categorical column summaries
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary_stats["categorical_summary"][col] = {
            "unique_values": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict(),
            "missing_count": df[col].isnull().sum()
        }
    
    # Save summary statistics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_statistics_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f"Summary statistics saved: {summary_file}")
    return summary_stats

def create_histograms(df, output_dir):
    """Create histograms for numeric columns"""
    print("Creating histograms...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found for histograms")
        return
    
    # Calculate subplot layout
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Distribution of Numeric Variables', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Create histogram
        ax.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{col} Distribution', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        
        # Add statistics text
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save histogram
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    histogram_file = output_dir / f"histograms_{timestamp}.png"
    plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histograms saved: {histogram_file}")

def create_box_plots(df, output_dir):
    """Create box plots for numeric columns"""
    print("Creating box plots...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("No numeric columns found for box plots")
        return
    
    # Calculate subplot layout
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Box Plots of Numeric Variables', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Create box plot
        df.boxplot(column=col, ax=ax, color='lightgreen')
        ax.set_title(f'{col} Box Plot', fontweight='bold')
        ax.set_ylabel(col)
        
        # Add statistics text
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        ax.text(0.02, 0.98, f'Q1: {q1:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save box plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    boxplot_file = output_dir / f"box_plots_{timestamp}.png"
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved: {boxplot_file}")

def create_correlation_heatmap(df, output_dir):
    """Create correlation heatmap for numeric columns"""
    print("Creating correlation heatmap...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns for correlation heatmap")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numeric Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save correlation heatmap
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_file = output_dir / f"correlation_heatmap_{timestamp}.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved: {heatmap_file}")

def create_categorical_analysis(df, output_dir):
    """Create visualizations for categorical columns"""
    print("Creating categorical analysis...")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        print("No categorical columns found")
        return
    
    # Select top 6 categorical columns for visualization
    top_categorical = categorical_cols[:6]
    
    # Calculate subplot layout
    n_cols = 2
    n_rows = (len(top_categorical) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Categorical Variable Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(top_categorical):
        ax = axes[idx]
        
        # Create bar plot
        value_counts = df[col].value_counts().head(10)
        value_counts.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title(f'{col} Value Counts', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + max(value_counts.values) * 0.01, str(v), 
                   ha='center', va='bottom', fontweight='bold')
    
    # Hide empty subplots
    for idx in range(len(top_categorical), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save categorical analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    categorical_file = output_dir / f"categorical_analysis_{timestamp}.png"
    plt.savefig(categorical_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Categorical analysis saved: {categorical_file}")

def create_target_analysis(df, output_dir):
    """Create target variable analysis (if Churn column exists)"""
    print("Creating target variable analysis...")
    
    if 'Churn' not in df.columns:
        print("No Churn column found for target analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Target Variable (Churn) Analysis', fontsize=16, fontweight='bold')
    
    # 1. Churn distribution
    churn_counts = df['Churn'].value_counts()
    axes[0, 0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
                   colors=['lightgreen', 'lightcoral'], startangle=90)
    axes[0, 0].set_title('Churn Distribution', fontweight='bold')
    
    # 2. Churn by tenure
    if 'tenure' in df.columns:
        tenure_churn = df.groupby('tenure')['Churn'].mean()
        axes[0, 1].plot(tenure_churn.index, tenure_churn.values, marker='o', linewidth=2)
        axes[0, 1].set_title('Churn Rate by Tenure', fontweight='bold')
        axes[0, 1].set_xlabel('Tenure (months)')
        axes[0, 1].set_ylabel('Churn Rate')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Churn by monthly charges
    if 'MonthlyCharges' in df.columns:
        df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1, 0])
        axes[1, 0].set_title('Monthly Charges by Churn Status', fontweight='bold')
        axes[1, 0].set_xlabel('Churn')
        axes[1, 0].set_ylabel('Monthly Charges')
    
    # 4. Churn by contract type
    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
        contract_churn.plot(kind='bar', ax=axes[1, 1], color='lightblue')
        axes[1, 1].set_title('Churn Rate by Contract Type', fontweight='bold')
        axes[1, 1].set_xlabel('Contract Type')
        axes[1, 1].set_ylabel('Churn Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save target analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_file = output_dir / f"target_analysis_{timestamp}.png"
    plt.savefig(target_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Target analysis saved: {target_file}")

def main():
    """Main visualization function"""
    print("Data Visualization and Summary Statistics")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, data_name = load_cleaned_data()
    if df is None:
        return
    
    print(f"Loaded dataset: {data_name}")
    print(f"Dataset shape: {df.shape}")
    
    # Generate all visualizations and statistics
    summary_stats = create_summary_statistics(df, output_dir)
    create_histograms(df, output_dir)
    create_box_plots(df, output_dir)
    create_correlation_heatmap(df, output_dir)
    create_categorical_analysis(df, output_dir)
    create_target_analysis(df, output_dir)
    
    print("\n" + "=" * 50)
    print("Visualization and summary statistics completed!")
    print(f"All files saved in: {output_dir}")
    
    # Print key summary statistics
    print(f"\nKey Statistics:")
    print(f"- Total records: {summary_stats['dataset_info']['total_records']}")
    print(f"- Total columns: {summary_stats['dataset_info']['total_columns']}")
    print(f"- Numeric columns: {summary_stats['dataset_info']['numeric_columns']}")
    print(f"- Categorical columns: {summary_stats['dataset_info']['categorical_columns']}")
    print(f"- Missing values: {summary_stats['dataset_info']['missing_values']}")

if __name__ == "__main__":
    main()
