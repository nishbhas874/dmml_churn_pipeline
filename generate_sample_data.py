#!/usr/bin/env python3
"""
Script to generate sample churn prediction data for testing the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_churn_data(n_samples=1000, random_state=42):
    """
    Generate sample churn prediction data.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Generated churn data
    """
    np.random.seed(random_state)
    
    # Generate customer IDs
    customer_ids = range(1, n_samples + 1)
    
    # Generate demographic features
    gender = np.random.choice(['Male', 'Female'], n_samples)
    age = np.random.normal(45, 15, n_samples).astype(int)
    age = np.clip(age, 18, 85)  # Clip to reasonable age range
    
    # Generate tenure (customer lifetime)
    tenure = np.random.exponential(30, n_samples).astype(int)
    tenure = np.clip(tenure, 1, 72)  # Clip to reasonable tenure range
    
    # Generate contract information
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
    
    # Generate charges
    monthly_charges = np.random.normal(65, 25, n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150)  # Clip to reasonable range
    
    # Calculate total charges based on tenure and monthly charges
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
    total_charges = np.clip(total_charges, 0, 10000)  # Clip to reasonable range
    
    # Generate payment method
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
        n_samples, 
        p=[0.3, 0.2, 0.25, 0.25]
    )
    
    # Generate internet service
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
    
    # Generate additional services (only for customers with internet)
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_samples)
    
    # Generate billing preferences
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples)
    
    # Generate demographic information
    partner = np.random.choice(['Yes', 'No'], n_samples)
    dependents = np.random.choice(['Yes', 'No'], n_samples)
    
    # Generate target variable (churn) with realistic patterns
    churn_prob = np.zeros(n_samples)
    
    # Higher churn probability for month-to-month contracts
    churn_prob += (contract_type == 'Month-to-month') * 0.3
    
    # Higher churn probability for higher monthly charges
    churn_prob += (monthly_charges > 80) * 0.2
    
    # Higher churn probability for shorter tenure
    churn_prob += (tenure < 12) * 0.2
    
    # Higher churn probability for electronic check payment
    churn_prob += (payment_method == 'Electronic check') * 0.1
    
    # Higher churn probability for no additional services
    churn_prob += ((online_security == 'No') & (online_backup == 'No') & 
                   (device_protection == 'No') & (tech_support == 'No')) * 0.1
    
    # Add some randomness
    churn_prob += np.random.normal(0, 0.1, n_samples)
    
    # Clip probabilities to [0, 1]
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate churn labels
    churn = np.random.binomial(1, churn_prob)
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'contract_type': contract_type,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'paperless_billing': paperless_billing,
        'partner': partner,
        'dependents': dependents,
        'churn': churn
    })
    
    return data


def main():
    """Main function to generate and save sample data."""
    parser = argparse.ArgumentParser(description='Generate sample churn prediction data')
    parser.add_argument('--n_samples', type=int, default=1000, 
                       help='Number of samples to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='data/raw/churn_data.csv',
                       help='Output file path (default: data/raw/churn_data.csv)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_samples} samples of churn data...")
    
    # Generate data
    data = generate_churn_data(n_samples=args.n_samples, random_state=args.random_state)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    data.to_csv(output_path, index=False)
    
    print(f"Data saved to {output_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"Churn rate: {data['churn'].mean():.2%}")
    
    # Display basic statistics
    print("\nBasic statistics:")
    print("=" * 50)
    print(data.describe())
    
    print("\nCategorical features:")
    print("=" * 50)
    categorical_cols = ['gender', 'contract_type', 'payment_method', 'internet_service']
    for col in categorical_cols:
        print(f"\n{col}:")
        print(data[col].value_counts())


if __name__ == "__main__":
    main()
