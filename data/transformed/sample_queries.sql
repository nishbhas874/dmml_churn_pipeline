-- Sample SQL Queries for Churn Prediction Database
-- Generated on: 2025-08-22T13:57:20.181055

-- Get High Risk Customers
SELECT c.customer_id, c.monthly_charges, cp.churn_probability
        FROM customers c
        JOIN churn_predictions cp ON c.customer_id = cp.customer_id
        WHERE cp.churn_probability > 0.7
        ORDER BY cp.churn_probability DESC;

-- Get Customer Features
SELECT c.customer_id, c.tenure, cf.tenure_group, 
               cf.charges_per_tenure, c.monthly_charges
        FROM customers c
        JOIN customer_features cf ON c.customer_id = cf.customer_id
        WHERE c.tenure > 12;

-- Monthly Churn Summary
SELECT 
            DATE(prediction_date) as date,
            COUNT(*) as total_predictions,
            AVG(churn_probability) as avg_churn_prob,
            SUM(churn_prediction) as predicted_churners
        FROM churn_predictions
        GROUP BY DATE(prediction_date)
        ORDER BY date DESC;

-- Feature Statistics
SELECT 
            tenure_group,
            COUNT(*) as customer_count,
            AVG(charges_per_tenure) as avg_charges_ratio,
            AVG(monthly_charges) as avg_monthly_charges
        FROM customer_features cf
        JOIN customers c ON cf.customer_id = c.customer_id
        GROUP BY tenure_group
        ORDER BY customer_count DESC;

