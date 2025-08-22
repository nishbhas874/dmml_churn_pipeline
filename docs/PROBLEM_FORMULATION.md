# Customer Churn Prediction Pipeline - Problem Formulation

## 1. Business Problem Definition

**Problem Statement**: Customer churn is a critical business challenge where customers stop using our services, leading to revenue loss and increased customer acquisition costs.

**Business Challenge**: We need to identify customers who are likely to churn before they actually leave, so we can take proactive retention actions.

## 2. Key Business Objectives

1. **Primary Objective**: Predict which customers are likely to churn in the next 30 days
2. **Secondary Objectives**:
   - Identify key factors that contribute to customer churn
   - Reduce customer churn rate by 15% through targeted interventions
   - Improve customer retention ROI by focusing on high-risk customers
   - Enable proactive customer service and retention campaigns

## 3. Key Data Sources and Attributes

### **Data Source 1: Kaggle - Telco Customer Churn Dataset**
- **Source**: `blastchar/telco-customer-churn`
- **Key Attributes**:
  - `customerID`: Unique customer identifier
  - `tenure`: Number of months customer has stayed
  - `MonthlyCharges`: Monthly charges amount
  - `TotalCharges`: Total charges amount
  - `Contract`: Contract type (Month-to-month, One year, Two year)
  - `PaymentMethod`: How customer pays
  - `Churn`: Target variable (Yes/No)

### **Data Source 2: Hugging Face - Customer Behavior Dataset**
- **Source**: `scikit-learn/adult-census-income`
- **Key Attributes**:
  - Customer demographics
  - Usage patterns
  - Service preferences
  - Historical interactions

## 4. Expected Pipeline Outputs

### **A. Clean Datasets for EDA**
- **Output**: `data/processed/preprocessed_data_YYYYMMDD_HHMMSS.csv`
- **Description**: Cleaned data with handled missing values, standardized formats
- **Usage**: Exploratory data analysis and visualization

### **B. Transformed Features for ML**
- **Output**: `data/transformed/transformed_features_YYYYMMDD_HHMMSS.csv`
- **Description**: Engineered features including:
  - Customer tenure groups
  - Charges per tenure ratios
  - Encoded categorical variables
  - Scaled numerical features

### **C. Deployable Churn Prediction Model**
- **Output**: `models/best_model_YYYYMMDD_HHMMSS.pkl`
- **Description**: Trained machine learning model ready for deployment
- **Algorithms**: Logistic Regression, Random Forest, Gradient Boosting
- **Format**: Pickle file with metadata

## 5. Measurable Evaluation Metrics

### **Primary Metrics**:
- **Accuracy**: Overall prediction correctness (Target: >85%)
- **Precision**: True positive rate for churn prediction (Target: >80%)
- **Recall**: Ability to identify actual churners (Target: >75%)
- **F1-Score**: Balanced precision and recall (Target: >78%)

### **Business Metrics**:
- **Customer Retention Rate**: Percentage of customers retained
- **Churn Reduction**: Decrease in churn rate after model deployment
- **ROI**: Return on investment from retention campaigns

### **Technical Metrics**:
- **Model Training Time**: Time to train the model
- **Prediction Latency**: Time to make predictions
- **Data Pipeline Execution Time**: End-to-end pipeline runtime

## 6. Success Criteria

### **Technical Success**:
- Pipeline runs successfully end-to-end
- Model achieves target performance metrics
- All data quality checks pass
- Automated deployment ready

### **Business Success**:
- Actionable insights for customer retention
- Reduced customer acquisition costs
- Improved customer lifetime value
- Measurable reduction in churn rate

## 7. Project Scope and Constraints

### **In Scope**:
- Data ingestion from Kaggle and Hugging Face
- Complete ML pipeline from data to model
- Automated orchestration and monitoring
- Model versioning and reproducibility

### **Out of Scope**:
- Real-time streaming data
- Production deployment infrastructure
- Customer communication systems
- A/B testing framework

### **Constraints**:
- Limited to publicly available datasets
- Local development environment
- Academic/learning project scope
- No real customer data privacy concerns

This problem formulation provides the foundation for our customer churn prediction pipeline, ensuring clear objectives and measurable outcomes.
