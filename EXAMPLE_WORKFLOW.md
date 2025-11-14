# Complete Workflow Example

This document shows a complete end-to-end workflow using the ML Agent System.

## Scenario: Customer Churn Prediction

You're tasked with building an ML model to predict customer churn for a subscription service.

### Step 1: Define the Problem

```python
from ml_agent_system import MLOrchestrator, ProblemType

# Initialize the orchestrator
orchestrator = MLOrchestrator()

# Define your business problem
problem_description = """
Predict whether a customer will cancel their subscription (churn) based on:
- Account age in months
- Monthly usage hours
- Number of support tickets opened
- Last payment amount
- Contract type (monthly, yearly)
- Customer satisfaction score
- Number of feature uses
- Days since last login
"""
```

### Step 2: Run the Complete Pipeline

```python
# Let the system handle everything
results = orchestrator.run_full_pipeline(
    problem_description=problem_description,
    problem_type=ProblemType.CLASSIFICATION,  # Binary classification
    num_samples=3000,  # Generate 3000 synthetic samples
    max_iterations=3    # Run 3 optimization iterations
)
```

### Step 3: What Happens Behind the Scenes

**Phase 1: Research (ResearchAgent)**
```
🔍 Analyzing problem...
✓ Problem type identified: CLASSIFICATION
✓ Recommended algorithms: Random Forest, XGBoost, LightGBM, Logistic Regression
✓ Suggested metrics: accuracy, precision, recall, f1_score, roc_auc
✓ Feature engineering ideas: interaction features, polynomial features
```

**Phase 2: Data Design (DataDesignAgent)**
```
📐 Designing data specification...
✓ Created 8 feature specifications
✓ Designed feature relationships and correlations
✓ Planned data quality issues: missing values, outliers, noise
✓ Target distribution: 70% no-churn, 30% churn (realistic imbalance)
```

**Phase 3: Data Generation (DataGenerationAgent)**
```
🔨 Generating synthetic dataset...
✓ Generated 3000 samples
✓ Applied feature correlations
✓ Introduced 15% missing values
✓ Added outliers to numerical features
✓ Saved to: data/synthetic_data_20241114_102030.csv
```

**Phase 4: Model Training (ModelBuilderAgent)**
```
🤖 Training models...

Iteration 1:
  ✓ Logistic Regression - F1: 0.7234
  ✓ Random Forest       - F1: 0.8156
  ✓ XGBoost            - F1: 0.8342
  ✓ LightGBM           - F1: 0.8298

Iteration 2 (with optimization):
  ✓ Logistic Regression - F1: 0.7456
  ✓ Random Forest       - F1: 0.8389
  ✓ XGBoost            - F1: 0.8567
  ✓ LightGBM           - F1: 0.8523

Iteration 3 (further tuning):
  ✓ Logistic Regression - F1: 0.7512
  ✓ Random Forest       - F1: 0.8423
  ✓ XGBoost            - F1: 0.8634 ⭐ BEST
  ✓ LightGBM           - F1: 0.8589

Best Model: XGBoost (Iteration 3)
```

**Phase 5: Evaluation (EvaluationAgent)**
```
📊 Generating evaluation report...
✓ Created model comparison chart
✓ Plotted performance by iteration
✓ Generated feature importance visualization
✓ LLM-generated recommendations
✓ HTML report: reports/evaluation_report_20241114_102530.html
```

### Step 4: Access Your Results

```python
# Best model information
print(f"Best Model: {results['best_model']['model_name']}")
# Output: Best Model: XGBoost

print(f"Metrics: {results['best_model']['metrics']}")
# Output: {'accuracy': 0.8634, 'precision': 0.8521, 'recall': 0.8234, ...}

# Generated data
print(f"Dataset: {results['generated_data']['data_path']}")
# Output: data/synthetic_data_20241114_102030.csv

# Trained model
print(f"Model saved: {results['best_model']['model_path']}")
# Output: outputs/model_XGBoost_3_20241114_102430.pkl

# Evaluation report
print(f"Report: {results['evaluation_report']['report_path']}")
# Output: reports/evaluation_report_20241114_102530.html
```

### Step 5: View the HTML Report

Open the HTML report in your browser:

```bash
# On Linux/Mac
xdg-open reports/evaluation_report_20241114_102530.html

# On Windows
start reports/evaluation_report_20241114_102530.html

# On Mac
open reports/evaluation_report_20241114_102530.html
```

**Report Contents:**
- 📊 Model performance comparison bar chart
- 📈 Performance improvement across iterations
- 🎯 Feature importance ranking
- 📋 Detailed metrics table for all models
- 💡 5-7 actionable recommendations

### Step 6: Review Recommendations

```python
for i, rec in enumerate(results['evaluation_report']['recommendations'], 1):
    print(f"{i}. {rec}")
```

**Example Output:**
```
1. Consider collecting more training data for underrepresented classes
2. Try ensemble methods combining XGBoost and LightGBM
3. Implement feature engineering: create interaction between usage_hours and account_age
4. Perform hyperparameter tuning using grid search or Bayesian optimization
5. Validate model on real customer data before production deployment
6. Monitor for data drift in production, especially in seasonal patterns
7. Consider cost-sensitive learning given business impact of false negatives
```

### Step 7: Use Your Model (Example)

```python
import joblib
import pandas as pd

# Load the best model
model_data = joblib.load(results['best_model']['model_path'])
model = model_data['model']
preprocessors = model_data['preprocessors']

# Prepare new customer data
new_customer = pd.DataFrame({
    'account_age_months': [24],
    'monthly_usage_hours': [15],
    'support_tickets': [3],
    'last_payment': [49.99],
    'contract_type': ['monthly'],
    'satisfaction_score': [6],
    'feature_uses': [12],
    'days_since_login': [5]
})

# Preprocess (encode, scale, etc.)
# ... apply same preprocessing as training ...

# Make prediction
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print(f"Churn prediction: {prediction[0]}")
print(f"Churn probability: {probability[0][1]:.2%}")
```

## Advanced: Custom Algorithms

```python
# Specify which algorithms to try
results = orchestrator.run_full_pipeline(
    problem_description=problem_description,
    custom_algorithms=[
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Logistic Regression"
    ],
    num_samples=5000,
    max_iterations=4
)
```

## Advanced: Individual Agent Usage

```python
from ml_agent_system.agents import ResearchAgent, DataDesignAgent

# Step 1: Just do research
research_agent = ResearchAgent()
research_result = research_agent.execute({
    "ml_task": {
        "description": problem_description,
        "problem_type": "classification"
    }
})

print(research_result['research_result']['recommended_algorithms'])
# ['Random Forest', 'XGBoost', 'LightGBM', ...]

# Step 2: Design data based on research
data_design_agent = DataDesignAgent()
data_spec = data_design_agent.execute({
    "problem_type": research_result['research_result']['problem_type'],
    "description": problem_description,
    "num_samples": 2000
})

print(f"Features designed: {len(data_spec['data_specification']['features'])}")
# Features designed: 8
```

## Monitoring Agent Status

```python
# Check all agent statuses
status = orchestrator.get_agent_status()

for agent_name, agent_status in status.items():
    print(f"{agent_name}:")
    print(f"  Executions: {agent_status['executions']}")
    print(f"  Last run: {agent_status['last_execution']}")
```

## Production Considerations

### 1. Data Quality
- Review the generated synthetic data
- Ensure it matches your domain characteristics
- Consider supplementing with real data

### 2. Model Validation
- Test on real customer data if available
- Perform cross-validation
- Check for bias in predictions

### 3. Monitoring
- Track model performance over time
- Watch for data drift
- Retrain periodically with new data

### 4. Business Integration
- Define action thresholds (e.g., >70% churn probability)
- Create intervention strategies
- Measure business impact

## Complete Example Script

Save this as `my_churn_project.py`:

```python
from ml_agent_system import MLOrchestrator, ProblemType

def main():
    # Initialize
    orchestrator = MLOrchestrator()

    # Define problem
    problem = """
    Predict customer churn based on usage patterns,
    support history, and account characteristics.
    """

    # Run pipeline
    print("Starting ML pipeline...")
    results = orchestrator.run_full_pipeline(
        problem_description=problem,
        problem_type=ProblemType.CLASSIFICATION,
        num_samples=3000,
        max_iterations=3
    )

    # Display results
    if results['status'] == 'success':
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nBest Model: {results['best_model']['model_name']}")
        print(f"F1 Score: {results['best_model']['metrics'].get('f1_score', 'N/A')}")
        print(f"\nDataset: {results['generated_data']['data_path']}")
        print(f"Model: {results['best_model']['model_path']}")
        print(f"Report: {results['evaluation_report']['report_path']}")
        print("\nNext steps:")
        print("1. Open the HTML report in your browser")
        print("2. Review the recommendations")
        print("3. Test the model with real data")
        print("="*60 + "\n")
    else:
        print(f"Pipeline failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python my_churn_project.py
```

## Summary

This workflow demonstrates:
1. ✅ Problem definition
2. ✅ Automated pipeline execution
3. ✅ Multi-algorithm comparison
4. ✅ Iterative optimization
5. ✅ Comprehensive reporting
6. ✅ Model deployment preparation

**Total Time**: 2-5 minutes (depending on dataset size)
**Result**: Production-ready model with full evaluation

The ML Agent System handles all the complexity, letting you focus on the business problem!
