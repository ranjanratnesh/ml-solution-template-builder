# Quick Start Guide - ML Agent System

Get up and running with the ML Agent System in 5 minutes!

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd ml-solution-template-builder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

## Test the Installation

```bash
python test_system.py
```

You should see: `🎉 All tests passed! System is ready to use.`

## Your First ML Pipeline

Create a file `my_first_pipeline.py`:

```python
from ml_agent_system import MLOrchestrator

# Initialize the orchestrator
orchestrator = MLOrchestrator()

# Run a complete ML pipeline
results = orchestrator.run_full_pipeline(
    problem_description="Predict if a customer will purchase a product based on age, income, and browsing history",
    num_samples=1000,
    max_iterations=2
)

# View results
print(f"Best Model: {results['best_model']['model_name']}")
print(f"Metrics: {results['best_model']['metrics']}")
print(f"Report: {results['evaluation_report']['report_path']}")
```

Run it:
```bash
python my_first_pipeline.py
```

## What Happens?

The system will automatically:

1. **Research** - Analyze your problem and identify it as classification/regression/clustering
2. **Design Data** - Create a realistic synthetic dataset specification
3. **Generate Data** - Build a dataset with real-world characteristics and issues
4. **Train Models** - Try multiple algorithms with iterative optimization
5. **Evaluate** - Generate a comprehensive HTML report with visualizations

## Output Files

After running, check these directories:

- `data/` - Your generated synthetic dataset
- `outputs/` - Trained model files (.pkl)
- `reports/` - HTML evaluation report with charts

Open the HTML report in your browser to see:
- Model performance comparison
- Feature importance analysis
- Recommendations for improvement

## Try the Examples

```bash
# Basic classification
python ml_agent_system/examples/basic_example.py

# Regression with auto-detection
python ml_agent_system/examples/regression_example.py

# Custom pipeline
python ml_agent_system/examples/custom_pipeline_example.py

# Advanced configuration
python ml_agent_system/examples/advanced_example.py
```

## Common Use Cases

### Classification

```python
results = orchestrator.run_full_pipeline(
    problem_description="Classify emails as spam or not spam",
    problem_type=ProblemType.CLASSIFICATION
)
```

### Regression

```python
results = orchestrator.run_full_pipeline(
    problem_description="Predict house prices based on features",
    problem_type=ProblemType.REGRESSION
)
```

### Auto-Detection

```python
# Let the system detect the problem type
results = orchestrator.run_full_pipeline(
    problem_description="Your problem here",
    problem_type=None  # Will auto-detect
)
```

## Customization

### Use Specific Algorithms

```python
results = orchestrator.run_full_pipeline(
    problem_description="Your problem",
    custom_algorithms=["Random Forest", "XGBoost", "LightGBM"]
)
```

### Adjust Dataset Size

```python
results = orchestrator.run_full_pipeline(
    problem_description="Your problem",
    num_samples=5000,  # Generate 5000 samples
    max_iterations=4   # Run 4 optimization iterations
)
```

### Custom Configuration

```python
from ml_agent_system.core import SystemConfig, set_config

config = SystemConfig(
    log_level="DEBUG",
    output_dir="./my_outputs",
    random_seed=123
)
set_config(config)

orchestrator = MLOrchestrator()
# Now uses your custom config
```

## Troubleshooting

### API Key Issues

```python
# Check if API key is loaded
from ml_agent_system import get_config
config = get_config()
print(config.llm.api_key)  # Should not be None
```

### Memory Issues

Reduce dataset size:
```python
results = orchestrator.run_full_pipeline(
    problem_description="Your problem",
    num_samples=500,  # Smaller dataset
    max_iterations=1  # Fewer iterations
)
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [examples](ml_agent_system/examples/) directory
- Check the architecture section to understand how agents work
- Extend the system with custom agents

## Support

- Issues: GitHub Issues
- Documentation: README.md
- Examples: ml_agent_system/examples/

## Quick Reference

```python
from ml_agent_system import MLOrchestrator, ProblemType

# Full pipeline
orchestrator = MLOrchestrator()
results = orchestrator.run_full_pipeline(
    problem_description="Your ML problem",
    problem_type=ProblemType.CLASSIFICATION,  # or REGRESSION, CLUSTERING, None
    num_samples=1000,
    max_iterations=3,
    custom_algorithms=["XGBoost", "Random Forest"]
)

# Individual agents
from ml_agent_system.agents import ResearchAgent
research = ResearchAgent()
result = research.execute({"ml_task": {"description": "Your problem"}})

# Agent status
status = orchestrator.get_agent_status()
```

Happy ML Development! 🚀
