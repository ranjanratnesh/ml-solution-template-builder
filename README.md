# ML Agent System - Multi-Agent ML Solution Development

A comprehensive multi-agent system for automating end-to-end ML solution development using Python and LLMs. This system orchestrates multiple specialized agents to handle different aspects of the ML pipeline, from problem analysis to model evaluation.

## Features

### Core Agents

1. **ResearchAgent** - Researches existing ML solutions, identifies problem types, and recommends algorithms
2. **DataDesignAgent** - Designs synthetic data specifications with realistic relationships and industry insights
3. **DataGenerationAgent** - Generates synthetic data with real-world characteristics and intentional data issues
4. **ModelBuilderAgent** - Builds, trains, and iteratively optimizes ML models
5. **EvaluationAgent** - Evaluates results and generates comprehensive reports with visualizations

### Intelligent Capabilities

- **Automatic problem type detection** (classification/regression/clustering)
- **Synthetic data generation** with realistic relationships and data quality issues
- **Iterative model improvement** through LLM feedback
- **Comprehensive evaluation** with visualizations and HTML reports
- **Multi-model comparison** across multiple iterations
- **Feature importance analysis**
- **Actionable recommendations** for improvement

### Extension Points

1. **Add New Agents**: Create specialized agents for specific tasks (e.g., AutoML, feature selection)
2. **Custom Data Generators**: Implement domain-specific data generation logic
3. **Model Libraries**: Integrate with AutoML libraries like AutoGluon or H2O
4. **Real Data Integration**: Connect to actual databases and data sources
5. **Deployment Agents**: Add agents for model deployment and monitoring

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-solution-template-builder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys:
- `ANTHROPIC_API_KEY` - For Claude models (recommended)
- `OPENAI_API_KEY` - For OpenAI models (optional)

## Quick Start

### Basic Usage

```python
from ml_agent_system import MLOrchestrator

# Initialize orchestrator
orchestrator = MLOrchestrator()

# Run complete ML pipeline
results = orchestrator.run_full_pipeline(
    problem_description="Predict customer churn based on usage patterns",
    num_samples=5000,
    max_iterations=3
)

# Access results
print(f"Best Model: {results['best_model']['model_name']}")
print(f"Report: {results['evaluation_report']['report_path']}")
```

### Run Examples

```bash
# Basic classification example
python ml_agent_system/examples/basic_example.py

# Regression with auto-detection
python ml_agent_system/examples/regression_example.py

# Custom pipeline
python ml_agent_system/examples/custom_pipeline_example.py

# Advanced configuration
python ml_agent_system/examples/advanced_example.py
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      MLOrchestrator                         │
│  Coordinates agents for end-to-end ML solution development  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Research   │     │ Data Design  │     │ Data Gen     │
│    Agent     │────▶│    Agent     │────▶│   Agent      │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │Model Builder │
                                          │   Agent      │
                                          └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │ Evaluation   │
                                          │   Agent      │
                                          └──────────────┘
```

### Agent Details

#### 1. ResearchAgent
- Analyzes problem descriptions
- Auto-detects problem type (classification/regression/clustering)
- Recommends suitable algorithms
- Suggests feature engineering techniques
- Identifies evaluation metrics

#### 2. DataDesignAgent
- Designs feature specifications
- Creates realistic data relationships
- Incorporates industry insights
- Plans data quality issues to simulate real-world data

#### 3. DataGenerationAgent
- Generates synthetic datasets based on specifications
- Creates numerical, categorical, and binary features
- Applies correlations and interactions
- Introduces realistic data quality issues:
  - Missing values
  - Outliers
  - Noise
  - Imbalanced classes

#### 4. ModelBuilderAgent
- Trains multiple algorithms in parallel
- Performs iterative optimization
- Adjusts hyperparameters based on LLM feedback
- Tracks feature importance
- Supports: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, etc.

#### 5. EvaluationAgent
- Generates comprehensive HTML reports
- Creates visualizations:
  - Model comparison charts
  - Performance by iteration
  - Feature importance plots
- Provides actionable recommendations
- Compares all models and iterations

## Configuration

### System Configuration

```python
from ml_agent_system import set_config
from ml_agent_system.core import SystemConfig, LLMConfig

config = SystemConfig(
    log_level="INFO",
    output_dir="./outputs",
    data_dir="./data",
    reports_dir="./reports",
    random_seed=42,
    llm=LLMConfig(
        provider="anthropic",  # or "openai"
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=4000
    )
)

set_config(config)
```

### Custom Algorithms

```python
results = orchestrator.run_full_pipeline(
    problem_description="Your problem here",
    custom_algorithms=[
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ]
)
```

## API Reference

### MLOrchestrator

#### `run_full_pipeline()`

Runs the complete ML solution development pipeline.

**Parameters:**
- `problem_description` (str): Description of the ML problem
- `problem_type` (ProblemType, optional): Problem type (auto-detected if None)
- `num_samples` (int): Number of synthetic samples to generate (default: 1000)
- `max_iterations` (int): Number of optimization iterations (default: 3)
- `custom_algorithms` (list, optional): Specific algorithms to use

**Returns:**
- Dictionary with complete pipeline results

#### `run_custom_pipeline()`

Runs a custom pipeline with specific agent steps.

**Parameters:**
- `steps` (list): List of step configurations

**Returns:**
- Dictionary with results from each step

### Individual Agents

Each agent can be used independently:

```python
from ml_agent_system.agents import ResearchAgent, DataDesignAgent

# Use research agent
research_agent = ResearchAgent()
result = research_agent.execute({
    "ml_task": {
        "description": "Your problem",
        "problem_type": "classification"
    }
})

# Use data design agent
data_design_agent = DataDesignAgent()
result = data_design_agent.execute({
    "problem_type": "classification",
    "num_samples": 1000
})
```

## Output Structure

```
project/
├── data/                      # Generated datasets
│   └── synthetic_data_*.csv
├── outputs/                   # Trained models
│   └── model_*.pkl
└── reports/                   # Evaluation reports
    ├── evaluation_report_*.html
    ├── model_comparison_*.png
    ├── performance_by_iteration_*.png
    └── feature_importance_*.png
```

## Examples

### Classification Problem

```python
orchestrator = MLOrchestrator()

results = orchestrator.run_full_pipeline(
    problem_description="""
    Predict whether an email is spam or not based on:
    - Email length
    - Number of links
    - Number of images
    - Sender reputation score
    - Keywords present
    """,
    problem_type=ProblemType.CLASSIFICATION,
    num_samples=3000
)
```

### Regression Problem

```python
results = orchestrator.run_full_pipeline(
    problem_description="""
    Predict house prices based on:
    - Square footage
    - Number of bedrooms
    - Location quality
    - Age of house
    - Recent renovations
    """,
    problem_type=ProblemType.REGRESSION,
    num_samples=2000,
    max_iterations=4
)
```

### Clustering Problem

```python
results = orchestrator.run_full_pipeline(
    problem_description="""
    Segment customers based on:
    - Purchase frequency
    - Average order value
    - Product categories
    - Time since last purchase
    """,
    problem_type=ProblemType.CLUSTERING,
    num_samples=5000
)
```

## Extension Guide

### Adding a New Agent

1. Create agent class inheriting from `BaseAgent`:

```python
from ml_agent_system.core import BaseAgent

class CustomAgent(BaseAgent):
    def execute(self, task):
        # Your implementation
        return {"status": "success", "result": ...}
```

2. Register in orchestrator:

```python
# In orchestrator.py
self.custom_agent = CustomAgent()
```

### Custom Data Generators

Extend `DataGenerationAgent`:

```python
class CustomDataGenerator(DataGenerationAgent):
    def _generate_custom_feature(self, n_samples, params):
        # Your custom logic
        return custom_values
```

### Integrating AutoML Libraries

Example with AutoGluon:

```python
from autogluon.tabular import TabularPredictor

class AutoGluonAgent(BaseAgent):
    def execute(self, task):
        predictor = TabularPredictor(label=task['target'])
        predictor.fit(task['data'])
        return {"model": predictor, "leaderboard": predictor.leaderboard()}
```

## Troubleshooting

### Common Issues

1. **LLM API Errors**
   - Verify API keys in `.env`
   - Check API rate limits
   - Ensure sufficient credits

2. **Memory Issues**
   - Reduce `num_samples`
   - Decrease `max_iterations`
   - Use smaller models

3. **Import Errors**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version (3.8+)

### Logging

Enable debug logging:

```python
config = SystemConfig(log_level="DEBUG")
set_config(config)
```

## Performance Tips

1. **Parallel Execution**: Run multiple experiments in parallel
2. **Caching**: Results are logged in agent history
3. **Incremental Development**: Use `run_custom_pipeline()` for specific steps
4. **Resource Management**: Adjust `num_samples` and `max_iterations` based on resources

## Contributing

Contributions are welcome! Areas for improvement:

- Additional agent types (AutoML, deployment, monitoring)
- More data generation patterns
- Advanced feature engineering
- Real data source integrations
- Additional ML libraries support
- Enhanced visualizations

## License

MIT License - See LICENSE file for details

## Citation

If you use this system in your research or projects, please cite:

```
@software{ml_agent_system,
  title={ML Agent System: Multi-Agent ML Solution Development},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ml-solution-template-builder}
}
```

## Contact

For questions, issues, or suggestions:
- GitHub Issues: [Project Issues](https://github.com/yourusername/ml-solution-template-builder/issues)
- Email: your.email@example.com

## Acknowledgments

- Built with Anthropic Claude and OpenAI APIs
- Inspired by AutoML and agent-based systems research
- Uses scikit-learn, XGBoost, LightGBM, and other ML libraries

---

**Happy ML Development! 🚀**
