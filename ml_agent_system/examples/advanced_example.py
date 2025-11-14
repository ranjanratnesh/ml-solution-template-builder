"""
Advanced example with custom configuration and algorithms.

This example demonstrates:
- Custom configuration
- Specific algorithm selection
- Extended pipeline settings
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml_agent_system import MLOrchestrator, ProblemType, set_config
from ml_agent_system.core import SystemConfig, LLMConfig


def main():
    """Run advanced example with custom configuration."""
    print("\n" + "="*80)
    print("ML Agent System - Advanced Example")
    print("="*80 + "\n")

    # Create custom configuration
    custom_config = SystemConfig(
        log_level="DEBUG",
        output_dir="./custom_outputs",
        data_dir="./custom_data",
        reports_dir="./custom_reports",
        random_seed=123,
        llm=LLMConfig(
            provider="anthropic",
            temperature=0.8,
            max_tokens=4000
        )
    )

    # Set custom configuration
    set_config(custom_config)
    print("✓ Custom configuration applied")

    # Initialize orchestrator
    orchestrator = MLOrchestrator()

    # Define complex problem
    problem_description = """
    Multi-class classification problem for predicting product categories
    in an e-commerce platform. Products should be classified into one of
    10 categories based on features like: title length, description length,
    price, seller rating, number of reviews, image quality score,
    keywords present, and historical sales volume.
    """

    # Specify custom algorithms to try
    custom_algorithms = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ]

    # Run pipeline with advanced settings
    results = orchestrator.run_full_pipeline(
        problem_description=problem_description,
        problem_type=ProblemType.CLASSIFICATION,
        num_samples=5000,  # Large dataset
        max_iterations=4,  # More iterations
        custom_algorithms=custom_algorithms
    )

    # Detailed results analysis
    if results['status'] == 'success':
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)

        print("\n1. Research Phase:")
        print(f"   - Problem Type: {results['research_results']['problem_type']}")
        print(f"   - Recommended Metrics: {results['research_results']['evaluation_metrics']}")
        print(f"   - Feature Engineering: {results['research_results']['feature_engineering_suggestions'][:2]}")

        print("\n2. Data Generation:")
        print(f"   - Samples: {results['generated_data']['num_samples']}")
        print(f"   - Features: {results['generated_data']['num_features']}")
        print(f"   - Quality Issues: {results['generated_data']['quality_issues_introduced']}")

        print("\n3. Model Training:")
        print(f"   - Models Trained: {len(results['model_results'])}")
        print(f"   - Best Model: {results['best_model']['model_name']}")
        print(f"   - Best Iteration: {results['best_model']['iteration']}")

        print("\n4. Performance Comparison:")
        for model in results['model_results']:
            print(f"   - {model['model_name']} (iter {model['iteration']}): ", end="")
            if 'f1_score' in model['metrics']:
                print(f"F1={model['metrics']['f1_score']:.4f}")
            elif 'r2_score' in model['metrics']:
                print(f"R²={model['metrics']['r2_score']:.4f}")

        print("\n5. Recommendations:")
        for i, rec in enumerate(results['evaluation_report']['recommendations'][:3], 1):
            print(f"   {i}. {rec}")

        print(f"\n6. Final Report: {results['evaluation_report']['report_path']}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
