"""
Regression example using the ML Agent System.

This example demonstrates solving a regression problem
with automatic problem detection.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml_agent_system import MLOrchestrator


def main():
    """Run a regression example."""
    print("\n" + "="*80)
    print("ML Agent System - Regression Example")
    print("="*80 + "\n")

    # Initialize orchestrator
    orchestrator = MLOrchestrator()

    # Define the problem (let the system auto-detect it's regression)
    problem_description = """
    Predict the price of a house based on its features such as:
    square footage, number of bedrooms, number of bathrooms, age of the house,
    location quality score, and neighborhood median income.
    """

    # Run pipeline with automatic problem type detection
    results = orchestrator.run_full_pipeline(
        problem_description=problem_description,
        problem_type=None,  # Auto-detect
        num_samples=3000,
        max_iterations=3
    )

    # Print results
    if results['status'] == 'success':
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"\n✓ Detected Problem Type: {results['research_results']['problem_type']}")
        print(f"✓ Recommended Algorithms: {results['research_results']['recommended_algorithms']}")
        print(f"✓ Best Model: {results['best_model']['model_name']}")
        print(f"✓ Performance Metrics:")
        for metric, value in results['best_model']['metrics'].items():
            print(f"   - {metric}: {value:.4f}")
        print(f"✓ Data Generated: {results['generated_data']['data_path']}")
        print(f"✓ Report: {results['evaluation_report']['report_path']}")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
