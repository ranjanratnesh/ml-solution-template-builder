"""
Basic example of using the ML Agent System.

This example demonstrates how to run a complete ML pipeline
for a simple classification problem.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml_agent_system import MLOrchestrator, ProblemType


def main():
    """Run a basic classification example."""
    print("\n" + "="*80)
    print("ML Agent System - Basic Example")
    print("="*80 + "\n")

    # Initialize orchestrator
    orchestrator = MLOrchestrator()

    # Define the problem
    problem_description = """
    Predict whether a customer will churn based on their behavior.
    Features include: usage frequency, support tickets, contract length, and monthly charges.
    """

    # Run the full pipeline
    results = orchestrator.run_full_pipeline(
        problem_description=problem_description,
        problem_type=ProblemType.CLASSIFICATION,  # Explicitly specify type
        num_samples=2000,  # Generate 2000 samples
        max_iterations=2   # 2 optimization iterations
    )

    # Print summary
    if results['status'] == 'success':
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"\n✓ Problem Type: {results['research_results']['problem_type']}")
        print(f"✓ Best Model: {results['best_model']['model_name']}")
        print(f"✓ Best Metrics: {results['best_model']['metrics']}")
        print(f"✓ Total Time: {results['total_time_seconds']:.2f} seconds")
        print(f"✓ Report: {results['evaluation_report']['report_path']}")
        print("\n" + "="*80 + "\n")
    else:
        print(f"\n✗ Pipeline failed: {results.get('error', 'Unknown error')}\n")


if __name__ == "__main__":
    main()
