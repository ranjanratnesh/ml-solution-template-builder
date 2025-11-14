"""
Custom pipeline example - run specific agents independently.

This example shows how to use individual agents and create
a custom workflow.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml_agent_system import (
    MLOrchestrator,
    ResearchAgent,
    DataDesignAgent,
    ProblemType
)


def main():
    """Run custom pipeline with individual agents."""
    print("\n" + "="*80)
    print("ML Agent System - Custom Pipeline Example")
    print("="*80 + "\n")

    # Method 1: Use individual agents
    print("Method 1: Using Individual Agents")
    print("-" * 80)

    # Research agent
    research_agent = ResearchAgent()
    research_result = research_agent.execute({
        "ml_task": {
            "description": "Classify emails as spam or not spam",
            "problem_type": "classification"
        }
    })
    print(f"✓ Research completed: {research_result['research_result']['problem_type']}")

    # Data design agent
    data_design_agent = DataDesignAgent()
    data_spec_result = data_design_agent.execute({
        "problem_type": research_result['research_result']['problem_type'],
        "description": "Email spam classification",
        "num_samples": 1000
    })
    print(f"✓ Data specification created: {len(data_spec_result['data_specification']['features'])} features")

    print("\n" + "-" * 80 + "\n")

    # Method 2: Use orchestrator with custom steps
    print("Method 2: Using Orchestrator with Custom Steps")
    print("-" * 80)

    orchestrator = MLOrchestrator()

    custom_steps = [
        {
            "agent": "research",
            "params": {
                "ml_task": {
                    "description": "Predict customer lifetime value",
                    "problem_type": "regression"
                }
            }
        },
        {
            "agent": "data_design",
            "params": {
                "problem_type": "regression",
                "description": "Customer lifetime value prediction",
                "num_samples": 1500
            }
        }
    ]

    results = orchestrator.run_custom_pipeline(custom_steps)
    print(f"✓ Custom pipeline completed with {len(results)} steps")

    # Check agent status
    print("\n" + "-" * 80)
    print("Agent Status:")
    print("-" * 80)
    status = orchestrator.get_agent_status()
    for agent_name, agent_status in status.items():
        print(f"  {agent_name}: {agent_status['executions']} executions")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
