"""
ML Agent System - Multi-Agent ML Solution Development

A comprehensive multi-agent system for automating ML solution development.
Orchestrates specialized agents to handle different aspects of the ML pipeline.

Core Components:
- ResearchAgent: Researches ML solutions and recommends algorithms
- DataDesignAgent: Designs synthetic data specifications
- DataGenerationAgent: Generates realistic synthetic data
- ModelBuilderAgent: Builds and optimizes ML models
- EvaluationAgent: Evaluates results and generates reports
- MLOrchestrator: Coordinates all agents

Example Usage:
    from ml_agent_system import MLOrchestrator

    orchestrator = MLOrchestrator()
    results = orchestrator.run_full_pipeline(
        problem_description="Predict customer churn based on usage patterns",
        num_samples=5000,
        max_iterations=3
    )
"""

__version__ = "1.0.0"

from .core import MLOrchestrator, ProblemType, get_config, set_config
from .agents import (
    ResearchAgent, DataDesignAgent, DataGenerationAgent,
    ModelBuilderAgent, EvaluationAgent
)

__all__ = [
    'MLOrchestrator',
    'ProblemType',
    'get_config',
    'set_config',
    'ResearchAgent',
    'DataDesignAgent',
    'DataGenerationAgent',
    'ModelBuilderAgent',
    'EvaluationAgent'
]
