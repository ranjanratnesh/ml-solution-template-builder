"""ML Agent System - Core Module

Contains base classes, configuration, and orchestrator.
"""

from .base_agent import BaseAgent
from .config import SystemConfig, LLMConfig, AgentConfig, get_config, set_config
from .models import (
    ProblemType, DataQualityIssue, MLTask, ResearchResult,
    DataSpecification, GeneratedData, ModelResult, EvaluationReport
)
from .orchestrator import MLOrchestrator

__all__ = [
    'BaseAgent',
    'SystemConfig',
    'LLMConfig',
    'AgentConfig',
    'get_config',
    'set_config',
    'ProblemType',
    'DataQualityIssue',
    'MLTask',
    'ResearchResult',
    'DataSpecification',
    'GeneratedData',
    'ModelResult',
    'EvaluationReport',
    'MLOrchestrator'
]
