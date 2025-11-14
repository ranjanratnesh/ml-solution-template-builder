"""Data models for inter-agent communication."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    """Types of ML problems."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    UNKNOWN = "unknown"


class DataQualityIssue(str, Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    IMBALANCED = "imbalanced"
    INCONSISTENT_FORMATS = "inconsistent_formats"
    NOISE = "noise"


class MLTask(BaseModel):
    """Represents an ML task to be solved."""
    description: str
    problem_type: Optional[ProblemType] = None
    target_variable: Optional[str] = None
    features: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None


class ResearchResult(BaseModel):
    """Result from ResearchAgent."""
    problem_type: ProblemType
    recommended_algorithms: List[str]
    feature_engineering_suggestions: List[str]
    similar_solutions: List[Dict[str, str]]
    evaluation_metrics: List[str]
    considerations: List[str]


class DataSpecification(BaseModel):
    """Specification for synthetic data generation."""
    num_samples: int = Field(default=1000, ge=100)
    features: List[Dict[str, Any]]
    target: Dict[str, Any]
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    quality_issues: List[DataQualityIssue] = Field(default_factory=list)
    issue_percentage: float = Field(default=0.1, ge=0.0, le=0.5)


class GeneratedData(BaseModel):
    """Container for generated data."""
    data_path: str
    num_samples: int
    num_features: int
    target_column: str
    feature_columns: List[str]
    quality_issues_introduced: List[DataQualityIssue]
    statistics: Dict[str, Any]


class ModelResult(BaseModel):
    """Result from model training."""
    model_name: str
    model_path: str
    metrics: Dict[str, float]
    training_time: float
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Dict[str, Any]
    iteration: int = 1


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report."""
    problem_type: ProblemType
    best_model: str
    best_metrics: Dict[str, float]
    all_results: List[ModelResult]
    recommendations: List[str]
    visualizations: List[str]
    report_path: str
    timestamp: str


class AgentMessage(BaseModel):
    """Message passed between agents."""
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    timestamp: str
