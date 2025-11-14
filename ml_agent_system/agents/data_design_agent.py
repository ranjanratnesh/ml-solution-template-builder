"""DataDesignAgent: Designs synthetic data specifications with realistic relationships."""

import json
from typing import Any, Dict, List

from ..core.base_agent import BaseAgent
from ..core.models import ProblemType, DataSpecification, DataQualityIssue


class DataDesignAgent(BaseAgent):
    """
    Agent responsible for designing synthetic data specifications with
    realistic relationships and intentional quality issues.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.logger.info("DataDesignAgent initialized")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design data specification for the ML problem.

        Args:
            task: Dictionary containing:
                - problem_type: ProblemType
                - description: Problem description
                - num_samples: Number of samples (optional)

        Returns:
            Dictionary with data specification
        """
        self.logger.info("Starting data design task")

        problem_type = task.get('problem_type')
        if isinstance(problem_type, str):
            problem_type = ProblemType(problem_type)

        description = task.get('description', '')
        num_samples = task.get('num_samples', 1000)

        # Design features
        features = self._design_features(problem_type, description)

        # Design target variable
        target = self._design_target(problem_type, description)

        # Design relationships between features
        relationships = self._design_relationships(problem_type, features, target)

        # Select quality issues to introduce
        quality_issues = self._select_quality_issues(problem_type)

        data_spec = DataSpecification(
            num_samples=num_samples,
            features=features,
            target=target,
            relationships=relationships,
            quality_issues=quality_issues,
            issue_percentage=0.15  # 15% of data will have issues
        )

        result = {
            "data_specification": data_spec.model_dump(),
            "status": "success"
        }

        self.log_execution(task, result)
        self.logger.info(f"Data specification designed: {len(features)} features, {num_samples} samples")

        return result

    def _design_features(self, problem_type: ProblemType, description: str) -> List[Dict[str, Any]]:
        """Design features using LLM to understand domain context."""
        system_prompt = """You are a data science expert. Design realistic features for the ML problem.
        Return a JSON array of feature specifications. Each feature should have:
        - name: feature name
        - type: "numerical", "categorical", or "binary"
        - description: what the feature represents
        - distribution: distribution type (for numerical: "normal", "uniform", "exponential"; for categorical: "uniform")
        - params: distribution parameters

        Example for numerical:
        {"name": "age", "type": "numerical", "description": "Person's age", "distribution": "normal", "params": {"mean": 35, "std": 10, "min": 18, "max": 80}}

        Example for categorical:
        {"name": "category", "type": "categorical", "description": "Product category", "distribution": "uniform", "params": {"categories": ["A", "B", "C"], "weights": [0.5, 0.3, 0.2]}}
        """

        prompt = f"""Design 5-8 realistic features for this ML problem:

Problem Type: {problem_type.value}
Description: {description}

Create features that would be realistic for this domain. Include a mix of numerical and categorical features.
Return ONLY a JSON array of feature specifications, nothing else."""

        try:
            response = self.call_llm(prompt, system_prompt)
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                features = json.loads(response[start_idx:end_idx])
                return features
        except Exception as e:
            self.logger.warning(f"Error parsing LLM response for features: {e}")

        # Fallback to default features
        return self._get_default_features(problem_type)

    def _get_default_features(self, problem_type: ProblemType) -> List[Dict[str, Any]]:
        """Get default feature specifications."""
        default_features = [
            {
                "name": "feature_1",
                "type": "numerical",
                "description": "Numerical feature 1",
                "distribution": "normal",
                "params": {"mean": 50, "std": 15, "min": 0, "max": 100}
            },
            {
                "name": "feature_2",
                "type": "numerical",
                "description": "Numerical feature 2",
                "distribution": "uniform",
                "params": {"min": 0, "max": 1}
            },
            {
                "name": "feature_3",
                "type": "categorical",
                "description": "Categorical feature",
                "distribution": "uniform",
                "params": {
                    "categories": ["A", "B", "C", "D"],
                    "weights": [0.4, 0.3, 0.2, 0.1]
                }
            },
            {
                "name": "feature_4",
                "type": "numerical",
                "description": "Numerical feature 4",
                "distribution": "exponential",
                "params": {"scale": 2.0, "min": 0, "max": 20}
            },
            {
                "name": "feature_5",
                "type": "binary",
                "description": "Binary feature",
                "distribution": "bernoulli",
                "params": {"p": 0.6}
            }
        ]
        return default_features

    def _design_target(self, problem_type: ProblemType, description: str) -> Dict[str, Any]:
        """Design target variable specification."""
        if problem_type == ProblemType.CLASSIFICATION:
            # Binary or multi-class classification
            return {
                "name": "target",
                "type": "categorical",
                "description": "Target class label",
                "classes": ["class_0", "class_1"],  # Binary by default
                "balance": [0.7, 0.3]  # Slightly imbalanced
            }
        elif problem_type == ProblemType.REGRESSION:
            return {
                "name": "target",
                "type": "numerical",
                "description": "Target continuous value",
                "distribution": "normal",
                "params": {"mean": 100, "std": 25}
            }
        elif problem_type == ProblemType.CLUSTERING:
            return {
                "name": "cluster",
                "type": "categorical",
                "description": "True cluster label (for evaluation)",
                "classes": ["cluster_0", "cluster_1", "cluster_2"],
                "balance": [0.33, 0.33, 0.34]
            }
        else:
            return {
                "name": "target",
                "type": "numerical",
                "description": "Target value",
                "distribution": "normal",
                "params": {"mean": 0, "std": 1}
            }

    def _design_relationships(
        self,
        problem_type: ProblemType,
        features: List[Dict[str, Any]],
        target: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Design relationships between features and target."""
        relationships = []

        # Create relationships for target based on features
        if problem_type == ProblemType.CLASSIFICATION:
            relationships.append({
                "type": "feature_to_target",
                "description": "Linear combination with sigmoid for classification",
                "formula": "sigmoid(sum(feature_i * weight_i))",
                "influential_features": [f["name"] for f in features[:3]],
                "weights": [0.8, 0.5, -0.6]
            })
        elif problem_type == ProblemType.REGRESSION:
            relationships.append({
                "type": "feature_to_target",
                "description": "Linear combination with noise for regression",
                "formula": "sum(feature_i * weight_i) + noise",
                "influential_features": [f["name"] for f in features[:3]],
                "weights": [2.5, -1.8, 3.2],
                "noise_std": 10
            })

        # Feature interactions
        if len(features) >= 2:
            relationships.append({
                "type": "feature_interaction",
                "description": "Polynomial interaction between features",
                "features": [features[0]["name"], features[1]["name"]],
                "interaction_type": "polynomial",
                "degree": 2
            })

        # Correlation between features
        if len(features) >= 3:
            relationships.append({
                "type": "feature_correlation",
                "description": "Correlation between numerical features",
                "features": [f["name"] for f in features[:2] if f["type"] == "numerical"],
                "correlation": 0.7
            })

        return relationships

    def _select_quality_issues(self, problem_type: ProblemType) -> List[DataQualityIssue]:
        """Select realistic quality issues to introduce."""
        # Always include these common issues
        issues = [
            DataQualityIssue.MISSING_VALUES,
            DataQualityIssue.OUTLIERS
        ]

        # Add problem-specific issues
        if problem_type == ProblemType.CLASSIFICATION:
            issues.append(DataQualityIssue.IMBALANCED)

        # Add noise for all problems
        issues.append(DataQualityIssue.NOISE)

        return issues
