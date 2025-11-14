"""ResearchAgent: Researches ML solutions and recommends algorithms."""

import json
from typing import Any, Dict

from ..core.base_agent import BaseAgent
from ..core.models import MLTask, ProblemType, ResearchResult


class ResearchAgent(BaseAgent):
    """
    Agent responsible for researching ML solutions, identifying problem types,
    and recommending algorithms and approaches.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.logger.info("ResearchAgent initialized")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research on the ML problem.

        Args:
            task: Dictionary containing 'ml_task' (MLTask object or dict)

        Returns:
            Dictionary with research results
        """
        self.logger.info("Starting research task")

        # Parse task
        if isinstance(task.get('ml_task'), MLTask):
            ml_task = task['ml_task']
        else:
            ml_task = MLTask(**task.get('ml_task', task))

        # Identify problem type
        problem_type = self._identify_problem_type(ml_task)

        # Research algorithms
        algorithms = self._recommend_algorithms(problem_type, ml_task)

        # Get feature engineering suggestions
        feature_suggestions = self._suggest_feature_engineering(problem_type, ml_task)

        # Find similar solutions
        similar_solutions = self._find_similar_solutions(problem_type, ml_task)

        # Recommend evaluation metrics
        metrics = self._recommend_metrics(problem_type)

        # Additional considerations
        considerations = self._get_considerations(problem_type, ml_task)

        result = ResearchResult(
            problem_type=problem_type,
            recommended_algorithms=algorithms,
            feature_engineering_suggestions=feature_suggestions,
            similar_solutions=similar_solutions,
            evaluation_metrics=metrics,
            considerations=considerations
        )

        result_dict = {
            "research_result": result.model_dump(),
            "status": "success"
        }

        self.log_execution(task, result_dict)
        self.logger.info(f"Research completed: Problem type = {problem_type}")

        return result_dict

    def _identify_problem_type(self, ml_task: MLTask) -> ProblemType:
        """Identify the ML problem type using LLM."""
        if ml_task.problem_type and ml_task.problem_type != ProblemType.UNKNOWN:
            return ml_task.problem_type

        system_prompt = """You are an ML expert. Analyze the problem description and identify the ML problem type.
        Respond with ONLY one of: classification, regression, clustering, time_series"""

        prompt = f"""Analyze this ML problem and identify its type:

Description: {ml_task.description}
Target Variable: {ml_task.target_variable or 'Not specified'}
Features: {ml_task.features or 'Not specified'}

What type of ML problem is this? Respond with only one word: classification, regression, clustering, or time_series."""

        try:
            response = self.call_llm(prompt, system_prompt).strip().lower()

            # Parse response
            for prob_type in ProblemType:
                if prob_type.value in response:
                    return prob_type

            return ProblemType.UNKNOWN
        except Exception as e:
            self.logger.error(f"Error identifying problem type: {e}")
            return ProblemType.UNKNOWN

    def _recommend_algorithms(self, problem_type: ProblemType, ml_task: MLTask) -> list:
        """Recommend algorithms based on problem type."""
        system_prompt = """You are an ML expert. Recommend the best algorithms for the given problem.
        Provide a JSON list of 3-5 algorithm names."""

        prompt = f"""Recommend the best ML algorithms for this problem:

Problem Type: {problem_type.value}
Description: {ml_task.description}
Constraints: {ml_task.constraints or 'None'}

Provide a JSON array of 3-5 algorithm names most suitable for this problem.
Example format: ["Random Forest", "XGBoost", "Logistic Regression"]

Only return the JSON array, nothing else."""

        try:
            response = self.call_llm(prompt, system_prompt)
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                algorithms = json.loads(response[start_idx:end_idx])
                return algorithms
        except Exception as e:
            self.logger.warning(f"Error parsing LLM response for algorithms: {e}")

        # Fallback to defaults
        return self._get_default_algorithms(problem_type)

    def _get_default_algorithms(self, problem_type: ProblemType) -> list:
        """Get default algorithms for a problem type."""
        defaults = {
            ProblemType.CLASSIFICATION: [
                "Logistic Regression", "Random Forest", "XGBoost",
                "LightGBM", "Support Vector Machine"
            ],
            ProblemType.REGRESSION: [
                "Linear Regression", "Random Forest Regressor",
                "XGBoost Regressor", "LightGBM Regressor", "Ridge Regression"
            ],
            ProblemType.CLUSTERING: [
                "K-Means", "DBSCAN", "Hierarchical Clustering",
                "Gaussian Mixture Models"
            ],
            ProblemType.TIME_SERIES: [
                "ARIMA", "Prophet", "LSTM", "XGBoost for Time Series"
            ]
        }
        return defaults.get(problem_type, ["Random Forest", "XGBoost"])

    def _suggest_feature_engineering(self, problem_type: ProblemType, ml_task: MLTask) -> list:
        """Suggest feature engineering techniques."""
        system_prompt = """You are an ML expert. Suggest feature engineering techniques.
        Provide a JSON list of 3-5 specific suggestions."""

        prompt = f"""Suggest feature engineering techniques for this problem:

Problem Type: {problem_type.value}
Description: {ml_task.description}
Features: {ml_task.features or 'Not specified'}

Provide a JSON array of 3-5 specific feature engineering suggestions.
Example: ["Create polynomial features", "Apply log transformation to skewed features"]

Only return the JSON array, nothing else."""

        try:
            response = self.call_llm(prompt, system_prompt)
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                suggestions = json.loads(response[start_idx:end_idx])
                return suggestions
        except Exception as e:
            self.logger.warning(f"Error parsing feature engineering suggestions: {e}")

        # Fallback defaults
        return [
            "Handle missing values appropriately",
            "Scale/normalize numerical features",
            "Encode categorical variables",
            "Create interaction features",
            "Remove highly correlated features"
        ]

    def _find_similar_solutions(self, problem_type: ProblemType, ml_task: MLTask) -> list:
        """Find similar ML solutions (simulated)."""
        # In a real system, this would query a database or use web search
        return [
            {
                "title": f"Similar {problem_type.value} problem",
                "approach": "Ensemble methods with feature engineering",
                "performance": "90%+ accuracy"
            },
            {
                "title": f"Industry standard {problem_type.value} solution",
                "approach": "Gradient boosting with hyperparameter tuning",
                "performance": "High performance on similar datasets"
            }
        ]

    def _recommend_metrics(self, problem_type: ProblemType) -> list:
        """Recommend evaluation metrics based on problem type."""
        metrics_map = {
            ProblemType.CLASSIFICATION: [
                "accuracy", "precision", "recall", "f1_score",
                "roc_auc", "confusion_matrix"
            ],
            ProblemType.REGRESSION: [
                "mse", "rmse", "mae", "r2_score", "mape"
            ],
            ProblemType.CLUSTERING: [
                "silhouette_score", "calinski_harabasz_score",
                "davies_bouldin_score"
            ],
            ProblemType.TIME_SERIES: [
                "mse", "rmse", "mae", "mape", "smape"
            ]
        }
        return metrics_map.get(problem_type, ["accuracy", "mse"])

    def _get_considerations(self, problem_type: ProblemType, ml_task: MLTask) -> list:
        """Get additional considerations for the problem."""
        considerations = [
            f"This is a {problem_type.value} problem",
            "Consider data quality and preprocessing requirements",
            "Plan for train/test split and cross-validation",
            "Monitor for overfitting during model training"
        ]

        if ml_task.constraints:
            considerations.append(f"Account for constraints: {ml_task.constraints}")

        return considerations
