"""ModelBuilderAgent: Builds, trains, and optimizes ML models."""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core.base_agent import BaseAgent
from ..core.models import ProblemType, ModelResult


class ModelBuilderAgent(BaseAgent):
    """
    Agent responsible for building, training, and iteratively optimizing ML models.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.logger.info("ModelBuilderAgent initialized")
        self.preprocessors = {}

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build and train ML models.

        Args:
            task: Dictionary containing:
                - data_path: Path to training data
                - target_column: Name of target column
                - problem_type: ProblemType
                - algorithms: List of algorithms to try
                - max_iterations: Maximum optimization iterations (default: 3)

        Returns:
            Dictionary with model results
        """
        self.logger.info("Starting model building task")

        data_path = task['data_path']
        target_column = task['target_column']
        problem_type = task['problem_type']
        if isinstance(problem_type, str):
            problem_type = ProblemType(problem_type)

        algorithms = task.get('algorithms', self._get_default_algorithms(problem_type))
        max_iterations = task.get('max_iterations', 3)

        # Load and preprocess data
        X_train, X_test, y_train, y_test = self._load_and_preprocess_data(
            data_path, target_column, problem_type
        )

        # Train models iteratively
        all_results = []
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"Training iteration {iteration}/{max_iterations}")

            for algo_name in algorithms:
                self.logger.info(f"Training {algo_name}")

                model_result = self._train_model(
                    algo_name, X_train, X_test, y_train, y_test,
                    problem_type, iteration
                )

                if model_result:
                    all_results.append(model_result)

            # Get feedback from LLM for next iteration
            if iteration < max_iterations:
                self._get_optimization_feedback(all_results, iteration)

        result = {
            "model_results": [r.model_dump() for r in all_results],
            "best_model": self._select_best_model(all_results, problem_type),
            "status": "success"
        }

        self.log_execution(task, result)
        self.logger.info(f"Model building completed: {len(all_results)} models trained")

        return result

    def _load_and_preprocess_data(
        self, data_path: str, target_column: str, problem_type: ProblemType
    ) -> tuple:
        """Load and preprocess data."""
        # Load data
        df = pd.read_csv(data_path)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.preprocessors[f'le_{col}'] = le

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )
        self.preprocessors['imputer'] = imputer

        # Encode target for classification
        if problem_type == ProblemType.CLASSIFICATION:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.preprocessors['le_target'] = le_target

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        self.preprocessors['scaler'] = scaler

        # Split data
        if problem_type != ProblemType.CLUSTERING:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.system_config.random_seed
            )
        else:
            # For clustering, use all data for training
            X_train, X_test = X, X
            y_train, y_test = y, y

        return X_train, X_test, y_train, y_test

    def _get_default_algorithms(self, problem_type: ProblemType) -> List[str]:
        """Get default algorithms for problem type."""
        if problem_type == ProblemType.CLASSIFICATION:
            algos = ["Logistic Regression", "Random Forest"]
            if XGBOOST_AVAILABLE:
                algos.append("XGBoost")
            return algos
        elif problem_type == ProblemType.REGRESSION:
            algos = ["Linear Regression", "Random Forest Regressor"]
            if XGBOOST_AVAILABLE:
                algos.append("XGBoost Regressor")
            return algos
        elif problem_type == ProblemType.CLUSTERING:
            return ["K-Means", "DBSCAN"]
        else:
            return ["Random Forest"]

    def _train_model(
        self, algo_name: str, X_train, X_test, y_train, y_test,
        problem_type: ProblemType, iteration: int
    ) -> ModelResult:
        """Train a single model."""
        start_time = time.time()

        try:
            # Get model instance
            model, hyperparameters = self._get_model_instance(algo_name, problem_type, iteration)

            # Train model
            if problem_type == ProblemType.CLUSTERING:
                model.fit(X_train)
                y_pred = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            training_time = time.time() - start_time

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, problem_type)

            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = {
                    f"feature_{i}": float(imp)
                    for i, imp in enumerate(model.feature_importances_)
                }

            # Save model
            model_path = self._save_model(model, algo_name, iteration)

            result = ModelResult(
                model_name=algo_name,
                model_path=model_path,
                metrics=metrics,
                training_time=training_time,
                feature_importance=feature_importance,
                hyperparameters=hyperparameters,
                iteration=iteration
            )

            return result

        except Exception as e:
            self.logger.error(f"Error training {algo_name}: {e}")
            return None

    def _get_model_instance(self, algo_name: str, problem_type: ProblemType, iteration: int):
        """Get model instance with hyperparameters."""
        # Base hyperparameters
        hyperparameters = {}

        if algo_name == "Logistic Regression":
            C = 1.0 * (1.5 ** (iteration - 1))
            hyperparameters = {"C": C, "max_iter": 1000}
            return LogisticRegression(**hyperparameters), hyperparameters

        elif algo_name == "Linear Regression":
            return LinearRegression(), hyperparameters

        elif algo_name == "Ridge":
            alpha = 1.0 * (1.5 ** (iteration - 1))
            hyperparameters = {"alpha": alpha}
            return Ridge(**hyperparameters), hyperparameters

        elif algo_name == "Random Forest" or algo_name == "Random Forest Classifier":
            n_estimators = 100 + (iteration - 1) * 50
            max_depth = 10 + (iteration - 1) * 5
            hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
            return RandomForestClassifier(**hyperparameters), hyperparameters

        elif algo_name == "Random Forest Regressor":
            n_estimators = 100 + (iteration - 1) * 50
            max_depth = 10 + (iteration - 1) * 5
            hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
            return RandomForestRegressor(**hyperparameters), hyperparameters

        elif algo_name == "SVM" or algo_name == "Support Vector Machine":
            C = 1.0 * (1.5 ** (iteration - 1))
            hyperparameters = {"C": C, "kernel": "rbf"}
            return SVC(**hyperparameters), hyperparameters

        elif algo_name == "SVR":
            C = 1.0 * (1.5 ** (iteration - 1))
            hyperparameters = {"C": C, "kernel": "rbf"}
            return SVR(**hyperparameters), hyperparameters

        elif "XGBoost" in algo_name and XGBOOST_AVAILABLE:
            n_estimators = 100 + (iteration - 1) * 50
            max_depth = 6 + (iteration - 1) * 2
            learning_rate = 0.1
            hyperparameters = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "random_state": 42
            }
            if problem_type == ProblemType.CLASSIFICATION:
                return xgb.XGBClassifier(**hyperparameters), hyperparameters
            else:
                return xgb.XGBRegressor(**hyperparameters), hyperparameters

        elif "LightGBM" in algo_name and LIGHTGBM_AVAILABLE:
            n_estimators = 100 + (iteration - 1) * 50
            max_depth = 6 + (iteration - 1) * 2
            hyperparameters = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42
            }
            if problem_type == ProblemType.CLASSIFICATION:
                return lgb.LGBMClassifier(**hyperparameters), hyperparameters
            else:
                return lgb.LGBMRegressor(**hyperparameters), hyperparameters

        elif algo_name == "K-Means":
            n_clusters = 3 + (iteration - 1)
            hyperparameters = {"n_clusters": n_clusters, "random_state": 42}
            return KMeans(**hyperparameters), hyperparameters

        elif algo_name == "DBSCAN":
            eps = 0.5 + (iteration - 1) * 0.1
            min_samples = 5
            hyperparameters = {"eps": eps, "min_samples": min_samples}
            return DBSCAN(**hyperparameters), hyperparameters

        else:
            # Default to Random Forest
            return RandomForestClassifier(random_state=42), {"n_estimators": 100}

    def _calculate_metrics(self, y_true, y_pred, problem_type: ProblemType) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        try:
            if problem_type == ProblemType.CLASSIFICATION:
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

            elif problem_type == ProblemType.REGRESSION:
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
                metrics['r2_score'] = float(r2_score(y_true, y_pred))

            elif problem_type == ProblemType.CLUSTERING:
                if len(np.unique(y_pred)) > 1:
                    metrics['silhouette_score'] = float(silhouette_score(y_true, y_pred))
                else:
                    metrics['silhouette_score'] = 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def _save_model(self, model, algo_name: str, iteration: int) -> str:
        """Save trained model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{algo_name.replace(' ', '_')}_{iteration}_{timestamp}.pkl"
        filepath = os.path.join(self.system_config.output_dir, filename)

        joblib.dump({
            'model': model,
            'preprocessors': self.preprocessors
        }, filepath)

        return filepath

    def _get_optimization_feedback(self, results: List[ModelResult], iteration: int):
        """Get LLM feedback for next iteration optimization."""
        system_prompt = """You are an ML expert. Analyze model results and provide feedback for optimization."""

        # Prepare results summary
        results_summary = []
        for r in results[-len(self._get_default_algorithms(ProblemType.CLASSIFICATION)):]:  # Last iteration results
            results_summary.append({
                "model": r.model_name,
                "metrics": r.metrics,
                "hyperparameters": r.hyperparameters
            })

        prompt = f"""Analyze these model results from iteration {iteration}:

{json.dumps(results_summary, indent=2)}

Provide brief suggestions for hyperparameter tuning in the next iteration."""

        try:
            feedback = self.call_llm(prompt, system_prompt)
            self.logger.info(f"Optimization feedback: {feedback}")
        except Exception as e:
            self.logger.warning(f"Could not get optimization feedback: {e}")

    def _select_best_model(self, results: List[ModelResult], problem_type: ProblemType) -> Dict[str, Any]:
        """Select the best model based on metrics."""
        if not results:
            return {}

        # Define primary metric for each problem type
        if problem_type == ProblemType.CLASSIFICATION:
            metric = 'f1_score'
            maximize = True
        elif problem_type == ProblemType.REGRESSION:
            metric = 'r2_score'
            maximize = True
        elif problem_type == ProblemType.CLUSTERING:
            metric = 'silhouette_score'
            maximize = True
        else:
            metric = 'accuracy'
            maximize = True

        # Find best model
        best_result = None
        best_score = float('-inf') if maximize else float('inf')

        for result in results:
            if metric in result.metrics:
                score = result.metrics[metric]
                if maximize and score > best_score:
                    best_score = score
                    best_result = result
                elif not maximize and score < best_score:
                    best_score = score
                    best_result = result

        if best_result:
            return {
                "model_name": best_result.model_name,
                "model_path": best_result.model_path,
                "metrics": best_result.metrics,
                "iteration": best_result.iteration
            }
        else:
            return {}
