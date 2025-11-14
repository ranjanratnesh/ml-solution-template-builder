"""DataGenerationAgent: Generates synthetic data with real-world characteristics."""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime

from ..core.base_agent import BaseAgent
from ..core.models import DataSpecification, DataQualityIssue, GeneratedData


class DataGenerationAgent(BaseAgent):
    """
    Agent responsible for generating synthetic data based on specifications,
    including realistic relationships and intentional quality issues.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.logger.info("DataGenerationAgent initialized")
        np.random.seed(self.system_config.random_seed)

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic data based on specification.

        Args:
            task: Dictionary containing 'data_specification' (DataSpecification or dict)

        Returns:
            Dictionary with generated data information
        """
        self.logger.info("Starting data generation task")

        # Parse specification
        if isinstance(task.get('data_specification'), DataSpecification):
            spec = task['data_specification']
        else:
            spec = DataSpecification(**task['data_specification'])

        # Generate base features
        df = self._generate_features(spec)

        # Apply relationships
        df = self._apply_relationships(df, spec)

        # Generate target variable
        df = self._generate_target(df, spec)

        # Introduce quality issues
        df, issues_introduced = self._introduce_quality_issues(df, spec)

        # Calculate statistics
        statistics = self._calculate_statistics(df, spec)

        # Save data
        data_path = self._save_data(df, spec)

        # Create result
        generated_data = GeneratedData(
            data_path=data_path,
            num_samples=len(df),
            num_features=len(spec.features),
            target_column=spec.target['name'],
            feature_columns=[f['name'] for f in spec.features],
            quality_issues_introduced=issues_introduced,
            statistics=statistics
        )

        result = {
            "generated_data": generated_data.model_dump(),
            "status": "success"
        }

        self.log_execution(task, result)
        self.logger.info(f"Data generated: {len(df)} samples, {len(spec.features)} features")

        return result

    def _generate_features(self, spec: DataSpecification) -> pd.DataFrame:
        """Generate features based on specification."""
        data = {}

        for feature in spec.features:
            if feature['type'] == 'numerical':
                data[feature['name']] = self._generate_numerical_feature(
                    spec.num_samples, feature
                )
            elif feature['type'] == 'categorical':
                data[feature['name']] = self._generate_categorical_feature(
                    spec.num_samples, feature
                )
            elif feature['type'] == 'binary':
                data[feature['name']] = self._generate_binary_feature(
                    spec.num_samples, feature
                )

        return pd.DataFrame(data)

    def _generate_numerical_feature(self, n_samples: int, feature: Dict[str, Any]) -> np.ndarray:
        """Generate numerical feature based on distribution."""
        dist = feature.get('distribution', 'normal')
        params = feature.get('params', {})

        if dist == 'normal':
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            values = np.random.normal(mean, std, n_samples)

            # Apply bounds if specified
            if 'min' in params:
                values = np.maximum(values, params['min'])
            if 'max' in params:
                values = np.minimum(values, params['max'])

        elif dist == 'uniform':
            low = params.get('min', 0)
            high = params.get('max', 1)
            values = np.random.uniform(low, high, n_samples)

        elif dist == 'exponential':
            scale = params.get('scale', 1.0)
            values = np.random.exponential(scale, n_samples)

            # Apply bounds if specified
            if 'min' in params:
                values = np.maximum(values, params['min'])
            if 'max' in params:
                values = np.minimum(values, params['max'])

        else:
            # Default to normal
            values = np.random.normal(0, 1, n_samples)

        return values

    def _generate_categorical_feature(self, n_samples: int, feature: Dict[str, Any]) -> np.ndarray:
        """Generate categorical feature."""
        params = feature.get('params', {})
        categories = params.get('categories', ['A', 'B', 'C'])
        weights = params.get('weights', None)

        if weights:
            # Normalize weights
            weights = np.array(weights) / np.sum(weights)

        values = np.random.choice(categories, size=n_samples, p=weights)
        return values

    def _generate_binary_feature(self, n_samples: int, feature: Dict[str, Any]) -> np.ndarray:
        """Generate binary feature."""
        params = feature.get('params', {})
        p = params.get('p', 0.5)
        values = np.random.binomial(1, p, n_samples)
        return values

    def _apply_relationships(self, df: pd.DataFrame, spec: DataSpecification) -> pd.DataFrame:
        """Apply relationships between features."""
        for rel in spec.relationships:
            if rel['type'] == 'feature_correlation':
                # Create correlation between features
                features = rel['features']
                if len(features) >= 2 and all(f in df.columns for f in features):
                    corr = rel.get('correlation', 0.7)
                    # Add correlated noise to second feature
                    f1, f2 = features[0], features[1]
                    if pd.api.types.is_numeric_dtype(df[f1]) and pd.api.types.is_numeric_dtype(df[f2]):
                        noise = np.random.normal(0, df[f2].std() * 0.3, len(df))
                        df[f2] = corr * (df[f1] - df[f1].mean()) / df[f1].std() * df[f2].std() + df[f2].mean() + noise

            elif rel['type'] == 'feature_interaction':
                # Create interaction feature
                features = rel['features']
                if len(features) >= 2 and all(f in df.columns for f in features):
                    interaction_name = f"{features[0]}_x_{features[1]}"
                    if pd.api.types.is_numeric_dtype(df[features[0]]) and pd.api.types.is_numeric_dtype(df[features[1]]):
                        if rel.get('interaction_type') == 'polynomial':
                            degree = rel.get('degree', 2)
                            df[interaction_name] = df[features[0]] * df[features[1]] ** degree
                        else:
                            df[interaction_name] = df[features[0]] * df[features[1]]

        return df

    def _generate_target(self, df: pd.DataFrame, spec: DataSpecification) -> pd.DataFrame:
        """Generate target variable based on features and relationships."""
        target_spec = spec.target

        # Find feature_to_target relationship
        target_rel = None
        for rel in spec.relationships:
            if rel['type'] == 'feature_to_target':
                target_rel = rel
                break

        if target_spec['type'] == 'categorical':
            # Classification target
            if target_rel:
                # Use weighted combination of features
                features = target_rel.get('influential_features', [])
                weights = target_rel.get('weights', [1.0] * len(features))

                score = np.zeros(len(df))
                for feat, weight in zip(features, weights):
                    if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                        # Normalize feature
                        normalized = (df[feat] - df[feat].mean()) / (df[feat].std() + 1e-8)
                        score += weight * normalized

                # Apply sigmoid and convert to classes
                prob = 1 / (1 + np.exp(-score))
                classes = target_spec.get('classes', ['class_0', 'class_1'])
                balance = target_spec.get('balance', [0.5] * len(classes))

                if len(classes) == 2:
                    # Binary classification
                    df[target_spec['name']] = [classes[1] if p > (1 - balance[1]) else classes[0] for p in prob]
                else:
                    # Multi-class (simplified)
                    thresholds = np.cumsum(balance)
                    df[target_spec['name']] = [classes[np.searchsorted(thresholds, p)] for p in prob]
            else:
                # Random assignment based on balance
                classes = target_spec.get('classes', ['class_0', 'class_1'])
                balance = target_spec.get('balance', None)
                if balance:
                    balance = np.array(balance) / np.sum(balance)
                df[target_spec['name']] = np.random.choice(classes, size=len(df), p=balance)

        elif target_spec['type'] == 'numerical':
            # Regression target
            if target_rel:
                features = target_rel.get('influential_features', [])
                weights = target_rel.get('weights', [1.0] * len(features))
                noise_std = target_rel.get('noise_std', 1.0)

                target = np.zeros(len(df))
                for feat, weight in zip(features, weights):
                    if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                        target += weight * df[feat]

                # Add noise
                target += np.random.normal(0, noise_std, len(df))
                df[target_spec['name']] = target
            else:
                # Use specified distribution
                dist = target_spec.get('distribution', 'normal')
                params = target_spec.get('params', {})
                mean = params.get('mean', 0)
                std = params.get('std', 1)
                df[target_spec['name']] = np.random.normal(mean, std, len(df))

        return df

    def _introduce_quality_issues(
        self, df: pd.DataFrame, spec: DataSpecification
    ) -> tuple[pd.DataFrame, list]:
        """Introduce realistic quality issues."""
        issues_introduced = []
        issue_pct = spec.issue_percentage

        for issue in spec.quality_issues:
            if issue == DataQualityIssue.MISSING_VALUES:
                # Randomly set some values to NaN
                for col in df.columns:
                    if col != spec.target['name']:  # Don't add missing to target
                        mask = np.random.random(len(df)) < issue_pct
                        df.loc[mask, col] = np.nan
                issues_introduced.append(issue)

            elif issue == DataQualityIssue.OUTLIERS:
                # Add outliers to numerical columns
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col != spec.target['name']:
                        n_outliers = int(len(df) * issue_pct)
                        outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
                        mean = df[col].mean()
                        std = df[col].std()
                        outliers = np.random.choice(
                            [mean + 5 * std, mean - 5 * std],
                            size=n_outliers
                        )
                        df.loc[outlier_indices, col] = outliers
                issues_introduced.append(issue)

            elif issue == DataQualityIssue.DUPLICATES:
                # Duplicate some rows
                n_duplicates = int(len(df) * issue_pct)
                duplicate_indices = np.random.choice(len(df), n_duplicates, replace=True)
                df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
                issues_introduced.append(issue)

            elif issue == DataQualityIssue.NOISE:
                # Add random noise to numerical columns
                for col in df.select_dtypes(include=[np.number]).columns:
                    std = df[col].std()
                    noise = np.random.normal(0, std * 0.1, len(df))
                    df[col] = df[col] + noise
                issues_introduced.append(issue)

        return df, issues_introduced

    def _calculate_statistics(self, df: pd.DataFrame, spec: DataSpecification) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[spec.target['name']].value_counts().to_dict() if spec.target['type'] == 'categorical' else {
                "mean": float(df[spec.target['name']].mean()),
                "std": float(df[spec.target['name']].std()),
                "min": float(df[spec.target['name']].min()),
                "max": float(df[spec.target['name']].max())
            }
        }
        return stats

    def _save_data(self, df: pd.DataFrame, spec: DataSpecification) -> str:
        """Save generated data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_data_{timestamp}.csv"
        filepath = os.path.join(self.system_config.data_dir, filename)

        df.to_csv(filepath, index=False)
        self.logger.info(f"Data saved to {filepath}")

        return filepath
