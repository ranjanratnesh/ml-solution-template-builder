"""EvaluationAgent: Evaluates models and generates comprehensive reports."""

import os
import json
from typing import Any, Dict, List
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.base_agent import BaseAgent
from ..core.models import ProblemType, ModelResult, EvaluationReport


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for evaluating model results and generating
    comprehensive reports with visualizations and recommendations.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.logger.info("EvaluationAgent initialized")
        sns.set_style("whitegrid")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate models and generate report.

        Args:
            task: Dictionary containing:
                - problem_type: ProblemType
                - model_results: List of ModelResult objects or dicts
                - best_model: Best model information

        Returns:
            Dictionary with evaluation report
        """
        self.logger.info("Starting evaluation task")

        problem_type = task['problem_type']
        if isinstance(problem_type, str):
            problem_type = ProblemType(problem_type)

        # Parse model results
        model_results = []
        for mr in task['model_results']:
            if isinstance(mr, ModelResult):
                model_results.append(mr)
            else:
                model_results.append(ModelResult(**mr))

        best_model_info = task['best_model']

        # Generate visualizations
        visualizations = self._generate_visualizations(model_results, problem_type)

        # Generate recommendations
        recommendations = self._generate_recommendations(model_results, problem_type)

        # Create HTML report
        report_path = self._generate_html_report(
            problem_type, model_results, best_model_info,
            visualizations, recommendations
        )

        # Create evaluation report
        evaluation_report = EvaluationReport(
            problem_type=problem_type,
            best_model=best_model_info.get('model_name', 'Unknown'),
            best_metrics=best_model_info.get('metrics', {}),
            all_results=model_results,
            recommendations=recommendations,
            visualizations=visualizations,
            report_path=report_path,
            timestamp=datetime.now().isoformat()
        )

        result = {
            "evaluation_report": evaluation_report.model_dump(),
            "status": "success"
        }

        self.log_execution(task, result)
        self.logger.info(f"Evaluation completed: Report saved to {report_path}")

        return result

    def _generate_visualizations(
        self, model_results: List[ModelResult], problem_type: ProblemType
    ) -> List[str]:
        """Generate visualization plots."""
        visualizations = []

        try:
            # 1. Model comparison chart
            viz_path = self._plot_model_comparison(model_results, problem_type)
            if viz_path:
                visualizations.append(viz_path)

            # 2. Performance by iteration
            viz_path = self._plot_performance_by_iteration(model_results, problem_type)
            if viz_path:
                visualizations.append(viz_path)

            # 3. Feature importance (if available)
            viz_path = self._plot_feature_importance(model_results)
            if viz_path:
                visualizations.append(viz_path)

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")

        return visualizations

    def _plot_model_comparison(
        self, model_results: List[ModelResult], problem_type: ProblemType
    ) -> str:
        """Plot model comparison chart."""
        if not model_results:
            return None

        # Determine primary metric
        if problem_type == ProblemType.CLASSIFICATION:
            metric = 'f1_score'
            metric_label = 'F1 Score'
        elif problem_type == ProblemType.REGRESSION:
            metric = 'r2_score'
            metric_label = 'R² Score'
        elif problem_type == ProblemType.CLUSTERING:
            metric = 'silhouette_score'
            metric_label = 'Silhouette Score'
        else:
            metric = 'accuracy'
            metric_label = 'Accuracy'

        # Extract data
        models = []
        scores = []
        for result in model_results:
            if metric in result.metrics:
                models.append(f"{result.model_name}\n(iter {result.iteration})")
                scores.append(result.metrics[metric])

        if not models:
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(models)), scores, color='skyblue', edgecolor='navy')

        # Highlight best model
        best_idx = scores.index(max(scores))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('darkorange')

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'Model Performance Comparison - {metric_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.png"
        filepath = os.path.join(self.system_config.reports_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def _plot_performance_by_iteration(
        self, model_results: List[ModelResult], problem_type: ProblemType
    ) -> str:
        """Plot performance improvement across iterations."""
        if not model_results:
            return None

        # Determine primary metric
        if problem_type == ProblemType.CLASSIFICATION:
            metric = 'f1_score'
            metric_label = 'F1 Score'
        elif problem_type == ProblemType.REGRESSION:
            metric = 'r2_score'
            metric_label = 'R² Score'
        else:
            metric = 'accuracy'
            metric_label = 'Accuracy'

        # Group by model name and iteration
        model_iterations = {}
        for result in model_results:
            if metric in result.metrics:
                if result.model_name not in model_iterations:
                    model_iterations[result.model_name] = {}
                model_iterations[result.model_name][result.iteration] = result.metrics[metric]

        if not model_iterations:
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for model_name, iterations in model_iterations.items():
            iters = sorted(iterations.keys())
            scores = [iterations[i] for i in iters]
            ax.plot(iters, scores, marker='o', label=model_name, linewidth=2)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title('Model Performance by Iteration', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_by_iteration_{timestamp}.png"
        filepath = os.path.join(self.system_config.reports_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def _plot_feature_importance(self, model_results: List[ModelResult]) -> str:
        """Plot feature importance from best model."""
        # Find result with feature importance
        best_with_importance = None
        for result in model_results:
            if result.feature_importance:
                if best_with_importance is None or \
                   len(result.feature_importance) > len(best_with_importance.feature_importance):
                    best_with_importance = result

        if not best_with_importance:
            return None

        # Extract feature importance
        features = list(best_with_importance.feature_importance.keys())
        importances = list(best_with_importance.feature_importance.values())

        # Sort by importance
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        features = [features[i] for i in sorted_idx[:15]]  # Top 15 features
        importances = [importances[i] for i in sorted_idx[:15]]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(features)), importances, color='lightcoral', edgecolor='darkred')

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {best_with_importance.model_name}',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_importance_{timestamp}.png"
        filepath = os.path.join(self.system_config.reports_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def _generate_recommendations(
        self, model_results: List[ModelResult], problem_type: ProblemType
    ) -> List[str]:
        """Generate recommendations using LLM."""
        system_prompt = """You are an ML expert. Analyze model results and provide actionable recommendations
        for improving model performance. Focus on:
        1. Feature engineering opportunities
        2. Hyperparameter tuning suggestions
        3. Data quality improvements
        4. Alternative algorithms to try
        5. Deployment considerations

        Provide 5-7 specific, actionable recommendations as a JSON array of strings."""

        # Prepare results summary
        results_summary = []
        for r in model_results:
            results_summary.append({
                "model": r.model_name,
                "iteration": r.iteration,
                "metrics": r.metrics,
                "training_time": r.training_time
            })

        prompt = f"""Analyze these ML model results for a {problem_type.value} problem:

{json.dumps(results_summary, indent=2)}

Provide 5-7 specific recommendations for improvement. Return ONLY a JSON array of strings, nothing else.
Example: ["Try feature scaling", "Increase model complexity", ...]"""

        try:
            response = self.call_llm(prompt, system_prompt)
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                recommendations = json.loads(response[start_idx:end_idx])
                return recommendations
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")

        # Fallback recommendations
        return [
            "Consider collecting more training data",
            "Experiment with feature engineering techniques",
            "Try ensemble methods for better performance",
            "Perform thorough hyperparameter tuning",
            "Validate model on unseen data before deployment",
            "Monitor for data drift in production",
            "Consider model interpretability requirements"
        ]

    def _generate_html_report(
        self, problem_type: ProblemType, model_results: List[ModelResult],
        best_model_info: Dict[str, Any], visualizations: List[str],
        recommendations: List[str]
    ) -> str:
        """Generate comprehensive HTML report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        h2 {{ color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        h3 {{ color: #764ba2; }}
        .metric {{
            display: inline-block;
            background: #f0f0f0;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .best-model {{
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{ background-color: #f5f5f5; }}
        .recommendation {{
            background: #e8f5e9;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #4caf50;
            border-radius: 4px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 ML Evaluation Report</h1>
        <p>Generated on: {timestamp}</p>
        <p>Problem Type: {problem_type.value.upper()}</p>
    </div>

    <div class="section">
        <h2>📊 Best Model</h2>
        <div class="best-model">
            <h3>{best_model_info.get('model_name', 'N/A')}</h3>
            <p>Iteration: {best_model_info.get('iteration', 'N/A')}</p>
            <div>
"""

        # Add best model metrics
        for metric, value in best_model_info.get('metrics', {}).items():
            html_content += f'                <span class="metric"><strong>{metric}:</strong> {value:.4f}</span>\n'

        html_content += """
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Visualizations</h2>
"""

        # Add visualizations
        for viz_path in visualizations:
            viz_filename = os.path.basename(viz_path)
            html_content += f'        <img src="{viz_filename}" alt="Visualization">\n'

        html_content += """
    </div>

    <div class="section">
        <h2>📋 All Model Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Iteration</th>
                    <th>Metrics</th>
                    <th>Training Time (s)</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add all model results
        for result in model_results:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in result.metrics.items()])
            html_content += f"""
                <tr>
                    <td>{result.model_name}</td>
                    <td>{result.iteration}</td>
                    <td>{metrics_str}</td>
                    <td>{result.training_time:.2f}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>💡 Recommendations</h2>
"""

        # Add recommendations
        for i, rec in enumerate(recommendations, 1):
            html_content += f'        <div class="recommendation">{i}. {rec}</div>\n'

        html_content += """
    </div>

    <div class="footer">
        <p>Generated by ML Agent System</p>
        <p>Multi-Agent ML Solution Template Builder</p>
    </div>
</body>
</html>
"""

        # Save HTML report
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{timestamp_file}.html"
        filepath = os.path.join(self.system_config.reports_dir, filename)

        with open(filepath, 'w') as f:
            f.write(html_content)

        return filepath
