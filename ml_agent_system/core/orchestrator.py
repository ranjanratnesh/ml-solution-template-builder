"""Orchestrator: Coordinates multiple agents for end-to-end ML solution development."""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from .config import get_config
from .models import MLTask, ProblemType
from ..agents.research_agent import ResearchAgent
from ..agents.data_design_agent import DataDesignAgent
from ..agents.data_generation_agent import DataGenerationAgent
from ..agents.model_builder_agent import ModelBuilderAgent
from ..agents.evaluation_agent import EvaluationAgent


class MLOrchestrator:
    """
    Orchestrates multiple specialized agents to automate ML solution development.

    The orchestrator coordinates the following workflow:
    1. ResearchAgent: Analyzes problem and recommends approaches
    2. DataDesignAgent: Designs synthetic data specifications
    3. DataGenerationAgent: Generates realistic synthetic data
    4. ModelBuilderAgent: Trains and optimizes models
    5. EvaluationAgent: Evaluates results and generates reports
    """

    def __init__(self):
        """Initialize the orchestrator and all agents."""
        self.config = get_config()
        self.logger = self._setup_logger()

        # Initialize agents
        self.logger.info("Initializing agents...")
        self.research_agent = ResearchAgent()
        self.data_design_agent = DataDesignAgent()
        self.data_generation_agent = DataGenerationAgent()
        self.model_builder_agent = ModelBuilderAgent()
        self.evaluation_agent = EvaluationAgent()

        self.logger.info("All agents initialized successfully")

        # Store execution context
        self.execution_context = {}

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for orchestrator."""
        logger = logging.getLogger("MLOrchestrator")
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - MLOrchestrator - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_full_pipeline(
        self,
        problem_description: str,
        problem_type: Optional[ProblemType] = None,
        num_samples: int = 1000,
        max_iterations: int = 3,
        custom_algorithms: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Run the complete ML solution development pipeline.

        Args:
            problem_description: Description of the ML problem to solve
            problem_type: Optional problem type (auto-detected if not provided)
            num_samples: Number of synthetic data samples to generate
            max_iterations: Number of model optimization iterations
            custom_algorithms: Optional list of specific algorithms to try

        Returns:
            Dictionary containing full pipeline results including:
            - research_results
            - data_specification
            - generated_data
            - model_results
            - evaluation_report
        """
        self.logger.info("="*80)
        self.logger.info("Starting ML Solution Development Pipeline")
        self.logger.info("="*80)

        start_time = datetime.now()

        try:
            # Step 1: Research
            self.logger.info("\n[1/5] Running ResearchAgent...")
            research_result = self._run_research(problem_description, problem_type)

            # Step 2: Data Design
            self.logger.info("\n[2/5] Running DataDesignAgent...")
            data_spec_result = self._run_data_design(research_result, num_samples)

            # Step 3: Data Generation
            self.logger.info("\n[3/5] Running DataGenerationAgent...")
            data_gen_result = self._run_data_generation(data_spec_result)

            # Step 4: Model Building
            self.logger.info("\n[4/5] Running ModelBuilderAgent...")
            model_result = self._run_model_building(
                research_result, data_gen_result, max_iterations, custom_algorithms
            )

            # Step 5: Evaluation
            self.logger.info("\n[5/5] Running EvaluationAgent...")
            eval_result = self._run_evaluation(research_result, model_result)

            # Compile final results
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            final_results = {
                "status": "success",
                "total_time_seconds": total_time,
                "problem_description": problem_description,
                "research_results": research_result,
                "data_specification": data_spec_result['data_specification'],
                "generated_data": data_gen_result['generated_data'],
                "model_results": model_result['model_results'],
                "best_model": model_result['best_model'],
                "evaluation_report": eval_result['evaluation_report'],
                "timestamp": datetime.now().isoformat()
            }

            self.execution_context = final_results

            self.logger.info("\n" + "="*80)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            self.logger.info(f"Best model: {model_result['best_model'].get('model_name', 'N/A')}")
            self.logger.info(f"Report saved to: {eval_result['evaluation_report']['report_path']}")
            self.logger.info("="*80 + "\n")

            return final_results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _run_research(
        self, problem_description: str, problem_type: Optional[ProblemType]
    ) -> Dict[str, Any]:
        """Run research agent."""
        task = {
            "ml_task": {
                "description": problem_description,
                "problem_type": problem_type.value if problem_type else "unknown"
            }
        }

        result = self.research_agent.execute(task)
        return result['research_result']

    def _run_data_design(
        self, research_result: Dict[str, Any], num_samples: int
    ) -> Dict[str, Any]:
        """Run data design agent."""
        task = {
            "problem_type": research_result['problem_type'],
            "description": "Design synthetic data based on research",
            "num_samples": num_samples
        }

        result = self.data_design_agent.execute(task)
        return result

    def _run_data_generation(self, data_spec_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run data generation agent."""
        task = {
            "data_specification": data_spec_result['data_specification']
        }

        result = self.data_generation_agent.execute(task)
        return result

    def _run_model_building(
        self,
        research_result: Dict[str, Any],
        data_gen_result: Dict[str, Any],
        max_iterations: int,
        custom_algorithms: Optional[list]
    ) -> Dict[str, Any]:
        """Run model builder agent."""
        generated_data = data_gen_result['generated_data']

        task = {
            "data_path": generated_data['data_path'],
            "target_column": generated_data['target_column'],
            "problem_type": research_result['problem_type'],
            "algorithms": custom_algorithms or research_result['recommended_algorithms'],
            "max_iterations": max_iterations
        }

        result = self.model_builder_agent.execute(task)
        return result

    def _run_evaluation(
        self, research_result: Dict[str, Any], model_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run evaluation agent."""
        task = {
            "problem_type": research_result['problem_type'],
            "model_results": model_result['model_results'],
            "best_model": model_result['best_model']
        }

        result = self.evaluation_agent.execute(task)
        return result

    def run_custom_pipeline(self, steps: list) -> Dict[str, Any]:
        """
        Run a custom pipeline with specific steps.

        Args:
            steps: List of step configurations, each containing:
                - agent: Agent name (research, data_design, data_generation, model_builder, evaluation)
                - params: Parameters for the agent

        Returns:
            Dictionary with results from each step
        """
        self.logger.info("Running custom pipeline...")
        results = {}

        for i, step in enumerate(steps, 1):
            agent_name = step['agent']
            params = step.get('params', {})

            self.logger.info(f"Step {i}: Running {agent_name}...")

            if agent_name == 'research':
                result = self.research_agent.execute(params)
            elif agent_name == 'data_design':
                result = self.data_design_agent.execute(params)
            elif agent_name == 'data_generation':
                result = self.data_generation_agent.execute(params)
            elif agent_name == 'model_builder':
                result = self.model_builder_agent.execute(params)
            elif agent_name == 'evaluation':
                result = self.evaluation_agent.execute(params)
            else:
                self.logger.warning(f"Unknown agent: {agent_name}")
                continue

            results[agent_name] = result

        return results

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "research_agent": self.research_agent.get_status(),
            "data_design_agent": self.data_design_agent.get_status(),
            "data_generation_agent": self.data_generation_agent.get_status(),
            "model_builder_agent": self.model_builder_agent.get_status(),
            "evaluation_agent": self.evaluation_agent.get_status()
        }

    def reset_all_agents(self):
        """Reset all agents to initial state."""
        self.logger.info("Resetting all agents...")
        self.research_agent.reset()
        self.data_design_agent.reset()
        self.data_generation_agent.reset()
        self.model_builder_agent.reset()
        self.evaluation_agent.reset()
        self.execution_context = {}
        self.logger.info("All agents reset successfully")
