"""Base agent class for all ML agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import anthropic
import openai

from .config import AgentConfig, get_config


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration. If None, uses default configuration.
        """
        self.config = config or AgentConfig(name=self.__class__.__name__)
        self.system_config = get_config()
        self.logger = self._setup_logger()
        self.execution_history: list = []

        # Initialize LLM client
        self.llm_client = self._initialize_llm()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the agent."""
        logger = logging.getLogger(self.config.name)
        logger.setLevel(getattr(logging, self.system_config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.config.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_llm(self):
        """Initialize the LLM client based on configuration."""
        llm_config = self.system_config.llm

        if llm_config.provider == "anthropic":
            return anthropic.Anthropic(api_key=llm_config.api_key)
        elif llm_config.provider == "openai":
            return openai.OpenAI(api_key=llm_config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            LLM response as string
        """
        llm_config = self.system_config.llm

        try:
            if llm_config.provider == "anthropic":
                messages = [{"role": "user", "content": prompt}]

                kwargs = {
                    "model": llm_config.model,
                    "max_tokens": llm_config.max_tokens,
                    "temperature": llm_config.temperature,
                    "messages": messages
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                response = self.llm_client.messages.create(**kwargs)
                return response.content[0].text

            elif llm_config.provider == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.llm_client.chat.completions.create(
                    model=llm_config.model,
                    messages=messages,
                    temperature=llm_config.temperature,
                    max_tokens=llm_config.max_tokens
                )
                return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task.

        Args:
            task: Task parameters as dictionary

        Returns:
            Execution results as dictionary
        """
        pass

    def log_execution(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Log execution details to history."""
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result
        })

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "executions": len(self.execution_history),
            "last_execution": self.execution_history[-1]["timestamp"] if self.execution_history else None
        }

    def reset(self):
        """Reset agent state."""
        self.execution_history = []
        self.logger.info(f"{self.config.name} reset successfully")
