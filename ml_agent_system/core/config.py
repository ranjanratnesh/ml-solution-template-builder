"""Configuration management for the ML Agent System."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM integration."""
    provider: str = Field(default="anthropic", description="LLM provider (anthropic or openai)")
    model: str = Field(default="claude-3-5-sonnet-20241022", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=1)

    def __init__(self, **data):
        super().__init__(**data)
        if self.api_key is None:
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")


class SystemConfig(BaseModel):
    """System-wide configuration."""
    log_level: str = Field(default="INFO")
    output_dir: str = Field(default="./outputs")
    data_dir: str = Field(default="./data")
    reports_dir: str = Field(default="./reports")
    random_seed: int = Field(default=42)

    llm: LLMConfig = Field(default_factory=LLMConfig)

    def __init__(self, **data):
        super().__init__(**data)
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    name: str
    enabled: bool = True
    timeout: int = Field(default=300, description="Timeout in seconds")
    retry_attempts: int = Field(default=3)
    system: SystemConfig = Field(default_factory=SystemConfig)


# Global configuration instance
_global_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get or create the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = SystemConfig()
    return _global_config


def set_config(config: SystemConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
