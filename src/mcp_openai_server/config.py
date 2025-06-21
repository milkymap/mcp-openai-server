"""Configuration management for MCP OpenAI Server."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration settings for the MCP OpenAI Server."""
    
    # OpenAI API Configuration
    openai_api_key: str = Field(
        description="OpenAI API key for chat completions",
        validation_alias="OPENAI_API_KEY"
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        description="Custom OpenAI API base URL (optional)"
    )
    openai_organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID (optional)"
    )
    
    # Default model settings
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Default model for chat completions"
    )
    default_max_tokens: int = Field(
        default=1000,
        description="Default maximum tokens for completions"
    )
    default_temperature: float = Field(
        default=0.7,
        description="Default temperature for completions"
    )
    
    # Server settings
    server_name: str = Field(
        default="MCP OpenAI Server",
        description="Name of the MCP server"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "MCP_OPENAI_",
        "case_sensitive": False,
    }


def get_config() -> Config:
    """Get the configuration instance."""
    return Config()