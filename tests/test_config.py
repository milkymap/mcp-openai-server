"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_openai_server.config import Config, get_config


class TestConfig:
    """Test configuration management."""
    
    def test_config_with_required_fields(self):
        """Test configuration with required fields."""
        config = Config(openai_api_key="test-key")
        
        assert config.openai_api_key == "test-key"
        assert config.default_model == "gpt-4o-mini"
        assert config.default_max_tokens == 1000
        assert config.default_temperature == 0.7
        assert config.server_name == "MCP OpenAI Server"
        assert config.debug is False
    
    def test_config_with_optional_fields(self):
        """Test configuration with optional fields."""
        config = Config(
            openai_api_key="test-key",
            openai_base_url="https://api.example.com/v1",
            openai_organization="org-123",
            default_model="gpt-4",
            default_max_tokens=2000,
            default_temperature=0.5,
            server_name="Custom Server",
            debug=True,
        )
        
        assert config.openai_api_key == "test-key"
        assert config.openai_base_url == "https://api.example.com/v1"
        assert config.openai_organization == "org-123"
        assert config.default_model == "gpt-4"
        assert config.default_max_tokens == 2000
        assert config.default_temperature == 0.5
        assert config.server_name == "Custom Server"
        assert config.debug is True
    
    def test_config_missing_required_field(self):
        """Test configuration with missing required field."""
        with pytest.raises(ValidationError):
            Config()
    
    def test_config_from_env_file(self):
        """Test configuration loading from .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "MCP_OPENAI_OPENAI_API_KEY=env-test-key\n"
                "MCP_OPENAI_DEFAULT_MODEL=gpt-4\n"
                "MCP_OPENAI_DEBUG=true\n"
            )
            
            # Change to the temp directory
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                config = Config(_env_file=".env")
                
                assert config.openai_api_key == "env-test-key"
                assert config.default_model == "gpt-4"
                assert config.debug is True
            finally:
                os.chdir(old_cwd)
    
    def test_config_env_variables(self):
        """Test configuration from environment variables."""
        env_vars = {
            "MCP_OPENAI_OPENAI_API_KEY": "env-var-key",
            "MCP_OPENAI_DEFAULT_MODEL": "gpt-3.5-turbo",
            "MCP_OPENAI_DEFAULT_MAX_TOKENS": "1500",
            "MCP_OPENAI_DEBUG": "true",
        }
        
        # Save original env vars
        original_env = {}
        for key in env_vars:
            original_env[key] = os.environ.get(key)
        
        try:
            # Set test env vars
            for key, value in env_vars.items():
                os.environ[key] = value
            
            config = Config()
            
            assert config.openai_api_key == "env-var-key"
            assert config.default_model == "gpt-3.5-turbo"
            assert config.default_max_tokens == 1500
            assert config.debug is True
            
        finally:
            # Restore original env vars
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    def test_get_config_function(self):
        """Test get_config helper function."""
        # This will fail without API key in env
        os.environ["MCP_OPENAI_OPENAI_API_KEY"] = "test-key"
        try:
            config = get_config()
            assert isinstance(config, Config)
            assert config.openai_api_key == "test-key"
        finally:
            os.environ.pop("MCP_OPENAI_OPENAI_API_KEY", None)