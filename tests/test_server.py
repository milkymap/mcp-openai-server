"""Tests for the MCP server implementation."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastmcp import Client

from mcp_openai_server.config import Config
from mcp_openai_server.server import MCPOpenAIServer


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return Config(
        openai_api_key="test-api-key",
        default_model="gpt-4o-mini",
        default_max_tokens=1000,
        default_temperature=0.7,
        server_name="Test MCP OpenAI Server",
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.id = "chatcmpl-test-123"
    response.created = 1677652288
    response.model = "gpt-4o-mini"
    response.system_fingerprint = "test-fingerprint"
    
    # Mock choice
    choice = Mock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.message = Mock()
    choice.message.role = "assistant"
    choice.message.content = "Hello! How can I help you today?"
    
    response.choices = [choice]
    
    # Mock usage
    usage = Mock()
    usage.prompt_tokens = 12
    usage.completion_tokens = 10
    usage.total_tokens = 22
    
    response.usage = usage
    
    return response


@pytest.fixture
def mock_models_response():
    """Mock OpenAI models list response."""
    response = Mock()
    
    model1 = Mock()
    model1.id = "gpt-4o-mini"
    model1.object = "model"
    model1.created = 1677610602
    model1.owned_by = "openai"
    
    model2 = Mock()
    model2.id = "gpt-4"
    model2.object = "model"
    model2.created = 1687882411
    model2.owned_by = "openai"
    
    response.data = [model1, model2]
    return response


class TestMCPOpenAIServer:
    """Test MCP OpenAI Server."""
    
    @pytest.fixture
    def server(self, mock_config):
        """Create a server instance for testing."""
        with patch('mcp_openai_server.server.AsyncOpenAI'):
            server = MCPOpenAIServer(mock_config)
            return server
    
    def test_server_initialization(self, mock_config):
        """Test server initialization."""
        with patch('mcp_openai_server.server.AsyncOpenAI') as mock_openai:
            server = MCPOpenAIServer(mock_config)
            
            assert server.config == mock_config
            assert server.mcp.name == "Test MCP OpenAI Server"
            mock_openai.assert_called_once_with(
                api_key="test-api-key",
                base_url=None,
                organization=None,
            )
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, server, mock_openai_response):
        """Test successful chat completion."""
        # Mock the OpenAI client
        server.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )
        
        # Test with the server's mcp instance directly
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "chat_completion",
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "model": "gpt-4o-mini",
                }
            )
            
            # Verify the result structure
            assert "id" in result[0].text
            assert "choices" in result[0].text
            assert "usage" in result[0].text
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_defaults(self, server, mock_openai_response):
        """Test chat completion using default parameters."""
        server.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )
        
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "chat_completion",
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )
            
            # Should use default model and parameters
            server.openai_client.chat.completions.create.assert_called_once()
            call_args = server.openai_client.chat.completions.create.call_args[1]
            assert call_args["model"] == "gpt-4o-mini"  # default model
            assert call_args["max_tokens"] == 1000      # default max_tokens
            assert call_args["temperature"] == 0.7      # default temperature
    
    @pytest.mark.asyncio
    async def test_chat_completion_validation_error(self, server):
        """Test chat completion with invalid input."""
        # Mock the OpenAI client to avoid issues with missing mock
        server.openai_client.chat.completions.create = AsyncMock()
        
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "chat_completion",
                {
                    "messages": [],  # Empty messages should cause validation error
                }
            )
            
            # Should return error response
            result_text = result[0].text
            assert "error" in result_text
            assert "validation_error" in result_text
    
    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, server):
        """Test chat completion with OpenAI API error."""
        # Mock API to raise exception
        server.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "chat_completion",
                {
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            )
            
            # Should return error response
            result_text = result[0].text
            assert "error" in result_text
            assert "api_error" in result_text
            assert "API Error" in result_text
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, server, mock_models_response):
        """Test successful model listing."""
        server.openai_client.models.list = AsyncMock(
            return_value=mock_models_response
        )
        
        async with Client(server.mcp) as client:
            result = await client.call_tool("list_models", {})
            
            result_text = result[0].text
            assert "models" in result_text
            assert "gpt-4o-mini" in result_text
            assert "gpt-4" in result_text
    
    @pytest.mark.asyncio
    async def test_list_models_error(self, server):
        """Test model listing with API error."""
        server.openai_client.models.list = AsyncMock(
            side_effect=Exception("Models API Error")
        )
        
        async with Client(server.mcp) as client:
            result = await client.call_tool("list_models", {})
            
            result_text = result[0].text
            assert "error" in result_text
            assert "Models API Error" in result_text
    
    @pytest.mark.asyncio
    async def test_get_server_info(self, server):
        """Test server info retrieval."""
        async with Client(server.mcp) as client:
            result = await client.call_tool("get_server_info", {})
            
            result_text = result[0].text
            assert "server_name" in result_text
            assert "Test MCP OpenAI Server" in result_text
            assert "capabilities" in result_text
            assert "chat_completion" in result_text
            assert "version" in result_text
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_all_parameters(self, server, mock_openai_response):
        """Test chat completion with all optional parameters."""
        server.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )
        
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "chat_completion",
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello"},
                    ],
                    "model": "gpt-4",
                    "max_tokens": 500,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.2,
                    "stop": ["END", "STOP"],
                }
            )
            
            # Verify all parameters were passed to OpenAI
            server.openai_client.chat.completions.create.assert_called_once()
            call_args = server.openai_client.chat.completions.create.call_args[1]
            
            assert call_args["model"] == "gpt-4"
            assert call_args["max_tokens"] == 500
            assert call_args["temperature"] == 0.5
            assert call_args["top_p"] == 0.9
            assert call_args["frequency_penalty"] == 0.1
            assert call_args["presence_penalty"] == 0.2
            assert call_args["stop"] == ["END", "STOP"]
            assert len(call_args["messages"]) == 2
    
    @pytest.mark.asyncio
    async def test_message_with_name(self, server, mock_openai_response):
        """Test chat completion with message that includes name."""
        server.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )
        
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "chat_completion",
                {
                    "messages": [
                        {
                            "role": "user", 
                            "content": "Hello",
                            "name": "TestUser"
                        }
                    ],
                }
            )
            
            # Verify name was included in the API call
            server.openai_client.chat.completions.create.assert_called_once()
            call_args = server.openai_client.chat.completions.create.call_args[1]
            
            assert call_args["messages"][0]["name"] == "TestUser"