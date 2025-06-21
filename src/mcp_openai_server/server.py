"""MCP server implementation with OpenAI chat completion capabilities."""

import time
from typing import Dict, List, Optional

from fastmcp import FastMCP
from openai import AsyncOpenAI
from pydantic import ValidationError

from .config import Config, get_config
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionChoice,
    ChatCompletionUsage,
)


class MCPOpenAIServer:
    """MCP server that provides OpenAI chat completion functionality."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the MCP server with configuration."""
        self.config = config or get_config()
        self.mcp = FastMCP(name=self.config.server_name)
        self.openai_client = AsyncOpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            organization=self.config.openai_organization,
        )
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register MCP tools."""
        
        @self.mcp.tool
        async def chat_completion(
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[List[str]] = None,
        ) -> Dict:
            """
            Generate a chat completion using OpenAI's API.
            
            This tool allows other LLMs to request chat completions by providing
            a list of messages and optional parameters. It forwards the request
            to OpenAI's API and returns the response in OpenAI-compatible format.
            
            Args:
                messages: List of message dictionaries with 'role' and 'content' keys
                model: Model to use (defaults to configured default model)
                max_tokens: Maximum tokens to generate (defaults to configured default)
                temperature: Sampling temperature 0-2 (defaults to configured default)
                top_p: Nucleus sampling parameter 0-1
                frequency_penalty: Frequency penalty -2.0 to 2.0
                presence_penalty: Presence penalty -2.0 to 2.0
                stop: List of stop sequences
            
            Returns:
                Dictionary containing the chat completion response in OpenAI format
            """
            try:
                # Convert message dictionaries to ChatMessage objects for validation
                chat_messages = [
                    ChatMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        name=msg.get("name")
                    )
                    for msg in messages
                ]
                
                # Create request with defaults from config
                request = ChatCompletionRequest(
                    messages=chat_messages,
                    model=model or self.config.default_model,
                    max_tokens=max_tokens or self.config.default_max_tokens,
                    temperature=temperature if temperature is not None else self.config.default_temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    stream=False,  # MCP doesn't support streaming
                )
                
                # Convert back to OpenAI format for API call
                openai_messages = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        **({"name": msg.name} if msg.name else {})
                    }
                    for msg in request.messages
                ]
                
                # Build OpenAI API parameters
                api_params = {
                    "model": request.model,
                    "messages": openai_messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }
                
                # Add optional parameters if provided
                if request.top_p is not None:
                    api_params["top_p"] = request.top_p
                if request.frequency_penalty is not None:
                    api_params["frequency_penalty"] = request.frequency_penalty
                if request.presence_penalty is not None:
                    api_params["presence_penalty"] = request.presence_penalty
                if request.stop is not None:
                    api_params["stop"] = request.stop
                
                # Make OpenAI API call
                response = await self.openai_client.chat.completions.create(**api_params)
                
                # Convert OpenAI response to our format
                choices = [
                    ChatCompletionChoice(
                        index=choice.index,
                        message=ChatMessage(
                            role=choice.message.role,
                            content=choice.message.content or "",
                        ),
                        finish_reason=choice.finish_reason,
                    )
                    for choice in response.choices
                ]
                
                usage = ChatCompletionUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                )
                
                completion_response = ChatCompletionResponse(
                    id=response.id,
                    created=response.created,
                    model=response.model,
                    choices=choices,
                    usage=usage,
                    system_fingerprint=response.system_fingerprint,
                )
                
                return completion_response.model_dump()
                
            except ValidationError as e:
                return {
                    "error": {
                        "type": "validation_error",
                        "message": f"Invalid request format: {str(e)}",
                        "details": e.errors()
                    }
                }
            except Exception as e:
                return {
                    "error": {
                        "type": "api_error",
                        "message": f"OpenAI API error: {str(e)}",
                    }
                }
        
        @self.mcp.tool
        async def list_models() -> Dict:
            """
            List available OpenAI models.
            
            Returns a list of available models that can be used with the chat_completion tool.
            
            Returns:
                Dictionary containing the list of available models
            """
            try:
                models = await self.openai_client.models.list()
                return {
                    "models": [
                        {
                            "id": model.id,
                            "object": model.object,
                            "created": model.created,
                            "owned_by": model.owned_by,
                        }
                        for model in models.data
                        if "gpt" in model.id.lower()  # Filter to chat models
                    ]
                }
            except Exception as e:
                return {
                    "error": {
                        "type": "api_error",
                        "message": f"Error listing models: {str(e)}",
                    }
                }
        
        @self.mcp.tool
        async def get_server_info() -> Dict:
            """
            Get information about the MCP OpenAI server.
            
            Returns server configuration and capabilities information.
            
            Returns:
                Dictionary containing server information
            """
            return {
                "server_name": self.config.server_name,
                "default_model": self.config.default_model,
                "default_max_tokens": self.config.default_max_tokens,
                "default_temperature": self.config.default_temperature,
                "capabilities": [
                    "chat_completion",
                    "list_models",
                    "custom_parameters",
                    "multiple_models",
                ],
                "supported_roles": ["system", "user", "assistant"],
                "version": "0.1.0",
            }
    
    def run(self, **kwargs) -> None:
        """Run the MCP server."""
        self.mcp.run(**kwargs)