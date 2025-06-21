"""Tests for data models."""

import pytest
from pydantic import ValidationError

from mcp_openai_server.models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
)


class TestChatMessage:
    """Test ChatMessage model."""
    
    def test_valid_chat_message(self):
        """Test valid chat message creation."""
        message = ChatMessage(role="user", content="Hello, world!")
        
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None
    
    def test_chat_message_with_name(self):
        """Test chat message with name."""
        message = ChatMessage(role="assistant", content="Hi there!", name="Claude")
        
        assert message.role == "assistant"
        assert message.content == "Hi there!"
        assert message.name == "Claude"
    
    def test_invalid_role(self):
        """Test invalid role validation."""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="Hello")
    
    def test_empty_content(self):
        """Test empty content is allowed."""
        message = ChatMessage(role="user", content="")
        assert message.content == ""


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""
    
    def test_minimal_request(self):
        """Test minimal valid request."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(messages=messages)
        
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello"
        assert request.model is None
        assert request.stream is False
    
    def test_full_request(self):
        """Test request with all parameters."""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello"),
        ]
        request = ChatCompletionRequest(
            messages=messages,
            model="gpt-4",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            stream=False,
        )
        
        assert len(request.messages) == 2
        assert request.model == "gpt-4"
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.top_p == 0.9
        assert request.frequency_penalty == 0.1
        assert request.presence_penalty == 0.1
        assert request.stop == ["END"]
        assert request.stream is False
    
    def test_empty_messages(self):
        """Test empty messages list."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[])


class TestChatCompletionResponse:
    """Test ChatCompletionResponse model."""
    
    def test_valid_response(self):
        """Test valid response creation."""
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello there!"),
            finish_reason="stop",
        )
        usage = ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1677652288,
            model="gpt-4",
            choices=[choice],
            usage=usage,
        )
        
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert response.created == 1677652288
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello there!"
        assert response.usage.total_tokens == 15
    
    def test_multiple_choices(self):
        """Test response with multiple choices."""
        choices = [
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Option 1"),
                finish_reason="stop",
            ),
            ChatCompletionChoice(
                index=1,
                message=ChatMessage(role="assistant", content="Option 2"),
                finish_reason="stop",
            ),
        ]
        usage = ChatCompletionUsage(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1677652288,
            model="gpt-4",
            choices=choices,
            usage=usage,
        )
        
        assert len(response.choices) == 2
        assert response.choices[0].message.content == "Option 1"
        assert response.choices[1].message.content == "Option 2"


class TestChatCompletionUsage:
    """Test ChatCompletionUsage model."""
    
    def test_valid_usage(self):
        """Test valid usage statistics."""
        usage = ChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    def test_zero_tokens(self):
        """Test usage with zero tokens."""
        usage = ChatCompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
    
    def test_reasoning_tokens(self):
        """Test usage with reasoning tokens."""
        usage = ChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=1000,
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.reasoning_tokens == 1000
    
    def test_optional_reasoning_tokens(self):
        """Test usage without reasoning tokens (traditional models)."""
        usage = ChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.reasoning_tokens is None


class TestReasoningModelSupport:
    """Test reasoning model specific functionality."""
    
    def test_request_with_max_completion_tokens(self):
        """Test request with max_completion_tokens for reasoning models."""
        messages = [ChatMessage(role="user", content="Solve this problem step by step.")]
        request = ChatCompletionRequest(
            messages=messages,
            model="o1-preview",
            max_completion_tokens=32768,
        )
        
        assert len(request.messages) == 1
        assert request.model == "o1-preview"
        assert request.max_completion_tokens == 32768
        assert request.max_tokens is None
        assert request.temperature is None
        assert request.top_p is None
        assert request.frequency_penalty is None
        assert request.presence_penalty is None
    
    def test_request_with_both_token_limits(self):
        """Test request with both max_tokens and max_completion_tokens."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(
            messages=messages,
            model="gpt-4",
            max_tokens=1000,
            max_completion_tokens=2000,
        )
        
        assert request.max_tokens == 1000
        assert request.max_completion_tokens == 2000