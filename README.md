# MCP OpenAI Server

An MCP (Model Context Protocol) server that exposes OpenAI chat completion API functionality, allowing other LLMs to generate chat completions through the MCP protocol.

## Features

-  **Chat Completion API**: Expose OpenAI-compatible chat completion functionality via MCP
-  **Model Listing**: List available OpenAI models
-  **Server Information**: Get server capabilities and configuration
-  **Full Parameter Support**: Support all OpenAI chat completion parameters
-  **Error Handling**: Robust error handling and validation
-  **Configuration Management**: Flexible configuration via environment variables
-  **Type Safety**: Full type hints and Pydantic validation
-  **Comprehensive Testing**: Unit tests with >95% coverage

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcp-openai-server
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Usage

#### Run with STDIO transport (default)
```bash
uv run mcp-openai-server run
```

#### Run with HTTP transport
```bash
uv run mcp-openai-server run --transport sse --port 8000
```

#### Test your OpenAI API connection
```bash
uv run mcp-openai-server test-connection
```

## Configuration

Configure the server using environment variables or a `.env` file:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_BASE_URL=https://api.openai.com/v1  # Custom API endpoint
OPENAI_ORGANIZATION=your_org_id            # OpenAI organization ID

# Server settings (can also use MCP_OPENAI_ prefix)
MCP_OPENAI_DEFAULT_MODEL=gpt-4o-mini
MCP_OPENAI_DEFAULT_MAX_TOKENS=1000
MCP_OPENAI_DEFAULT_TEMPERATURE=0.7
MCP_OPENAI_SERVER_NAME=MCP OpenAI Server
MCP_OPENAI_DEBUG=false
```

## MCP Tools

The server exposes three MCP tools:

### 1. `chat_completion`
Generate chat completions using OpenAI's API.

**Parameters:**
- `messages` (required): List of message objects with `role` and `content`
- `model` (optional): Model to use (defaults to configured default)
- `max_tokens` (optional): Maximum tokens to generate
- `temperature` (optional): Sampling temperature (0-2)
- `top_p` (optional): Nucleus sampling parameter (0-1)
- `frequency_penalty` (optional): Frequency penalty (-2.0 to 2.0)
- `presence_penalty` (optional): Presence penalty (-2.0 to 2.0)
- `stop` (optional): List of stop sequences

**Example:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "model": "gpt-4o-mini",
  "temperature": 0.7
}
```

### 2. `list_models`
List available OpenAI models.

**Returns:** List of available chat models with their metadata.

### 3. `get_server_info`
Get information about the server configuration and capabilities.

**Returns:** Server name, default settings, supported features, and version.

## Development

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=mcp_openai_server

# Run specific test file
uv run pytest tests/test_server.py -v
```

### Code Quality
```bash
# Install development dependencies
uv sync --extra dev

# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

## Architecture

```
src/mcp_openai_server/
   __init__.py          # Package initialization
   config.py            # Configuration management
   models.py            # Pydantic data models
   server.py            # MCP server implementation
   main.py              # CLI entry point
```

## Use Cases

This MCP server enables other LLMs to:

1. **Generate completions**: Use OpenAI's models for text generation
2. **Multi-model workflows**: Combine different LLMs in a single pipeline
3. **API abstraction**: Provide a standardized interface to OpenAI's API
4. **Testing and development**: Test LLM applications without direct API calls

## Example Integration

Here's how another LLM could use this MCP server:

```python
from fastmcp import Client

async def example_usage():
    async with Client("mcp-openai-server") as client:
        # Generate a chat completion
        result = await client.call_tool("chat_completion", {
            "messages": [
                {"role": "user", "content": "Explain quantum computing"}
            ],
            "model": "gpt-4o-mini",
            "max_tokens": 500
        })
        
        print(result[0].text)  # The completion response
        
        # List available models
        models = await client.call_tool("list_models", {})
        print(models[0].text)  # Available models
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Support

- =� Documentation: See this README and inline code documentation
- = Issues: Report bugs via GitHub Issues
- =� Feature Requests: Submit ideas via GitHub Issues