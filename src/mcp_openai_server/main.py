"""Main entry point for the MCP OpenAI Server."""

import asyncio
import os
import sys
from typing import Optional

import typer
from dotenv import load_dotenv

from .config import get_config
from .server import MCPOpenAIServer

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(
    name="mcp-openai-server",
    help="MCP server that exposes OpenAI chat completion API for other LLMs to use.",
)


@app.command()
def run(
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to run the server on (for HTTP transport)",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport protocol to use (stdio, sse, or streamable-http)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
) -> None:
    """Run the MCP OpenAI server."""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MCP_OPENAI_OPENAI_API_KEY"):
        typer.echo(
            "Error: OpenAI API key is required. Set OPENAI_API_KEY or MCP_OPENAI_OPENAI_API_KEY environment variable.",
            err=True
        )
        typer.echo(
            "You can also create a .env file with: OPENAI_API_KEY=your_api_key_here",
            err=True
        )
        raise typer.Exit(1)
    
    try:
        # Get configuration
        config = get_config()
        if debug:
            config.debug = True
        
        # Create and run server
        server = MCPOpenAIServer(config)
        
        # Configure run parameters
        run_kwargs = {"transport": transport}
        if port:
            run_kwargs["port"] = port
        
        typer.echo(f"Starting MCP OpenAI Server with {transport} transport...")
        if port:
            typer.echo(f"Server will run on port {port}")
        
        server.run(**run_kwargs)
        
    except KeyboardInterrupt:
        typer.echo("\nServer stopped by user")
    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def test_connection() -> None:
    """Test OpenAI API connection."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MCP_OPENAI_OPENAI_API_KEY"):
        typer.echo(
            "Error: OpenAI API key is required. Set OPENAI_API_KEY or MCP_OPENAI_OPENAI_API_KEY environment variable.",
            err=True
        )
        raise typer.Exit(1)
    
    async def _test_connection():
        from openai import AsyncOpenAI
        
        config = get_config()
        client = AsyncOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            organization=config.openai_organization,
        )
        
        try:
            # Test with a simple API call
            models = await client.models.list()
            typer.echo("✅ OpenAI API connection successful!")
            typer.echo(f"Available models: {len(models.data)}")
            
            # Show a few example models
            chat_models = [m for m in models.data if "gpt" in m.id.lower()][:5]
            if chat_models:
                typer.echo("Example chat models:")
                for model in chat_models:
                    typer.echo(f"  - {model.id}")
                    
        except Exception as e:
            typer.echo(f"❌ OpenAI API connection failed: {e}", err=True)
            raise typer.Exit(1)
    
    asyncio.run(_test_connection())


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()