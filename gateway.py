#!/usr/bin/env python3
"""DeepSeek4Free Gateway - OpenAI-compatible API proxy to DeepSeek web chat."""

import argparse
import logging
import sys

from src.config import Config
from src.deepseek_client import DeepSeekClient
from src.server import create_server


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepSeek4Free Gateway - OpenAI-compatible API server"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind host (default: 0.0.0.0, or SERVER_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: 8080, or SERVER_PORT env var)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="DeepSeek bearer token (or DEEPSEEK_TOKEN env var)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="DeepSeek API base URL (default: https://chat.deepseek.com)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = Config(
        token=args.token,
        host=args.host,
        port=args.port,
        base_url=args.base_url,
        log_level=args.log_level,
    )

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize DeepSeek client
    client = DeepSeekClient(token=config.token, base_url=config.base_url)

    # Create and start server
    server = create_server(config.host, config.port, client)

    print(f"""
{'='*60}
  DeepSeek4Free Gateway
  OpenAI-compatible API proxy
{'='*60}

  Server:    http://{config.host}:{config.port}
  DeepSeek:  {config.base_url}

  Endpoints:
    POST /v1/chat/completions  - Chat completions
    POST /v1/completions       - Legacy completions
    GET  /v1/models            - List models
    POST /v1/embeddings        - (not supported)

  Press Ctrl+C to stop
{'='*60}
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
