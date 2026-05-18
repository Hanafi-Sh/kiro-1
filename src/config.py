"""Configuration management for DeepSeek4Free gateway."""

import os


class Config:
    """Configuration loaded from environment variables and/or arguments."""

    def __init__(self, token=None, host=None, port=None, base_url=None, log_level=None,
                 gateway_api_key=None, request_timeout=None):
        self.token = token or os.environ.get("DEEPSEEK_TOKEN", "")
        self.host = host or os.environ.get("SERVER_HOST", "0.0.0.0")
        self.port = int(port or os.environ.get("SERVER_PORT", "8080"))
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", "https://chat.deepseek.com"
        )
        self.log_level = log_level or os.environ.get("LOG_LEVEL", "INFO")
        self.gateway_api_key = gateway_api_key or os.environ.get("GATEWAY_API_KEY", "")
        self.request_timeout = int(
            request_timeout or os.environ.get("REQUEST_TIMEOUT", "30")
        )

    def validate(self):
        """Validate required configuration."""
        if not self.token:
            raise ValueError(
                "DEEPSEEK_TOKEN is required. Set via environment variable or --token argument."
            )
        return self
