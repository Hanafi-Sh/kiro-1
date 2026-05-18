"""HTTP server with routing for the OpenAI-compatible API gateway."""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from .handlers import (
    handle_chat_completions,
    handle_completions,
    handle_embeddings,
    handle_models,
)
from .errors import not_found_error, invalid_request_error, server_error, format_error_json

logger = logging.getLogger("gateway")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""
    daemon_threads = True
    allow_reuse_address = True


class GatewayHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the OpenAI-compatible API gateway."""

    # Class-level reference to the DeepSeek client (set before server starts)
    deepseek_client = None

    def log_message(self, format, *args):
        """Override to use Python logging."""
        logger.info("%s - %s", self.address_string(), format % args)

    def _set_cors_headers(self):
        """Set CORS headers on the response."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json_response(self, status_code, body):
        """Send a JSON response.

        Args:
            status_code: HTTP status code
            body: dict to serialize as JSON
        """
        response_bytes = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_streaming_headers(self):
        """Send headers for an SSE streaming response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._set_cors_headers()
        self.end_headers()

    def _read_body(self):
        """Read and parse the request body as JSON.

        Returns:
            dict or None on parse error
        """
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._set_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        path = self.path.split("?")[0]  # Strip query string

        if path == "/v1/models":
            status_code, body = handle_models()
            self._send_json_response(status_code, body)
        elif path == "/health" or path == "/":
            self._send_json_response(200, {"status": "ok", "service": "deepseek4free-gateway"})
        else:
            status_code, body = not_found_error(f"Endpoint not found: {path}")
            self._send_json_response(status_code, body)

    def do_POST(self):
        """Handle POST requests."""
        path = self.path.split("?")[0]  # Strip query string

        # Read request body
        body = self._read_body()
        if body is None:
            status_code, error_body = invalid_request_error("Invalid JSON in request body.")
            self._send_json_response(status_code, error_body)
            return

        if path == "/v1/chat/completions":
            self._handle_chat_completions(body)
        elif path == "/v1/completions":
            self._handle_completions(body)
        elif path == "/v1/embeddings":
            status_code, response_body = handle_embeddings(body)
            self._send_json_response(status_code, response_body)
        else:
            status_code, error_body = not_found_error(f"Endpoint not found: {path}")
            self._send_json_response(status_code, error_body)

    def _handle_chat_completions(self, body):
        """Handle the chat completions endpoint with streaming support."""
        stream = body.get("stream", False)

        if stream:
            self._send_streaming_headers()
            handle_chat_completions(body, self.deepseek_client, wfile=self.wfile)
            self.close_connection = True
        else:
            result = handle_chat_completions(body, self.deepseek_client)
            if result is not None:
                status_code, response_body = result
                self._send_json_response(status_code, response_body)

    def _handle_completions(self, body):
        """Handle the legacy completions endpoint with streaming support."""
        stream = body.get("stream", False)

        if stream:
            self._send_streaming_headers()
            handle_completions(body, self.deepseek_client, wfile=self.wfile)
            self.close_connection = True
        else:
            result = handle_completions(body, self.deepseek_client)
            if result is not None:
                status_code, response_body = result
                self._send_json_response(status_code, response_body)


def create_server(host, port, deepseek_client):
    """Create and configure the gateway HTTP server.

    Args:
        host: bind host
        port: bind port
        deepseek_client: configured DeepSeekClient instance

    Returns:
        ThreadingHTTPServer instance
    """
    GatewayHandler.deepseek_client = deepseek_client
    server = ThreadingHTTPServer((host, port), GatewayHandler)
    return server
