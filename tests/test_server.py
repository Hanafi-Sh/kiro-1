"""Integration tests for the HTTP server."""

import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from unittest.mock import MagicMock, patch

from src.server import create_server, GatewayHandler
from src.deepseek_client import DeepSeekAPIError


def find_free_port():
    """Find a free port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class TestServerIntegration(unittest.TestCase):
    """Integration tests that start a real server and make HTTP requests."""

    @classmethod
    def setUpClass(cls):
        """Start a test server on a random port."""
        cls.mock_client = MagicMock()
        cls.port = find_free_port()
        cls.server = create_server("127.0.0.1", cls.port, cls.mock_client)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        time.sleep(0.1)  # Give server time to start

    @classmethod
    def tearDownClass(cls):
        """Shut down the test server."""
        cls.server.shutdown()
        cls.server_thread.join(timeout=5)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def test_get_models(self):
        """GET /v1/models should return model list."""
        url = f"{self.base_url}/v1/models"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["object"], "list")
            self.assertGreater(len(data["data"]), 0)

    def test_get_models_content_type(self):
        """GET /v1/models should return application/json."""
        url = f"{self.base_url}/v1/models"
        with urllib.request.urlopen(url) as resp:
            content_type = resp.headers.get("Content-Type")
            self.assertIn("application/json", content_type)

    def test_get_health(self):
        """GET /health should return OK status."""
        url = f"{self.base_url}/health"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "ok")

    def test_post_chat_completions(self):
        """POST /v1/chat/completions should return completion."""
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ]
        }
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["object"], "chat.completion")
            self.assertEqual(data["choices"][0]["message"]["content"], "Test response")

    def test_post_chat_completions_invalid_json(self):
        """POST /v1/chat/completions with invalid JSON should return 400."""
        url = f"{self.base_url}/v1/chat/completions"
        body = b"not valid json"
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", str(len(body)))

        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            data = json.loads(e.read().decode())
            self.assertIn("error", data)

    def test_post_chat_completions_missing_messages(self):
        """POST /v1/chat/completions without messages should return 400."""
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps({"model": "deepseek-chat"}).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_post_chat_completions_streaming(self):
        """POST /v1/chat/completions with stream=true should return SSE."""
        chunks = [
            {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
        ]
        self.mock_client.chat_completion_stream.return_value = iter(chunks)

        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("Content-Type")
            self.assertIn("text/event-stream", content_type)
            output = resp.read().decode()
            self.assertIn("data: ", output)
            self.assertIn("[DONE]", output)

    def test_post_embeddings_not_supported(self):
        """POST /v1/embeddings should return error."""
        url = f"{self.base_url}/v1/embeddings"
        body = json.dumps({"input": "test"}).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            data = json.loads(e.read().decode())
            self.assertIn("not supported", data["error"]["message"])

    def test_unknown_endpoint_returns_404(self):
        """GET /v1/unknown should return 404."""
        url = f"{self.base_url}/v1/unknown"
        try:
            urllib.request.urlopen(url)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_cors_headers(self):
        """Responses should include CORS headers."""
        url = f"{self.base_url}/v1/models"
        with urllib.request.urlopen(url) as resp:
            cors = resp.headers.get("Access-Control-Allow-Origin")
            self.assertEqual(cors, "*")

    def test_options_preflight(self):
        """OPTIONS request should return 204 with CORS headers."""
        url = f"{self.base_url}/v1/chat/completions"
        req = urllib.request.Request(url, method="OPTIONS")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 204)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")

    def test_post_completions_legacy(self):
        """POST /v1/completions should work with prompt field."""
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Legacy response"},
                    "finish_reason": "stop",
                }
            ]
        }
        url = f"{self.base_url}/v1/completions"
        body = json.dumps({
            "model": "deepseek-chat",
            "prompt": "Hello world",
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["object"], "chat.completion")


class TestServerAuth(unittest.TestCase):
    """Tests for gateway-level API key authentication."""

    @classmethod
    def setUpClass(cls):
        """Start a test server with auth enabled."""
        cls.mock_client = MagicMock()
        cls.port = find_free_port()
        cls.api_key = "test-secret-key-123"
        cls.server = create_server("127.0.0.1", cls.port, cls.mock_client, gateway_api_key=cls.api_key)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        """Shut down the test server."""
        cls.server.shutdown()
        cls.server_thread.join(timeout=5)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def test_no_auth_header_returns_401(self):
        """Request without Authorization header should return 401."""
        url = f"{self.base_url}/v1/models"
        try:
            urllib.request.urlopen(url)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)
            data = json.loads(e.read().decode())
            self.assertIn("error", data)
            self.assertEqual(data["error"]["type"], "authentication_error")

    def test_wrong_key_returns_401(self):
        """Request with wrong API key should return 401."""
        url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer wrong-key")
        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)

    def test_correct_key_allows_access(self):
        """Request with correct API key should succeed."""
        url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["object"], "list")

    def test_health_endpoint_no_auth_required(self):
        """Health endpoint should work without auth."""
        url = f"{self.base_url}/health"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "ok")

    def test_post_with_correct_key(self):
        """POST request with correct key should succeed."""
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Auth OK"},
                    "finish_reason": "stop",
                }
            ]
        }
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)

    def test_post_without_key_returns_401(self):
        """POST request without key should return 401."""
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 401)

    def test_streaming_missing_messages_returns_400_not_200(self):
        """Streaming request with missing messages should return 400, not SSE stream."""
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps({
            "model": "deepseek-chat",
            "stream": True,
        }).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            data = json.loads(e.read().decode())
            self.assertIn("messages", data["error"]["message"])


if __name__ == "__main__":
    unittest.main()
