"""Unit tests for endpoint handlers."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from src.handlers import (
    handle_chat_completions,
    handle_completions,
    handle_embeddings,
    handle_models,
)
from src.deepseek_client import DeepSeekAPIError


class TestHandleModels(unittest.TestCase):
    """Tests for the models endpoint handler."""

    def test_returns_model_list(self):
        status_code, body = handle_models()
        self.assertEqual(status_code, 200)
        self.assertEqual(body["object"], "list")
        self.assertIsInstance(body["data"], list)
        self.assertGreater(len(body["data"]), 0)

    def test_model_format(self):
        _, body = handle_models()
        for model in body["data"]:
            self.assertIn("id", model)
            self.assertEqual(model["object"], "model")
            self.assertIn("created", model)
            self.assertIn("owned_by", model)

    def test_includes_expected_models(self):
        _, body = handle_models()
        model_ids = [m["id"] for m in body["data"]]
        self.assertIn("deepseek-chat", model_ids)
        self.assertIn("deepseek-coder", model_ids)
        self.assertIn("deepseek-reasoner", model_ids)


class TestHandleChatCompletions(unittest.TestCase):
    """Tests for the chat completions handler."""

    def setUp(self):
        self.mock_client = MagicMock()

    def test_valid_request(self):
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ]
        }
        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello!")

    def test_missing_messages_returns_error(self):
        request = {"model": "deepseek-chat"}
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 400)
        self.assertIn("messages", body["error"]["message"])

    def test_empty_messages_returns_error(self):
        request = {"model": "deepseek-chat", "messages": []}
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 400)

    def test_invalid_body_type(self):
        status_code, body = handle_chat_completions("not a dict", self.mock_client)
        self.assertEqual(status_code, 400)

    def test_invalid_message_format(self):
        request = {
            "model": "deepseek-chat",
            "messages": [{"content": "no role"}],
        }
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 400)
        self.assertIn("role", body["error"]["message"])

    def test_deepseek_auth_error(self):
        self.mock_client.chat_completion.side_effect = DeepSeekAPIError(401, "Unauthorized")
        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 401)

    def test_deepseek_rate_limit_error(self):
        self.mock_client.chat_completion.side_effect = DeepSeekAPIError(429, "Rate limited")
        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 429)

    def test_deepseek_server_error(self):
        self.mock_client.chat_completion.side_effect = DeepSeekAPIError(500, "Server error")
        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 500)

    def test_unexpected_exception(self):
        self.mock_client.chat_completion.side_effect = RuntimeError("Unexpected")
        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        status_code, body = handle_chat_completions(request, self.mock_client)
        self.assertEqual(status_code, 500)

    def test_streaming_request(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        ]
        self.mock_client.chat_completion_stream.return_value = iter(chunks)

        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        wfile = io.BytesIO()
        result = handle_chat_completions(request, self.mock_client, wfile=wfile)
        self.assertIsNone(result)

        output = wfile.getvalue().decode("utf-8")
        self.assertIn("data: ", output)
        self.assertIn("[DONE]", output)
        # Verify each data line is valid JSON
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                self.assertEqual(data["object"], "chat.completion.chunk")

    def test_streaming_error(self):
        self.mock_client.chat_completion_stream.side_effect = DeepSeekAPIError(500, "Stream error")
        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        wfile = io.BytesIO()
        result = handle_chat_completions(request, self.mock_client, wfile=wfile)
        self.assertIsNone(result)
        output = wfile.getvalue().decode("utf-8")
        self.assertIn("error", output)
        self.assertIn("[DONE]", output)


class TestHandleCompletions(unittest.TestCase):
    """Tests for the legacy completions handler."""

    def setUp(self):
        self.mock_client = MagicMock()

    def test_basic_prompt(self):
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Completion"},
                    "finish_reason": "stop",
                }
            ]
        }
        request = {"prompt": "Hello", "model": "deepseek-chat"}
        status_code, body = handle_completions(request, self.mock_client)
        self.assertEqual(status_code, 200)

        # Verify the messages sent to client
        call_args = self.mock_client.chat_completion.call_args
        messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")

    def test_list_prompt(self):
        self.mock_client.chat_completion.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ]
        }
        request = {"prompt": ["line1", "line2"], "model": "deepseek-chat"}
        status_code, body = handle_completions(request, self.mock_client)
        self.assertEqual(status_code, 200)

    def test_invalid_body(self):
        status_code, body = handle_completions("not a dict", self.mock_client)
        self.assertEqual(status_code, 400)


class TestHandleEmbeddings(unittest.TestCase):
    """Tests for the embeddings handler (not supported)."""

    def test_returns_error(self):
        status_code, body = handle_embeddings({"input": "test"})
        self.assertEqual(status_code, 400)
        self.assertIn("not supported", body["error"]["message"])

    def test_error_format(self):
        status_code, body = handle_embeddings({})
        self.assertIn("error", body)
        self.assertIn("message", body["error"])
        self.assertIn("type", body["error"])


if __name__ == "__main__":
    unittest.main()
