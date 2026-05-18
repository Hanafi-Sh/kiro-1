"""Unit tests for the translation layer."""

import unittest
import time

from src.translator import (
    openai_to_deepseek,
    deepseek_to_openai,
    deepseek_stream_to_openai,
    create_final_stream_chunk,
    generate_request_id,
    get_model_class,
    MODEL_MAP,
)


class TestModelMapping(unittest.TestCase):
    """Tests for model name mapping."""

    def test_deepseek_chat_maps_correctly(self):
        self.assertEqual(get_model_class("deepseek-chat"), "deepseek_chat")

    def test_gpt35_maps_to_deepseek_chat(self):
        self.assertEqual(get_model_class("gpt-3.5-turbo"), "deepseek_chat")

    def test_gpt4_maps_to_deepseek_chat(self):
        self.assertEqual(get_model_class("gpt-4"), "deepseek_chat")

    def test_deepseek_coder_maps_correctly(self):
        self.assertEqual(get_model_class("deepseek-coder"), "deepseek_code")

    def test_deepseek_reasoner_maps_correctly(self):
        self.assertEqual(get_model_class("deepseek-reasoner"), "deepseek_reasoner")

    def test_unknown_model_defaults_to_chat(self):
        self.assertEqual(get_model_class("unknown-model"), "deepseek_chat")

    def test_none_model_defaults_to_chat(self):
        self.assertEqual(get_model_class(None), "deepseek_chat")

    def test_empty_model_defaults_to_chat(self):
        self.assertEqual(get_model_class(""), "deepseek_chat")


class TestOpenAIToDeepSeek(unittest.TestCase):
    """Tests for OpenAI -> DeepSeek request translation."""

    def test_basic_request(self):
        openai_req = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = openai_to_deepseek(openai_req)
        self.assertEqual(result["model_class"], "deepseek_chat")
        self.assertEqual(result["messages"], [{"role": "user", "content": "Hello"}])
        self.assertFalse(result["stream"])

    def test_streaming_request(self):
        openai_req = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        result = openai_to_deepseek(openai_req)
        self.assertTrue(result["stream"])

    def test_optional_parameters(self):
        openai_req = {
            "model": "deepseek-coder",
            "messages": [{"role": "user", "content": "Code"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }
        result = openai_to_deepseek(openai_req)
        self.assertEqual(result["model_class"], "deepseek_code")
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["max_tokens"], 100)
        self.assertEqual(result["top_p"], 0.9)

    def test_missing_optional_params_not_included(self):
        openai_req = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "test"}],
        }
        result = openai_to_deepseek(openai_req)
        self.assertNotIn("temperature", result)
        self.assertNotIn("max_tokens", result)
        self.assertNotIn("top_p", result)

    def test_empty_messages(self):
        openai_req = {
            "model": "deepseek-chat",
            "messages": [],
        }
        result = openai_to_deepseek(openai_req)
        self.assertEqual(result["messages"], [])

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        openai_req = {"model": "gpt-4", "messages": messages}
        result = openai_to_deepseek(openai_req)
        self.assertEqual(result["messages"], messages)
        self.assertEqual(result["model_class"], "deepseek_chat")


class TestDeepSeekToOpenAI(unittest.TestCase):
    """Tests for DeepSeek -> OpenAI response translation."""

    def test_basic_response(self):
        deepseek_resp = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ]
        }
        result = deepseek_to_openai(deepseek_resp, model="deepseek-chat", request_id="test-id")
        self.assertEqual(result["id"], "test-id")
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["model"], "deepseek-chat")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(result["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    def test_response_has_usage(self):
        deepseek_resp = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ]
        }
        result = deepseek_to_openai(deepseek_resp)
        self.assertIn("usage", result)
        self.assertIn("prompt_tokens", result["usage"])
        self.assertIn("completion_tokens", result["usage"])
        self.assertIn("total_tokens", result["usage"])

    def test_response_with_delta_format(self):
        deepseek_resp = {
            "choices": [{"delta": {"content": "test content"}, "finish_reason": "stop"}]
        }
        result = deepseek_to_openai(deepseek_resp)
        self.assertEqual(result["choices"][0]["message"]["content"], "test content")

    def test_empty_choices(self):
        deepseek_resp = {"choices": []}
        result = deepseek_to_openai(deepseek_resp)
        self.assertEqual(result["choices"][0]["message"]["content"], "")

    def test_generated_request_id_format(self):
        result = deepseek_to_openai({"choices": []})
        self.assertTrue(result["id"].startswith("chatcmpl-"))

    def test_created_timestamp(self):
        before = int(time.time())
        result = deepseek_to_openai({"choices": []})
        after = int(time.time())
        self.assertGreaterEqual(result["created"], before)
        self.assertLessEqual(result["created"], after)


class TestStreamChunkTranslation(unittest.TestCase):
    """Tests for streaming chunk translation."""

    def test_content_chunk(self):
        chunk = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        result = deepseek_stream_to_openai(chunk, model="deepseek-chat", request_id="test-stream")
        self.assertEqual(result["id"], "test-stream")
        self.assertEqual(result["object"], "chat.completion.chunk")
        self.assertEqual(result["choices"][0]["delta"]["content"], "Hello")
        self.assertIsNone(result["choices"][0]["finish_reason"])

    def test_first_chunk_includes_role(self):
        chunk = {"choices": [{"delta": {"content": ""}, "finish_reason": None}]}
        result = deepseek_stream_to_openai(chunk, is_first=True)
        self.assertEqual(result["choices"][0]["delta"]["role"], "assistant")

    def test_non_first_chunk_no_role(self):
        chunk = {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
        result = deepseek_stream_to_openai(chunk, is_first=False)
        self.assertNotIn("role", result["choices"][0]["delta"])

    def test_final_chunk(self):
        result = create_final_stream_chunk(model="deepseek-chat", request_id="test-final")
        self.assertEqual(result["id"], "test-final")
        self.assertEqual(result["choices"][0]["delta"], {})
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")


class TestGenerateRequestId(unittest.TestCase):
    """Tests for request ID generation."""

    def test_format(self):
        rid = generate_request_id()
        self.assertTrue(rid.startswith("chatcmpl-"))

    def test_uniqueness(self):
        ids = {generate_request_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
