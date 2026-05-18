"""Unit tests for tool calling / function calling emulation."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from src.tool_calling import (
    has_tool_definitions,
    normalize_tools,
    build_tool_system_prompt,
    inject_tools_into_messages,
    convert_tool_messages_to_text,
    parse_tool_calls,
    has_tool_call_response,
    build_tool_call_response,
    build_tool_call_stream_chunks,
    preprocess_request,
    postprocess_response,
    generate_tool_call_id,
)


class TestHasToolDefinitions(unittest.TestCase):
    """Tests for detecting tool definitions in requests."""

    def test_no_tools(self):
        self.assertFalse(has_tool_definitions({"messages": []}))

    def test_with_tools(self):
        self.assertTrue(has_tool_definitions({"tools": [{"type": "function"}]}))

    def test_with_functions(self):
        self.assertTrue(has_tool_definitions({"functions": [{"name": "test"}]}))

    def test_empty_tools(self):
        self.assertFalse(has_tool_definitions({"tools": []}))

    def test_non_dict_input(self):
        self.assertFalse(has_tool_definitions("not a dict"))
        self.assertFalse(has_tool_definitions(None))


class TestNormalizeTools(unittest.TestCase):
    """Tests for normalizing tools and legacy functions format."""

    def test_tools_passthrough(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        result_tools, result_choice = normalize_tools({"tools": tools, "tool_choice": "auto"})
        self.assertEqual(result_tools, tools)
        self.assertEqual(result_choice, "auto")

    def test_legacy_functions_conversion(self):
        request = {
            "functions": [
                {"name": "get_weather", "description": "Get weather", "parameters": {}}
            ],
            "function_call": "auto",
        }
        tools, choice = normalize_tools(request)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "get_weather")
        self.assertEqual(choice, "auto")

    def test_legacy_function_call_none(self):
        request = {
            "functions": [{"name": "test"}],
            "function_call": "none",
        }
        _, choice = normalize_tools(request)
        self.assertEqual(choice, "none")

    def test_legacy_function_call_specific(self):
        request = {
            "functions": [{"name": "get_weather"}],
            "function_call": {"name": "get_weather"},
        }
        _, choice = normalize_tools(request)
        self.assertEqual(choice["type"], "function")
        self.assertEqual(choice["function"]["name"], "get_weather")

    def test_no_tools_or_functions(self):
        tools, choice = normalize_tools({"messages": []})
        self.assertEqual(tools, [])
        self.assertEqual(choice, "auto")


class TestBuildToolSystemPrompt(unittest.TestCase):
    """Tests for building tool system prompt."""

    def test_basic_prompt(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        prompt = build_tool_system_prompt(tools)
        self.assertIn("get_weather", prompt)
        self.assertIn("Get weather for a city", prompt)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("</tool_call>", prompt)

    def test_tool_choice_none_returns_empty(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        prompt = build_tool_system_prompt(tools, tool_choice="none")
        self.assertEqual(prompt, "")

    def test_empty_tools_returns_empty(self):
        prompt = build_tool_system_prompt([])
        self.assertEqual(prompt, "")

    def test_tool_choice_required(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        prompt = build_tool_system_prompt(tools, tool_choice="required")
        self.assertIn("MUST call", prompt)

    def test_tool_choice_specific_function(self):
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        choice = {"type": "function", "function": {"name": "get_weather"}}
        prompt = build_tool_system_prompt(tools, tool_choice=choice)
        self.assertIn("get_weather", prompt)
        self.assertIn("MUST call", prompt)

    def test_multiple_tools(self):
        tools = [
            {"type": "function", "function": {"name": "func_a", "description": "A"}},
            {"type": "function", "function": {"name": "func_b", "description": "B"}},
        ]
        prompt = build_tool_system_prompt(tools)
        self.assertIn("func_a", prompt)
        self.assertIn("func_b", prompt)


class TestInjectToolsIntoMessages(unittest.TestCase):
    """Tests for injecting tool definitions into messages."""

    def test_injects_new_system_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = inject_tools_into_messages(messages, tools)
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("test", result[0]["content"])
        self.assertEqual(result[1]["role"], "user")

    def test_prepends_to_existing_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        tools = [{"type": "function", "function": {"name": "greet"}}]
        result = inject_tools_into_messages(messages, tools)
        self.assertEqual(len(result), 2)
        self.assertIn("greet", result[0]["content"])
        self.assertIn("You are helpful.", result[0]["content"])

    def test_no_mutation_of_original(self):
        messages = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        original_content = messages[0]["content"]
        inject_tools_into_messages(messages, tools)
        self.assertEqual(messages[0]["content"], original_content)

    def test_empty_tools_no_injection(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = inject_tools_into_messages(messages, [], tool_choice="none")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "Hi")


class TestConvertToolMessages(unittest.TestCase):
    """Tests for converting tool-calling messages to plain text."""

    def test_assistant_tool_calls_message(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            }
        ]
        result = convert_tool_messages_to_text(messages)
        self.assertEqual(result[0]["role"], "assistant")
        self.assertIn("<tool_call>", result[0]["content"])
        self.assertIn("get_weather", result[0]["content"])
        self.assertIn("Paris", result[0]["content"])

    def test_tool_result_message(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_weather",
                "content": "72F and sunny",
            }
        ]
        result = convert_tool_messages_to_text(messages)
        self.assertEqual(result[0]["role"], "user")
        self.assertIn("get_weather", result[0]["content"])
        self.assertIn("72F and sunny", result[0]["content"])
        self.assertIn("call_123", result[0]["content"])

    def test_legacy_function_result_message(self):
        messages = [
            {
                "role": "function",
                "name": "get_weather",
                "content": "72F and sunny",
            }
        ]
        result = convert_tool_messages_to_text(messages)
        self.assertEqual(result[0]["role"], "user")
        self.assertIn("get_weather", result[0]["content"])
        self.assertIn("72F and sunny", result[0]["content"])

    def test_regular_messages_passthrough(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = convert_tool_messages_to_text(messages)
        self.assertEqual(result, messages)

    def test_multi_turn_flow(self):
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": "65F, cloudy",
            },
        ]
        result = convert_tool_messages_to_text(messages)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(result[2]["role"], "user")
        self.assertIn("<tool_call>", result[1]["content"])
        self.assertIn("65F, cloudy", result[2]["content"])


class TestParseToolCalls(unittest.TestCase):
    """Tests for parsing tool calls from model response."""

    def test_single_tool_call(self):
        content = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
        result = parse_tool_calls(content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Paris")

    def test_multiple_tool_calls(self):
        content = (
            '<tool_call>\n{"name": "func_a", "arguments": {"x": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "func_b", "arguments": {"y": 2}}\n</tool_call>'
        )
        result = parse_tool_calls(content)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["function"]["name"], "func_a")
        self.assertEqual(result[1]["function"]["name"], "func_b")

    def test_no_tool_calls(self):
        content = "Just a normal response without any tool calls."
        result = parse_tool_calls(content)
        self.assertEqual(result, [])

    def test_empty_content(self):
        self.assertEqual(parse_tool_calls(""), [])
        self.assertEqual(parse_tool_calls(None), [])

    def test_malformed_json_skipped(self):
        content = '<tool_call>\n{invalid json}\n</tool_call>'
        result = parse_tool_calls(content)
        self.assertEqual(result, [])

    def test_tool_call_id_generated(self):
        content = '<tool_call>\n{"name": "test", "arguments": {}}\n</tool_call>'
        result = parse_tool_calls(content)
        self.assertTrue(result[0]["id"].startswith("call_"))

    def test_arguments_as_string(self):
        content = '<tool_call>\n{"name": "test", "arguments": "{\\"key\\": \\"val\\"}"}\n</tool_call>'
        result = parse_tool_calls(content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["arguments"], '{"key": "val"}')


class TestHasToolCallResponse(unittest.TestCase):
    """Tests for detecting tool call markers in response."""

    def test_with_marker(self):
        self.assertTrue(has_tool_call_response("Some text <tool_call> stuff"))

    def test_without_marker(self):
        self.assertFalse(has_tool_call_response("Normal response"))

    def test_empty(self):
        self.assertFalse(has_tool_call_response(""))
        self.assertFalse(has_tool_call_response(None))


class TestBuildToolCallResponse(unittest.TestCase):
    """Tests for building OpenAI-format tool call response."""

    def test_basic_response(self):
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ]
        result = build_tool_call_response(tool_calls, "deepseek-chat", "req-123")
        self.assertEqual(result["id"], "req-123")
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(result["choices"][0]["message"]["content"])
        self.assertEqual(result["choices"][0]["message"]["tool_calls"], tool_calls)
        self.assertEqual(result["choices"][0]["message"]["role"], "assistant")


class TestBuildToolCallStreamChunks(unittest.TestCase):
    """Tests for streaming tool call chunks."""

    def test_single_tool_call_chunks(self):
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ]
        chunks = build_tool_call_stream_chunks(tool_calls, "deepseek-chat", "req-123")
        # Should have: 1 initial chunk (role + name), 1 arguments chunk, 1 final chunk
        self.assertEqual(len(chunks), 3)

        # First chunk has role and tool call name
        first = chunks[0]
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")
        tc_delta = first["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc_delta["id"], "call_abc")
        self.assertEqual(tc_delta["function"]["name"], "get_weather")
        self.assertEqual(tc_delta["function"]["arguments"], "")

        # Second chunk has arguments
        second = chunks[1]
        tc_args = second["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(tc_args["function"]["arguments"], '{"location": "NYC"}')

        # Final chunk has finish_reason
        final = chunks[-1]
        self.assertEqual(final["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(final["choices"][0]["delta"], {})

    def test_multiple_tool_calls_chunks(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "func_a", "arguments": '{"x": 1}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "func_b", "arguments": '{"y": 2}'},
            },
        ]
        chunks = build_tool_call_stream_chunks(tool_calls, "deepseek-chat", "req-456")
        # 1 initial chunk (with both names), 2 argument chunks, 1 final
        self.assertEqual(len(chunks), 4)

        # First chunk should have both tool calls listed
        first_delta = chunks[0]["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(first_delta), 2)
        self.assertEqual(first_delta[0]["function"]["name"], "func_a")
        self.assertEqual(first_delta[1]["function"]["name"], "func_b")


class TestPreprocessRequest(unittest.TestCase):
    """Tests for the full request preprocessing."""

    def test_no_tools_passthrough(self):
        request = {"messages": [{"role": "user", "content": "Hi"}]}
        result, has_tools = preprocess_request(request)
        self.assertFalse(has_tools)
        self.assertEqual(result, request)

    def test_with_tools_injects_prompt(self):
        request = {
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {},
                    },
                }
            ],
            "tool_choice": "auto",
        }
        result, has_tools = preprocess_request(request)
        self.assertTrue(has_tools)
        # Should have system message injected
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("get_weather", result["messages"][0]["content"])
        # Should not have tools field in result
        self.assertNotIn("tools", result)
        self.assertNotIn("tool_choice", result)

    def test_tool_choice_none_no_injection(self):
        request = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "tool_choice": "none",
        }
        result, has_tools = preprocess_request(request)
        self.assertFalse(has_tools)

    def test_with_tool_messages_converted(self):
        request = {
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "get_weather",
                    "content": "Sunny",
                },
                {"role": "user", "content": "Thanks"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
            ],
        }
        result, has_tools = preprocess_request(request)
        self.assertTrue(has_tools)
        # Tool messages should be converted
        for msg in result["messages"]:
            self.assertNotEqual(msg.get("role"), "tool")
            self.assertNotIn("tool_calls", msg)


class TestPostprocessResponse(unittest.TestCase):
    """Tests for response postprocessing."""

    def test_no_tools_passthrough(self):
        response = {
            "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}]
        }
        result = postprocess_response(response, has_tools=False)
        self.assertEqual(result, response)

    def test_tool_call_detected(self):
        content = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>'
        response = {
            "id": "test",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
        }
        result = postprocess_response(response, has_tools=True)
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(result["choices"][0]["message"]["content"])
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")

    def test_no_tool_call_in_response(self):
        response = {
            "choices": [
                {"message": {"role": "assistant", "content": "Just text"}, "finish_reason": "stop"}
            ]
        }
        result = postprocess_response(response, has_tools=True)
        self.assertEqual(result["choices"][0]["message"]["content"], "Just text")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")

    def test_empty_choices(self):
        response = {"choices": []}
        result = postprocess_response(response, has_tools=True)
        self.assertEqual(result, response)


class TestStreamingToolCalls(unittest.TestCase):
    """Tests for streaming tool call handling in the handler."""

    def test_streaming_with_tool_call_response(self):
        from src.handlers import handle_chat_completions

        mock_client = MagicMock()
        tool_response = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>'
        chunks = [
            {"choices": [{"delta": {"content": "<tool_call>"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": '\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n'}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "</tool_call>"}, "finish_reason": None}]},
        ]
        mock_client.chat_completion_stream.return_value = iter(chunks)

        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Weather in NYC?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                }
            ],
            "stream": True,
        }
        wfile = io.BytesIO()
        result = handle_chat_completions(request, mock_client, wfile=wfile)
        self.assertIsNone(result)

        output = wfile.getvalue().decode("utf-8")
        self.assertIn("[DONE]", output)
        # Should contain tool_calls in the stream
        self.assertIn("tool_calls", output)
        self.assertIn("get_weather", output)

    def test_streaming_without_tool_call(self):
        from src.handlers import handle_chat_completions

        mock_client = MagicMock()
        chunks = [
            {"choices": [{"delta": {"content": "The weather is "}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "sunny today."}, "finish_reason": None}]},
        ]
        mock_client.chat_completion_stream.return_value = iter(chunks)

        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
            "stream": True,
        }
        wfile = io.BytesIO()
        handle_chat_completions(request, mock_client, wfile=wfile)

        output = wfile.getvalue().decode("utf-8")
        self.assertIn("[DONE]", output)
        self.assertIn("sunny today", output)
        # Should NOT contain tool_calls
        self.assertNotIn('"tool_calls"', output)


class TestNonStreamingToolCalls(unittest.TestCase):
    """Tests for non-streaming tool call handling in the handler."""

    def test_non_streaming_tool_call(self):
        from src.handlers import handle_chat_completions

        mock_client = MagicMock()
        content = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "London"}}\n</tool_call>'
        mock_client.chat_completion.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ]
        }

        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Weather in London?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    },
                }
            ],
        }
        status_code, body = handle_chat_completions(request, mock_client)
        self.assertEqual(status_code, 200)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(body["choices"][0]["message"]["content"])
        tc = body["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["function"]["name"], "get_weather")
        args = json.loads(tc[0]["function"]["arguments"])
        self.assertEqual(args["location"], "London")

    def test_non_streaming_no_tool_call(self):
        from src.handlers import handle_chat_completions

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": "It is sunny."}, "finish_reason": "stop"}
            ]
        }

        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
            ],
        }
        status_code, body = handle_chat_completions(request, mock_client)
        self.assertEqual(status_code, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "It is sunny.")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")


class TestGenerateToolCallId(unittest.TestCase):
    """Tests for tool call ID generation."""

    def test_format(self):
        tid = generate_tool_call_id()
        self.assertTrue(tid.startswith("call_"))

    def test_uniqueness(self):
        ids = {generate_tool_call_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestLegacyFunctionCallIntegration(unittest.TestCase):
    """Integration tests for legacy function_call support."""

    def test_legacy_functions_in_handler(self):
        from src.handlers import handle_chat_completions

        mock_client = MagicMock()
        content = '<tool_call>\n{"name": "search", "arguments": {"query": "python"}}\n</tool_call>'
        mock_client.chat_completion.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ]
        }

        request = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Search for python"}],
            "functions": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
            "function_call": "auto",
        }
        status_code, body = handle_chat_completions(request, mock_client)
        self.assertEqual(status_code, 200)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        tc = body["choices"][0]["message"]["tool_calls"]
        self.assertEqual(tc[0]["function"]["name"], "search")


if __name__ == "__main__":
    unittest.main()
