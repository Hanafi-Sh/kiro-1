"""Tool calling / function calling emulation via prompt injection and response parsing.

DeepSeek web chat does NOT natively support function/tool calling.
This module emulates it by:
1. Injecting tool definitions into the system prompt
2. Instructing the model to respond with a specific format when calling tools
3. Parsing the model's response to detect and extract tool calls
4. Converting multi-turn tool calling messages to plain text
"""

import json
import re
import uuid

# Regex to find <tool_call>...</tool_call> blocks in model output
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def generate_tool_call_id():
    """Generate a unique tool call ID in OpenAI format."""
    return f"call_{uuid.uuid4().hex[:24]}"


def has_tool_definitions(request_body):
    """Check if the request contains tool/function definitions.

    Args:
        request_body: parsed OpenAI request dict

    Returns:
        True if tools or functions are present
    """
    if not isinstance(request_body, dict):
        return False
    return bool(request_body.get("tools")) or bool(request_body.get("functions"))


def normalize_tools(request_body):
    """Normalize legacy functions format to tools format.

    Converts deprecated 'functions'/'function_call' fields to 'tools'/'tool_choice'.

    Args:
        request_body: parsed OpenAI request dict

    Returns:
        tuple: (tools_list, tool_choice) where tools_list is a list of tool dicts
               and tool_choice is the resolved tool_choice value
    """
    tools = request_body.get("tools")
    tool_choice = request_body.get("tool_choice", "auto")

    if tools:
        return tools, tool_choice

    # Convert legacy functions format
    functions = request_body.get("functions", [])
    if functions:
        tools = []
        for func in functions:
            tools.append({
                "type": "function",
                "function": func,
            })

        # Convert legacy function_call to tool_choice
        function_call = request_body.get("function_call", "auto")
        if function_call == "auto":
            tool_choice = "auto"
        elif function_call == "none":
            tool_choice = "none"
        elif isinstance(function_call, dict) and "name" in function_call:
            tool_choice = {"type": "function", "function": {"name": function_call["name"]}}
        else:
            tool_choice = "auto"

        return tools, tool_choice

    return [], "auto"


def build_tool_system_prompt(tools, tool_choice="auto"):
    """Build a system prompt that describes available tools to the model.

    Args:
        tools: list of tool definitions in OpenAI format
        tool_choice: "auto", "none", "required", or specific function dict

    Returns:
        str: system prompt text to inject
    """
    if not tools or tool_choice == "none":
        return ""

    tool_descriptions = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            desc = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
            tool_descriptions.append(desc)

    tools_json = json.dumps(tool_descriptions, indent=2)

    # Build the instruction based on tool_choice
    if tool_choice == "required":
        choice_instruction = "You MUST call one or more of the available tools in your response. Do NOT respond with plain text."
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        func_name = tool_choice["function"]["name"]
        choice_instruction = f'You MUST call the function "{func_name}" in your response. Do NOT respond with plain text.'
    else:
        # "auto" - model decides
        choice_instruction = "If you need to call a tool, use the format below. If you can answer without tools, respond normally with plain text."

    prompt = f"""You have access to the following tools:

{tools_json}

To call a tool, you MUST use exactly this format (one block per tool call):

<tool_call>
{{"name": "function_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

Rules:
- {choice_instruction}
- The "arguments" field must be a JSON object matching the function's parameters schema.
- You may call multiple tools by using multiple <tool_call> blocks.
- Do NOT include any other text inside <tool_call> blocks.
- If calling tools, ONLY output <tool_call> blocks with no additional text outside them."""

    return prompt


def inject_tools_into_messages(messages, tools, tool_choice="auto"):
    """Inject tool definitions into the message list as a system prompt.

    If a system message already exists, prepend the tool prompt to it.
    Otherwise, insert a new system message at the beginning.

    Args:
        messages: list of message dicts
        tools: list of tool definitions
        tool_choice: tool choice setting

    Returns:
        list: modified messages with tool definitions injected
    """
    tool_prompt = build_tool_system_prompt(tools, tool_choice)
    if not tool_prompt:
        return messages

    messages = [msg.copy() for msg in messages]  # shallow copy to avoid mutation

    # Check if there's already a system message at the start
    if messages and messages[0].get("role") == "system":
        existing_content = messages[0].get("content", "")
        messages[0]["content"] = tool_prompt + "\n\n" + existing_content
    else:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    return messages


def convert_tool_messages_to_text(messages):
    """Convert tool-calling multi-turn messages to plain text format.

    Handles:
    - Assistant messages with tool_calls field -> convert to text showing the calls
    - Tool result messages (role: "tool") -> convert to text showing the result

    Args:
        messages: list of message dicts (may include tool calling messages)

    Returns:
        list: messages converted to plain text format DeepSeek can understand
    """
    converted = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant" and msg.get("tool_calls"):
            # Convert assistant tool call message to text
            tool_calls = msg["tool_calls"]
            parts = []
            content = msg.get("content")
            if content:
                parts.append(content)
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", "{}")
                # If arguments is a string, try to parse it for clean formatting
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        pass
                call_obj = {"name": name, "arguments": arguments}
                parts.append(f"<tool_call>\n{json.dumps(call_obj)}\n</tool_call>")
            converted.append({
                "role": "assistant",
                "content": "\n".join(parts),
            })

        elif role == "tool":
            # Convert tool result message to a user message with context
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "")
            content = msg.get("content", "")
            result_text = f'Tool "{tool_name}" (call_id: {tool_call_id}) returned:\n{content}'
            converted.append({
                "role": "user",
                "content": result_text,
            })

        elif role == "function":
            # Legacy function result message
            func_name = msg.get("name", "")
            content = msg.get("content", "")
            result_text = f'Function "{func_name}" returned:\n{content}'
            converted.append({
                "role": "user",
                "content": result_text,
            })

        else:
            # Pass through system, user, and regular assistant messages
            converted.append(msg)

    return converted


def parse_tool_calls(content):
    """Parse tool calls from model response content.

    Looks for <tool_call>...</tool_call> blocks and extracts function calls.

    Args:
        content: string response content from the model

    Returns:
        list: list of parsed tool call dicts in OpenAI format, or empty list if none found
    """
    if not content:
        return []

    matches = TOOL_CALL_PATTERN.findall(content)
    if not matches:
        return []

    tool_calls = []
    for match in matches:
        try:
            call_data = json.loads(match)
        except json.JSONDecodeError:
            continue

        name = call_data.get("name", "")
        arguments = call_data.get("arguments", {})

        # Ensure arguments is a JSON string (OpenAI format)
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        elif isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments)

        tool_calls.append({
            "id": generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments_str,
            },
        })

    return tool_calls


def has_tool_call_response(content):
    """Check if the response content contains tool call markers.

    Args:
        content: response text

    Returns:
        bool: True if content contains <tool_call> markers
    """
    if not content:
        return False
    return "<tool_call>" in content


def build_tool_call_response(tool_calls, model, request_id):
    """Build an OpenAI-format response with tool calls.

    Args:
        tool_calls: list of parsed tool call dicts
        model: model name
        request_id: request ID

    Returns:
        dict: OpenAI chat completion response with tool_calls
    """
    import time

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def build_tool_call_stream_chunks(tool_calls, model, request_id):
    """Build streaming chunks for tool call responses.

    OpenAI streams tool calls as:
    - First chunk: role + tool_calls[i].id + tool_calls[i].function.name + empty arguments
    - Subsequent chunks: arguments string split across chunks
    - Final chunk: finish_reason = "tool_calls"

    For simplicity (and because we buffer the full response), we emit:
    - One chunk per tool call with full id, name, and arguments
    - Final chunk with finish_reason

    Args:
        tool_calls: list of parsed tool call dicts
        model: model name
        request_id: request ID

    Returns:
        list: list of chunk dicts to emit as SSE events
    """
    import time

    chunks = []
    created = int(time.time())

    # First chunk with role and first tool call info
    first_tool_calls_delta = []
    for i, tc in enumerate(tool_calls):
        first_tool_calls_delta.append({
            "index": i,
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": "",
            },
        })

    chunks.append({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": first_tool_calls_delta,
                },
                "finish_reason": None,
            }
        ],
    })

    # Argument chunks - emit arguments for each tool call
    for i, tc in enumerate(tool_calls):
        args = tc["function"]["arguments"]
        if args:
            chunks.append({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": i,
                                    "function": {
                                        "arguments": args,
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            })

    # Final chunk with finish_reason
    chunks.append({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }
        ],
    })

    return chunks


def preprocess_request(request_body):
    """Preprocess a request to handle tool calling.

    This is the main entry point for tool calling preprocessing.
    It normalizes tools, injects them into the system prompt,
    and converts tool messages to text.

    Args:
        request_body: parsed OpenAI request dict

    Returns:
        tuple: (modified_request_body, has_tools) where has_tools indicates
               whether tool calling is active for this request
    """
    if not has_tool_definitions(request_body):
        return request_body, False

    tools, tool_choice = normalize_tools(request_body)

    if not tools or tool_choice == "none":
        return request_body, False

    messages = request_body.get("messages", [])

    # Convert any tool-calling messages to plain text
    messages = convert_tool_messages_to_text(messages)

    # Inject tool definitions into system prompt
    messages = inject_tools_into_messages(messages, tools, tool_choice)

    # Build modified request (strip tool-related fields)
    modified = {}
    for key, value in request_body.items():
        if key not in ("tools", "tool_choice", "functions", "function_call"):
            modified[key] = value
    modified["messages"] = messages

    return modified, True


def postprocess_response(openai_response, has_tools):
    """Postprocess a non-streaming response to detect and format tool calls.

    Args:
        openai_response: OpenAI-format response dict
        has_tools: whether tool calling was active for this request

    Returns:
        dict: modified response with tool_calls if detected
    """
    if not has_tools:
        return openai_response

    choices = openai_response.get("choices", [])
    if not choices:
        return openai_response

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if not has_tool_call_response(content):
        return openai_response

    tool_calls = parse_tool_calls(content)
    if not tool_calls:
        return openai_response

    # Replace message with tool_calls format
    openai_response["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    openai_response["choices"][0]["finish_reason"] = "tool_calls"

    return openai_response
