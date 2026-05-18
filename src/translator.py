"""Translation layer between OpenAI API format and DeepSeek web chat format."""

import json
import time
import uuid


# Model mapping: OpenAI model name -> DeepSeek model_class
MODEL_MAP = {
    "deepseek-chat": "deepseek_chat",
    "gpt-3.5-turbo": "deepseek_chat",
    "gpt-4": "deepseek_chat",
    "gpt-4o": "deepseek_chat",
    "gpt-4o-mini": "deepseek_chat",
    "deepseek-coder": "deepseek_code",
    "deepseek-reasoner": "deepseek_reasoner",
}

DEFAULT_MODEL_CLASS = "deepseek_chat"


def generate_request_id():
    """Generate a unique request ID in OpenAI format."""
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def get_model_class(model_name):
    """Map an OpenAI model name to a DeepSeek model_class."""
    if not model_name:
        return DEFAULT_MODEL_CLASS
    return MODEL_MAP.get(model_name, DEFAULT_MODEL_CLASS)


def openai_to_deepseek(openai_request):
    """Convert an OpenAI chat completion request to DeepSeek web chat API format.

    Args:
        openai_request: dict with OpenAI API fields (messages, model, temperature, etc.)

    Returns:
        dict in DeepSeek web chat API format
    """
    model_name = openai_request.get("model", "deepseek-chat")
    model_class = get_model_class(model_name)

    messages = openai_request.get("messages", [])

    deepseek_request = {
        "messages": messages,
        "model_class": model_class,
        "stream": openai_request.get("stream", False),
    }

    # Optional parameters
    if "temperature" in openai_request:
        deepseek_request["temperature"] = openai_request["temperature"]
    if "max_tokens" in openai_request:
        deepseek_request["max_tokens"] = openai_request["max_tokens"]
    if "top_p" in openai_request:
        deepseek_request["top_p"] = openai_request["top_p"]

    return deepseek_request


def deepseek_to_openai(deepseek_response, model="deepseek-chat", request_id=None):
    """Convert a DeepSeek non-streaming response to OpenAI format.

    Args:
        deepseek_response: dict from DeepSeek API response
        model: model name to include in response
        request_id: optional request ID (generated if not provided)

    Returns:
        dict in OpenAI chat completion format
    """
    if request_id is None:
        request_id = generate_request_id()

    # Extract content from DeepSeek response
    choices = deepseek_response.get("choices", [])
    content = ""
    finish_reason = "stop"

    if choices:
        choice = choices[0]
        if "message" in choice:
            content = choice["message"].get("content", "")
        elif "delta" in choice:
            content = choice["delta"].get("content", "")
        finish_reason = choice.get("finish_reason", "stop") or "stop"

    # Estimate token usage
    prompt_tokens = 0
    completion_tokens = len(content) // 4 if content else 0
    total_tokens = prompt_tokens + completion_tokens

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def deepseek_stream_to_openai(chunk, model="deepseek-chat", request_id=None, is_first=False):
    """Convert a DeepSeek streaming chunk to OpenAI SSE format.

    Args:
        chunk: dict from a DeepSeek SSE data line
        model: model name
        request_id: request ID for this stream
        is_first: if True, include role in delta

    Returns:
        dict in OpenAI chat.completion.chunk format
    """
    if request_id is None:
        request_id = generate_request_id()

    choices = chunk.get("choices", [])
    delta = {}
    finish_reason = None

    if choices:
        choice = choices[0]
        if "delta" in choice:
            delta = choice["delta"]
        finish_reason = choice.get("finish_reason")

    if is_first and "role" not in delta:
        delta["role"] = "assistant"

    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def create_final_stream_chunk(model="deepseek-chat", request_id=None):
    """Create the final stream chunk with finish_reason=stop."""
    if request_id is None:
        request_id = generate_request_id()

    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
