"""Endpoint handlers for the OpenAI-compatible API."""

import json
import time

from .translator import (
    openai_to_deepseek,
    deepseek_to_openai,
    deepseek_stream_to_openai,
    create_final_stream_chunk,
    generate_request_id,
    get_model_class,
)
from .streaming import StreamingResponseWriter, format_sse_event, format_sse_done
from .errors import invalid_request_error, server_error
from .deepseek_client import DeepSeekAPIError


AVAILABLE_MODELS = [
    {
        "id": "deepseek-chat",
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek",
        "permission": [],
        "root": "deepseek-chat",
        "parent": None,
    },
    {
        "id": "deepseek-coder",
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek",
        "permission": [],
        "root": "deepseek-coder",
        "parent": None,
    },
    {
        "id": "deepseek-reasoner",
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek",
        "permission": [],
        "root": "deepseek-reasoner",
        "parent": None,
    },
]


def handle_models():
    """Handle GET /v1/models - return available models list.

    Returns:
        tuple: (status_code, response_body_dict)
    """
    return 200, {
        "object": "list",
        "data": AVAILABLE_MODELS,
    }


def handle_chat_completions(request_body, deepseek_client, wfile=None):
    """Handle POST /v1/chat/completions.

    Args:
        request_body: parsed JSON request body (dict)
        deepseek_client: DeepSeekClient instance
        wfile: socket file for streaming responses (None for non-streaming)

    Returns:
        tuple: (status_code, response_body_dict) for non-streaming
        None for streaming (writes directly to wfile)
    """
    # Validate required fields
    if not isinstance(request_body, dict):
        return invalid_request_error("Request body must be a JSON object.")

    messages = request_body.get("messages")
    if not messages or not isinstance(messages, list):
        return invalid_request_error("'messages' is required and must be a non-empty array.", param="messages")

    # Check message format
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            return invalid_request_error("Each message must have a 'role' field.", param="messages")

    model_name = request_body.get("model", "deepseek-chat")
    stream = request_body.get("stream", False)
    request_id = generate_request_id()

    # Translate to DeepSeek format
    deepseek_request = openai_to_deepseek(request_body)
    model_class = deepseek_request["model_class"]

    temperature = deepseek_request.get("temperature", 1.0)
    max_tokens = deepseek_request.get("max_tokens")
    top_p = deepseek_request.get("top_p", 1.0)

    if stream and wfile is not None:
        # Streaming response
        try:
            writer = StreamingResponseWriter(wfile)
            is_first = True

            for chunk in deepseek_client.chat_completion_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                model_class=model_class,
            ):
                openai_chunk = deepseek_stream_to_openai(
                    chunk, model=model_name, request_id=request_id, is_first=is_first
                )
                writer.write_event(openai_chunk)
                is_first = False

            # Send final chunk with finish_reason
            final_chunk = create_final_stream_chunk(model=model_name, request_id=request_id)
            writer.write_event(final_chunk)
            writer.write_done()
        except DeepSeekAPIError as e:
            # For streaming errors, write an error event
            error_data = {
                "error": {
                    "message": f"DeepSeek API error: {e.message}",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            }
            writer.write_event(error_data)
            writer.write_done()
        except Exception as e:
            error_data = {
                "error": {
                    "message": f"Internal error: {str(e)}",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            }
            try:
                writer.write_event(error_data)
                writer.write_done()
            except Exception:
                pass
        return None  # Response already written
    else:
        # Non-streaming response
        try:
            deepseek_response = deepseek_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                model_class=model_class,
            )
            openai_response = deepseek_to_openai(
                deepseek_response, model=model_name, request_id=request_id
            )
            return 200, openai_response
        except DeepSeekAPIError as e:
            if e.status_code == 401:
                from .errors import authentication_error
                return authentication_error("DeepSeek token is invalid or expired.")
            elif e.status_code == 429:
                from .errors import rate_limit_error
                return rate_limit_error()
            else:
                return server_error(f"DeepSeek API error: {e.message}")
        except Exception as e:
            return server_error(f"Internal error: {str(e)}")


def handle_completions(request_body, deepseek_client, wfile=None):
    """Handle POST /v1/completions (legacy completions endpoint).

    Converts prompt-based format to messages and delegates to chat_completions.

    Args:
        request_body: parsed JSON request body
        deepseek_client: DeepSeekClient instance
        wfile: socket file for streaming

    Returns:
        tuple: (status_code, response_body_dict) or None for streaming
    """
    if not isinstance(request_body, dict):
        return invalid_request_error("Request body must be a JSON object.")

    prompt = request_body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(prompt)

    # Convert to chat format
    chat_request = {
        "model": request_body.get("model", "deepseek-chat"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": request_body.get("stream", False),
    }

    # Copy optional parameters
    for key in ("temperature", "max_tokens", "top_p"):
        if key in request_body:
            chat_request[key] = request_body[key]

    return handle_chat_completions(chat_request, deepseek_client, wfile=wfile)


def handle_embeddings(request_body):
    """Handle POST /v1/embeddings - not supported.

    Returns:
        tuple: (status_code, error_response_dict)
    """
    return invalid_request_error(
        "Embeddings are not supported by this gateway. DeepSeek web chat does not provide embedding functionality.",
        param="input",
    )


def handle_chat_completions_stream_preflight(request_body, deepseek_client):
    """Validate a streaming chat completions request before sending SSE headers.

    This performs request validation so that errors can be returned as proper
    HTTP error responses rather than being embedded in the SSE stream.

    Args:
        request_body: parsed JSON request body (dict)
        deepseek_client: DeepSeekClient instance

    Returns:
        tuple (status_code, error_body) if there is an error, or None if OK to proceed.
    """
    if not isinstance(request_body, dict):
        return invalid_request_error("Request body must be a JSON object.")

    messages = request_body.get("messages")
    if not messages or not isinstance(messages, list):
        return invalid_request_error("'messages' is required and must be a non-empty array.", param="messages")

    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            return invalid_request_error("Each message must have a 'role' field.", param="messages")

    return None


def handle_completions_stream_preflight(request_body, deepseek_client):
    """Validate a streaming legacy completions request before sending SSE headers.

    Args:
        request_body: parsed JSON request body (dict)
        deepseek_client: DeepSeekClient instance

    Returns:
        tuple (status_code, error_body) if there is an error, or None if OK to proceed.
    """
    if not isinstance(request_body, dict):
        return invalid_request_error("Request body must be a JSON object.")

    return None
