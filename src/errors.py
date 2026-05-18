"""OpenAI-compatible error response builder."""

import json


def create_error_response(message, error_type="invalid_request_error", param=None, code=None, status_code=400):
    """Create an OpenAI-format error response.

    Returns a tuple of (status_code, response_body_dict).
    """
    body = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    return status_code, body


def invalid_request_error(message, param=None):
    """Create an invalid request error."""
    return create_error_response(message, "invalid_request_error", param=param, status_code=400)


def authentication_error(message="Invalid API key or token."):
    """Create an authentication error."""
    return create_error_response(message, "authentication_error", status_code=401)


def rate_limit_error(message="Rate limit exceeded. Please try again later."):
    """Create a rate limit error."""
    return create_error_response(message, "rate_limit_error", status_code=429)


def server_error(message="An internal server error occurred."):
    """Create a server error."""
    return create_error_response(message, "server_error", status_code=500)


def not_found_error(message="The requested resource was not found."):
    """Create a not found error."""
    return create_error_response(message, "not_found_error", status_code=404)


def format_error_json(status_code, body):
    """Format error body as JSON string."""
    return json.dumps(body)
