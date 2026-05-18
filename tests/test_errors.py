"""Unit tests for error response formatting."""

import json
import unittest

from src.errors import (
    create_error_response,
    invalid_request_error,
    authentication_error,
    rate_limit_error,
    server_error,
    not_found_error,
    format_error_json,
)


class TestCreateErrorResponse(unittest.TestCase):
    """Tests for the base error response creator."""

    def test_basic_error_format(self):
        status_code, body = create_error_response("Something went wrong")
        self.assertEqual(status_code, 400)
        self.assertIn("error", body)
        self.assertEqual(body["error"]["message"], "Something went wrong")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertIsNone(body["error"]["param"])
        self.assertIsNone(body["error"]["code"])

    def test_custom_error_type(self):
        status_code, body = create_error_response(
            "Auth failed", error_type="authentication_error", status_code=401
        )
        self.assertEqual(status_code, 401)
        self.assertEqual(body["error"]["type"], "authentication_error")

    def test_with_param(self):
        status_code, body = create_error_response(
            "Invalid value", param="temperature"
        )
        self.assertEqual(body["error"]["param"], "temperature")

    def test_with_code(self):
        status_code, body = create_error_response(
            "Rate limited", error_type="rate_limit_error", code="rate_limit_exceeded", status_code=429
        )
        self.assertEqual(body["error"]["code"], "rate_limit_exceeded")

    def test_json_serializable(self):
        _, body = create_error_response("test")
        # Should not raise
        json_str = json.dumps(body)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["error"]["message"], "test")


class TestInvalidRequestError(unittest.TestCase):
    """Tests for invalid_request_error helper."""

    def test_default_message(self):
        status_code, body = invalid_request_error("Bad request")
        self.assertEqual(status_code, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["message"], "Bad request")

    def test_with_param(self):
        status_code, body = invalid_request_error("Invalid", param="model")
        self.assertEqual(body["error"]["param"], "model")


class TestAuthenticationError(unittest.TestCase):
    """Tests for authentication_error helper."""

    def test_default_message(self):
        status_code, body = authentication_error()
        self.assertEqual(status_code, 401)
        self.assertEqual(body["error"]["type"], "authentication_error")
        self.assertIn("Invalid", body["error"]["message"])

    def test_custom_message(self):
        status_code, body = authentication_error("Token expired")
        self.assertEqual(body["error"]["message"], "Token expired")


class TestRateLimitError(unittest.TestCase):
    """Tests for rate_limit_error helper."""

    def test_default_message(self):
        status_code, body = rate_limit_error()
        self.assertEqual(status_code, 429)
        self.assertEqual(body["error"]["type"], "rate_limit_error")


class TestServerError(unittest.TestCase):
    """Tests for server_error helper."""

    def test_default_message(self):
        status_code, body = server_error()
        self.assertEqual(status_code, 500)
        self.assertEqual(body["error"]["type"], "server_error")

    def test_custom_message(self):
        status_code, body = server_error("Database connection failed")
        self.assertEqual(body["error"]["message"], "Database connection failed")


class TestNotFoundError(unittest.TestCase):
    """Tests for not_found_error helper."""

    def test_default_message(self):
        status_code, body = not_found_error()
        self.assertEqual(status_code, 404)
        self.assertEqual(body["error"]["type"], "not_found_error")


class TestFormatErrorJson(unittest.TestCase):
    """Tests for JSON formatting."""

    def test_format_produces_valid_json(self):
        _, body = create_error_response("test error")
        json_str = format_error_json(400, body)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["error"]["message"], "test error")

    def test_special_characters(self):
        _, body = create_error_response('Error with "quotes" and \\ backslash')
        json_str = format_error_json(400, body)
        parsed = json.loads(json_str)
        self.assertIn('"quotes"', parsed["error"]["message"])


if __name__ == "__main__":
    unittest.main()
