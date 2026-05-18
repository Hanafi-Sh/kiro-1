"""DeepSeek web chat API client using urllib."""

import json
import urllib.request
import urllib.error


class DeepSeekClient:
    """Client for the DeepSeek web chat API."""

    def __init__(self, token, base_url="https://chat.deepseek.com", timeout=30):
        """Initialize the client.

        Args:
            token: Bearer token from DeepSeek browser session
            base_url: DeepSeek API base URL
            timeout: Request timeout in seconds (default 30)
        """
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v0/chat/completions"
        self.timeout = timeout

    def _build_request(self, payload):
        """Build a urllib Request object for the DeepSeek API.

        Args:
            payload: dict to send as JSON body

        Returns:
            urllib.request.Request object
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")
        return req

    def chat_completion(self, messages, temperature=1.0, max_tokens=None, top_p=1.0, model_class="deepseek_chat"):
        """Send a non-streaming chat completion request.

        Args:
            messages: list of message dicts with role/content
            temperature: sampling temperature
            max_tokens: max tokens to generate
            top_p: nucleus sampling parameter
            model_class: DeepSeek model class (deepseek_chat, deepseek_code, deepseek_reasoner)

        Returns:
            dict: parsed JSON response from DeepSeek

        Raises:
            DeepSeekAPIError: on API errors
        """
        payload = {
            "messages": messages,
            "model_class": model_class,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = self._build_request(payload)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise DeepSeekAPIError(e.code, error_body) from e
        except urllib.error.URLError as e:
            raise DeepSeekAPIError(503, str(e.reason)) from e
        except Exception as e:
            raise DeepSeekAPIError(500, str(e)) from e

    def chat_completion_stream(self, messages, temperature=1.0, max_tokens=None, top_p=1.0, model_class="deepseek_chat"):
        """Send a streaming chat completion request.

        Args:
            messages: list of message dicts with role/content
            temperature: sampling temperature
            max_tokens: max tokens to generate
            top_p: nucleus sampling parameter
            model_class: DeepSeek model class

        Yields:
            dict: parsed JSON chunks from SSE stream

        Raises:
            DeepSeekAPIError: on API errors
        """
        payload = {
            "messages": messages,
            "model_class": model_class,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = self._build_request(payload)
        req.add_header("Accept", "text/event-stream")

        try:
            response = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise DeepSeekAPIError(e.code, error_body) from e
        except urllib.error.URLError as e:
            raise DeepSeekAPIError(503, str(e.reason)) from e
        except Exception as e:
            raise DeepSeekAPIError(500, str(e)) from e

        # Set socket-level timeout for streaming reads to prevent indefinite blocking
        if hasattr(response, 'fp') and hasattr(response.fp, 'raw'):
            try:
                sock = response.fp.raw._sock if hasattr(response.fp.raw, '_sock') else None
                if sock is not None:
                    sock.settimeout(self.timeout)
            except (AttributeError, OSError):
                pass
        elif hasattr(response, 'fp') and hasattr(response.fp, '_sock'):
            try:
                response.fp._sock.settimeout(self.timeout)
            except (AttributeError, OSError):
                pass

        try:
            for line in response:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()


class DeepSeekAPIError(Exception):
    """Error from the DeepSeek API."""

    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"DeepSeek API error {status_code}: {message}")
