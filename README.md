# DeepSeek4Free Gateway

An OpenAI-compatible API gateway that proxies requests to DeepSeek's free web chat interface. Zero external dependencies - uses only Python standard library.

## Features

- 100% OpenAI-compatible API endpoints
- Proxies to DeepSeek's free web chat (chat.deepseek.com)
- Server-Sent Events (SSE) streaming support
- Concurrent request handling (ThreadingHTTPServer)
- Model mapping (GPT model names map to DeepSeek equivalents)
- Proper OpenAI-format error responses
- CORS support for browser-based clients
- Zero dependencies - Python standard library only

## Quick Start

### 1. Get your DeepSeek token

1. Go to [chat.deepseek.com](https://chat.deepseek.com) and log in
2. Open browser Developer Tools (F12)
3. Go to Application > Cookies or Network tab
4. Find the Bearer token in request headers (look for `Authorization: Bearer ...`)
5. Copy the token value

### 2. Run the gateway

```bash
# Set token via environment variable
export DEEPSEEK_TOKEN="your-token-here"
python3 gateway.py

# Or pass token as argument
python3 gateway.py --token "your-token-here"

# Custom host and port
python3 gateway.py --token "your-token" --host 127.0.0.1 --port 3000
```

### 3. Use with any OpenAI-compatible client

```bash
# Point your client to the gateway
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=dummy  # Not validated by gateway
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming and non-streaming) |
| POST | `/v1/completions` | Legacy text completions |
| GET | `/v1/models` | List available models |
| POST | `/v1/embeddings` | Not supported (returns error) |
| GET | `/health` | Health check |

## Example Usage

### Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

### Legacy Completions

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "prompt": "Once upon a time"
  }'
```

## Available Models

| Model Name | DeepSeek Class | Description |
|-----------|---------------|-------------|
| `deepseek-chat` | deepseek_chat | General chat model |
| `deepseek-coder` | deepseek_code | Code-focused model |
| `deepseek-reasoner` | deepseek_reasoner | Reasoning model |
| `gpt-3.5-turbo` | deepseek_chat | Mapped to DeepSeek chat |
| `gpt-4` | deepseek_chat | Mapped to DeepSeek chat |
| `gpt-4o` | deepseek_chat | Mapped to DeepSeek chat |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DEEPSEEK_TOKEN` | (required) | Bearer token from DeepSeek browser session |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port |
| `DEEPSEEK_BASE_URL` | `https://chat.deepseek.com` | DeepSeek API base URL |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

Command-line arguments override environment variables:

```
python3 gateway.py --help
```

## Project Structure

```
gateway.py              - Main entry point
src/
  __init__.py           - Package init
  config.py             - Configuration management
  errors.py             - OpenAI error format helpers
  translator.py         - Request/response translation
  deepseek_client.py    - DeepSeek web chat API client
  streaming.py          - SSE streaming utilities
  handlers.py           - Endpoint handlers
  server.py             - HTTP server with routing
tests/
  __init__.py           - Test package init
  test_translator.py    - Translation tests
  test_errors.py        - Error format tests
  test_handlers.py      - Handler tests
  test_server.py        - Server integration tests
```

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

## Limitations

- Token usage values are approximate estimates (based on character count / 4)
- Embeddings endpoint is not supported (DeepSeek web chat does not offer this)
- The gateway depends on DeepSeek's web chat API which may change without notice
- Rate limiting depends on DeepSeek's policies for the web chat interface
- Token must be refreshed when the browser session expires
