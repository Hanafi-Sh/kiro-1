"""SSE (Server-Sent Events) streaming utilities."""

import json


def format_sse_event(data):
    """Format a dict as an SSE data event.

    Args:
        data: dict to serialize as JSON in the SSE event

    Returns:
        str: formatted SSE event string "data: {json}\n\n"
    """
    return f"data: {json.dumps(data)}\n\n"


def format_sse_done():
    """Return the SSE done marker.

    Returns:
        str: "data: [DONE]\n\n"
    """
    return "data: [DONE]\n\n"


class StreamingResponseWriter:
    """Writer that sends SSE events over a socket wfile with immediate flushing."""

    def __init__(self, wfile):
        """Initialize with the response wfile (socket file object).

        Args:
            wfile: socket file object to write to
        """
        self.wfile = wfile

    def write_event(self, data):
        """Write an SSE event and flush immediately.

        Args:
            data: dict to send as SSE event
        """
        event = format_sse_event(data)
        self.wfile.write(event.encode("utf-8"))
        self.wfile.flush()

    def write_done(self):
        """Write the [DONE] event and flush."""
        done = format_sse_done()
        self.wfile.write(done.encode("utf-8"))
        self.wfile.flush()

    def write_raw(self, text):
        """Write raw text and flush.

        Args:
            text: raw string to write
        """
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()
