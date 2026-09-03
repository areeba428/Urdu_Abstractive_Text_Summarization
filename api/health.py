"""Vercel Python serverless proxy — GET /health.

Forwards the request to the persistent backend (Hugging Face Space) named by the
BACKEND_URL environment variable and relays its status, body and content type.

Vercel's Python runtime detects a top-level ``handler`` that subclasses
``http.server.BaseHTTPRequestHandler`` and dispatches to ``do_GET`` / ``do_POST``.
Only the standard library is used so the function needs no extra dependencies.
"""

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

BACKEND_URL = os.environ.get("BACKEND_URL")
TIMEOUT = 55  # seconds; keep below the 60s maxDuration configured in vercel.json


def _forward(url, method="GET", body=None):
    """Proxy ``url`` with ``method`` and return ``(status, payload_bytes, ctype)``."""
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as err:
        return err.code, err.read(), err.headers.get("Content-Type", "application/json")
    except Exception as err:  # noqa: BLE001 - relay any failure as a clean 502 JSON
        payload = json.dumps(
            {"detail": "Backend request failed: %s" % err.__class__.__name__}
        ).encode()
        return 502, payload, "application/json"


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        if not BACKEND_URL:
            payload = json.dumps(
                {"detail": "BACKEND_URL is not configured on this deployment."}
            ).encode()
            return self._send(503, payload, "application/json")
        status, payload, ctype = _forward(BACKEND_URL.rstrip("/") + "/health", "GET")
        self._send(status, payload, ctype)
