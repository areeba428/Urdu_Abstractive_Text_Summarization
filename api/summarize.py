"""Vercel Python serverless proxy — POST /summarize.

Reads the JSON request body and forwards it to the persistent backend (Hugging
Face Space) named by the BACKEND_URL environment variable, relaying the backend's
status, body, content type and any ``Set-Cookie`` headers. The incoming ``Cookie``
header is forwarded so the backend can enforce the authenticated session that now
guards ``/summarize``. Beam decoding can take several seconds on CPU, so this
function is configured with ``maxDuration: 60`` in vercel.json and uses a 55s
client timeout.

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


def _forward(url, method="GET", body=None, cookie=None):
    """Proxy ``url``; return ``(status, payload_bytes, ctype, set_cookie_list)``."""
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    if cookie:
        req.add_header("Cookie", cookie)  # carry the session to the backend
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return (resp.status, resp.read(),
                    resp.headers.get("Content-Type", "application/json"),
                    resp.headers.get_all("Set-Cookie") or [])
    except urllib.error.HTTPError as err:
        return (err.code, err.read(),
                err.headers.get("Content-Type", "application/json"),
                err.headers.get_all("Set-Cookie") or [])
    except Exception as err:  # noqa: BLE001 - relay any failure as a clean 502 JSON
        payload = json.dumps(
            {"detail": "Backend request failed: %s" % err.__class__.__name__}
        ).encode()
        return 502, payload, "application/json", []


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, ctype, set_cookies=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        for cookie in (set_cookies or []):
            self.send_header("Set-Cookie", cookie)  # relay the session cookie back
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        if not BACKEND_URL:
            payload = json.dumps(
                {"detail": "BACKEND_URL is not configured on this deployment."}
            ).encode()
            return self._send(503, payload, "application/json")

        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        cookie = self.headers.get("Cookie")

        status, payload, ctype, set_cookies = _forward(
            BACKEND_URL.rstrip("/") + "/summarize", "POST", body, cookie
        )
        self._send(status, payload, ctype, set_cookies)
