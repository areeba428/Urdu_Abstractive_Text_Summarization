"""Vercel Python serverless proxy — POST /auth/logout.

Forwards the incoming ``Cookie`` header to the backend named by BACKEND_URL so it
can delete the session row, and relays the backend's expiring ``Set-Cookie``
(``Max-Age=0``) back to the browser. No request body is required.

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
            self.send_header("Set-Cookie", cookie)  # relay the expiring cookie back
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
            BACKEND_URL.rstrip("/") + "/auth/logout", "POST", body, cookie
        )
        self._send(status, payload, ctype, set_cookies)
