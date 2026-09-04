"""Vercel Python serverless proxy — GET /health.

Forwards the request to the persistent backend (Hugging Face Space) named by the
BACKEND_URL environment variable and relays its status, body, content type and
any ``Set-Cookie`` headers. The incoming ``Cookie`` and ``X-Forwarded-For``
headers are forwarded so the backend can recognise an authenticated session and
attribute its per-IP rate limit to the real client.

Hardening (Task #28):
* Content-Length is parsed defensively, clamped to ``0..1_000_000``, and a
  ``Transfer-Encoding: chunked`` body is rejected with a clean 411 (W5).
* The upstream call is bounded by a monotonic 50s total deadline — connect
  ``min(10, remaining)``, read ``remaining`` — so the 502 handler can run before
  Vercel's 60s kill (W6).
* Any upstream failure returns a constant 502 ``{"detail": "Backend unavailable."}``
  and logs only the exception class name to stderr (N4).

Vercel's Python runtime detects a top-level ``handler`` that subclasses
``http.server.BaseHTTPRequestHandler`` and dispatches to ``do_GET`` / ``do_POST``.
Only the standard library is used so the function needs no extra dependencies.
"""

import http.client
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

BACKEND_URL = os.environ.get("BACKEND_URL")
DEADLINE = 50.0        # total seconds; keep below the 60s maxDuration in vercel.json
CONNECT_CAP = 10.0     # max seconds for the TCP/TLS connect phase
MAX_BODY = 1_000_000   # clamp a declared Content-Length (W5)
BACKEND_UNAVAILABLE = json.dumps({"detail": "Backend unavailable."}).encode()


def _forward(url, method="GET", body=None, headers=None):
    """Proxy ``url`` within a monotonic 50s deadline (W6).

    ``headers`` is a dict of extra request headers (Cookie, X-Forwarded-For).
    Returns ``(status, payload_bytes, ctype, set_cookie_list)``.
    """
    parts = urlsplit(url)
    host = parts.hostname
    port = parts.port
    path = parts.path or "/"
    if parts.query:
        path = path + "?" + parts.query
    deadline = time.monotonic() + DEADLINE
    conn = None
    try:
        send_headers = {"Accept": "application/json", "Connection": "close"}
        if body is not None:
            send_headers["Content-Type"] = "application/json"
            send_headers["Content-Length"] = str(len(body))
        for key, value in (headers or {}).items():
            if value:
                send_headers[key] = value  # C1b: carry Cookie + X-Forwarded-For
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("deadline exceeded before connect")
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(host, port or 443,
                                               timeout=min(CONNECT_CAP, remaining))
        else:
            conn = http.client.HTTPConnection(host, port or 80,
                                              timeout=min(CONNECT_CAP, remaining))
        conn.request(method, path, body=body, headers=send_headers)
        # W6: bound the read phase by whatever is left of the total deadline.
        remaining = deadline - time.monotonic()
        if conn.sock is not None and remaining > 0:
            conn.sock.settimeout(remaining)
        resp = conn.getresponse()
        payload = resp.read()
        ctype = resp.getheader("Content-Type", "application/json")
        set_cookies = resp.msg.get_all("Set-Cookie") or []
        return resp.status, payload, ctype, set_cookies
    except Exception as err:  # noqa: BLE001 - relay any failure as a clean 502 JSON
        # N4: constant detail to the client; only the class name goes to stderr.
        sys.stderr.write("proxy %s %s -> backend failed: %s\n"
                         % (method, path, err.__class__.__name__))
        return 502, BACKEND_UNAVAILABLE, "application/json", []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, ctype, set_cookies=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        for cookie in (set_cookies or []):
            self.send_header("Set-Cookie", cookie)  # relay the session cookie back
        self.end_headers()
        self.wfile.write(payload)

    def _relay_headers(self):
        """C1b: forward the browser Cookie + the real-client X-Forwarded-For."""
        headers = {}
        cookie = self.headers.get("Cookie")
        if cookie:
            headers["Cookie"] = cookie
        xff = self.headers.get("X-Forwarded-For")
        if xff:
            headers["X-Forwarded-For"] = xff
        return headers

    def _read_body(self):
        """W5: guarded Content-Length parse + clamp; ``chunked`` -> clean 411."""
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            payload = json.dumps({"detail": "chunked bodies unsupported"}).encode()
            self._send(411, payload, "application/json")
            return None, False
        raw = self.headers.get("Content-Length", "0")
        try:
            declared = int(raw)
        except (TypeError, ValueError):
            declared = 0
        length = max(0, min(declared, MAX_BODY))
        if declared > MAX_BODY:
            self.close_connection = True  # leftover bytes would desync keep-alive
        body = self.rfile.read(length) if length else None
        return body, True

    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        if not BACKEND_URL:
            payload = json.dumps(
                {"detail": "BACKEND_URL is not configured on this deployment."}
            ).encode()
            return self._send(503, payload, "application/json")
        _body, ok = self._read_body()  # W5 guard (a GET forwards no body)
        if not ok:
            return
        status, payload, ctype, set_cookies = _forward(
            BACKEND_URL.rstrip("/") + "/health", "GET", None, self._relay_headers()
        )
        self._send(status, payload, ctype, set_cookies)
