"""Email + password authentication for the Urdu summarizer (stdlib only).

Design constraints (fixed by Task #23):

* **No new dependencies** — only :mod:`sqlite3`, :func:`hashlib.pbkdf2_hmac`,
  :mod:`secrets` and :func:`secrets.compare_digest` from the standard library.
* **DB file** — ``webapp/data/accounts.db`` by default, overridable with the
  ``AUTH_DB_PATH`` environment variable (so a Hugging Face Space can point it at
  persistent storage and keep accounts across restarts).
* **Passwords** — a versioned ``pbkdf2_sha256$<iters>$<salt>$<hash>`` string
  (600_000 iterations for new accounts; older rows keep their stored count),
  compared with :func:`secrets.compare_digest`. The raw password is never stored
  or logged.
* **Sessions** — an opaque ``secrets.token_urlsafe(32)`` is returned to the
  caller; only its SHA-256 hex digest is stored. Expired rows are purged
  opportunistically on write.
* **Brute force** — a per-IP token bucket throttles signup (10/300s) and login
  (20/300s) BEFORE any PBKDF2 work; a hard 15-minute lockout is keyed by
  ``(ip, email)`` after 5 failures, and a soft per-email counter adds only
  incremental delay (never a hard reject).

A single :class:`threading.Lock` serialises writes; the app runs one worker, and
each call opens and closes its own connection on the calling thread (FastAPI runs
the sync endpoints on a thread pool), so connections are never shared across
threads.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'accounts.db'

# N3: versioned password hashing. New accounts use 600_000 iterations (measured
# ~0.79 s per hash on the dev machine, well under the ~2.5 s budget); rows written
# by older builds keep their stored count and stay verifiable.
PBKDF2_ITERATIONS = 600_000          # default for NEW hashes and the dummy equalizer
LEGACY_PBKDF2_ITERATIONS = 200_000   # pre-versioned rows (raw hex + salt column)
HASH_PREFIX = 'pbkdf2_sha256'
SALT_BYTES = 16

SESSION_COOKIE = 'session'
SESSION_TTL_DAYS = 7
SESSION_MAX_AGE = SESSION_TTL_DAYS * 24 * 3600  # seconds, for the cookie Max-Age

EMAIL_PATTERN = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
MAX_EMAIL_LEN = 254
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 200

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

# W1: soft per-email delay — incremental only, never a hard reject.
SOFT_DELAY_SECONDS = 0.5
SOFT_DELAY_MAX_STEPS = 4

# C1: per-IP token buckets, checked BEFORE any PBKDF2 work.
IP_BUDGETS = {'signup': (10, 300.0), 'login': (20, 300.0)}  # kind -> (limit, window_s)

# W2: bound the in-memory throttle maps.
_THROTTLE_CAP = 50_000

# Bilingual, matching the existing OOV-warning style ("English / Urdu").
LOCKOUT_MESSAGE = (
    f'Too many failed sign-in attempts — try again in {LOCKOUT_MINUTES} minutes. / '
    f'بہت زیادہ ناکام کوششیں — {LOCKOUT_MINUTES} منٹ بعد دوبارہ کوشش کریں۔'
)

_email_re = re.compile(EMAIL_PATTERN)
_write_lock = threading.Lock()
_schema_ready = False

# In-memory brute-force throttle.
#   _failures:        (ip, email) -> {'count', 'locked_until', 'last'}  HARD lockout (W1)
#   _email_soft:      email -> int   SOFT counter, incremental delay only (W1)
#   _email_soft_last: email -> datetime  (TTL sweeps, W2)
#   _ip_buckets:      (kind, ip) -> {'tokens', 'updated'}  C1 token bucket (monotonic)
_failures: dict[tuple, dict] = {}
_email_soft: dict[str, int] = {}
_email_soft_last: dict[str, datetime] = {}
_ip_buckets: dict[tuple, dict] = {}
_failures_lock = threading.Lock()
_dir_ready = False       # W8: cache the parent-dir mkdir (not a per-request syscall)
_dummy_cache = None      # N1: lazily-computed dummy hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    UNIQUE NOT NULL,
    password_hash TEXT    NOT NULL,
    salt          TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    created_at TEXT    NOT NULL,
    expires_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions (expires_at);
"""


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class DuplicateEmail(Exception):
    """Raised by :func:`create_user` when the address is already registered."""


class InvalidCredentials(Exception):
    """Generic authentication failure — never reveals which part was wrong."""


class AccountLocked(InvalidCredentials):
    """Too many consecutive failures; carries the bilingual retry-later message."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def db_path() -> Path:
    """Absolute path of the SQLite file (env ``AUTH_DB_PATH`` wins)."""
    raw = os.environ.get('AUTH_DB_PATH')
    return Path(raw).expanduser() if raw else _DEFAULT_DB_PATH


def cookie_secure() -> bool:
    """W3: ``Secure`` flag for the session cookie.

    True by default (HF Spaces / Vercel are HTTPS). Set ``AUTH_COOKIE_INSECURE=1``
    to drop it for plain-HTTP local development; :mod:`run` sets that automatically
    when binding to 127.0.0.1/localhost.
    """
    return os.environ.get('AUTH_COOKIE_INSECURE', '').strip().lower() not in {'1', 'true', 'yes', 'on'}


def normalize_email(email: str) -> str:
    return (email or '').strip().lower()


def validate_email(email: str) -> str:
    email = normalize_email(email)
    if len(email) > MAX_EMAIL_LEN or not _email_re.match(email):
        raise ValueError('Enter a valid email address. / ایک درست ای میل پتہ درج کریں۔')
    return email


def validate_password(password: str) -> str:
    if not (MIN_PASSWORD_LEN <= len(password or '') <= MAX_PASSWORD_LEN):
        raise ValueError(
            f'Password must be {MIN_PASSWORD_LEN}-{MAX_PASSWORD_LEN} characters. / '
            f'پاس ورڈ {MIN_PASSWORD_LEN} سے {MAX_PASSWORD_LEN} حروف کا ہونا چاہیے۔'
        )
    return password


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    """Fixed-length UTC stamp so lexicographic comparison == chronological."""
    return moment.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _parse_iso(stamp: str) -> datetime:
    return datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _derive(password: str, salt_hex: str, iterations: int) -> str:
    """Raw PBKDF2-SHA256 hex digest for ``password`` at ``iterations``."""
    salt = bytes.fromhex(salt_hex)
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations).hex()


def _encode_password(password: str, salt_hex: str | None = None,
                     iterations: int | None = None) -> tuple[str, str]:
    """Return ``(salt_hex, versioned_hash)``; the hash is
    ``pbkdf2_sha256$<iters>$<salt_hex>$<hash_hex>`` (N3)."""
    salt_hex = salt_hex or secrets.token_hex(SALT_BYTES)
    iterations = iterations or PBKDF2_ITERATIONS
    return salt_hex, f'{HASH_PREFIX}${iterations}${salt_hex}${_derive(password, salt_hex, iterations)}'


def _verify_password(password: str, stored: str, legacy_salt: str) -> bool:
    """Verify against a versioned hash, falling back to legacy raw-hex rows.

    New rows store ``pbkdf2_sha256$<iters>$<salt>$<hash>`` in ``password_hash``;
    pre-versioned rows store the raw hex digest there and keep the salt in the
    separate ``salt`` column at :data:`LEGACY_PBKDF2_ITERATIONS`.
    """
    if stored.startswith(HASH_PREFIX + '$'):
        parts = stored.split('$')
        if len(parts) != 4:
            return False
        _, iters_raw, salt_hex, expected = parts
        try:
            iterations = int(iters_raw)
        except ValueError:
            return False
        return secrets.compare_digest(_derive(password, salt_hex, iterations), expected)
    # Legacy row: raw hex digest + separate salt column.
    return secrets.compare_digest(_derive(password, legacy_salt, LEGACY_PBKDF2_ITERATIONS), stored)


def _dummy_hash() -> tuple[str, str]:
    """Lazily build a fixed dummy ``(salt, versioned_hash)`` so a login/signup for
    an unknown-or-duplicate address costs the same PBKDF2 work as a real one —
    closing the timing side-channel. Computed on first use, not at import (N1)."""
    global _dummy_cache
    if _dummy_cache is None:
        _dummy_cache = _encode_password('timing-equalizer-not-a-real-password')
    return _dummy_cache


def equalize_timing(password: str) -> None:
    """Burn one PBKDF2 at the current default work factor (W4/N1)."""
    salt_hex, stored = _dummy_hash()
    _verify_password(password or '', stored, salt_hex)


def _warn_if_db_in_model_dir() -> None:
    """N2: warn once if the default DB would live inside UATS_MODEL_DIR (ephemeral
    on HF Spaces) while AUTH_DB_PATH is unset."""
    if os.environ.get('AUTH_DB_PATH'):
        return
    model_dir = os.environ.get('UATS_MODEL_DIR')
    if not model_dir:
        return
    try:
        resolved_db = db_path().resolve()
        resolved_md = Path(model_dir).expanduser().resolve()
    except OSError:
        return
    if resolved_md in resolved_db.parents:
        print(f'[auth] WARNING: the accounts DB defaults to {resolved_db}, inside '
              f'UATS_MODEL_DIR ({resolved_md}); set AUTH_DB_PATH to persistent '
              f'storage or all accounts are lost on every restart.', flush=True)


def _connect() -> sqlite3.Connection:
    """Open a per-call connection, creating the schema once per process."""
    global _schema_ready, _dir_ready
    path = db_path()
    if not _dir_ready:                       # W8: mkdir once, not per request
        path.parent.mkdir(parents=True, exist_ok=True)
        _dir_ready = True
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    if not _schema_ready:
        with _write_lock:
            if not _schema_ready:
                conn.executescript(_SCHEMA)
                conn.commit()
                _schema_ready = True
                _warn_if_db_in_model_dir()   # N2: one-line startup warning
    return conn


# --------------------------------------------------------------------------- #
# Brute-force throttle (in-memory).
#   * C1  per-IP token bucket, checked BEFORE any PBKDF2 work.
#   * W1  HARD 15-minute lockout keyed by (ip, email); SOFT per-email counter
#         that only adds incremental delay, never a hard reject.
#   * W2  both maps are TTL-swept and capped at _THROTTLE_CAP entries.
# --------------------------------------------------------------------------- #
def _sweep_locked(now: datetime) -> None:
    """Drop stale throttle records; called with ``_failures_lock`` held (W2)."""
    if len(_failures) <= _THROTTLE_CAP and len(_email_soft) <= _THROTTLE_CAP:
        return
    cutoff = now - timedelta(minutes=LOCKOUT_MINUTES)
    for key in [k for k, v in _failures.items() if v.get('last', now) < cutoff]:
        _failures.pop(key, None)
    for em in [e for e, ts in _email_soft_last.items() if ts < cutoff]:
        _email_soft.pop(em, None)
        _email_soft_last.pop(em, None)


def check_ip_budget(ip: str, kind: str) -> tuple[bool, int]:
    """C1 token bucket. Return ``(allowed, retry_after_seconds)``.

    ``kind`` is ``'signup'`` (10/300s) or ``'login'`` (20/300s). The bucket refills
    continuously; a denied call reports how long until one token is available.
    """
    limit, window = IP_BUDGETS[kind]
    rate = limit / window                      # tokens per second
    now = time.monotonic()
    key = (kind, ip or '?')
    with _failures_lock:
        # W2: bound the bucket map (monotonic TTL sweep when over the cap).
        if len(_ip_buckets) > _THROTTLE_CAP:
            for k in [k for k, v in _ip_buckets.items() if now - v['updated'] > window * 2]:
                _ip_buckets.pop(k, None)
        rec = _ip_buckets.get(key)
        if rec is None:
            rec = {'tokens': float(limit), 'updated': now}
            _ip_buckets[key] = rec
        rec['tokens'] = min(float(limit), rec['tokens'] + (now - rec['updated']) * rate)
        rec['updated'] = now
        if rec['tokens'] >= 1.0:
            rec['tokens'] -= 1.0
            return True, 0
        return False, max(1, math.ceil((1.0 - rec['tokens']) / rate))


def _soft_delay(email: str) -> float:
    """W1: incremental per-email delay (0.5s * min(count, 4)); never a reject."""
    with _failures_lock:
        count = _email_soft.get(email, 0)
    return SOFT_DELAY_SECONDS * min(count, SOFT_DELAY_MAX_STEPS)


def _register_failure(ip: str, email: str) -> None:
    now = _now()
    with _failures_lock:
        key = (ip or '?', email)
        record = _failures.setdefault(key, {'count': 0, 'locked_until': None, 'last': now})
        record['count'] += 1
        record['last'] = now
        if record['count'] >= MAX_FAILED_ATTEMPTS:
            record['locked_until'] = now + timedelta(minutes=LOCKOUT_MINUTES)
            record['count'] = 0
        _email_soft[email] = _email_soft.get(email, 0) + 1
        _email_soft_last[email] = now
        _sweep_locked(now)                     # W2: opportunistic purge


def _reset_failures(ip: str, email: str) -> None:
    with _failures_lock:
        _failures.pop((ip or '?', email), None)
        _email_soft.pop(email, None)
        _email_soft_last.pop(email, None)


def _locked_out(ip: str, email: str) -> bool:
    with _failures_lock:
        record = _failures.get((ip or '?', email))
        if not record or not record['locked_until']:
            return False
        if _now() >= record['locked_until']:
            _failures.pop((ip or '?', email), None)  # lockout window elapsed
            return False
        return True


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
def create_user(email: str, password: str) -> int:
    """Insert a new user and return its id. Raises :class:`DuplicateEmail`.

    A cheap existence probe runs BEFORE the PBKDF2 so the duplicate path burns no
    hash here; :func:`equalize_timing` is then called by the API layer (W4) so a
    duplicate signup costs exactly one PBKDF2 — the same as a real one. The UNIQUE
    constraint remains the authoritative guard against a probe/insert race.
    """
    email = validate_email(email)
    validate_password(password)

    conn = _connect()
    try:
        if conn.execute('SELECT 1 FROM users WHERE email = ?', (email,)).fetchone():
            raise DuplicateEmail(email)
        salt_hex, password_hash = _encode_password(password)   # N3 versioned format
        created = _iso(_now())
        with _write_lock:
            try:
                cur = conn.execute(
                    'INSERT INTO users (email, password_hash, salt, created_at) VALUES (?, ?, ?, ?)',
                    (email, password_hash, salt_hex, created),
                )
                conn.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                conn.rollback()
                raise DuplicateEmail(email) from None
    finally:
        conn.close()


def authenticate(email: str, password: str, ip: str | None = None) -> int:
    """Return the user id on success.

    ``ip`` keys the HARD lockout (W1). Raises :class:`AccountLocked` while
    throttled and :class:`InvalidCredentials` otherwise; both map to a generic 401
    at the API layer. A soft per-email counter adds incremental delay on failures.
    """
    email = normalize_email(email)
    if _locked_out(ip, email):
        raise AccountLocked(LOCKOUT_MESSAGE)

    delay = _soft_delay(email)                 # W1: incremental, never a reject

    conn = _connect()
    try:
        row = conn.execute(
            'SELECT id, password_hash, salt FROM users WHERE email = ?', (email,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        # Equalise timing, count the failure, then fail generically.
        equalize_timing(password or '')
        _register_failure(ip, email)
        if delay:
            time.sleep(delay)
        raise InvalidCredentials()

    if _verify_password(password or '', row['password_hash'], row['salt']):
        _reset_failures(ip, email)
        return int(row['id'])

    _register_failure(ip, email)
    if delay:
        time.sleep(delay)
    if _locked_out(ip, email):
        raise AccountLocked(LOCKOUT_MESSAGE)
    raise InvalidCredentials()


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def _purge_expired(conn: sqlite3.Connection, now: datetime) -> None:
    conn.execute('DELETE FROM sessions WHERE expires_at <= ?', (_iso(now),))


def create_session(user_id: int) -> str:
    """Create a session for ``user_id`` and return the raw opaque token."""
    token = secrets.token_urlsafe(32)
    token_hash = _sha256_hex(token)
    now = _now()
    expires = now + timedelta(days=SESSION_TTL_DAYS)

    conn = _connect()
    try:
        with _write_lock:
            _purge_expired(conn, now)  # opportunistic cleanup
            conn.execute(
                'INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)',
                (token_hash, int(user_id), _iso(now), _iso(expires)),
            )
            conn.commit()
    finally:
        conn.close()
    return token


def get_user_by_session(token: str | None) -> dict | None:
    """Return ``{'user_id', 'email'}`` for a valid token, else ``None``.

    Expired sessions are deleted on sight and reported as invalid.
    """
    if not token:
        return None
    token_hash = _sha256_hex(token)
    now = _now()

    conn = _connect()
    try:
        row = conn.execute(
            'SELECT s.user_id AS user_id, s.expires_at AS expires_at, u.email AS email '
            'FROM sessions s JOIN users u ON u.id = s.user_id '
            'WHERE s.token_hash = ?',
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        if _parse_iso(row['expires_at']) <= now:
            with _write_lock:
                conn.execute('DELETE FROM sessions WHERE token_hash = ?', (token_hash,))
                conn.commit()
            return None
        return {'user_id': int(row['user_id']), 'email': row['email']}
    finally:
        conn.close()


def delete_session(token: str | None) -> None:
    """Remove a session row (logout). No-op for a missing/empty token."""
    if not token:
        return
    token_hash = _sha256_hex(token)
    conn = _connect()
    try:
        with _write_lock:
            conn.execute('DELETE FROM sessions WHERE token_hash = ?', (token_hash,))
            conn.commit()
    finally:
        conn.close()
