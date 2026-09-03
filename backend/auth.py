"""Email + password authentication for the Urdu summarizer (stdlib only).

Design constraints (fixed by Task #23):

* **No new dependencies** — only :mod:`sqlite3`, :func:`hashlib.pbkdf2_hmac`,
  :mod:`secrets` and :mod:`hmac` from the standard library.
* **DB file** — ``webapp/data/accounts.db`` by default, overridable with the
  ``AUTH_DB_PATH`` environment variable (so a Hugging Face Space can point it at
  persistent storage and keep accounts across restarts).
* **Passwords** — ``pbkdf2_hmac('sha256', password, salt, 200_000)``, compared
  with :func:`secrets.compare_digest`. The raw password is never stored or logged.
* **Sessions** — an opaque ``secrets.token_urlsafe(32)`` is returned to the
  caller; only its SHA-256 hex digest is stored. Expired rows are purged
  opportunistically on write.
* **Brute force** — an in-memory per-email counter locks an address out for 15
  minutes after 5 consecutive failures.

A single :class:`threading.Lock` serialises writes; the app runs one worker, and
each call opens and closes its own connection on the calling thread (FastAPI runs
the sync endpoints on a thread pool), so connections are never shared across
threads.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'accounts.db'

PBKDF2_ITERATIONS = 200_000
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

# Bilingual, matching the existing OOV-warning style ("English / Urdu").
LOCKOUT_MESSAGE = (
    f'Too many failed sign-in attempts — try again in {LOCKOUT_MINUTES} minutes. / '
    f'بہت زیادہ ناکام کوششیں — {LOCKOUT_MINUTES} منٹ بعد دوبارہ کوشش کریں۔'
)

_email_re = re.compile(EMAIL_PATTERN)
_write_lock = threading.Lock()
_schema_ready = False

# In-memory brute-force throttle: email -> {'count': int, 'locked_until': datetime|None}
_failures: dict[str, dict] = {}
_failures_lock = threading.Lock()

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


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    derived = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PBKDF2_ITERATIONS)
    return derived.hex()


def _verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    return secrets.compare_digest(_hash_password(password, salt_hex), expected_hash)


# A fixed dummy credential so a login for an unknown address costs the same
# PBKDF2 work as a known one — closing the timing side-channel that would
# otherwise reveal whether an email is registered.
_DUMMY_SALT = secrets.token_hex(SALT_BYTES)
_DUMMY_HASH = _hash_password('timing-equalizer-not-a-real-password', _DUMMY_SALT)


def _connect() -> sqlite3.Connection:
    """Open a per-call connection, creating the schema once per process."""
    global _schema_ready
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    if not _schema_ready:
        with _write_lock:
            if not _schema_ready:
                conn.executescript(_SCHEMA)
                conn.commit()
                _schema_ready = True
    return conn


# --------------------------------------------------------------------------- #
# Brute-force throttle (in-memory, per-email)
# --------------------------------------------------------------------------- #
def _register_failure(email: str) -> None:
    with _failures_lock:
        record = _failures.setdefault(email, {'count': 0, 'locked_until': None})
        record['count'] += 1
        if record['count'] >= MAX_FAILED_ATTEMPTS:
            record['locked_until'] = _now() + timedelta(minutes=LOCKOUT_MINUTES)
            record['count'] = 0


def _reset_failures(email: str) -> None:
    with _failures_lock:
        _failures.pop(email, None)


def _locked_out(email: str) -> bool:
    with _failures_lock:
        record = _failures.get(email)
        if not record or not record['locked_until']:
            return False
        if _now() >= record['locked_until']:
            _failures.pop(email, None)  # lockout window elapsed
            return False
        return True


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
def create_user(email: str, password: str) -> int:
    """Insert a new user and return its id. Raises :class:`DuplicateEmail`."""
    email = validate_email(email)
    validate_password(password)
    salt_hex = secrets.token_hex(SALT_BYTES)
    password_hash = _hash_password(password, salt_hex)
    created = _iso(_now())

    conn = _connect()
    try:
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


def authenticate(email: str, password: str) -> int:
    """Return the user id on success.

    Raises :class:`AccountLocked` while throttled and :class:`InvalidCredentials`
    otherwise. Both map to a generic 401 at the API layer.
    """
    email = normalize_email(email)
    if _locked_out(email):
        raise AccountLocked(LOCKOUT_MESSAGE)

    conn = _connect()
    try:
        row = conn.execute(
            'SELECT id, password_hash, salt FROM users WHERE email = ?', (email,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        # Equalise timing, count the failure, then fail generically.
        _verify_password(password or '', _DUMMY_SALT, _DUMMY_HASH)
        _register_failure(email)
        raise InvalidCredentials()

    if _verify_password(password or '', row['salt'], row['password_hash']):
        _reset_failures(email)
        return int(row['id'])

    _register_failure(email)
    if _locked_out(email):
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
