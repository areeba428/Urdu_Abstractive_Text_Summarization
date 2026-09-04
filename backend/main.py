"""FastAPI service around the trained Urdu Coverage-GRU summarizer.

The model, the 46k vocabulary and the Stanza Urdu tokenizer are all loaded
ONCE, during application startup; every request reuses that single instance.
CPU beam search takes seconds per article, so ``/summarize`` is declared with
``def`` (not ``async def``) and FastAPI runs it on the thread pool.  The light
endpoints (``/``, ``/health``, ``/sample``) are ``async def`` so they are served
on the event loop and never queue behind blocked ``/summarize`` workers.

Responses never carry filesystem paths or exception internals; failures are
logged server-side and reported with a constant message.

Run from the ``webapp`` directory:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import auth
from .inference import (
    BEAM_SIZE,
    MAX_SOURCE_WORDS,
    MAX_TARGET_WORDS,
    MAX_TOKENIZE_CHARS,
    MIN_SUMMARY_WORDS,
    NO_REPEAT_NGRAM,
    REPETITION_PENALTY,
    UrduSummarizer,
)
from .sample import TEST_ARTICLE

logger = logging.getLogger('uats.web')

FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'
INDEX_HTML = FRONTEND_DIR / 'index.html'
LOGIN_HTML = FRONTEND_DIR / 'login.html'
SIGNUP_HTML = FRONTEND_DIR / 'signup.html'

# Stanza is only ever fed MAX_TOKENIZE_CHARS, and the encoder only sees
# MAX_SOURCE_WORDS - 2 tokens, so a 20k-character body is already far more text
# than the model can use.
MAX_ARTICLE_CHARS = 20_000
# The training corpus was filtered to Article >= 200 characters; anything
# shorter is out-of-distribution and produces junk headlines.
MIN_ARTICLE_CHARS = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    print('[startup] loading vocabulary, checkpoint and Stanza tokenizer ...', flush=True)
    app.state.summarizer = UrduSummarizer()
    app.state.load_seconds = round(time.perf_counter() - started, 2)
    print(f'[startup] ready in {app.state.load_seconds}s', flush=True)
    yield
    app.state.summarizer = None


app = FastAPI(
    title='Urdu Abstractive Summarization',
    description='Coverage-GRU (Bahdanau + coverage) seq2seq over 100k Urdu news pairs.',
    version='1.0.0',
    lifespan=lifespan,
)


class SummarizeRequest(BaseModel):
    article: str = Field(..., min_length=1, max_length=MAX_ARTICLE_CHARS,
                         description=f'Raw Urdu article text ({MIN_ARTICLE_CHARS}-'
                                     f'{MAX_ARTICLE_CHARS} characters).')
    mode: Literal['greedy', 'beam'] = Field('beam', description="Decoding strategy.")


class SummarizeResponse(BaseModel):
    summary: str
    summary_words: int
    source_words: int
    truncated_source: bool
    repetition_rate: float
    mode: str
    elapsed_seconds: float
    # Transparency (Task #11): let the user confirm their own text was used and
    # warn when it was mostly outside the model's Urdu-news vocabulary.
    oov_ratio: float = 0.0
    warning: Optional[str] = None
    received_first_words: str = ''


def summarizer() -> UrduSummarizer:
    instance = getattr(app.state, 'summarizer', None)
    if instance is None:
        raise HTTPException(status_code=503, detail='Model is not loaded yet.')
    return instance


# --- authentication (Task #23): email + password, SQLite-backed sessions ---
SIGNUP_MESSAGE = 'Account created. / اکاؤنٹ بن گیا۔'
DUPLICATE_MESSAGE = ('This email is already registered — sign in instead. / '
                     'یہ ای میل پہلے سے رجسٹرڈ ہے — براہ کرم لاگ ان کریں۔')
# C1: bilingual 429 detail for the per-IP token bucket.
THROTTLE_MESSAGE = ('Too many attempts — please try again later. / '
                    'بہت زیادہ کوششیں — براہ کرم بعد میں دوبارہ کوشش کریں۔')
# W8: constant detail when the account store is unreachable.
ACCOUNT_SERVICE_DOWN = 'Account service unavailable.'


def client_ip(request: Request) -> str:
    """C1: best-effort client IP for the per-IP token bucket.

    Uses the first ``X-Forwarded-For`` value (the real client as seen by the
    Vercel/HF proxy chain) when present, else the socket peer. CAVEAT: a client
    talking to the backend directly can SPOOF ``X-Forwarded-For`` to spread its
    budget across fake IPs; the bucket is a best-effort throttle, not a hard
    security boundary — the ``(ip, email)`` lockout and the PBKDF2 cost remain the
    real brute-force defence.
    """
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        first = forwarded.split(',')[0].strip()
        if first:
            return first
    return request.client.host if request.client else '?'


class CredentialsRequest(BaseModel):
    """Shared body for /auth/signup and /auth/login; validation yields 422."""
    # N5: no Pydantic pattern — auth.validate_email (after normalize/strip) is the
    # single source of truth and yields the bilingual 422. Bounds only guard size.
    email: str = Field(..., min_length=3, max_length=256,
                       description='Account email (a Gmail or any valid address).')
    password: str = Field(..., min_length=auth.MIN_PASSWORD_LEN,
                          max_length=auth.MAX_PASSWORD_LEN,
                          description=f'Password, {auth.MIN_PASSWORD_LEN}-'
                                      f'{auth.MAX_PASSWORD_LEN} characters.')


def require_auth(request: Request) -> dict:
    """Dependency: a valid, unexpired ``session`` cookie is required."""
    token = request.cookies.get(auth.SESSION_COOKIE)
    try:
        user = auth.get_user_by_session(token) if token else None
    except (OSError, sqlite3.Error):
        # W8: the account store is unreachable — 503, not a misleading 401.
        logger.exception('Account store unavailable during auth')
        raise HTTPException(status_code=503, detail=ACCOUNT_SERVICE_DOWN) from None
    if not user:
        raise HTTPException(status_code=401, detail='authentication required')
    return user


@app.get('/', include_in_schema=False)
async def index():
    if not INDEX_HTML.exists():
        logger.error('UI asset missing: %s', INDEX_HTML)
        raise HTTPException(status_code=500, detail='The web UI asset is missing on the server.')
    return FileResponse(INDEX_HTML, media_type='text/html; charset=utf-8')


@app.get('/sample', include_in_schema=False)
async def sample():
    """The notebook's TEST_ARTICLE (cell 10), for the UI's example button."""
    return {'article': TEST_ARTICLE.strip()}


@app.get('/health')
async def health():
    instance = getattr(app.state, 'summarizer', None)
    if instance is None:
        return JSONResponse({'status': 'loading'}, status_code=503)
    return {
        'status': 'ok',
        'startup_seconds': app.state.load_seconds,
        'vocab_size': instance.ACTUAL_VOCAB_SIZE,
        'checkpoint': instance.checkpoint_path.name,
        'beam_size': BEAM_SIZE,
        'repetition_penalty': REPETITION_PENALTY,
        'no_repeat_ngram': NO_REPEAT_NGRAM,
        'min_summary_words': MIN_SUMMARY_WORDS,
        'max_source_words': MAX_SOURCE_WORDS,
        'max_target_words': MAX_TARGET_WORDS,
        'max_tokenize_chars': MAX_TOKENIZE_CHARS,
        'max_article_chars': MAX_ARTICLE_CHARS,
        'min_article_chars': MIN_ARTICLE_CHARS,
    }


@app.post('/summarize', response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest, user: dict = Depends(require_auth)):
    article = (payload.article or '').strip()
    if not article:
        raise HTTPException(status_code=400, detail='Article text is empty.')
    # Pydantic's max_length normally rejects an over-long body with 422 first;
    # this guard stays as a safety net for callers that bypass the model.
    if len(article) > MAX_ARTICLE_CHARS:
        raise HTTPException(status_code=413, detail=f'Article exceeds {MAX_ARTICLE_CHARS} characters.')
    if len(article) < MIN_ARTICLE_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(f'Article is too short ({len(article)} characters); at least '
                    f'{MIN_ARTICLE_CHARS} characters are required. / '
                    f'مضمون بہت مختصر ہے، کم از کم {MIN_ARTICLE_CHARS} حروف درکار ہیں۔')
        )

    # Resolved outside the try block: a 503 from here must stay a 503.
    instance = summarizer()

    started = time.perf_counter()
    try:
        result = instance.summarize(article, mode=payload.mode)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception:  # keep the server alive; internals stay in the log
        logger.exception('Summarization failed (mode=%s, chars=%d)', payload.mode, len(article))
        raise HTTPException(status_code=500, detail='Summarization failed.') from None

    result['elapsed_seconds'] = round(time.perf_counter() - started, 2)
    # Echo the first few words of the trimmed text we actually processed so the
    # user can confirm the summary came from THEIR input, not a cached sample.
    result['received_first_words'] = ' '.join(article.split()[:8])
    return result


# --------------------------------------------------------------------------- #
# Authentication routes (Task #23). Every handler is a sync ``def`` so its
# SQLite work runs on FastAPI's thread pool exactly like ``/summarize`` — never
# on the event loop. Password hashes and exception internals are never returned.
# --------------------------------------------------------------------------- #
def _session_response(status_code: int, token: str, content: dict) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content=content)
    response.set_cookie(
        auth.SESSION_COOKIE, token,
        max_age=auth.SESSION_MAX_AGE, httponly=True, path='/', samesite='lax',
        secure=auth.cookie_secure(),   # W3: Secure by default; AUTH_COOKIE_INSECURE=1 opts out
    )
    return response


@app.post('/auth/signup', status_code=201)
def auth_signup(payload: CredentialsRequest, request: Request):
    # C1: per-IP budget, checked BEFORE any PBKDF2 work.
    allowed, retry_after = auth.check_ip_budget(client_ip(request), 'signup')
    if not allowed:
        raise HTTPException(status_code=429, detail=THROTTLE_MESSAGE,
                            headers={'Retry-After': str(retry_after)})
    body = {'email': auth.normalize_email(payload.email), 'message': SIGNUP_MESSAGE}
    try:
        user_id = auth.create_user(payload.email, payload.password)
    except auth.DuplicateEmail:
        # W4: enumeration-safe. Burn one PBKDF2 (create_user probed before hashing)
        # and return the IDENTICAL 201 body — but WITHOUT a Set-Cookie, since no new
        # session exists. Residual oracle: the presence/absence of Set-Cookie still
        # distinguishes duplicate from fresh; accepted and mitigated by the per-IP
        # throttle above (an attacker cannot cheaply probe many addresses).
        auth.equalize_timing(payload.password)
        return JSONResponse(status_code=201, content=body)
    except ValueError as error:  # N5: validate_email is the source of truth -> 422
        raise HTTPException(status_code=422, detail=str(error)) from error
    token = auth.create_session(user_id)  # auto-login straight after signup
    return _session_response(201, token, body)


@app.post('/auth/login')
def auth_login(payload: CredentialsRequest, request: Request):
    ip = client_ip(request)
    # C1: per-IP budget, checked BEFORE any PBKDF2 work.
    allowed, retry_after = auth.check_ip_budget(ip, 'login')
    if not allowed:
        raise HTTPException(status_code=429, detail=THROTTLE_MESSAGE,
                            headers={'Retry-After': str(retry_after)})
    try:
        user_id = auth.authenticate(payload.email, payload.password, ip)  # W1: (ip, email) lockout
    except auth.AccountLocked as locked:
        raise HTTPException(status_code=401, detail=str(locked)) from None
    except auth.InvalidCredentials:
        # Constant message: no hint about whether the email exists.
        raise HTTPException(status_code=401, detail='Invalid email or password.') from None
    except ValueError as error:  # N5: validate_email -> 422 (mirrors signup)
        raise HTTPException(status_code=422, detail=str(error)) from error
    token = auth.create_session(user_id)
    return _session_response(200, token, {
        'email': auth.normalize_email(payload.email),
        'message': 'Signed in. / خوش آمدید۔',
    })


@app.post('/auth/logout')
def auth_logout(request: Request):
    token = request.cookies.get(auth.SESSION_COOKIE)
    if token:
        auth.delete_session(token)
    response = JSONResponse(status_code=200, content={'message': 'Signed out. / الوداع۔'})
    response.delete_cookie(auth.SESSION_COOKIE, path='/', httponly=True, samesite='lax',
                           secure=auth.cookie_secure())  # W3: identical flags on delete
    return response


@app.get('/auth/me')
def auth_me(request: Request):
    token = request.cookies.get(auth.SESSION_COOKIE)
    user = auth.get_user_by_session(token) if token else None
    if not user:
        return JSONResponse(status_code=401, content={'authenticated': False})
    return {'authenticated': True, 'email': user['email']}


@app.get('/login', include_in_schema=False)
async def login_page():
    if not LOGIN_HTML.exists():
        logger.error('UI asset missing: %s', LOGIN_HTML)
        raise HTTPException(status_code=500, detail='The login page asset is missing on the server.')
    return FileResponse(LOGIN_HTML, media_type='text/html; charset=utf-8')


@app.get('/signup', include_in_schema=False)
async def signup_page():
    if not SIGNUP_HTML.exists():
        logger.error('UI asset missing: %s', SIGNUP_HTML)
        raise HTTPException(status_code=500, detail='The signup page asset is missing on the server.')
    return FileResponse(SIGNUP_HTML, media_type='text/html; charset=utf-8')
