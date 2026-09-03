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
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

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
def summarize(payload: SummarizeRequest):
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
