# خلاصہ نگار — Urdu Abstractive Summarization Web App

A FastAPI service + single-page RTL Urdu UI around the trained **CoverageGRUSummarizer**
(shared `nn.Embedding` → bidirectional GRU encoder → coverage-attended GRU decoder,
133.7M fp32 parameters, 46,000-word Stanza vocabulary).

Model code, decoding math and preprocessing are transcribed verbatim from
`urdu_gru_100k_pretrained_colab.ipynb` (cells 2, 5, 6, 7, 10).

---

## 1. Prerequisites

* Windows / Linux / macOS, **Python 3.10–3.12** (tested on 3.12.10)
* **≈2 GB free RAM** for the fp32 model at load time (510 MB on disk)
* Internet access for the one-time Stanza resource download
* The trained assets, by default under `D:\Urdu_text`:

```
<UATS_MODEL_DIR>/
├── checkpoints/best_model.pt     # raw state_dict, ~510 MB (strict load)
└── cache/vocab_100k.json         # JSON array `itos`, 46,000 entries
```

## 2. Install — in this order

```powershell
cd webapp

# (a) CPU-only torch FIRST, or pip pulls the multi-GB CUDA build
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# (b) everything else
python -m pip install -r requirements.txt

# (c) once, while online: the Urdu tokenizer weights (~15 MB)
python -c "import stanza; stanza.download('ur')"
```

Tested combination: `torch 2.14.0+cpu`, `fastapi 0.141.1`, `uvicorn 0.52.4`,
`stanza 1.14.0`, `numpy 2.5.2`. `requirements.txt` pins those major bounds.

## 3. Run

```powershell
cd webapp
python run.py
```

or the raw uvicorn command, from the `webapp` directory:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>. Startup takes a few seconds (vocabulary +
checkpoint + tokenizer); the UI keeps re-probing `/health` every 2 s until the
model reports ready, so it is safe to open the page immediately.

**Single worker by design.** The model lives in-process; extra workers would each
duplicate 510 MB and re-read the checkpoint. To run a second instance, use a
different `--port`.

## 4. Options

| Flag / env var | Default | Meaning |
|---|---|---|
| `python run.py --port N` / `UATS_PORT` | `8000` | Listen port |
| `python run.py --host H` / `UATS_HOST` | `127.0.0.1` | Bind address |
| `python run.py --model-dir D` / `UATS_MODEL_DIR` | `D:\Urdu_text` | Directory holding `checkpoints/best_model.pt` and `cache/vocab_100k.json` |

## 5. API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/` | The single-page UI (`frontend/index.html`) |
| `GET` | `/health` | `async`; readiness, vocab size, decode settings, file **names** only |
| `GET` | `/sample` | `async`; the notebook's `TEST_ARTICLE` (cell 10) |
| `POST` | `/summarize` | sync `def` → runs on the thread pool; body `{"article": str, "mode": "greedy"\|"beam"}` |
| `GET` | `/docs` | Interactive OpenAPI schema |

`POST /summarize` returns:

```json
{
  "summary": "…",
  "summary_words": 12,
  "source_words": 396,
  "truncated_source": false,
  "repetition_rate": 0.0,
  "mode": "beam",
  "elapsed_seconds": 1.15
}
```

Status codes: **200** ok · **400** empty article (after strip) ·
**413** over `MAX_ARTICLE_CHARS` · **422** under 200 characters, bad `mode`, or
over 20,000 characters (schema validation) · **500** inference failure
(constant message; the traceback goes to the server log) · **503** model not loaded.

## 6. Request bounds

| Bound | Value | Why |
|---|---|---|
| `MAX_ARTICLE_CHARS` | 20,000 | Body ceiling; enforced by the schema and by the textarea `maxlength` |
| `MIN_ARTICLE_CHARS` | 200 | Training corpus was filtered to `Article >= 200` chars; shorter input is out-of-distribution |
| `MAX_TOKENIZE_CHARS` | 8,000 | Stanza input cap — only the first 398 tokens are ever encoded |
| `MAX_SOURCE_WORDS` | 400 | Encoder limit (`BOS` + 398 words + `EOS`) |
| `MAX_TARGET_WORDS` | 32 | Decoder limit |
| Request timeout (UI) | 120 s | `AbortController`; the UI resets instead of wedging |

Reserved surface forms (`<pad>`, `<unk>`, `<s>`, `</s>`) pasted by a user are
dropped before encoding so they cannot corrupt the attention mask.

## 7. Layout

```
webapp/
├── backend/
│   ├── model.py        # constants + Encoder / CoverageAttention / Decoder /
│   │                   #   CoverageGRUSummarizer  (cell 6, verbatim)
│   ├── inference.py    # UrduSummarizer: vocab, checkpoint, Stanza, encode/decode,
│   │                   #   greedy_generate, beam_search_ids, repetition helpers
│   ├── main.py         # FastAPI app: lifespan load-once, /summarize, /, /health, /sample
│   └── sample.py       # TEST_ARTICLE (cell 10)
├── frontend/index.html # single page, dir="rtl", inline CSS/JS
├── requirements.txt
├── run.py
└── README.md
```

Decoding defaults (cell 7): beam size 4, length α 0.9, repetition penalty 1.3,
no-repeat 3-gram, coverage β 0.5, minimum 6 summary words. A degenerate first
pass (repetition > 0.3 or < 6 words) is retried once with penalties 1.8 / 1.0
and beam ≥ 6, exactly as in cell 10.

## 8. Troubleshooting

* **`Port 8000 is already in use`** — `run.py` prints this hint; use `python run.py --port 8001`.
* **`RuntimeError: Stanza Urdu resources are missing…`** — you are offline and the
  tokenizer was never downloaded. Run step 2(c) while online, then restart.
* **`FileNotFoundError: checkpoint/vocabulary not found`** — set `UATS_MODEL_DIR`
  (or `--model-dir`) to the directory that actually contains `checkpoints/` and `cache/`.
* **Mojibake in the console** — Windows cp1252 cannot print Urdu. `run.py` forces
  UTF-8; for ad-hoc scripts set `$env:PYTHONIOENCODING="utf-8"` or call
  `sys.stdout.reconfigure(encoding='utf-8')`.
* **PowerShell** — use `;` as a statement separator, never `&&`.
* **Beam search is slow** — expected on CPU (≈1–3 s per article). Use `greedy`
  mode for a faster, slightly less fluent result.
* **Fonts look wrong** — the UI loads Noto Nastaliq Urdu / Noto Naskh Arabic /
  Fraunces / IBM Plex Mono from Google Fonts and falls back to local Urdu faces.

---

## 9. Repository layout

This directory is both the live development copy and the git repository root.

```
webapp/
├── backend/            # FastAPI app: main.py, model.py, inference.py, sample.py, …
├── frontend/index.html # single-page RTL Urdu UI (same-origin fetch calls)
├── tools/              # offline evaluation harnesses + generated reports
├── api/                # Vercel Python serverless proxy functions
│   ├── health.py       #   GET  /health  -> BACKEND_URL/health
│   ├── sample.py       #   GET  /sample  -> BACKEND_URL/sample
│   └── summarize.py    #   POST /summarize -> BACKEND_URL/summarize
├── spaces/             # Hugging Face Spaces (Docker) backend deployment
│   ├── Dockerfile      #   python:3.12-slim, CPU torch, stanza 'ur', port 7860
│   └── README.md       #   step-by-step Space + LFS + UATS_MODEL_DIR guide
├── vercel.json         # rewrites (/ -> frontend, /summarize|/health|/sample -> api/*)
├── .python-version     # pins the Vercel Python runtime to 3.12
├── requirements.txt
├── run.py              # local launcher (uvicorn backend.main:app)
├── LICENSE             # MIT
├── .gitignore
├── .gitattributes      # *.pt / *.npy routed through Git LFS
└── README.md
```

The trained weights (**~510 MB `best_model.pt`** + `vocab_100k.json`) are **not**
in this repository — they live in `UATS_MODEL_DIR` (default `D:\Urdu_text`).
Nothing copies them in, so they are never committed; `.gitattributes` routes
`*.pt` / `*.npy` through Git LFS should weights ever be added on purpose.

## 10. Push to GitHub

```bash
cd webapp
git init                 # already done for this checkout
git add -A
git status               # confirm no __pycache__/, *.log, or model weights are staged
git commit -m "Urdu abstractive summarizer: FastAPI app, Vercel frontend proxy, HF Spaces backend"

# create an EMPTY repo on GitHub (no README/license), then:
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If you ever add `best_model.pt` / `*.npy`, install LFS first (`git lfs install`)
so the `.gitattributes` rules take effect and the large files upload correctly.

## 11. Deploy: Vercel (frontend + proxy) + Hugging Face Spaces (backend)

Vercel cannot run the torch model, so the deployment is split in two:

* **Backend — Hugging Face Space (Docker, free).** Hosts `backend.main:app` on
  port 7860. Copy `spaces/Dockerfile` to the Space root, upload the weights via
  Git LFS, and set `UATS_MODEL_DIR`. Full walkthrough: **[`spaces/README.md`](spaces/README.md)**.
* **Frontend + API proxy — Vercel.** Serves `frontend/index.html` at `/` and
  proxies `/summarize`, `/health`, `/sample` to the Space through the `api/*.py`
  functions (see `vercel.json`). The UI already calls these paths same-origin, so
  **no frontend change is needed**.

Wiring:

1. Deploy the Space, copy its URL `https://<user>-<space>.hf.space`.
2. Import this repo into Vercel.
3. In **Vercel → Settings → Environment Variables**, set
   `BACKEND_URL = https://<user>-<space>.hf.space`.
4. Deploy. `/` serves the UI; the proxy functions forward API calls to the Space.
   Missing `BACKEND_URL` returns a clear **503** JSON; backend failures return
   **502**. Proxies allow up to 60 s (`maxDuration: 60`) for CPU beam decoding.

## 12. Run locally

For local development you do **not** need Vercel or a Space — run the whole app
in one process (see sections 2–3 for the one-time install):

```powershell
cd webapp
python run.py                              # http://127.0.0.1:8000  (UI + API together)
# or, equivalently:
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Set `UATS_MODEL_DIR` if your weights are not in the default `D:\Urdu_text`. The
local server serves the UI and the API from the same origin, exactly like the
split Vercel + Space deployment but without the network hop.
