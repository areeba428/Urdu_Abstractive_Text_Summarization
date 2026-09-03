# Hugging Face Spaces — Docker SDK backend for the Urdu abstractive summarizer.
#
# This image serves the SAME FastAPI app that runs locally, but on port 7860
# (the port HF Spaces expects). It does NOT bake in the ~510 MB checkpoint: the
# weights are supplied at run time from UATS_MODEL_DIR (see spaces/README.md).
#
# Build context = the Space repository root, which must contain:
#   backend/  frontend/  run.py  requirements.txt  (and this Dockerfile)

FROM python:3.12-slim

# Keep Python unbuffered and disable pip's cache so the image stays small.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Where the model weights live inside the container. The app expects
#   $UATS_MODEL_DIR/checkpoints/best_model.pt
#   $UATS_MODEL_DIR/cache/vocab_100k.json
# Override this in the Space's "Variables and secrets" if you mount them elsewhere
# (e.g. /data when using HF persistent storage).
ENV UATS_MODEL_DIR=/app/data

WORKDIR /app

# 1) CPU-only torch FIRST from the PyTorch CPU index, otherwise pip pulls the
#    multi-gigabyte CUDA build. Then the rest of the runtime dependencies.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "fastapi>=0.110,<1" "uvicorn[standard]>=0.29,<1" \
        "stanza>=1.8,<2" "numpy>=1.26,<3"

# 2) Download the Urdu tokenizer weights at build time so the first request does
#    not pay for it (and so the Space works even if outbound traffic is limited).
RUN python -c "import stanza; stanza.download('ur')"

# 3) Application code. requirements.txt is copied for reference/reproducibility;
#    dependencies are already installed above.
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY run.py ./run.py
COPY requirements.txt ./requirements.txt

# HF Spaces routes traffic to port 7860.
EXPOSE 7860

# Single worker by design: the 510 MB checkpoint is held in-process, so extra
# workers would duplicate it in RAM and re-pay the load cost.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
