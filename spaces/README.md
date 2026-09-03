# Deploying the backend on Hugging Face Spaces (free)

Vercel serves the UI and the three `api/*.py` proxy functions, but it **cannot**
run the torch model. This guide hosts the FastAPI backend on a free **Hugging Face
Space (Docker SDK)** and points the Vercel proxies at it via `BACKEND_URL`.

The Space runs the exact same app as local development — `backend.main:app` —
on port **7860** (the port HF Spaces expects). The ~510 MB checkpoint is **not**
baked into the image; you upload it to the Space and tell the app where it lives
with `UATS_MODEL_DIR`.

---

## 1. Create the Space

1. Sign in at <https://huggingface.co>, then **New → Space**.
2. **Space name**: e.g. `urdu-abstractive-summarizer`.
3. **SDK**: select **Docker**.
4. **License**: pick one (MIT matches this repo).
5. **Visibility**: Public (free) or Private. Click **Create Space**.

## 2. Add the application files

The Docker build context is the **Space repository root**, so it must contain:

```
Dockerfile          <- copy from spaces/Dockerfile in this repo (to the ROOT)
backend/            <- the whole backend package
frontend/           <- the single-page UI
run.py
requirements.txt
```

Copy this repo's `spaces/Dockerfile` to the Space **root** as `Dockerfile`, then
add `backend/`, `frontend/`, `run.py` and `requirements.txt` alongside it. You can
drag-and-drop them in the Space's **Files** tab, or push with git:

```bash
git clone https://huggingface.co/spaces/<your-user>/<your-space>
cd <your-space>
# copy backend/ frontend/ run.py requirements.txt and spaces/Dockerfile -> ./Dockerfile
git lfs install
git add .
git commit -m "Add Urdu summarizer backend"
git push
```

## 3. Upload the model weights (Git LFS)

The app expects, under `UATS_MODEL_DIR`:

```
checkpoints/best_model.pt      (~510 MB)
cache/vocab_100k.json
```

Large files must go through **Git LFS** (already enabled on HF Spaces):

```bash
git lfs track "*.pt"
git lfs track "*.json"        # only if vocab_100k.json exceeds the normal-file limit
# place the files so the tree is:  data/checkpoints/best_model.pt
#                                  data/cache/vocab_100k.json
git add data/ .gitattributes
git commit -m "Add model weights + vocabulary (LFS)"
git push
```

Alternatively, upload `best_model.pt` and `vocab_100k.json` through the Space's
**Files → Upload** UI (it uses LFS automatically for big files), keeping the
`checkpoints/` and `cache/` sub-folder structure.

> Using **persistent storage** instead? Enable it in *Space → Settings → Storage*,
> upload the weights to the mounted `/data` volume, and set
> `UATS_MODEL_DIR=/data` (with `checkpoints/` and `cache/` inside it).

## 4. Set `UATS_MODEL_DIR`

In **Space → Settings → Variables and secrets**, add a variable:

| Name              | Value                            |
| ----------------- | -------------------------------- |
| `UATS_MODEL_DIR`  | `/app/data` (default in the image) |

Set it to the directory that actually contains `checkpoints/` and `cache/`
(`/app/data` if you committed the weights under `data/`, or `/data` for the
persistent-storage volume). Restart the Space after changing it.

## 5. Get the Space URL

Once the Space finishes building, its public API is reachable at:

```
https://<your-user>-<your-space>.hf.space
```

Check it works:

```bash
curl https://<your-user>-<your-space>.hf.space/health
curl https://<your-user>-<your-space>.hf.space/sample
```

Both should return JSON (the `/health` payload reports the loaded vocabulary and
checkpoint). The first request after a cold start is slow while the 510 MB
checkpoint loads — subsequent ones are fast.

## 6. Point Vercel at the Space

In your **Vercel project → Settings → Environment Variables**, add:

| Name           | Value                                        |
| -------------- | -------------------------------------------- |
| `BACKEND_URL`  | `https://<your-user>-<your-space>.hf.space`  |

Redeploy Vercel. The `api/health.py`, `api/sample.py` and `api/summarize.py`
functions read `BACKEND_URL` and proxy to the Space, so the UI's same-origin
`fetch('/summarize')`, `fetch('/health')` and `fetch('/sample')` calls (routed by
`vercel.json` rewrites) now reach the real model.

---

### Notes & limits

* **Cold starts / sleep**: free Spaces idle-sleep; the first hit after inactivity
  can take ~30–60 s to load the model. The Vercel proxies allow up to 60 s
  (`maxDuration: 60`, client timeout 55 s).
* **Single worker** by design — the checkpoint is held in-process.
* **Weights are not in this repo**: they live in `UATS_MODEL_DIR`. Nothing in the
  build copies a 510 MB file into git; you supply it to the Space via LFS.
