"""Launcher for the Urdu summarization web app.

Equivalent to running, from this directory:

    uvicorn backend.main:app --host 127.0.0.1 --port 8000

This wrapper only exists so that the Windows cp1252 console does not mangle the
Urdu text the model prints at startup, and so the app can be started from any
working directory.

    python run.py                # http://127.0.0.1:8000
    python run.py --port 9000

Single worker by design: the 510 MB checkpoint is held in-process, so every
additional worker would duplicate it in RAM and pay the load cost again. Use
``--port`` to run a second, independent instance instead of ``--workers``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve the Urdu Coverage-GRU summarizer.')
    parser.add_argument('--host', default=os.environ.get('UATS_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('UATS_PORT', '8000')))
    parser.add_argument('--model-dir', default=os.environ.get('UATS_MODEL_DIR', r'D:\Urdu_text'),
                        help='Directory holding checkpoints/best_model.pt and cache/vocab_100k.json')
    args = parser.parse_args()

    # cp1252 consoles raise UnicodeEncodeError on Urdu output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

    os.environ.setdefault('UATS_MODEL_DIR', args.model_dir)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import uvicorn

    print(f'Model directory : {os.environ["UATS_MODEL_DIR"]}')
    print(f'Opening         : http://{args.host}:{args.port}/')
    print('First start takes a minute: 510 MB state_dict + Stanza Urdu tokenizer.')
    print('Single worker by design (the model cache lives in-process).\n')

    # Single worker, no reload: the 510 MB model must not be re-read per worker.
    try:
        uvicorn.run('backend.main:app', host=args.host, port=args.port,
                    workers=1, reload=False, log_level='info')
    except OSError as error:
        print(f'\nCould not bind {args.host}:{args.port}.')
        print(f'Port {args.port} is already in use — try `python run.py --port {args.port + 1}`.')
        print(f'Details: {error}')
        raise SystemExit(1) from error
    except SystemExit as error:
        # uvicorn logs the bind failure itself and then calls sys.exit(1).
        code = error.code if isinstance(error.code, int) else 1
        if code:
            print(f'\nCould not bind {args.host}:{args.port}.')
            print(f'Port {args.port} is already in use — try `python run.py --port {args.port + 1}`.')
        raise SystemExit(code)


if __name__ == '__main__':
    main()
