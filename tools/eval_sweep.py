"""Offline decoding-parameter optimisation harness (Task #15).

This tool loads its OWN ``UrduSummarizer`` (real 46k-vocab Coverage-GRU on CPU)
in a SEPARATE process and never touches the running web server on port 8000.

It sweeps beam-search decoding parameters against the 30 article/reference pairs
in ``D:\\Urdu_text\\checkpoints\\manual_evaluation.csv`` (utf-8-sig) and scores
every candidate summary with an in-script ROUGE-1/2/L F1 (no new pip deps).

Parameter override mechanism
----------------------------
``inference.beam_search_ids`` already accepts ``beam_size``, ``repetition_penalty``
and ``coverage_beta`` as keyword arguments.  The remaining three knobs
(``LENGTH_ALPHA``, ``NO_REPEAT_NGRAM``, ``MIN_SUMMARY_WORDS``) are read as MODULE
GLOBALS *at call time* inside that method, so the harness overrides them by
temporarily injecting values into the ``backend.inference`` module namespace and
restoring them afterwards.  This requires ZERO changes to ``inference.py`` and is
fully contained in this harness.

Decoding path
-------------
Each config performs a SINGLE beam decode (no degenerate-output retry) so the
measured effect is attributable purely to the swept parameters.  Post-processing
(``decode_ids`` -> ``collapse_immediate_repeats``) matches the production path.

Outputs (all UTF-8):
    tools/sweep_results.json   per-config mean metrics + provenance
    tools/sweep_summary.txt    human-readable ranking table
    tools/sweep_examples.txt   article -> baseline / best / reference triples
"""

from __future__ import annotations

import csv
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# --- make the backend package importable (add webapp root to sys.path) -------
WEBAPP_ROOT = Path(__file__).resolve().parent.parent
if str(WEBAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBAPP_ROOT))

import torch  # noqa: E402  (import after sys.path is set)
from backend import inference as inference_module  # noqa: E402
from backend.inference import UrduSummarizer  # noqa: E402

# --- paths -------------------------------------------------------------------
CSV_PATH = Path(r'D:\Urdu_text\checkpoints\manual_evaluation.csv')
TOOLS_DIR = WEBAPP_ROOT / 'tools'
RESULTS_PATH = TOOLS_DIR / 'sweep_results.json'
SUMMARY_PATH = TOOLS_DIR / 'sweep_summary.txt'
EXAMPLES_PATH = TOOLS_DIR / 'sweep_examples.txt'

# --- baseline == current production config ----------------------------------
BASELINE = {
    'beam_size': 4,
    'length_alpha': 0.9,
    'repetition_penalty': 1.3,
    'no_repeat_ngram': 3,
    'coverage_beta': 0.5,
    'min_summary_words': 6,
}

# --- coordinate-sweep grids (one param varies at a time off the baseline) ----
COARSE_GRIDS = {
    'beam_size': [4, 6, 8],
    'length_alpha': [0.6, 0.9, 1.2],
    'repetition_penalty': [1.0, 1.3, 1.6, 2.0],
    'no_repeat_ngram': [2, 3, 4],
    'coverage_beta': [0.0, 0.5, 1.0],
}

N_SUBSET = 12          # coarse-sweep subset size
TOP_K_FINE = 3         # how many coarse winners to re-score on all 30 pairs

# Which params are read as module globals inside beam_search_ids (injected).
GLOBAL_PARAMS = {
    'length_alpha': 'LENGTH_ALPHA',
    'no_repeat_ngram': 'NO_REPEAT_NGRAM',
    'min_summary_words': 'MIN_SUMMARY_WORDS',
}

# Console is cp1252 on this machine; force UTF-8 so the log file is readable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ===========================================================================
# ROUGE-1/2/L F1 (whitespace tokens, NFC-normalised; simple LCS for ROUGE-L)
# ===========================================================================
def tokenize(text):
    """Whitespace-tokenise + NFC-normalise. Accepts a string or a token list."""
    toks = text if isinstance(text, list) else str(text).split()
    return [unicodedata.normalize('NFC', t).strip() for t in toks if t and t.strip()]


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(cand, ref, n):
    """Clipped n-gram F1 (standard ROUGE-N)."""
    cand_g, ref_g = _ngrams(cand, n), _ngrams(ref, n)
    if not cand_g or not ref_g:
        return 0.0
    ref_counts = Counter(ref_g)
    clipped = sum(min(c, ref_counts.get(g, 0)) for g, c in Counter(cand_g).items())
    p = clipped / len(cand_g)
    r = clipped / len(ref_g)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l(cand, ref):
    """LCS-based F1 (beta = 1)."""
    if not cand or not ref:
        return 0.0
    lcs = _lcs_len(cand, ref)
    p = lcs / len(cand)
    r = lcs / len(ref)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def repetition_rate(words):
    return 0.0 if not words else 1 - len(set(words)) / len(words)


def combined_score(m):
    """Ranking objective: content-weighted ROUGE blend (documented in report)."""
    return 0.4 * m['mean_rouge1'] + 0.2 * m['mean_rouge2'] + 0.4 * m['mean_rougel']


def lexical_coverage(cand, src_set):
    """Unsupervised relevance proxy: fraction of summary tokens seen in source."""
    uniq = set(cand)
    return len(uniq & src_set) / len(uniq) if uniq else 0.0


def proxy_score(cand, rep, src_set):
    """Best-of-K reranker proxy: low repetition + length in [8,20] + coverage."""
    n = len(cand)
    length_ok = 1.0 if 8 <= n <= 20 else 0.0
    return (1.0 - rep) + length_ok + lexical_coverage(cand, src_set)


# ===========================================================================
# Decoding with overridable parameters
# ===========================================================================
def decode_config(summarizer, source, lengths, cfg):
    """Beam-decode one pre-encoded article under ``cfg`` (module-global injection)."""
    saved = {attr: getattr(inference_module, attr) for attr in GLOBAL_PARAMS.values()}
    try:
        inference_module.LENGTH_ALPHA = cfg['length_alpha']
        inference_module.NO_REPEAT_NGRAM = cfg['no_repeat_ngram']
        inference_module.MIN_SUMMARY_WORDS = cfg['min_summary_words']
        ids = summarizer.beam_search_ids(
            source, lengths,
            beam_size=cfg['beam_size'],
            max_words=inference_module.MAX_TARGET_WORDS,
            repetition_penalty=cfg['repetition_penalty'],
            coverage_beta=cfg['coverage_beta'],
        )
    finally:
        for attr, val in saved.items():
            setattr(inference_module, attr, val)
    words = summarizer.collapse_immediate_repeats(summarizer.decode_ids(ids).split())
    return words


def eval_config(summarizer, sources, ref_tokens, indices, cfg, collect_details=False):
    """Score ``cfg`` over ``indices``; return (mean_metrics, per_article_details)."""
    r1, r2, rl, rep, tm = [], [], [], [], []
    details = []
    for i in indices:
        words_src, source, lengths = sources[i]
        t0 = time.perf_counter()
        cand_words = decode_config(summarizer, source, lengths, cfg)
        dt = time.perf_counter() - t0
        cand = tokenize(cand_words)
        ref = ref_tokens[i]
        s1, s2, sl = rouge_n(cand, ref, 1), rouge_n(cand, ref, 2), rouge_l(cand, ref)
        rp = repetition_rate(cand)
        r1.append(s1); r2.append(s2); rl.append(sl); rep.append(rp); tm.append(dt)
        if collect_details:
            src_set = set(tokenize(words_src))
            details.append({
                'index': i,
                'summary': ' '.join(cand_words),
                'summary_words': len(cand_words),
                'repetition_rate': round(rp, 4),
                'rouge1': round(s1, 4), 'rouge2': round(s2, 4), 'rougel': round(sl, 4),
                'lexical_coverage': round(lexical_coverage(cand, src_set), 4),
                'proxy_score': round(proxy_score(cand, rp, src_set), 4),
                'decode_seconds': round(dt, 3),
            })
    n = len(indices)
    metrics = {
        'n_pairs': n,
        'mean_rouge1': round(sum(r1) / n, 4),
        'mean_rouge2': round(sum(r2) / n, 4),
        'mean_rougel': round(sum(rl) / n, 4),
        'mean_repetition_rate': round(sum(rep) / n, 4),
        'mean_decode_seconds': round(sum(tm) / n, 3),
    }
    metrics['combined_score'] = round(combined_score(metrics), 4)
    return metrics, details


# ===========================================================================
# Sweep orchestration
# ===========================================================================
def build_coarse_configs():
    """One-param-at-a-time configs off the baseline, de-duplicated, baseline first."""
    seen = {tuple(sorted(BASELINE.items()))}
    configs = [dict(BASELINE)]
    for param, values in COARSE_GRIDS.items():
        for value in values:
            cfg = dict(BASELINE)
            cfg[param] = value
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(cfg)
    return configs


def subset_indices(total, k):
    """Deterministic spread of ``k`` indices across ``range(total)``."""
    if k >= total:
        return list(range(total))
    return sorted(set(int(i * total / k) for i in range(k)))


def main():
    t_start = time.perf_counter()
    print('=' * 78, flush=True)
    print('Task #15 decoding-parameter sweep (own model instance; server untouched)', flush=True)
    print('=' * 78, flush=True)

    # --- load evaluation pairs ------------------------------------------
    with CSV_PATH.open(encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    articles = [r['article'] for r in rows]
    references = [r['reference'] for r in rows]
    ref_tokens = [tokenize(r) for r in references]
    total = len(rows)
    print(f'Loaded {total} article/reference pairs from {CSV_PATH}', flush=True)

    # --- load the real model (this process only) -------------------------
    summarizer = UrduSummarizer()
    model_meta = {
        'model_dir': str(summarizer.model_dir),
        'checkpoint': summarizer.checkpoint_path.name,
        'vocab_size': summarizer.ACTUAL_VOCAB_SIZE,
        'parameters': sum(p.numel() for p in summarizer.model.parameters()),
        'device': str(summarizer.device),
    }
    print(f'Model ready: {model_meta}', flush=True)

    # --- pre-encode every article ONCE (Stanza is deterministic) ---------
    sources = []
    for art in articles:
        words, source, lengths = summarizer._prepare(art)
        sources.append((words, source, lengths))
    print(f'Pre-encoded {len(sources)} articles (source tokens cached).', flush=True)

    all_idx = list(range(total))
    sub_idx = subset_indices(total, N_SUBSET)
    print(f'Coarse subset ({len(sub_idx)} pairs): {sub_idx}', flush=True)

    # --- 1) baseline on ALL 30 ------------------------------------------
    print('\n[1/4] Baseline on all 30 pairs ...', flush=True)
    baseline_metrics, baseline_details = eval_config(
        summarizer, sources, ref_tokens, all_idx, BASELINE, collect_details=True)
    print(f'   baseline: R1={baseline_metrics["mean_rouge1"]} '
          f'R2={baseline_metrics["mean_rouge2"]} RL={baseline_metrics["mean_rougel"]} '
          f'rep={baseline_metrics["mean_repetition_rate"]} '
          f't={baseline_metrics["mean_decode_seconds"]}s', flush=True)

    # --- 2) coarse coordinate sweep on the subset -----------------------
    print(f'\n[2/4] Coarse coordinate sweep on {len(sub_idx)} pairs ...', flush=True)
    coarse = []
    coarse_configs = build_coarse_configs()
    for cfg in coarse_configs:
        m, _ = eval_config(summarizer, sources, ref_tokens, sub_idx, cfg)
        coarse.append({'config': cfg, 'subset_metrics': m})
        varied = {k: v for k, v in cfg.items() if v != BASELINE[k]}
        label = varied if varied else 'BASELINE'
        print(f'   {str(label):52s} R1={m["mean_rouge1"]:.4f} R2={m["mean_rouge2"]:.4f} '
              f'RL={m["mean_rougel"]:.4f} comb={m["combined_score"]:.4f} '
              f'rep={m["mean_repetition_rate"]:.4f}', flush=True)

    # rank coarse by combined score (tie-break: lower repetition)
    coarse_ranked = sorted(
        coarse,
        key=lambda e: (e['subset_metrics']['combined_score'],
                       -e['subset_metrics']['mean_repetition_rate']),
        reverse=True)
    top_fine = [e['config'] for e in coarse_ranked[:TOP_K_FINE]]
    print(f'\n   Top-{TOP_K_FINE} coarse winners -> fine pass on all 30:', flush=True)
    for cfg in top_fine:
        print(f'      {cfg}', flush=True)

    # --- 3) fine pass on ALL 30 -----------------------------------------
    print('\n[3/4] Fine pass on all 30 pairs ...', flush=True)
    fine = []
    fine_details = []
    for cfg in top_fine:
        if cfg == BASELINE:
            m, details = baseline_metrics, baseline_details
            tag = ' (reuse baseline)'
        else:
            m, details = eval_config(summarizer, sources, ref_tokens, all_idx, cfg,
                                     collect_details=True)
            tag = ''
        fine.append({'config': cfg, 'all30_metrics': m})
        fine_details.append(details)
        print(f'   {cfg}{tag} R1={m["mean_rouge1"]:.4f} R2={m["mean_rouge2"]:.4f} '
              f'RL={m["mean_rougel"]:.4f} comb={m["combined_score"]:.4f} '
              f'rep={m["mean_repetition_rate"]:.4f}', flush=True)

    # best on all 30 = argmax combined among {baseline} U {fine winners}
    candidates_all30 = [{'config': BASELINE, 'all30_metrics': baseline_metrics}] + fine
    best_entry = max(candidates_all30,
                     key=lambda e: (e['all30_metrics']['combined_score'],
                                    -e['all30_metrics']['mean_repetition_rate']))
    best_cfg = best_entry['config']
    best_metrics = best_entry['all30_metrics']
    print(f'\n   BEST on all 30: {best_cfg} comb={best_metrics["combined_score"]}', flush=True)

    # details for the best config (reuse if it is baseline or a fine winner)
    if best_cfg == BASELINE:
        best_details = baseline_details
    else:
        pos = next((k for k, c in enumerate(top_fine) if c == best_cfg), None)
        best_details = fine_details[pos] if pos is not None else []

    # --- 4) optional best-of-3 unsupervised reranking -------------------
    print('\n[4/4] Best-of-3 unsupervised reranking test ...', flush=True)
    rerank = run_reranking(top_fine, fine_details, best_metrics, ref_tokens)

    # --- write outputs ---------------------------------------------------
    runtime = round(time.perf_counter() - t_start, 1)
    results = {
        'meta': {
            'task': 15,
            'csv_path': str(CSV_PATH),
            'n_pairs': total,
            'subset_indices': sub_idx,
            'model': model_meta,
            'ranking_metric': 'combined = 0.4*ROUGE1 + 0.2*ROUGE2 + 0.4*ROUGEL (higher better); tie-break lower repetition',
            'decode_path': 'single beam_search_ids per config (no degenerate retry); production post-processing',
            'runtime_seconds': runtime,
        },
        'baseline_config': BASELINE,
        'baseline_all30': baseline_metrics,
        'coarse_subset': coarse,
        'coarse_ranked_configs': [e['config'] for e in coarse_ranked],
        'fine_top_configs': top_fine,
        'fine_all30': fine,
        'best_config': best_cfg,
        'best_all30': best_metrics,
        'reranking': rerank,
    }
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    write_summary(results)
    write_examples(articles, references, baseline_details, best_details, best_cfg)

    print('\n' + '=' * 78, flush=True)
    print(f'DONE. total runtime = {runtime}s ({runtime/60:.1f} min)', flush=True)
    print(f'Wrote: {RESULTS_PATH}', flush=True)
    print(f'       {SUMMARY_PATH}', flush=True)
    print(f'       {EXAMPLES_PATH}', flush=True)
    print('=' * 78, flush=True)


def run_reranking(top_fine, fine_details, best_metrics, ref_tokens):
    """Best-of-K reranking: pick per-article candidate maximising the proxy.

    ``fine_details`` is aligned 1:1 with ``top_fine`` (both built in the fine
    pass), so the K candidate summaries per article are already available and
    NO extra decoding is performed here.
    """
    k = len(top_fine)
    cfg_details = fine_details
    if k < 2 or len(cfg_details) != k or any(not d for d in cfg_details):
        return {'performed': False, 'reason': 'fewer than 2 valid candidate sets'}

    n = len(ref_tokens)
    sel_r1, sel_r2, sel_rl, sel_rep = [], [], [], []
    oracle_r1, oracle_r2, oracle_rl = [], [], []
    for i in range(n):
        cands = [cfg_details[c][i] for c in range(k)]
        chosen = max(cands, key=lambda d: d['proxy_score'])
        sel_r1.append(chosen['rouge1']); sel_r2.append(chosen['rouge2'])
        sel_rl.append(chosen['rougel']); sel_rep.append(chosen['repetition_rate'])
        # oracle: per-article best by combined ROUGE among the K candidates
        orc = max(cands, key=lambda d: 0.4 * d['rouge1'] + 0.2 * d['rouge2'] + 0.4 * d['rougel'])
        oracle_r1.append(orc['rouge1']); oracle_r2.append(orc['rouge2']); oracle_rl.append(orc['rougel'])

    def mean(x):
        return round(sum(x) / len(x), 4)

    rerank_metrics = {
        'n_pairs': n, 'k_candidates': k,
        'mean_rouge1': mean(sel_r1), 'mean_rouge2': mean(sel_r2),
        'mean_rougel': mean(sel_rl), 'mean_repetition_rate': mean(sel_rep),
    }
    rerank_metrics['combined_score'] = round(combined_score(rerank_metrics), 4)
    oracle_metrics = {
        'n_pairs': n,
        'mean_rouge1': mean(oracle_r1), 'mean_rouge2': mean(oracle_r2),
        'mean_rougel': mean(oracle_rl),
    }
    oracle_metrics['combined_score'] = round(combined_score(oracle_metrics), 4)
    beats_best = rerank_metrics['combined_score'] > best_metrics['combined_score']
    return {
        'performed': True,
        'candidate_configs': top_fine,
        'proxy': 'score = (1 - repetition_rate) + [1 if 8<=words<=20 else 0] + lexical_coverage(summary, source)',
        'reranked_all30': rerank_metrics,
        'oracle_best_all30': oracle_metrics,
        'single_best_all30': best_metrics,
        'reranker_beats_single_best': bool(beats_best),
        'delta_vs_single_best': round(rerank_metrics['combined_score'] - best_metrics['combined_score'], 4),
    }


# ===========================================================================
# Output writers (UTF-8)
# ===========================================================================
def write_summary(results):
    lines = []
    ap = lines.append
    b = results['baseline_all30']
    best = results['best_all30']
    ap('=' * 92)
    ap('TASK #15  DECODING-PARAMETER SWEEP  —  RESULTS SUMMARY')
    ap('=' * 92)
    meta = results['meta']
    ap(f"model: {meta['model']['checkpoint']}  vocab={meta['model']['vocab_size']}  "
       f"params={meta['model']['parameters']:,}  device={meta['model']['device']}")
    ap(f"pairs: {meta['n_pairs']}   coarse subset: {meta['subset_indices']}")
    ap(f"ranking: {meta['ranking_metric']}")
    ap(f"total runtime: {meta['runtime_seconds']}s ({meta['runtime_seconds']/60:.1f} min)")
    ap('')
    ap('--- BASELINE (all 30) vs BEST (all 30) ---')
    hdr = f"{'config':44s} {'R1':>7s} {'R2':>7s} {'RL':>7s} {'comb':>7s} {'rep':>7s} {'t(s)':>6s}"
    ap(hdr)
    ap('-' * len(hdr))
    ap(f"{fmt_cfg(results['baseline_config']):44s} {b['mean_rouge1']:7.4f} {b['mean_rouge2']:7.4f} "
       f"{b['mean_rougel']:7.4f} {b['combined_score']:7.4f} {b['mean_repetition_rate']:7.4f} "
       f"{b['mean_decode_seconds']:6.2f}")
    ap(f"{fmt_cfg(results['best_config']):44s} {best['mean_rouge1']:7.4f} {best['mean_rouge2']:7.4f} "
       f"{best['mean_rougel']:7.4f} {best['combined_score']:7.4f} {best['mean_repetition_rate']:7.4f} "
       f"{best['mean_decode_seconds']:6.2f}")
    d1 = best['mean_rouge1'] - b['mean_rouge1']
    d2 = best['mean_rouge2'] - b['mean_rouge2']
    dl = best['mean_rougel'] - b['mean_rougel']
    dr = best['mean_repetition_rate'] - b['mean_repetition_rate']
    ap('')
    ap(f"delta (best - baseline):  dR1={d1:+.4f}  dR2={d2:+.4f}  dRL={dl:+.4f}  drep={dr:+.4f}")
    ap('')
    ap('--- COARSE COORDINATE SWEEP (subset of %d) — ranked by combined score ---' % len(meta['subset_indices']))
    ap(hdr)
    ap('-' * len(hdr))
    for e in sorted(results['coarse_subset'],
                    key=lambda x: x['subset_metrics']['combined_score'], reverse=True):
        m = e['subset_metrics']
        ap(f"{fmt_cfg(e['config']):44s} {m['mean_rouge1']:7.4f} {m['mean_rouge2']:7.4f} "
           f"{m['mean_rougel']:7.4f} {m['combined_score']:7.4f} {m['mean_repetition_rate']:7.4f} "
           f"{m['mean_decode_seconds']:6.2f}")
    ap('')
    ap('--- FINE PASS (top-%d coarse winners, all 30) ---' % len(results['fine_all30']))
    ap(hdr)
    ap('-' * len(hdr))
    for e in results['fine_all30']:
        m = e['all30_metrics']
        ap(f"{fmt_cfg(e['config']):44s} {m['mean_rouge1']:7.4f} {m['mean_rouge2']:7.4f} "
           f"{m['mean_rougel']:7.4f} {m['combined_score']:7.4f} {m['mean_repetition_rate']:7.4f} "
           f"{m['mean_decode_seconds']:6.2f}")
    ap('')
    rr = results['reranking']
    ap('--- BEST-OF-3 UNSUPERVISED RERANKING (all 30) ---')
    if not rr.get('performed'):
        ap(f"skipped: {rr.get('reason')}")
    else:
        rk, orc, sb = rr['reranked_all30'], rr['oracle_best_all30'], rr['single_best_all30']
        ap(f"proxy: {rr['proxy']}")
        ap(f"{'single best config':30s} R1={sb['mean_rouge1']:.4f} R2={sb['mean_rouge2']:.4f} "
           f"RL={sb['mean_rougel']:.4f} comb={sb['combined_score']:.4f}")
        ap(f"{'proxy reranker (best-of-3)':30s} R1={rk['mean_rouge1']:.4f} R2={rk['mean_rouge2']:.4f} "
           f"RL={rk['mean_rougel']:.4f} comb={rk['combined_score']:.4f}")
        ap(f"{'oracle best-of-3':30s} R1={orc['mean_rouge1']:.4f} R2={orc['mean_rouge2']:.4f} "
           f"RL={orc['mean_rougel']:.4f} comb={orc['combined_score']:.4f}")
        verdict = 'YES' if rr['reranker_beats_single_best'] else 'NO'
        ap(f"reranker beats single best config? {verdict} (delta comb={rr['delta_vs_single_best']:+.4f})")
    ap('')
    ap('--- RECOMMENDED PARAMETER SET (best on all 30) ---')
    for k, v in results['best_config'].items():
        ap(f"   {k:20s} = {v}")
    ap('=' * 92)
    text = '\n'.join(lines)
    SUMMARY_PATH.write_text(text, encoding='utf-8')
    print('\n' + text, flush=True)


def write_examples(articles, references, baseline_details, best_details, best_cfg, count=3):
    lines = []
    ap = lines.append
    ap('=' * 92)
    ap('TASK #15  EXAMPLE TRIPLES  —  article -> (baseline summary / best summary / reference)')
    ap(f'best config: {best_cfg}')
    ap('=' * 92)
    idxs = [d['index'] for d in baseline_details[:count]] if baseline_details else list(range(count))
    bmap = {d['index']: d for d in baseline_details}
    nmap = {d['index']: d for d in best_details}
    for i in idxs:
        art = ' '.join(articles[i].split())
        shown = art[:420] + (' ...' if len(art) > 420 else '')
        ap('')
        ap(f'### Article #{i}  ({len(articles[i].split())} source words)')
        ap(f'ARTICLE : {shown}')
        ap(f'REFERENCE: {references[i].strip()}')
        bd = bmap.get(i, {})
        nd = nmap.get(i, {})
        ap(f'BASELINE : {bd.get("summary","")}')
        ap(f'           [R1={bd.get("rouge1")} R2={bd.get("rouge2")} RL={bd.get("rougel")} '
           f'rep={bd.get("repetition_rate")}]')
        ap(f'BEST     : {nd.get("summary","")}')
        ap(f'           [R1={nd.get("rouge1")} R2={nd.get("rouge2")} RL={nd.get("rougel")} '
           f'rep={nd.get("repetition_rate")}]')
    ap('')
    ap('=' * 92)
    EXAMPLES_PATH.write_text('\n'.join(lines), encoding='utf-8')


def fmt_cfg(cfg):
    if cfg == BASELINE:
        return 'BASELINE ' + json.dumps(cfg, ensure_ascii=False)
    varied = {k: v for k, v in cfg.items() if v != BASELINE[k]}
    return json.dumps(varied if varied else cfg, ensure_ascii=False)


if __name__ == '__main__':
    main()
