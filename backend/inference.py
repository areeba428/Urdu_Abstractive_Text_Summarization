"""Inference layer for the Urdu abstractive summarizer.

Function bodies are transcribed VERBATIM from cells 5, 7 and 10 of
``urdu_gru_100k_pretrained_colab.ipynb``.  Notebook globals become instance
state (``self.model``, ``self.stoi``, ``self.itos``); the two decoding
penalties that cell 10 rewrites in-place are passed as arguments instead so
that concurrent web requests cannot corrupt each other's beam search.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path

import stanza
import torch

from . import model as model_module
from .model import (
    BOS_ID,
    EOS_ID,
    MAX_SOURCE_WORDS,
    MAX_TARGET_WORDS,
    PAD_ID,
    UNK_ID,
    CoverageGRUSummarizer,
)

# ---------------------------------------------------------------------------
# cell 3 -- WHITESPACE
# ---------------------------------------------------------------------------
WHITESPACE = re.compile(r'\s+')

# ---------------------------------------------------------------------------
# cell 7 -- decoding constants (verbatim)
# ---------------------------------------------------------------------------
BEAM_SIZE = 4
LENGTH_ALPHA = 0.9          # >0 favours longer summaries; 0 disables length normalisation.
REPETITION_PENALTY = 1.3    # Log-space penalty on words the hypothesis already used.
NO_REPEAT_NGRAM = 3         # Hard block on repeating any trigram.
COVERAGE_BETA = 0.5         # Fluency penalty on source positions attended more than once.
MIN_SUMMARY_WORDS = 6

# cell 10 -- harsher penalties used for the single degenerate-output retry.
HARSH_REPETITION_PENALTY = 1.8
HARSH_COVERAGE_BETA = 1.0

# ---------------------------------------------------------------------------
# Web hardening (not in the notebook).
#
# encode_words() only ever keeps the first MAX_SOURCE_WORDS - 2 tokens, so
# handing Stanza megabytes of user text buys nothing and costs seconds of CPU:
# cap the tokenizer input.  ``RESERVED`` lists the surface forms that stoi maps
# to PAD/UNK/BOS/EOS; if a user pastes them literally they would land in the
# encoder as real ids and silently corrupt the attention mask, so they are
# dropped before encoding.
# ---------------------------------------------------------------------------
MAX_TOKENIZE_CHARS = 8_000
RESERVED = frozenset({'<pad>', '<unk>', '<s>', '</s>'})

# Transparency: when more than this fraction of the encoded source words fall
# outside the 46k Urdu-news vocabulary (i.e. map to UNK), the output is almost
# certainly the model's generic headline prior rather than a summary of the
# user's own text, so we warn them bilingually.
OOV_WARNING_THRESHOLD = 0.6
OOV_WARNING_MESSAGE = (
    "Most of your text is outside the model's Urdu-news vocabulary — the summary "
    "may be generic. / بیشتر متن ماڈل کے اردو اخباری لغت سے باہر ہے — خلاصہ عمومی ہو سکتا ہے۔"
)

DEFAULT_MODEL_DIR = r'D:\Urdu_text'
CHECKPOINT_RELPATH = Path('checkpoints') / 'best_model.pt'
VOCAB_RELPATH = Path('cache') / 'vocab_100k.json'


class UrduSummarizer:
    """Loads the trained checkpoint + vocabulary + Stanza tokenizer exactly once."""

    def __init__(self, model_dir=None, device='cpu', verbose=True):
        model_dir = Path(model_dir or os.environ.get('UATS_MODEL_DIR') or DEFAULT_MODEL_DIR)
        self.model_dir = model_dir
        self.checkpoint_path = model_dir / CHECKPOINT_RELPATH
        self.vocab_path = model_dir / VOCAB_RELPATH
        self.device = torch.device(device)
        self.verbose = verbose
        # One shared fp32 model on CPU: serialise generation so requests do not
        # fight over the same BLAS threads (beam search is memory-bound anyway).
        self._lock = threading.Lock()

        for path, label in ((self.vocab_path, 'vocabulary'), (self.checkpoint_path, 'checkpoint')):
            if not path.exists():
                raise FileNotFoundError(f'{label} not found at {path}')

        self._load_vocab()
        self._load_model()
        self._load_tokenizer()

    # -- loading ----------------------------------------------------------
    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _load_vocab(self):
        """cell 5: ``itos`` is the JSON array; ``stoi`` is its inverse map."""
        self.itos = json.loads(self.vocab_path.read_text(encoding='utf-8'))
        self.stoi = {word: index for index, word in enumerate(self.itos)}
        self.ACTUAL_VOCAB_SIZE = len(self.itos)
        # cell 6 reads ACTUAL_VOCAB_SIZE as a module global when Decoder is built.
        model_module.configure_vocab_size(self.ACTUAL_VOCAB_SIZE)
        self._log(f'Vocabulary size: {self.ACTUAL_VOCAB_SIZE}')

    def _load_model(self):
        # The checkpoint is a RAW state_dict saved from cuda:0. No embedding
        # matrix is passed to the constructor: the table comes from the file.
        network = CoverageGRUSummarizer()
        state = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        network.load_state_dict(state)
        self.model = network.to(self.device)
        self.model.eval()
        parameter_count = sum(p.numel() for p in self.model.parameters())
        self._log(f'Trainable parameters: {parameter_count:,} | device: {self.device}')

    def _load_tokenizer(self):
        try:
            self.urdu_nlp = stanza.Pipeline(lang='ur', processors='tokenize', use_gpu=False, verbose=False)
        except Exception:  # resources not downloaded yet
            try:
                stanza.download('ur')
            except Exception as error:
                raise RuntimeError(
                    'Stanza Urdu resources are missing and could not be downloaded (offline?). '
                    'Run: python -c "import stanza; stanza.download(\'ur\')" while online, then restart.'
                ) from error
            self.urdu_nlp = stanza.Pipeline(lang='ur', processors='tokenize', use_gpu=False, verbose=False)
        self._log('Stanza Urdu tokenizer ready')

    # -- cell 5: vocabulary codecs (verbatim) ------------------------------
    def encode_words(self, text, max_words):
        ids = [self.stoi.get(word, UNK_ID) for word in str(text).split()[:max_words - 2]]
        return [BOS_ID] + ids + [EOS_ID]

    def decode_ids(self, ids):
        words = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == EOS_ID:
                break
            if token_id not in (PAD_ID, BOS_ID):
                words.append(self.itos[token_id])
        return ' '.join(words)

    # -- cell 7: greedy decoding (verbatim) --------------------------------
    @torch.no_grad()
    def greedy_generate(self, source, lengths, max_words=MAX_TARGET_WORDS):
        model = self.model
        model.eval()
        encoder_outputs, hidden, source_mask, coverage = model.encode(source, lengths)
        token = torch.full((source.size(0),), BOS_ID, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        generated = []
        for _ in range(max_words - 1):
            logits, hidden, attention = model.decoder(token, hidden, encoder_outputs, source_mask, coverage)
            coverage = coverage + attention
            logits[:, UNK_ID] = float('-inf')
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, PAD_ID), token)
            generated.append(token)
            finished |= token.eq(EOS_ID)
            if bool(finished.all()):
                break
        return torch.stack(generated, dim=1) if generated else source.new_zeros((source.size(0), 0))

    # -- cell 7: fluency helpers (verbatim) --------------------------------
    @staticmethod
    def blocked_by_ngram(tokens, size):
        if len(tokens) < size:
            return ()
        prefix = tuple(tokens[-(size - 1):])
        return tuple(
            tokens[index + size - 1]
            for index in range(len(tokens) - size + 1)
            if tuple(tokens[index:index + size - 1]) == prefix
        )

    @staticmethod
    def repetition_rate(words):
        return 0.0 if not words else 1 - len(set(words)) / len(words)

    @staticmethod
    def collapse_immediate_repeats(words):
        cleaned = []
        for word in words:
            if not cleaned or word != cleaned[-1]:
                cleaned.append(word)
        return cleaned

    # -- cell 7: beam search (verbatim) ------------------------------------
    @torch.no_grad()
    def beam_search_ids(self, source, lengths, beam_size=BEAM_SIZE, max_words=MAX_TARGET_WORDS,
                        repetition_penalty=REPETITION_PENALTY, coverage_beta=COVERAGE_BETA):
        """Beam search with repetition, trigram and coverage (fluency) penalties. Batch size 1."""
        model = self.model
        model.eval()
        encoder_outputs, hidden, source_mask, coverage = model.encode(source, lengths)
        encoder_outputs = encoder_outputs.repeat(beam_size, 1, 1)
        source_mask = source_mask.repeat(beam_size, 1)
        hidden = hidden.repeat(beam_size, 1)
        coverage = coverage.repeat(beam_size, 1)

        sequences = [[] for _ in range(beam_size)]
        scores = torch.full((beam_size,), float('-inf'), device=source.device)
        scores[0] = 0.0
        tokens = torch.full((beam_size,), BOS_ID, dtype=torch.long, device=source.device)
        log_penalty = math.log(repetition_penalty)
        finished = []

        for step in range(max_words - 1):
            logits, next_hidden, attention = model.decoder(tokens, hidden, encoder_outputs,
                                                           source_mask, coverage)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            log_probs[:, UNK_ID] = float('-inf')
            log_probs[:, PAD_ID] = float('-inf')
            if step + 1 < MIN_SUMMARY_WORDS:
                log_probs[:, EOS_ID] = float('-inf')
            for beam, sequence in enumerate(sequences):
                if sequence:
                    used = torch.tensor(sorted(set(sequence)), device=source.device)
                    log_probs[beam, used] -= log_penalty
                for banned in self.blocked_by_ngram(sequence, NO_REPEAT_NGRAM):
                    log_probs[beam, banned] = float('-inf')

            next_coverage = coverage + attention
            candidates = (scores.unsqueeze(1) + log_probs).view(-1)
            best_scores, best_indices = candidates.topk(2 * beam_size)

            new_sequences, new_scores, new_tokens, parents = [], [], [], []
            for score, index in zip(best_scores.tolist(), best_indices.tolist()):
                if not math.isfinite(score):
                    continue
                beam, token_id = divmod(index, log_probs.size(1))
                if token_id == EOS_ID:
                    over_coverage = torch.relu(next_coverage[beam] - 1.0).sum().item()
                    length_penalty = ((5 + len(sequences[beam]) + 1) / 6) ** LENGTH_ALPHA
                    finished.append(((score - coverage_beta * over_coverage) / length_penalty,
                                     list(sequences[beam])))
                elif len(new_sequences) < beam_size:
                    new_sequences.append(sequences[beam] + [token_id])
                    new_scores.append(score)
                    new_tokens.append(token_id)
                    parents.append(beam)
            if not new_sequences or len(finished) >= beam_size:
                break

            while len(new_sequences) < beam_size:
                new_sequences.append(list(new_sequences[0]))
                new_scores.append(float('-inf'))
                new_tokens.append(new_tokens[0])
                parents.append(parents[0])

            parent_index = torch.tensor(parents, device=source.device)
            sequences = new_sequences
            scores = torch.tensor(new_scores, device=source.device)
            tokens = torch.tensor(new_tokens, dtype=torch.long, device=source.device)
            hidden = next_hidden.index_select(0, parent_index)
            coverage = next_coverage.index_select(0, parent_index)

        if not finished:
            over_coverage = torch.relu(coverage[0] - 1.0).sum().item()
            length_penalty = ((5 + len(sequences[0])) / 6) ** LENGTH_ALPHA
            finished.append(((scores[0].item() - coverage_beta * over_coverage) / length_penalty,
                             sequences[0]))
        return max(finished, key=lambda item: item[0])[1]

    # -- cell 10: preprocessing -------------------------------------------
    def _prepare(self, article):
        """Whitespace collapse -> Stanza tokenize -> encode -> batch of one."""
        article = WHITESPACE.sub(' ', str(article)).strip()
        if not article:
            raise ValueError('Article text is empty.')

        document = self.urdu_nlp(article[:MAX_TOKENIZE_CHARS])
        words = [word.text for sentence in document.sentences for word in sentence.words
                 if word.text not in RESERVED]
        ids = self.encode_words(' '.join(words), MAX_SOURCE_WORDS)
        source = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)])
        return words, source, lengths

    def _oov_ratio(self, source):
        """Fraction of encoded source words (excluding BOS/EOS/PAD) that are UNK.

        ``source`` is the batch-of-one id tensor from ``_prepare``; a content id
        equal to ``UNK_ID`` means that surface form was absent from the 46k vocab.
        """
        ids = source[0].tolist()
        content = [token for token in ids if token not in (BOS_ID, EOS_ID, PAD_ID)]
        if not content:
            return 0.0
        unknown = sum(1 for token in content if token == UNK_ID)
        return round(unknown / len(content), 3)

    def _result(self, summary_words, source_words):
        return {
            'summary': ' '.join(summary_words),
            'summary_words': len(summary_words),
            'source_words': source_words,
            'truncated_source': source_words > MAX_SOURCE_WORDS - 2,
            'repetition_rate': round(self.repetition_rate(summary_words), 3),
        }

    def _beam_with_retry(self, source, lengths, beam_size=BEAM_SIZE, max_words=MAX_TARGET_WORDS):
        summary_words = self.collapse_immediate_repeats(
            self.decode_ids(self.beam_search_ids(source, lengths, beam_size=beam_size,
                                                 max_words=max_words)).split()
        )

        # If the hypothesis is still degenerate, retry once with harsher fluency penalties.
        if self.repetition_rate(summary_words) > 0.3 or len(summary_words) < MIN_SUMMARY_WORDS:
            summary_words = self.collapse_immediate_repeats(
                self.decode_ids(self.beam_search_ids(source, lengths, beam_size=max(beam_size, 6),
                                                     max_words=max_words,
                                                     repetition_penalty=HARSH_REPETITION_PENALTY,
                                                     coverage_beta=HARSH_COVERAGE_BETA)).split()
            )
        return summary_words

    # -- cell 10: summarize_article (verbatim shape) -----------------------
    def summarize_article(self, article, beam_size=BEAM_SIZE, max_words=MAX_TARGET_WORDS):
        """Summarise raw Urdu text into a short, non-repetitive headline-style summary."""
        with self._lock:
            words, source, lengths = self._prepare(article)
            summary_words = self._beam_with_retry(source, lengths, beam_size, max_words)
            return self._result(summary_words, len(words))

    # -- public web entry point -------------------------------------------
    def summarize(self, article, mode='beam'):
        """Summarise ``article`` with ``mode='greedy'`` or ``mode='beam'``."""
        mode = (mode or 'beam').strip().lower()
        if mode not in ('greedy', 'beam'):
            raise ValueError(f"Unknown mode {mode!r}; expected 'greedy' or 'beam'.")

        with self._lock:
            words, source, lengths = self._prepare(article)
            if mode == 'greedy':
                generated = self.greedy_generate(source, lengths)
                summary_words = self.collapse_immediate_repeats(
                    self.decode_ids(generated[0].cpu().tolist()).split()
                )
            else:
                summary_words = self._beam_with_retry(source, lengths, BEAM_SIZE, MAX_TARGET_WORDS)
            result = self._result(summary_words, len(words))
            # Transparency: how much of the user's own text was in-vocabulary?
            result['oov_ratio'] = self._oov_ratio(source)

        if result['oov_ratio'] > OOV_WARNING_THRESHOLD:
            result['warning'] = OOV_WARNING_MESSAGE
        else:
            result['warning'] = None
        result['mode'] = mode
        return result
