"""Network definition transcribed VERBATIM from cell 6 of
``urdu_gru_100k_pretrained_colab.ipynb`` (constants from cells 2 and 5).

Nothing here is reinterpreted: the class bodies below are copy-pasted from the
notebook.  The only additions are (a) the imports the notebook performed in
earlier cells and (b) ``configure_vocab_size`` / ``DEFAULT_MODEL_DIR``, which
exist because ``ACTUAL_VOCAB_SIZE`` was a notebook global derived from
``len(itos)`` and this module is imported before the vocabulary file is read.
"""

import random  # used by CoverageGRUSummarizer.forward (teacher forcing)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---------------------------------------------------------------------------
# cell 2 -- architecture constants
# ---------------------------------------------------------------------------
SEED = 42

# Word-level vocabulary. 46,000 types keeps best_model.pt near the 510 MB target.
VOCAB_SIZE = 46_000
EMBED_DIM = 300            # Fixed by the pretrained UrduVec vectors.
ENC_HIDDEN_DIM = 512       # Decoder state is 2 * ENC_HIDDEN_DIM = 1024.
DROPOUT = 0.3
MAX_SOURCE_WORDS = 400     # Covers 91.6% of articles in full (median 163 words).
MAX_TARGET_WORDS = 32      # Titles are 11 words at the median, 19 at the 95th percentile.

# ---------------------------------------------------------------------------
# cell 5 -- token constants
# ---------------------------------------------------------------------------
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = '<pad>', '<unk>', '<s>', '</s>'
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# cell 5: ACTUAL_VOCAB_SIZE = len(itos).  The released vocab_100k.json holds
# exactly 46,000 entries; configure_vocab_size() re-asserts that at load time.
ACTUAL_VOCAB_SIZE = VOCAB_SIZE


def configure_vocab_size(vocab_size):
    """Set the module global the cell-6 classes read when they are constructed."""
    global ACTUAL_VOCAB_SIZE
    ACTUAL_VOCAB_SIZE = vocab_size


# ===========================================================================
# cell 6 -- VERBATIM
# ===========================================================================
class Encoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(DROPOUT)
        self.gru = nn.GRU(EMBED_DIM, ENC_HIDDEN_DIM, batch_first=True, bidirectional=True)

    def forward(self, source, lengths):
        embedded = self.dropout(self.embedding(source))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, torch.cat((hidden[-2], hidden[-1]), dim=1)


class CoverageAttention(nn.Module):
    """Bahdanau attention with the coverage term of See et al. (2017)."""

    def __init__(self, state_dim):
        super().__init__()
        self.energy = nn.Linear(state_dim * 2, state_dim)
        self.coverage = nn.Linear(1, state_dim)
        self.score = nn.Linear(state_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, source_mask, coverage):
        steps = encoder_outputs.size(1)
        repeated_hidden = decoder_hidden.unsqueeze(1).expand(-1, steps, -1)
        features = self.energy(torch.cat((repeated_hidden, encoder_outputs), dim=-1))
        features = features + self.coverage(coverage.unsqueeze(-1))
        scores = self.score(torch.tanh(features)).squeeze(-1)
        scores = scores.masked_fill(~source_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, embedding, state_dim):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(DROPOUT)
        self.attention = CoverageAttention(state_dim)
        self.gru = nn.GRU(EMBED_DIM + state_dim, state_dim, batch_first=True)
        self.output = nn.Linear(state_dim * 2 + EMBED_DIM, ACTUAL_VOCAB_SIZE)

    def forward(self, token, hidden, encoder_outputs, source_mask, coverage):
        embedded = self.dropout(self.embedding(token)).unsqueeze(1)
        context, attention = self.attention(hidden, encoder_outputs, source_mask, coverage)
        output, next_hidden = self.gru(torch.cat((embedded, context.unsqueeze(1)), dim=-1),
                                       hidden.unsqueeze(0))
        output = output.squeeze(1)
        logits = self.output(torch.cat((output, context, embedded.squeeze(1)), dim=-1))
        return logits, next_hidden.squeeze(0), attention


class CoverageGRUSummarizer(nn.Module):
    def __init__(self, embedding_weights=None):
        super().__init__()
        state_dim = ENC_HIDDEN_DIM * 2
        # One embedding table shared by encoder and decoder: same language, half the parameters.
        self.embedding = nn.Embedding(ACTUAL_VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        if embedding_weights is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(torch.from_numpy(embedding_weights))
        self.encoder = Encoder(self.embedding)
        self.decoder = Decoder(self.embedding, state_dim)

    def encode(self, source, lengths):
        encoder_outputs, hidden = self.encoder(source, lengths)
        source_mask = source.ne(PAD_ID)[:, :encoder_outputs.size(1)]
        coverage = torch.zeros_like(source_mask, dtype=encoder_outputs.dtype)
        return encoder_outputs, hidden, source_mask, coverage

    def forward(self, source, lengths, target, teacher_forcing_ratio=1.0):
        encoder_outputs, hidden, source_mask, coverage = self.encode(source, lengths)
        batch_size, target_steps = target.shape
        logits = torch.zeros(batch_size, target_steps - 1, ACTUAL_VOCAB_SIZE,
                             device=source.device, dtype=encoder_outputs.dtype)
        coverage_loss = source.new_zeros((), dtype=encoder_outputs.dtype)
        token = target[:, 0]
        for step in range(target_steps - 1):
            step_logits, hidden, attention = self.decoder(token, hidden, encoder_outputs,
                                                          source_mask, coverage)
            logits[:, step] = step_logits
            # Penalise attention that revisits source positions already covered.
            step_mask = target[:, step + 1].ne(PAD_ID).to(attention.dtype)
            coverage_loss = coverage_loss + (torch.min(attention, coverage).sum(dim=1) * step_mask).sum()
            coverage = coverage + attention
            teacher_forced = random.random() < teacher_forcing_ratio
            token = target[:, step + 1] if teacher_forced else step_logits.argmax(dim=-1)
        return logits, coverage_loss
