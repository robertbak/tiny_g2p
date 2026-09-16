"""Decoding with a trained model: argmax, blank-drop, restore, score.

Checkpoints come in two flavours, mirroring gpu-lexer's float + deployed pair:

* ``model.pt``           float ``state_dict`` + vocabularies + metadata;
* ``model_quantized.pt`` the converted static-int8 module (CPU-only).

Selection compares float and quantized candidates on the same held-out
metric; promotion persists both, and inference defaults to the quantized
twin -- the thing actually deployed is the thing being measured.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
from pl_g2p.metrics import Score, score

from .align import DIGRAPHS
from .data import ACTIVE_DIR, MANNERS, EvalRow, references_by_word
from .model import build
from .phones import restore
from .vocab import BLANK, PAD, UNK, Vocab

FLOAT_NAME = "model.pt"
QUANTIZED_NAME = "model_quantized.pt"
SRC_VOCAB_NAME = "src_vocab.json"
TGT_VOCAB_NAME = "tgt_vocab.json"
META_NAME = "meta.json"


def normalize_input(word: str) -> str:
    """NFC-normalise and lowercase: the lexicon is all-lowercase."""
    return unicodedata.normalize("NFC", word).strip().lower()


def resolve_device(device: str = "auto") -> torch.device:
    """Map "auto"/"cuda"/"cpu" to a torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TinyPhonemizer:
    """A loaded model that turns words into MFA phones."""

    model: torch.nn.Module
    src_vocab: Vocab
    tgt_vocab: Vocab
    device: torch.device
    quantized: bool = False

    def predict_word(self, word: str) -> list[str]:
        return self.predict_batch([word])[0]

    @torch.no_grad()
    def predict_batch(self, words: list[str]) -> list[list[str]]:
        words = [normalize_input(w) for w in words]
        if not words:
            return []
        width = max(len(w) for w in words)
        ids = torch.full((len(words), width), self.src_vocab.pad_id,
                         dtype=torch.long)
        for i, word in enumerate(words):
            encoded = self.src_vocab.encode(word)
            ids[i, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        logits, _ = self.model(ids.to(self.device))
        best = logits.argmax(dim=-1).cpu()
        out = []
        for word, row in zip(words, best):
            labels: list[str | None] = []
            for pos, idx in enumerate(row[: len(word)].tolist()):
                token = self.tgt_vocab.itos[idx]
                labels.append(None if token in (PAD, UNK, BLANK) else token)
            out.append(restore(word, labels))
        return out

    @classmethod
    def from_dir(cls, directory: Path | str, *,
                 quantized: bool = True,
                 device: str | torch.device = "auto") -> "TinyPhonemizer":
        """Load a checkpoint directory (``active/`` or a run)."""
        directory = Path(directory)
        src_vocab = Vocab.from_json(directory / SRC_VOCAB_NAME)
        tgt_vocab = Vocab.from_json(directory / TGT_VOCAB_NAME)
        if quantized:
            model = torch.load(directory / QUANTIZED_NAME, map_location="cpu",
                               weights_only=False)
            return cls(model, src_vocab, tgt_vocab, torch.device("cpu"),
                       quantized=True)
        resolved = resolve_device(device) if isinstance(device, str) else device
        model = build(len(src_vocab), len(tgt_vocab), len(MANNERS))
        payload = torch.load(directory / FLOAT_NAME, map_location=resolved,
                             weights_only=True)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(model.to(resolved), src_vocab, tgt_vocab, resolved)

    @classmethod
    def from_active(cls, **kwargs) -> "TinyPhonemizer":
        """Load the promoted checkpoint; fails loudly when absent."""
        if not (ACTIVE_DIR / FLOAT_NAME).is_file():
            raise FileNotFoundError(
                f"no promoted model at {ACTIVE_DIR}. Train one with "
                "`make train` (it auto-promotes when eligible).")
        return cls.from_dir(ACTIVE_DIR, **kwargs)


def evaluate_words(phonemizer: TinyPhonemizer, rows: list[EvalRow],
                   *, batch_size: int = 512, keep_errors: int = 0) -> Score:
    """Score decoded pronunciations against (polyphony-aware) references."""
    refs = references_by_word(rows)
    words = sorted(refs)
    predictions: list[list[str]] = []
    for i in range(0, len(words), batch_size):
        predictions.extend(phonemizer.predict_batch(words[i:i + batch_size]))
    return score(predictions, [refs[w] for w in words], words=words,
                 keep_errors=keep_errors)


def guard_groups(rows: list[EvalRow], rare: set[str]) -> dict[str, list[EvalRow]]:
    """Split eval rows into the promotion-guard subsets.

    ``nasal`` (ą/ę), ``digraph`` (cz/sz/rz/...) and ``palatal`` (ʲ phones)
    pin the phenomena the per-character model finds hardest; ``rare`` pins
    the long tail. Guards compare these against the promoted baseline on
    the same rows, so a run cannot buy overall accuracy by sacrificing
    a phenomenon.
    """
    groups: dict[str, list[EvalRow]] = {
        "all": list(rows), "nasal": [], "digraph": [], "palatal": [],
        "rare": [],
    }
    for row in rows:
        if "ą" in row.word or "ę" in row.word:
            groups["nasal"].append(row)
        if any(d in row.word for d in DIGRAPHS):
            groups["digraph"].append(row)
        if any("ʲ" in p for p in row.phones):
            groups["palatal"].append(row)
        if any(p in rare for p in row.phones):
            groups["rare"].append(row)
    return groups


def evaluate_guards(phonemizer: TinyPhonemizer, rows: list[EvalRow],
                    rare: set[str]) -> dict[str, Score]:
    """Score every guard group; used by selection and promotion."""
    return {name: evaluate_words(phonemizer, subset)
            for name, subset in guard_groups(rows, rare).items()}


def write_meta(directory: Path, meta: dict) -> None:
    """Write selection metadata (accuracy, guards, taxonomy, provenance)."""
    (directory / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def read_meta(directory: Path) -> dict:
    return json.loads((directory / META_NAME).read_text(encoding="utf-8"))
