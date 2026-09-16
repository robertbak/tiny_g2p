"""Dataset: the same words as pl_g2p, labelled per character.

Three deliberate choices:

**Same splits.** Partitioning reuses :func:`pl_g2p.data.split_words` with the
same seed, so the test words are identical to the transformer's -- the two
models' numbers are directly comparable.

**Labels are a train-only teacher artifact.** Only train rows are aligned;
val/test keep just (word, raw phones) and are scored by decoding, never by
label loss. Rows the aligner rejects (structurally unalignable acronyms, or
forced-noise alignments above the cost gate) are excluded from *training* but
stay in val/test: the model cannot learn them, but eval must stay honest.

**Weak-group multipliers.** Examples with nasal splits or rare phones are
upweighted (x2 / x3), the analogue of gpu-lexer's gentle multipliers for weak
language families.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from phoneme_lab.features import to_features
from pl_g2p.data import build_examples, split_words
from pl_g2p.lexicon import MFA_DICT_SHA256, Lexicon, fetch_lexicon, sha256_of

from .align import align_entry
from .vocab import BLANK, Vocab

# NOTE: pl_g2p.baseline.normalize() is intentionally NOT used for training
# targets. It merges every vowel+nasal sequence (including genuine clusters
# like "pan" -> "p ã"), which is fair for scoring both sides symmetrically
# but destroys information a classifier must predict. The aligner's
# word-aware nasal merge (ą/ę only) replaces it.

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "data" / "lexicons" / "polish_mfa.dict"
SIBLING_PATH = REPO_ROOT.parent / "pl_g2p" / "data" / "lexicons" / "polish_mfa.dict"

#: Promoted artifacts (tracked, like gpu-lexer's active/): the float model,
#: its quantized twin, vocabularies, and selection metadata.
ACTIVE_DIR = REPO_ROOT / "active"
#: Per-run outputs (ignored): weights, history, metadata.
RUNS_DIR = REPO_ROOT / "runs"
#: Local failure words for fine-tuning (ignored, never val/test words).
FAILURE_BANK_DIR = REPO_ROOT / "failure_bank"

SEED = 20260214
RATIOS = (0.90, 0.05, 0.05)

#: Raw phones rarer than this in train count as the weak "rare" group.
RARE_THRESHOLD = 500

#: Static example multipliers (weak-group curriculum).
WEIGHT_RARE = 3.0
WEIGHT_NASAL = 2.0

#: Per-position loss weight at a label boundary (phone changes, including
#: to/from blank) versus inside a run. The boundary-weighted-loss analogue.
BOUNDARY_WEIGHT = 2.0

#: Auxiliary head classes: articulatory manner of the label phone.
MANNERS = ("plosive", "affricate", "fricative", "nasal", "trill", "lateral",
           "approximant", "vowel", "blank", "other")


def resolve_lexicon(*, auto_fetch: bool = True) -> Path:
    """Locate a checksum-valid dictionary: own copy, sibling copy, or fetch."""
    for candidate in (DEFAULT_PATH, SIBLING_PATH):
        if candidate.is_file() and sha256_of(candidate) == MFA_DICT_SHA256:
            return candidate
    if not auto_fetch:
        raise FileNotFoundError(
            f"no valid lexicon at {DEFAULT_PATH} or {SIBLING_PATH}")
    return fetch_lexicon(DEFAULT_PATH)


@dataclass
class Example:
    """One train row: a word, its raw phones, and per-character labels."""

    word: str
    phones: tuple[str, ...]
    labels: tuple[str | None, ...]
    weight: float = 1.0
    boundary: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.boundary and self.labels:
            self.boundary = tuple(_boundary_weights(self.labels))

    @property
    def src_len(self) -> int:
        return len(self.word)

    @property
    def tgt_len(self) -> int:
        return len(self.word)


@dataclass
class EvalRow:
    """One val/test row: scored by decoding, never by label loss."""

    word: str
    phones: tuple[str, ...]


@dataclass
class DataSet:
    """Train examples plus val/test rows and the train-derived vocabularies."""

    train: list[Example]
    val: list[EvalRow]
    test: list[EvalRow]
    src_vocab: Vocab
    tgt_vocab: Vocab
    manner_of: dict[str, int] = field(default_factory=dict)
    excluded: int = 0
    rare: frozenset = frozenset()

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val),
                "test": len(self.test), "excluded": self.excluded}


def _manner(phone: str | None) -> str:
    if phone is None:
        return "blank"
    feats = to_features(phone)
    return feats.manner if feats is not None else "other"


def _boundary_weights(labels: Sequence[str | None]) -> list[float]:
    weights = [1.0] * len(labels)
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            weights[i] = BOUNDARY_WEIGHT
    return weights


def make_dataset(lexicon: Lexicon, *, seed: int = SEED,
                 ratios: tuple[float, float, float] = RATIOS,
                 align_train: bool = True) -> DataSet:
    """Partition by word (same code and seed as pl_g2p) and label train.

    ``align_train=False`` skips train labelling (and train vocabs stay empty),
    for eval-only flows that need just the val/test rows.
    """
    examples = build_examples(lexicon)
    parts = split_words((e.word for e in examples), seed=seed, ratios=ratios)
    where = {w: name for name, words in parts.items() for w in words}

    phone_counts: Counter[str] = Counter()
    for example in examples:
        if where[example.word] == "train":
            phone_counts.update(example.phonemes)
    rare = {p for p, n in phone_counts.items() if n < RARE_THRESHOLD}

    train: list[Example] = []
    val: list[EvalRow] = []
    test: list[EvalRow] = []
    excluded = 0
    for example in examples:
        split = where[example.word]
        if split != "train":
            row = EvalRow(example.word, example.phonemes)
            (val if split == "val" else test).append(row)
            continue
        if not align_train:
            continue
        alignment = align_entry(example.word, example.phonemes)
        if alignment is None:
            excluded += 1
            continue
        labels = alignment.labels
        weight = 1.0
        if any(ch in "ąę" for ch in example.word):
            weight = max(weight, WEIGHT_NASAL)
        if any(p in rare for p in example.phonemes):
            weight = max(weight, WEIGHT_RARE)
        train.append(Example(example.word, example.phonemes, labels, weight,
                             tuple(_boundary_weights(labels))))

    src_vocab = Vocab.build_chars(e.word for e in train)
    tgt_vocab = Vocab.build_phones(e.labels for e in train)
    manner_of = {phone: MANNERS.index(_manner(phone))
                 for phone in tgt_vocab.itos if phone != BLANK}
    manner_of[BLANK] = MANNERS.index("blank")
    return DataSet(train, val, test, src_vocab, tgt_vocab, manner_of,
                   excluded, frozenset(rare))


def references_by_word(rows: Sequence[EvalRow]) -> dict[str, list[tuple[str, ...]]]:
    """Group rows into {word: [acceptable pronunciations]} for scoring."""
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for row in rows:
        grouped.setdefault(row.word, []).append(row.phones)
    return grouped


class CharDataset(Dataset):
    """Encodes train examples into per-character id/weight tensors."""

    def __init__(self, examples: Sequence[Example], src_vocab: Vocab,
                 tgt_vocab: Vocab, manner_of: dict[str, int]):
        self.examples = list(examples)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.manner_of = manner_of
        self.blank_manner = MANNERS.index("blank")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        chars = torch.tensor(self.src_vocab.encode(example.word),
                             dtype=torch.long)
        labels = torch.tensor(
            [self.tgt_vocab.blank_id if p is None else self.tgt_vocab.stoi[p]
             for p in example.labels], dtype=torch.long)
        manner = torch.tensor(
            [self.manner_of.get(p if p is not None else BLANK,
                                self.blank_manner)
             for p in example.labels], dtype=torch.long)
        return {
            "chars": chars,
            "labels": labels,
            "manner": manner,
            "boundary": torch.tensor(example.boundary, dtype=torch.float),
            "weight": torch.tensor(example.weight, dtype=torch.float),
            "length": len(example.word),
        }


def collate(batch: Sequence[dict], *, pad_id: int = 0) -> dict:
    """Pad a batch of per-character examples to the longest word."""
    width = max(item["length"] for item in batch)
    out: dict[str, torch.Tensor] = {}
    for key, dtype, fill in (("chars", torch.long, pad_id),
                             ("labels", torch.long, pad_id),
                             ("manner", torch.long, 0),
                             ("boundary", torch.float, 0.0)):
        stacked = torch.full((len(batch), width), fill, dtype=dtype)
        for i, item in enumerate(batch):
            stacked[i, : item["length"]] = item[key][: item["length"]]
        out[key] = stacked
    out["weight"] = torch.stack([item["weight"] for item in batch])
    out["lengths"] = torch.tensor([item["length"] for item in batch],
                                  dtype=torch.long)
    return out
