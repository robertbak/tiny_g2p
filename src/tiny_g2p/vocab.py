"""Tiny vocabularies: characters in, phones-or-blank out.

A per-character classifier needs PAD (batching), UNK (unseen characters) and
BLANK (the label for characters that emit nothing) -- and nothing else. This
is deliberately not :mod:`pl_g2p.vocab`, whose BOS/EOS serve an autoregressive
decoder this model does not have.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PAD = "<pad>"
UNK = "<unk>"
BLANK = "<blank>"


@dataclass
class Vocab:
    """A bidirectional token<->id mapping with fixed special-token ids."""

    itos: list[str]
    pad_id: int = 0

    def __post_init__(self) -> None:
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def blank_id(self) -> int:
        return self.stoi[BLANK]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def taxonomy(self) -> str:
        """Short hash of the inventory: a changed taxonomy means retraining.

        The analogue of gpu-lexer's label-taxonomy version -- a checkpoint is
        only a valid warm start for data with the same taxonomy.
        """
        return hashlib.sha1(" ".join(self.itos).encode("utf-8")).hexdigest()[:12]

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(data["itos"]))

    @classmethod
    def build_chars(cls, words: Iterable[str]) -> "Vocab":
        seen: set[str] = set()
        for word in words:
            seen.update(word)
        return cls([PAD, UNK] + sorted(seen - {PAD, UNK}))

    @classmethod
    def build_phones(cls, labels: Iterable[Sequence[str | None]]) -> "Vocab":
        seen: set[str] = set()
        for seq in labels:
            seen.update(p for p in seq if p is not None)
        return cls([PAD, UNK, BLANK] + sorted(seen - {PAD, UNK, BLANK}))
