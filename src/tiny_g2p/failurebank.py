"""The failure bank: words the model got wrong, kept for fine-tuning.

Mirrors gpu-lexer's local failure bank: a snippet is added, then trained with
balanced failure-heavy updates followed by ordinary rehearsal. Two rules keep
this honest:

* bank words are labelled by prg2p (the rule-based external teacher -- the
  Shiki analogue), never by the model itself;
* val/test words are refused outright: fine-tuning on verification items is
  leakage, however tempting the score improvement.
"""

from __future__ import annotations

from pathlib import Path

from .data import FAILURE_BANK_DIR

BANK_NAME = "words.txt"


def bank_path() -> Path:
    return FAILURE_BANK_DIR / BANK_NAME


def read_bank() -> list[str]:
    """Bank words in first-added order (blanks and #-comments skipped)."""
    path = bank_path()
    if not path.is_file():
        return []
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            words.append(word)
    return words


def add_words(words: list[str]) -> list[str]:
    """Append words to the bank, skipping duplicates; returns the bank."""
    seen = set(read_bank())
    FAILURE_BANK_DIR.mkdir(parents=True, exist_ok=True)
    with bank_path().open("a", encoding="utf-8") as fh:
        for word in words:
            word = word.strip()
            if word and word not in seen:
                fh.write(word + "\n")
                seen.add(word)
    return read_bank()


def check_leakage(words: list[str], val_words: set[str],
                  test_words: set[str]) -> list[str]:
    """Bank candidates that are verification items (must be refused)."""
    return [w for w in words if w in val_words or w in test_words]
