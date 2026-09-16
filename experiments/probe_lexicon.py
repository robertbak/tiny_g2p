#!/usr/bin/env python
"""Probe the MFA lexicon to settle tiny_g2p's design questions.

A per-character classifier needs targets no longer than the word (one label
per character), so this measures, on raw and normalised MFA phones:

* how often the phone sequence is longer than the word (U > T), and why;
* the raw and normalised phone inventories;
* nasal-vowel contexts (can normalisation be *restored* deterministically?);
* palatalisation (ʲ) contexts;
* what the letter ``x`` transcribes to;
* the longest word (the conv stack's receptive field must cover it).

    uv run python experiments/probe_lexicon.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_g2p.baseline import normalize  # noqa: E402
from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402

SIBLING_DICT = ROOT.parent / "pl_g2p" / "data" / "lexicons" / "polish_mfa.dict"

STOPS = frozenset({"p", "b", "t", "d", "k", "g", "c", "ɟ",
                   "tʂ", "dʐ", "tɕ", "dʑ", "ts", "dz", "ʔ"})
FRICATIVES = frozenset({"f", "v", "s", "z", "ʂ", "ʐ", "ɕ", "ʑ", "x", "ç"})
NASALS = frozenset({"m", "n", "ɲ", "ŋ"})
LIQUIDS = frozenset({"r", "l", "ʎ", "w", "j"})


def follower_class(phone: str | None) -> str:
    if phone is None:
        return "END"
    base = phone[0]
    if phone in STOPS or base in {"p", "b", "t", "d", "k", "g"}:
        return "STOP"
    if phone in FRICATIVES or base in {"f", "v", "s", "z", "ʂ", "ʐ", "ɕ", "ʑ", "x"}:
        return "FRICATIVE"
    if base in {"m", "n", "ɲ", "ŋ"}:
        return "NASAL"
    if base in {"r", "l", "ʎ", "w", "j"}:
        return "LIQUID"
    return "VOWEL"


def main() -> int:
    if not SIBLING_DICT.is_file():
        print(f"no lexicon at {SIBLING_DICT}; run `make lexicon` in ../pl_g2p", file=sys.stderr)
        return 2
    lex = Lexicon(parse_lexicon(SIBLING_DICT))
    print(f"entries {len(lex)}  unique words {lex.unique_words}  "
          f"polyphonic {len(lex.polyphonic)}")
    print(f"max word length {max(len(e.word) for e in lex.entries)}")

    raw_inv: Counter[str] = Counter()
    norm_inv: Counter[str] = Counter()
    raw_longer: list[tuple[str, tuple[str, ...]]] = []
    norm_longer: list[tuple[str, list[str]]] = []
    nasal_followers: Counter[str] = Counter()
    split_trigrams: Counter[tuple[str, str, str]] = Counter()
    palatalised: Counter[str] = Counter()
    palatal_samples: list[tuple[str, tuple[str, ...]]] = []
    x_samples: list[tuple[str, tuple[str, ...]]] = []

    for entry in lex.entries:
        raw = list(entry.phonemes)
        norm = normalize(raw)
        raw_inv.update(raw)
        norm_inv.update(norm)
        if len(raw) > len(entry.word):
            raw_longer.append((entry.word, entry.phonemes))
        if len(norm) > len(entry.word):
            norm_longer.append((entry.word, norm))
        for i, phone in enumerate(norm):
            if phone in ("ɔ̃", "ɛ̃"):
                follower = norm[i + 1] if i + 1 < len(norm) else None
                nasal_followers[follower_class(follower)] += 1
        for i in range(len(raw) - 2):
            vow, nasal, foll = raw[i], raw[i + 1], raw[i + 2]
            if vow in ("ɔ", "ɛ", "a") and nasal[0] in {"m", "n", "ɲ", "ŋ"}:
                split_trigrams[(vow, nasal, follower_class(foll))] += 1
        for phone in raw:
            if "ʲ" in phone:
                palatalised[phone] += 1
                if len(palatal_samples) < 12:
                    palatal_samples.append((entry.word, entry.phonemes))
        if "x" in entry.word and len(x_samples) < 12:
            x_samples.append((entry.word, entry.phonemes))

    print(f"\nU > T raw: {len(raw_longer)} / {len(lex)}")
    for word, phones in raw_longer[:10]:
        print(f"  {word} -> {' '.join(phones)}")
    print(f"U > T normalised: {len(norm_longer)} / {len(lex)}")
    for word, phones in norm_longer[:10]:
        print(f"  {word} -> {' '.join(phones)}")

    print(f"\nraw inventory ({len(raw_inv)}): {' '.join(sorted(raw_inv))}")
    print(f"normalised inventory ({len(norm_inv)}): {' '.join(sorted(norm_inv))}")
    plain_alveolars = [p for p in ("t", "d", "n", "s", "z", "ts", "dz") if p in raw_inv]
    print(f"plain alveolars present raw (restore ambiguity): {plain_alveolars or 'none'}")

    print("\nnormalised nasal vowel followers:")
    for follower, count in nasal_followers.most_common():
        print(f"  {follower:10} {count:6d}")
    print("\nraw oral-vowel + nasal + follower trigrams (top 15):")
    for (vow, nasal, foll), count in split_trigrams.most_common(15):
        print(f"  {vow} {nasal} + {foll:10} {count:6d}")

    print(f"\npalatalised phones: {dict(palatalised.most_common())}")
    for word, phones in palatal_samples:
        print(f"  {word} -> {' '.join(phones)}")
    print("\nletter x samples:")
    for word, phones in x_samples:
        print(f"  {word} -> {' '.join(phones)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
