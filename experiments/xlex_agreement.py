#!/usr/bin/env python
"""Measure MFA<->CV agreement and dissect the disagreements.

Reports the overall agreement rate on shared words, then groups
disagreements by phenomenon (nasal splits, qu/x, glide-j, length) so the
canonicalizer's gaps show up as patterns rather than anecdotes.

    uv run python experiments/xlex_agreement.py [--limit N] [--json out]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402

from tiny_g2p.data import resolve_lexicon  # noqa: E402
from tiny_g2p.xlex import (  # noqa: E402
    DEFAULT_CV_PATH,
    TIE,
    canonicalize,
    compare,
    fetch_cv,
    parse_cv,
)


def _has_nasal_split(word: str, readings) -> bool:
    return "ą" in word or "ę" in word


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only compare the first N shared words (smoke test)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not DEFAULT_CV_PATH.is_file():
        print("fetching the CV dictionary...", flush=True)
        fetch_cv()
    cv = parse_cv()
    print(f"cv: {cv.kept_rows} rows, {len(cv.words)} unique words "
          f"({cv.raw_lines} raw lines)")

    mfa = Lexicon(parse_lexicon(resolve_lexicon()))
    mfa_entries: dict[str, list[tuple[str, ...]]] = {}
    for entry in mfa.entries:
        mfa_entries.setdefault(entry.word, []).append(entry.phonemes)
    print(f"mfa: {len(mfa_entries)} unique words")

    shared = sorted(set(mfa_entries) & cv.words)
    if args.limit:
        shared = shared[: args.limit]
    subset = {w: mfa_entries[w] for w in shared}
    agreement = compare(subset, cv)
    print(f"\nshared words: {agreement.shared}")
    print(f"agreed: {agreement.agreed} "
          f"({agreement.rate * 100:.2f}%)  "
          f"disputed: {len(agreement.disagreed)}")

    # Group the disputes so canonicalizer gaps show as patterns.
    kinds: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    for word in agreement.disagreed:
        mfa_canon = sorted({canonicalize(r, source="mfa")
                            for r in subset[word]})
        cv_canon = sorted({canonicalize(r, source="cv")
                           for r in cv.entries[word]})
        if _has_nasal_split(word, subset[word]):
            kind = "nasal-word"
        elif "qu" in word or "x" in word:
            kind = "qu/x-word"
        elif any(len(a) != len(b) for a in mfa_canon for b in cv_canon):
            kind = "length"
        else:
            kind = "substitution"
        kinds[kind] += 1
        samples.setdefault(kind, []).append(
            f"{word}: mfa {' | '.join(' '.join(r) for r in mfa_canon)}  "
            f"<> cv {' | '.join(' '.join(r) for r in cv_canon)}")

    print("\ndispute groups:")
    for kind, count in kinds.most_common():
        print(f"  {kind:12} {count:6d}")
        for sample in samples[kind][:6]:
            print(f"    {sample}")

    # Nasal-split probe: how does CV render ą/ę before a stop -- split like
    # MFA (oral + nasal) or an unsplit nasal vowel? Decides whether the
    # canonicalizer needs an MFA-side merge rule.
    from tiny_g2p.phones import STOPS  # noqa: E402

    stops_plain = {s.replace("̪", "") for s in STOPS}

    def cv_unsplit_before_stop(reading) -> bool:
        return any(phone in ("ɔ̃", "ɛ̃") and i + 1 < len(reading)
                   and reading[i + 1].replace(TIE, "") in stops_plain
                   for i, phone in enumerate(reading))

    def mfa_split_pair(reading) -> bool:
        return any(phone in ("ɔ", "ɛ") and i + 2 < len(reading)
                   and reading[i + 1][0] in {"m", "n", "ɲ", "ŋ"}
                   for i, phone in enumerate(reading))

    unsplit = split = 0
    for word in shared:
        if "ą" not in word and "ę" not in word:
            continue
        unsplit += sum(1 for r in cv.entries[word]
                       if cv_unsplit_before_stop(r))
        split += sum(1 for r in mfa_entries[word] if mfa_split_pair(r))
    print("\nnasal-split probe (readings of ą/ę-words):")
    print(f"  CV nasal vowel directly before a stop: {unsplit}")
    print(f"  MFA oral+nasal pairs in the same words: {split}")
    if unsplit == 0 and split > 0:
        print("  CV splits nasals before stops exactly like MFA: "
              "no merge rule needed.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"shared": agreement.shared, "agreed": agreement.agreed,
             "rate": round(agreement.rate, 6),
             "groups": dict(kinds), "disputed": agreement.disagreed},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
