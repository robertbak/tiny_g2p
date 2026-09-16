#!/usr/bin/env python
"""Align every lexicon row and round-trip restore() back to raw MFA.

Two design gates in one pass:

1. **Aligner coverage.** How many rows align, how many are structurally
   unalignable (more phones than characters even with the nasal 1:2 op),
   and what the mean-cost distribution looks like -- the quality gate
   threshold is set from this tail, not guessed.
2. **Restore round-trip.** ``restore(word, align(word, raw)) == raw`` must
   hold for every aligned row: it proves the nasal merge loses nothing and
   the homorganic table is complete.

    uv run python experiments/align_coverage.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402

from tiny_g2p.align import align  # noqa: E402
from tiny_g2p.phones import restore  # noqa: E402

SIBLING_DICT = ROOT.parent / "pl_g2p" / "data" / "lexicons" / "polish_mfa.dict"


def main() -> int:
    if not SIBLING_DICT.is_file():
        print(f"no lexicon at {SIBLING_DICT}", file=sys.stderr)
        return 2
    lex = Lexicon(parse_lexicon(SIBLING_DICT))
    started = time.time()

    aligned = 0
    unalignable: list[tuple[str, tuple[str, ...]]] = []
    costs: list[float] = []
    cost_hist: Counter[str] = Counter()
    costly: list[tuple[float, str, tuple[str, ...], tuple[str | None, ...]]] = []
    mismatches: list[tuple[str, list[str], list[str]]] = []

    for k, entry in enumerate(lex.entries):
        if k % 20000 == 0:
            print(f"  {k}/{len(lex)}...", flush=True)
        result = align(entry.word, entry.phonemes)
        if result is None:
            unalignable.append((entry.word, entry.phonemes))
            continue
        aligned += 1
        costs.append(result.mean_cost)
        cost_hist[f"{result.mean_cost:.2f}"] += 1
        if result.mean_cost >= 0.3:
            costly.append((result.mean_cost, entry.word, entry.phonemes,
                           result.labels))
        back = restore(entry.word, list(result.labels))
        if back != list(entry.phonemes):
            mismatches.append((entry.word, back, list(entry.phonemes)))

    print(f"\nentries {len(lex)}  aligned {aligned} "
          f"({aligned / len(lex) * 100:.2f}%)  "
          f"unalignable {len(unalignable)}  ({time.time() - started:.0f}s)")
    print("\nunalignable samples:")
    for word, phones in unalignable[:12]:
        print(f"  {word} -> {' '.join(phones)}")

    costs.sort()
    print("\nmean-cost percentiles (aligned rows):")
    for pct in (50, 90, 95, 99, 99.5, 99.9, 100):
        idx = min(int(len(costs) * pct / 100), len(costs) - 1)
        print(f"  p{pct:<5} {costs[idx]:.4f}")
    print("\nmean-cost histogram (bin -> rows):")
    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 2.0]
    for low, high in zip(edges, edges[1:]):
        count = sum(1 for c in costs if low <= c < high or (low == 0.0 and c == 0.0 and high == 0.05))
        print(f"  {low:.2f}-{high:.2f}  {count}")

    print(f"\nhigh-cost tail (mean >= 0.30): {len(costly)}")
    for cost, word, phones, labels in sorted(costly, reverse=True)[:15]:
        print(f"  {cost:.3f} {word} -> {' '.join(phones)}")
        print(f"         labels {' '.join(p or '_' for p in labels)}")

    print(f"\nrestore mismatches: {len(mismatches)}")
    for word, back, raw in mismatches[:15]:
        print(f"  {word}\n    back {' '.join(back)}\n    raw  {' '.join(raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
