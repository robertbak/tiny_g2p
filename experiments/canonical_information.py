#!/usr/bin/env python
"""What does the canonical form cost? An ablation, step by step.

The canonical form is not an alignment. An alignment would pair the two
sequences up (`ɔ̃` against `ɔ n̪`) and keep both; the canonical form instead
*rewrites both sides into one alphabet* and compares the results. That is a
many-to-one projection, so the honest question is not "does it lose anything"
(it does) but "does it lose contrasts, or only notation".

This measures both halves for each step of the pipeline:

* **what it costs** -- how many distinct raw readings collapse onto one
  canonical key (fan-in), and how many *different words* end up
  indistinguishable ("homophone classes");
* **what it buys** -- how much MFA/CV cross-lexicon agreement the step is
  responsible for, i.e. how many disagreements it resolves.

A step that resolves a lot of disagreement and adds no homophone classes is
notation. A step that adds classes is making a linguistic claim, and the
examples say whether the claim is true (``morze``/``może`` really are
homophones) or a real contrast being flattened.

    uv run python experiments/canonical_information.py [--json out]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402

from tiny_g2p.data import resolve_lexicon  # noqa: E402
from tiny_g2p.xlex import parse_cv  # noqa: E402

# The pipeline lives in one place; ablating a copy would measure the copy.
from distance_for_asr import STEPS, pipeline  # noqa: E402

# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------

def fan_in(entries: list[tuple[str, tuple[str, ...]]],
           skip: str | None) -> tuple[int, int, list]:
    """(distinct keys, largest collapse group, examples)."""
    groups: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for _, phones in entries:
        groups[pipeline(list(phones), skip)].add(phones)
    largest = max((len(v) for v in groups.values()), default=0)
    examples = sorted(groups.values(), key=len, reverse=True)[:3]
    return len(groups), largest, examples


def homophone_classes(entries: list[tuple[str, tuple[str, ...]]],
                      skip: str | None) -> tuple[int, int, list]:
    """Spellings that become indistinguishable: (classes, words, examples)."""
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for word, phones in entries:
        groups[pipeline(list(phones), skip)].add(word)
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    words = sum(len(v) for v in merged.values())
    examples = sorted(merged.values(), key=len, reverse=True)
    return len(merged), words, examples


def agreement_partial(mfa: dict[str, list[tuple[str, ...]]],
                      cv_entries: dict[str, list[tuple[str, ...]]],
                      upto: list[str]) -> float:
    """Agreement after applying only the first ``upto`` steps."""
    def project(phones):
        out = list(phones)
        for name, step in STEPS:
            if name in upto:
                out = step(out)
        return tuple(out)

    shared = sorted(set(mfa) & set(cv_entries))
    agreed = 0
    for word in shared:
        left = {project(r) for r in mfa[word]}
        right = {project(r) for r in cv_entries[word]}
        if not left.isdisjoint(right):
            agreed += 1
    return agreed / len(shared) if shared else 0.0


def agreement(mfa: dict[str, list[tuple[str, ...]]],
              cv_entries: dict[str, list[tuple[str, ...]]],
              skip: str | None, *, use_canonical: bool = True) -> float:
    """Share of shared words whose MFA and CV readings meet after projection."""
    shared = sorted(set(mfa) & set(cv_entries))
    agreed = 0
    for word in shared:
        left = {pipeline(list(r), skip) if use_canonical else tuple(r)
                for r in mfa[word]}
        right = {pipeline(list(r), skip) if use_canonical else tuple(r)
                 for r in cv_entries[word]}
        if not left.isdisjoint(right):
            agreed += 1
    return agreed / len(shared) if shared else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    entries = [(e.word, tuple(e.phonemes)) for e in lexicon.entries]
    mfa: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for word, phones in entries:
        mfa[word].append(phones)
    cv = parse_cv()

    full_keys, _, _ = fan_in(entries, None)
    full_classes, full_words, class_examples = homophone_classes(entries, None)
    print(f"lexicon: {len(entries):,} entries, {len({p for _, p in entries}):,} distinct readings")
    print(f"canonical (all steps): {full_keys:,} keys, {full_classes:,} homophone "
          f"classes covering {full_words:,} words")
    print(f"\nagreement MFA<->CV: raw {agreement(mfa, cv.entries, None, use_canonical=False) * 100:.2f}%"
          f"   canonical {agreement(mfa, cv.entries, None) * 100:.2f}%")

    print("\n--- what each step costs: drop it, see what stops collapsing")
    print(f"{'step':42} {'keys':>8} {'Δkeys':>7} {'classes':>8} {'Δclasses':>9}")
    rows = []
    for name, _ in STEPS:
        keys, _, _ = fan_in(entries, name)
        classes, words, _ = homophone_classes(entries, name)
        rows.append({"step": name, "keys": keys, "classes": classes, "words": words,
                     "delta_keys": keys - full_keys, "delta_classes": classes - full_classes})
        print(f"{name:42} {keys:8,} {keys - full_keys:+7,} {classes:8,} "
              f"{classes - full_classes:+9,}")

    # Cumulative, in pipeline order: each step's marginal contribution. A
    # leave-one-out view would understate the early steps (without the dental
    # strip, *nothing* agrees, so every later step looks useless).
    print("\n--- what each step buys, in order (cumulative MFA<->CV agreement)")
    print(f"{'after':42} {'agreement':>10} {'Δ step':>9}")
    previous = agreement(mfa, cv.entries, None, use_canonical=False)
    print(f"{'(raw readings)':42} {previous * 100:9.2f}%")
    for name, _ in STEPS:
        upto = [s for s, _ in STEPS][: [s for s, _ in STEPS].index(name) + 1]
        rate = agreement_partial(mfa, cv.entries, upto)
        print(f"{name:42} {rate * 100:9.2f}% {(rate - previous) * 100:+8.2f}pp")
        for row in rows:
            if row["step"] == name:
                row["agreement_after"] = round(rate, 6)
                row["agreement_step"] = round(rate - previous, 6)
        previous = rate

    print("\n--- the homophone classes the full pipeline creates (largest)")
    for spellings in class_examples:
        print(f"  {sorted(spellings)[:8]}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "entries": len(entries),
            "distinct_readings": len({p for _, p in entries}),
            "canonical_keys": full_keys,
            "homophone_classes": full_classes,
            "homophone_words": full_words,
            "agreement_raw": round(agreement(mfa, cv.entries, None,
                                             use_canonical=False), 6),
            "agreement_canonical": round(baseline, 6),
            "steps": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
