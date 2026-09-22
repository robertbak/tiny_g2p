#!/usr/bin/env python
"""Search the distance metric against the ASR-recovery objective.

``distance_for_asr.py`` compares variants by running each one and eyeballing two
independent top-1 numbers. At n=150 a 4pp gap is inside the binomial interval, so
that design cannot resolve the differences it is used to decide. This harness
scores every variant on the *same* corrupted samples -- the corruption RNG is
re-seeded per variant exactly as the original does -- and reports the paired
outcome plus an exact sign test on the discordant samples.

It also carries the two distance models the original never tried:

* **affine gaps** -- a run of L skipped phones costs ``gap_open + gap_extend*(L-1)``
  instead of ``L * indel``, which is what word-boundary drift actually looks like
  (``AuraSync`` heard as ``aurora sink`` is one gap, not four indels);
* **fitted parameters** -- ``WEIGHTS`` and every alignment constant are swept
  against MRR rather than set by hand. ``Lakretz et al. 2018`` learn exactly this
  from confusion data; here the recovery objective plays that role.

    uv run python experiments/metric_search.py validate
    uv run python experiments/metric_search.py gaps
    uv run python experiments/metric_search.py weights
    uv run python experiments/metric_search.py compare --a polish --b polish-affine
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import distance_for_asr as D  # noqa: E402
from phoneme_lab.features import WEIGHTS, to_features  # noqa: E402
from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402
from tiny_g2p.data import make_dataset, references_by_word, resolve_lexicon  # noqa: E402

# --------------------------------------------------------------------------
# the vocabulary and the corruptions
# --------------------------------------------------------------------------

FAMILIES: dict[str, object] = {
    "notation": D.corrupt_notation,
    "mishearing1": lambda p, r: D.corrupt_mishearing(p, r, edits=1),
    "mishearing2": lambda p, r: D.corrupt_mishearing(p, r, edits=2),
    "both1": lambda p, r: D.corrupt_mishearing(D.corrupt_notation(p, r), r, edits=1),
    # The families above are near-ceiling for the canonical form: it solves the
    # notation outright. The two-error families are where any further gain has
    # to show up.
    "both2": lambda p, r: D.corrupt_mishearing(D.corrupt_notation(p, r), r, edits=2),
}

HARD = ("mishearing2", "both2")


def load(limit: int, seed: int) -> tuple[list, dict[str, list]]:
    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    data = make_dataset(lexicon, align_train=False)
    refs = references_by_word(data.test)
    vocabulary = [(w, list(r[0])) for w, r in sorted(refs.items())]
    rng = random.Random(seed)
    sample = rng.sample(vocabulary, min(limit, len(vocabulary)))
    # One rng per family, advanced across the whole sample -- ``main`` in
    # distance_for_asr.py re-seeds per *variant*, so every variant sees the same
    # corruption. Re-seeding per *sample* instead would give each word the first
    # draw of the stream and quietly measure a different benchmark.
    heard: dict[str, list] = {}
    for name, corrupt in FAMILIES.items():
        family_rng = random.Random(seed)
        heard[name] = [corrupt(list(phones), family_rng) for _, phones in sample]
    return sample, heard


# --------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------

def make_feature_distance(weights: dict[str, float]):
    """``phoneme_lab``'s metric with replaceable weights.

    Same shape as the shipped one: axes inapplicable to a phone are dropped from
    the denominator, and unmarked agreement on palatalisation/nasality is
    dropped too. Only the weights move.
    """
    keys = tuple(weights)

    # ~100 distinct phones -> a pair table of ~10k entries, but the DP below
    # calls this O(len(sample) * len(vocabulary) * len(phone)^2) times. Rebuilding
    # both feature dicts per call dominated every sweep; memoize the pair.
    @lru_cache(maxsize=None)
    def distance(a: str, b: str) -> float:
        if a == b:
            return 0.0
        fa, fb = to_features(a), to_features(b)
        if fa is None or fb is None:
            return 1.0
        da, db = fa.as_dict(), fb.as_dict()
        mismatched = total = 0.0
        for key in keys:
            va, vb = da[key], db[key]
            if va is None or vb is None:
                continue
            if key in ("palatalized", "nasal") and va is False and vb is False:
                continue
            total += weights[key]
            if va != vb:
                mismatched += weights[key]
        return mismatched / total if total else 1.0

    return distance


def flat_edit(sub, *, indel: float = 1.0):
    """Levenshtein with a substitution cost function. ``distance_for_asr``'s DP."""
    def distance(a: list[str], b: list[str]) -> float:
        if not a:
            return len(b) * indel
        if not b:
            return len(a) * indel
        previous = [j * indel for j in range(len(b) + 1)]
        for i, pa in enumerate(a, 1):
            current = [i * indel] + [0.0] * len(b)
            for j, pb in enumerate(b, 1):
                current[j] = min(previous[j] + indel, current[j - 1] + indel,
                                 previous[j - 1] + sub(pa, pb))
            previous = current
        return previous[-1]
    return distance


def affine_edit(sub, *, gap_open: float = 0.6, gap_extend: float = 0.2):
    """A run of L skipped phones costs ``gap_open + gap_extend*(L-1)``.

    Substitution-only, with the gap penalty shared by insertions and deletions
    (nothing here distinguishes them). Two matrices: ``match`` ends in a
    substitution, ``gap`` ends in a skipped phone.
    """
    INF = float("inf")

    def distance(a: list[str], b: list[str]) -> float:
        n, m = len(a), len(b)
        if not n:
            return gap_open + gap_extend * (m - 1) if m else 0.0
        if not m:
            return gap_open + gap_extend * (n - 1)
        match = [[INF] * (m + 1) for _ in range(n + 1)]
        gap = [[INF] * (m + 1) for _ in range(n + 1)]
        match[0][0] = 0.0
        for i in range(1, n + 1):
            gap[i][0] = gap_open + gap_extend * (i - 1)
        for j in range(1, m + 1):
            gap[0][j] = gap_open + gap_extend * (j - 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match[i][j] = min(match[i - 1][j - 1], gap[i - 1][j - 1]) + sub(a[i - 1], b[j - 1])
                gap[i][j] = min(match[i - 1][j] + gap_open, gap[i - 1][j] + gap_extend,
                                match[i][j - 1] + gap_open, gap[i][j - 1] + gap_extend)
        return min(match[n][m], gap[n][m])

    return distance


#: How a variant turns a phone list into the sequence it compares.
PREPARES = {
    "identity": lambda phones: list(phones),
    "canonical": D.canonical,
    "expressive": D.expressive,
}


# --------------------------------------------------------------------------
# scoring, paired
# --------------------------------------------------------------------------

def prepared_vocabulary(prepare, vocabulary) -> list:
    return [(prepare(list(phones)), word) for word, phones in vocabulary]


def ranks(sample, heard, prepare, distance, vocabulary, prepared=None) -> list[int | None]:
    """Rank of the true word for each sample. Lower is better."""
    prepared = prepared if prepared is not None else prepared_vocabulary(prepare, vocabulary)
    out: list[int | None] = []
    for (word, phones), heard_phones in zip(sample, heard):
        heard_seq = prepare(list(heard_phones))
        target_score = distance(prepare(list(phones)), heard_seq)
        worse = sum(1 for seq, other in prepared
                    if (distance(seq, heard_seq), other) < (target_score, word))
        out.append(worse + 1)
    return out


def score(sample, heard, prepare, distance, vocabulary, prepared=None) -> dict:
    ranks_ = ranks(sample, heard, PREPARES[prepare] if isinstance(prepare, str) else prepare,
                   distance, vocabulary, prepared)
    n = len(ranks_)
    known = [r for r in ranks_ if r is not None]
    return {
        "n": n,
        "top1": sum(1 for r in known if r == 1) / n,
        "top10": sum(1 for r in known if r <= 10) / n,
        "mrr": sum(1.0 / r for r in known) / n,
        "mean_rank": sum(known) / len(known),
        "ranks": ranks_,
    }


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial test over the discordant samples."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a: list[int | None], b: list[int | None]) -> dict:
    """Paired outcome of variant ``b`` against variant ``a``."""
    wins = losses = ties = 0
    for ra, rb in zip(a, b):
        a_ok, b_ok = ra == 1, rb == 1
        if a_ok == b_ok:
            ties += 1
        elif b_ok:
            wins += 1
        else:
            losses += 1
    return {"better": wins, "worse": losses, "tied": ties,
            "p": round(sign_test(wins, losses), 4)}


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def build(weights: dict[str, float] | None = None) -> dict:
    sub_feature = make_feature_distance(weights or WEIGHTS)
    sub_canonical = sub_feature
    return {
        "feature": ("identity", flat_edit(sub_feature)),
        "polish": ("canonical", flat_edit(sub_canonical)),
        "polish-affine": ("canonical", affine_edit(sub_canonical)),
        "expressive-affine": ("expressive", affine_edit(sub_feature)),
        "polish-affine-g05": ("canonical", affine_edit(sub_canonical, gap_open=0.5)),
    }


def cmd_validate(args) -> int:
    """Reproduce ``distance_for_asr.py``'s published numbers with this harness.

    Same four variants, defined the same way -- including ``feature``, which
    normalises by the term length (what ``Span.error_rate`` does) rather than
    symmetrically. If these do not match the recorded run to the sample, the
    harness is not measuring the same thing and nothing downstream is usable.
    """
    sample, heard = load(args.limit, args.seed)
    vocabulary = [(w, list(r[0])) for w, r in sorted(
        references_by_word(make_dataset(Lexicon(parse_lexicon(resolve_lexicon())),
                                        align_train=False).test).items())]
    published_variants = {
        "unit": ("identity", lambda a, b: D.edit_distance(a, b) / max(len(a), len(b), 1)),
        "feature": ("identity", lambda a, b: D.weighted(a, b) / len(a) if a else 0.0),
        "feature-sym": ("identity", lambda a, b: D.weighted(a, b) / max(len(a), len(b), 1)),
        "polish": ("canonical", lambda a, b: D.weighted(D.canonical(a), D.canonical(b))
                   / max(len(a), len(b), 1)),
    }
    assert make_feature_distance(WEIGHTS)("tʂ", "b") == D.feature_distance("tʂ", "b")
    path = ROOT / "runs" / "distance_for_asr.json"
    reference = json.loads(path.read_text()) if path.exists() else {}
    print(f"vocabulary {len(vocabulary)}, {len(sample)} corrupted per family, seed {args.seed}\n")
    print(f"{'family':<13}{'variant':<14}{'top1':>8}{'published':>11}{'match':>8}")
    ok = True
    for fam, published_fam in (("notation", "notation"), ("mishearing1", "mishearing"),
                               ("both1", "both")):
        for name, (prepare, distance) in published_variants.items():
            r = score(sample, heard[fam], prepare, distance, vocabulary)
            ref = reference.get("results", {}).get(published_fam, {}).get(name, {}).get("top1")
            same = "" if ref is None else ("yes" if abs(r["top1"] - ref) < 5e-5 else "NO")
            ok &= same != "NO"
            print(f"{fam:<13}{name:<14}{r['top1'] * 100:7.1f}%"
                  + (f"{ref * 100:10.1f}%" if ref is not None else f"{'--':>11}")
                  + f"{same:>8}")
    print("\nreproduces the published run" if ok else "\nMISMATCH - harness differs")
    return 0 if ok else 1


def cmd_gaps(args) -> int:
    """Sweep the affine gap penalty against the flat indel."""
    sample, heard = load(args.limit, args.seed)
    vocabulary = [(w, list(r[0])) for w, r in sorted(
        references_by_word(make_dataset(Lexicon(parse_lexicon(resolve_lexicon())),
                                        align_train=False).test).items())]
    sub = make_feature_distance(WEIGHTS)
    flat = flat_edit(sub)
    # One pass over the vocabulary, reused by every configuration below: the
    # ranking itself is ~len(sample) * len(vocabulary) distance calls, so this
    # hoist is what makes a sweep affordable at all.
    prepared = prepared_vocabulary(PREPARES["canonical"], vocabulary)
    print(f"{len(sample)} samples x {len(vocabulary)} candidates")
    print(f"{'family':<13}{'model':<30}{'top1':>8}{'mrr':>8}{'d mrr':>9}")
    baseline: dict[str, list[int | None]] = {}
    for fam in FAMILIES:
        base = score(sample, heard[fam], "canonical", flat, vocabulary, prepared)
        baseline[fam] = base["ranks"]
        print(f"{fam:<13}{'canonical + flat indel (current)':<30}"
              f"{base['top1'] * 100:7.1f}%{base['mrr']:8.4f}{0.0:9.4f}")
        for gap_open, gap_extend in ((0.5, 0.2), (0.7, 0.2), (0.9, 0.2), (0.7, 0.1)):
            t0 = time.time()
            r = score(sample, heard[fam], "canonical",
                      affine_edit(sub, gap_open=gap_open, gap_extend=gap_extend),
                      vocabulary, prepared)
            cmp_ = compare(baseline[fam], r["ranks"])
            label = f"affine open={gap_open} ext={gap_extend}"
            print(f"{'':<13}{label:<30}{r['top1'] * 100:7.1f}%{r['mrr']:8.4f}"
                  f"{r['mrr'] - base['mrr']:+9.4f}   {cmp_['better']}/{cmp_['worse']}"
                  f" p={cmp_['p']:.4f}  [{time.time() - t0:.0f}s]")
        print()
    return 0


def cmd_weights(args) -> int:
    """Coordinate descent on the feature weights, against MRR on the hard split."""
    sample, heard = load(args.limit, args.seed)
    vocabulary = [(w, list(r[0])) for w, r in sorted(
        references_by_word(make_dataset(Lexicon(parse_lexicon(resolve_lexicon())),
                                        align_train=False).test).items())]

    def objective(weights: dict[str, float], prepare: str = "canonical") -> float:
        sub = make_feature_distance(weights)
        return sum(score(sample, heard[fam], prepare, flat_edit(sub), vocabulary)["mrr"]
                   for fam in HARD) / len(HARD)

    best = dict(WEIGHTS)
    best_score = objective(best)
    print(f"start  mrr(hard)={best_score:.4f}  weights={ {k: round(v, 2) for k, v in best.items()} }")
    for round_ in range(args.rounds):
        improved = False
        for key in WEIGHTS:
            for factor in (0.5, 0.75, 1.5, 2.0):
                trial = dict(best)
                trial[key] = round(best[key] * factor, 4)
                value = objective(trial)
                if value > best_score + 1e-6:
                    print(f"  round {round_ + 1} {key}: {best[key]} -> {trial[key]}"
                          f"   mrr {best_score:.4f} -> {value:.4f}")
                    best, best_score, improved = trial, value, True
        if not improved:
            break
    print(f"\nbest   mrr(hard)={best_score:.4f}")
    print(json.dumps({k: round(v, 3) for k, v in best.items()}, indent=2))
    if args.json:
        args.json.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return 0


def cmd_compare(args) -> int:
    """Paired comparison of two named variants across every family."""
    sample, heard = load(args.limit, args.seed)
    vocabulary = [(w, list(r[0])) for w, r in sorted(
        references_by_word(make_dataset(Lexicon(parse_lexicon(resolve_lexicon())),
                                        align_train=False).test).items())]
    variants = build()
    prepare_a, distance_a = variants[args.a]
    prepare_b, distance_b = variants[args.b]
    print(f"a={args.a}  b={args.b}   n={len(sample)}")
    print(f"{'family':<13}{'a top1':>8}{'b top1':>8}{'better':>8}{'worse':>7}{'p':>8}")
    for fam in FAMILIES:
        ra = score(sample, heard[fam], prepare_a, distance_a, vocabulary)
        rb = score(sample, heard[fam], prepare_b, distance_b, vocabulary)
        c = compare(ra["ranks"], rb["ranks"])
        print(f"{fam:<13}{ra['top1'] * 100:7.1f}%{rb['top1'] * 100:7.1f}%"
              f"{c['better']:8}{c['worse']:7}{c['p']:8.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["validate", "gaps", "weights", "compare"])
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--a", default="polish")
    ap.add_argument("--b", default="polish-affine")
    args = ap.parse_args()
    start = time.time()
    status = {"validate": cmd_validate, "gaps": cmd_gaps,
              "weights": cmd_weights, "compare": cmd_compare}[args.command](args)
    print(f"\n{time.time() - start:.0f}s")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
