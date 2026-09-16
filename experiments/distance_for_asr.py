#!/usr/bin/env python
"""Does the phonetic distance need Polish-specific tuning for ASR correction?

The question a correction system actually asks is "given what the ASR heard,
which vocabulary word is it?", so this measures *recovery*: corrupt a known
word the way Polish ASR corrupts words, rank the vocabulary by distance, and
see whether the intended word comes back.

Two families of corruption, because they are different problems:

``notation``
    The same pronunciation, transcribed differently -- nasal vowels written
    merged or split, affricates one token or two, obstruents devoiced the way
    Polish obligatorily devoices them, ``ł`` as ``w`` or ``v``. A matcher that
    charges for these charges for nothing, and it is the *lexicons themselves*
    that cannot agree (MFA writes ``dʐ`` in one word and ``d̪ ʐ`` in the next;
    CV writes ``t ʂ`` for ``cz`` where MFA writes ``tʂ``).

``mishearing``
    A genuine acoustic confusion -- one phone swapped for a member of its
    Polish confusion class (s/ʂ/ɕ, t/d, ɨ/i, ...). No matcher can make this
    free; the only question is whether systematic notation noise is drowning
    it out.

Variants compared:

1. ``unit``             plain Levenshtein (``pl_g2p.metrics``)
2. ``feature``          ``weighted_edit_distance / len(term)`` -- what
                        ``phrase.Span.error_rate`` does today
3. ``feature-sym``      the same, normalised by ``max(len)`` instead
4. ``polish``           both sides through the canonical form (below), then
                        symmetric normalisation
5. ``polish-indel``     as 4, with indels at 0.6 rather than 1.0

    uv run python experiments/distance_for_asr.py [--limit N] [--json out]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402
from pl_g2p.metrics import edit_distance  # noqa: E402
from phoneme_lab.features import feature_distance  # noqa: E402
from phoneme_lab.phrase import lcs_ratio  # noqa: E402

from tiny_g2p.data import make_dataset, references_by_word, resolve_lexicon  # noqa: E402
from tiny_g2p.phones import HOMORGANIC, STOPS  # noqa: E402

# --------------------------------------------------------------------------
# the proposal: a canonical form for *distance*, not for pronunciation
# --------------------------------------------------------------------------

DENTAL = "\u032a"
PALATALIZATION = "\u02b2"

#: Written one way by one lexicon and another way by the next.
ALIASES = {"w": "v", "ʎ": "l", "ç": "x", "c": "k", "ɟ": "ɡ", "j\u0303": "ɲ"}

#: One phone on one side, two on the other.
AFFRICATES = {
    "t\u0361s": ["t", "s"], "d\u0361z": ["d", "z"],
    "t\u0361\u0255": ["t", "\u0255"], "d\u0361\u0291": ["d", "\u0291"],
    "t\u0282": ["t", "\u0282"], "d\u0290": ["d", "\u0290"],
    "t\u0255": ["t", "\u0255"], "d\u0291": ["d", "\u0291"], "ts": ["t", "s"],
}

#: Obstruents that devoice, and their voiceless partner.
VOICELESS_OF = {
    "b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s", "ʐ": "ʂ", "ʑ": "ɕ",
    "d\u0361z": "t\u0361s", "d\u0290": "t\u0282", "d\u0291": "t\u0255",
    "ɣ": "x",
}
OBSTRUENTS = set(VOICELESS_OF) | set(VOICELESS_OF.values())
VOWELS = {"a", "ɛ", "ɔ", "i", "ɨ", "u", "ɔ\u0303", "ɛ\u0303"}
NASALS = {"m", "n", "ɲ", "ŋ"}


def homorganic_of(follower: str | None) -> str:
    """The homorganic nasal for a follower, in the *canonical* alphabet.

    The table is keyed on MFA's spelling, where the alveolar series carries a
    dental mark -- but by the time this runs the marks are gone. Looking up
    both spellings and stripping the result keeps one alphabet end to end;
    getting this wrong is how ``n̪`` and ``n`` end up as different keys.
    """
    if not follower:
        return "n"
    key = follower.replace(PALATALIZATION, "")
    nasal = HOMORGANIC.get(key) or HOMORGANIC.get(key + DENTAL) or "n"
    return nasal.replace(DENTAL, "")


def strip_dentals(seq: list[str]) -> list[str]:
    """``t̪`` -> ``t``: MFA always marks the alveolar series, no Polish contrast."""
    return [p for p in (phone.replace(DENTAL, "") for phone in seq)
            if p and p != "ʔ"]


def drop_palatalization(seq: list[str]) -> list[str]:
    """``bʲ`` -> ``b``: the mark spells the following ``i``.

    Its own step because it is the one mark that *can* carry a contrast --
    ``żabie`` [ʒabʲɛ] against ``żabę`` [ʒabɛ].
    """
    return [phone.replace(PALATALIZATION, "") for phone in seq]


def split_affricates(seq: list[str]) -> list[str]:
    out: list[str] = []
    for phone in seq:
        out.extend(AFFRICATES.get(phone, [phone]))
    return out


def alias_allophones(seq: list[str]) -> list[str]:
    """``ç``->``x``, ``j̃``->``ɲ``, ``c``/``ɟ``->``k``/``ɡ``, ``ʎ``->``l``."""
    return [{"ç": "x", "j\u0303": "ɲ", "c": "k", "ɟ": "ɡ", "ʎ": "l"}.get(p, p)
            for p in seq]


def merge_l_v(seq: list[str]) -> list[str]:
    """``ł``/``w`` -> ``v``: one phoneme for most speakers, a real distinction
    for some -- and it also erases the spelling."""
    return ["v" if p == "w" else p for p in seq]


def expand_nasals(seq: list[str]) -> list[str]:
    out: list[str] = []
    for i, phone in enumerate(seq):
        if phone in ("ɔ\u0303", "ɛ\u0303"):
            follower = seq[i + 1] if i + 1 < len(seq) else None
            out.extend([phone[0], homorganic_of(follower)])
            continue
        out.append(phone)
    return out


def drop_glide(seq: list[str]) -> list[str]:
    """Post-consonant pre-vowel ``j`` is the palatalisation, written twice."""
    out: list[str] = []
    for i, phone in enumerate(seq):
        if (phone == "j" and out and out[-1] not in VOWELS
                and i + 1 < len(seq) and seq[i + 1] in VOWELS):
            continue
        out.append(phone)
    return out


def collapse_geminates(seq: list[str]) -> list[str]:
    return [p for i, p in enumerate(seq) if i == 0 or p != seq[i - 1]]


def devoice_coda(seq: list[str]) -> list[str]:
    """Obligatory devoicing: word-final, or before an obstruent."""
    out = []
    for i, phone in enumerate(seq):
        follower = seq[i + 1] if i + 1 < len(seq) else None
        coda = follower is None or follower in OBSTRUENTS
        out.append(VOICELESS_OF.get(phone, phone) if coda else phone)
    return out


#: name -> step, in order. Exposed so ``canonical_information.py`` can ablate
#: them one at a time instead of re-implementing the pipeline.
STEPS: list[tuple[str, object]] = [
    ("dental marks t̪→t", strip_dentals),
    ("drop ʲ (żabie/żabę)", drop_palatalization),
    ("affricates to two tokens", split_affricates),
    ("allophones ç→x, j̃→ɲ, c→k", alias_allophones),
    ("merge ł/v w→v", merge_l_v),
    ("nasal vowel → oral+nasal", expand_nasals),
    ("drop post-consonant j", drop_glide),
    ("collapse geminates", collapse_geminates),
    ("devoice the coda", devoice_coda),
]


def pipeline(phones: list[str], skip: str | None = None) -> tuple[str, ...]:
    """Apply every step, or every step but ``skip``."""
    out = list(phones)
    for name, step in STEPS:
        if name == skip:
            continue
        out = step(out)  # type: ignore[operator]
    return tuple(out)


def canonical(phones: list[str]) -> list[str]:
    """One common form for two notations of the same pronunciation."""
    return list(pipeline(phones))


def weighted(a: list[str], b: list[str], *, indel: float = 1.0) -> float:
    """Levenshtein with articulatory substitution costs (``phoneme_lab``'s)."""
    if not a:
        return len(b) * indel
    if not b:
        return len(a) * indel
    previous = [j * indel for j in range(len(b) + 1)]
    for i, pa in enumerate(a, 1):
        current = [i * indel] + [0.0] * len(b)
        for j, pb in enumerate(b, 1):
            current[j] = min(previous[j] + indel, current[j - 1] + indel,
                             previous[j - 1] + feature_distance(pa, pb))
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------
# corruptions
# --------------------------------------------------------------------------

def corrupt_notation(phones: list[str], rng: random.Random) -> list[str]:
    """Re-transcribe the same pronunciation in the other lexicon's style."""
    out: list[str] = []
    for i, phone in enumerate(phones):
        bare = phone.replace(DENTAL, "")
        follower = phones[i + 1] if i + 1 < len(phones) else None
        last = i == len(phones) - 1
        choice = rng.random()
        if bare in ("ɔ\u0303", "ɛ\u0303") and choice < 0.6:
            # split the nasal, as MFA does before a stop and CV does everywhere
            out.extend([bare[0], HOMORGANIC.get(
                "".join(c for c in (follower or "") if c != PALATALIZATION), "n\u032a")])
            continue
        if bare in ("t\u0282", "d\u0290", "ts") and choice < 0.6:
            out.extend(AFFRICATES[bare])
            continue
        if bare in VOICELESS_OF and (last or (follower in OBSTRUENTS)) and choice < 0.7:
            out.append(VOICELESS_OF[bare] if rng.random() < 0.5 else bare)
            continue
        if bare == "w" and choice < 0.5:
            out.append("v")
            continue
        if bare == "\u0272" and choice < 0.4:
            out.extend(["j\u0303"] if rng.random() < 0.5 else ["n", "j"])
            continue
        out.append(bare)
    return out


#: Polish confusion classes: phones an ASR actually mixes up.
CONFUSIONS: dict[str, list[str]] = {
    "p": ["b"], "b": ["p"], "t": ["d"], "d": ["t"], "k": ["ɡ"], "ɡ": ["k"],
    "f": ["v"], "v": ["f", "w"], "s": ["ʂ", "ɕ", "z"], "z": ["s", "ʐ"],
    "ʂ": ["s", "ɕ", "ʐ"], "ʐ": ["ʂ", "z"], "ɕ": ["s", "ʂ"], "ʑ": ["ʐ", "z"],
    "t\u0282": ["t\u0255", "t\u0361s"], "t\u0255": ["t\u0282", "t\u0361s"],
    "ts": ["t\u0282"], "x": ["k", "f"],
    "m": ["n"], "n": ["m", "ɲ"], "ɲ": ["n", "j\u0303"], "ŋ": ["n"],
    "r": ["l"], "l": ["r", "w"], "w": ["v", "l"], "j": ["i"],
    "a": ["ɔ", "ɛ"], "ɔ": ["u", "a"], "u": ["ɔ"], "ɛ": ["a", "ɨ"], "ɨ": ["i", "ɛ"],
    "i": ["ɨ", "j"],
}

def corrupt_mishearing(phones: list[str], rng: random.Random,
                       *, edits: int = 1) -> list[str]:
    """Swap up to ``edits`` phones for a near neighbour."""
    out = list(phones)
    for _ in range(edits):
        spots = [i for i, p in enumerate(out)
                 if p.replace(DENTAL, "") in CONFUSIONS]
        if not spots:
            break
        i = rng.choice(spots)
        options = CONFUSIONS[out[i].replace(DENTAL, "")]
        out[i] = rng.choice(options)
    return out


# --------------------------------------------------------------------------
# recovery
# --------------------------------------------------------------------------

def variants() -> dict[str, tuple]:
    """name -> (prepare, score): normalise a phone list, then compare two.

    ``prepare`` is hoisted out of the ranking loop, so a variant that
    canonicalises pays for the vocabulary once rather than once per query.
    """
    identity = list

    def unit(a, b):
        return edit_distance(a, b) / max(len(a), len(b), 1)

    def feature(a, b):
        return weighted(a, b) / len(a) if a else 0.0        # as Span does today

    def feature_sym(a, b):
        return weighted(a, b) / max(len(a), len(b), 1)

    def polish(a, b):
        return weighted(a, b) / max(len(a), len(b), 1)

    def polish_indel(a, b):
        return weighted(a, b, indel=0.6) / max(len(a), len(b), 1)

    def score_current(a, b):
        """``1 - sqrt(phonetic * overlap)``: the number ``Span.score`` thresholds.

        Reported as a distance so the ranking machinery is shared; the
        ``margin`` column then reads as "how far the true match beats the best
        wrong candidate", which is the band a threshold has to fit into.
        """
        error = feature(a, b)
        return 1.0 - (max(0.0, 1.0 - error) * lcs_ratio(a, b)) ** 0.5

    def score_polish(a, b):
        error = polish(a, b)
        return 1.0 - (max(0.0, 1.0 - error) * lcs_ratio(a, b)) ** 0.5

    def expressive_align(a, b):
        return align_distance(a, b) / max(len(a), len(b), 1)

    return {
        "unit": (identity, unit),
        "feature": (identity, feature),
        "feature-sym": (identity, feature_sym),
        "polish": (canonical, polish),
        "polish-indel": (canonical, polish_indel),
        "score-current": (identity, score_current),
        "score-polish": (canonical, score_polish),
        "expressive-align": (expressive, expressive_align),
    }


def evaluate(prepare, distance, corrupt, sample: list[tuple[str, list[str]]],
             candidates: list[tuple[str, list[str]]], rng: random.Random,
             *, top_k: int = 10) -> dict:
    """Corrupt each sampled word, rank the whole vocabulary, report what came back."""
    prepared = [(prepare(list(phones)), word) for word, phones in candidates]
    ranked = []
    for word, phones in sample:
        heard = prepare(corrupt(list(phones), rng))
        scores = sorted(
            ((distance(phones_c, heard), word_c) for phones_c, word_c in prepared),
            key=lambda pair: (pair[0], pair[1]))
        position = next((i for i, (_, w) in enumerate(scores, 1) if w == word), None)
        best_wrong = next((s for s, w in scores if w != word), None)
        correct = next(s for s, w in scores if w == word)
        ranked.append({
            "word": word, "heard": heard, "rank": position,
            "correct": correct, "margin": (best_wrong - correct)
                      if best_wrong is not None else 1.0,
        })
    ranks = [r["rank"] for r in ranked if r["rank"] is not None]
    return {
        "n": len(ranked),
        "top1": sum(1 for r in ranks if r == 1) / len(ranked),
        "top_k": sum(1 for r in ranks if r <= top_k) / len(ranked),
        "mean_distance": sum(r["correct"] for r in ranked) / len(ranked),
        "mean_margin": sum(r["margin"] for r in ranked) / len(ranked),
        "examples": [r for r in ranked if r["rank"] != 1][:8],
    }


CONTRASTIVE_PAIRS = [
    ("żabie", "żabę"), ("partie", "partję"), ("żyła", "żywa"), ("masą", "mason"),
    ("morze", "może"), ("kody", "koty"), ("wąs", "wons"), ("rodzinny", "rodziny"),
]


def contrast_report() -> None:
    """What does each form keep apart? A projection is only worth its losses."""
    from pl_g2p.lexicon import Lexicon, parse_lexicon
    from tiny_g2p.data import resolve_lexicon

    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    readings: dict[str, list[str]] = {}
    for entry in lexicon.entries:
        readings.setdefault(entry.word, list(entry.phonemes))
    print("\n--- what each form keeps distinct")
    print(f"  {'pair':24} {'coarse':>8} {'expressive':>11}   phones")
    for left, right in CONTRASTIVE_PAIRS:
        if left not in readings or right not in readings:
            continue
        a, b = readings[left], readings[right]
        coarse_same = tuple(pipeline(a)) == tuple(pipeline(b))
        fine_same = tuple(expressive(a)) == tuple(expressive(b))
        print(f"  {left + ' / ' + right:24} "
              f"{'merged' if coarse_same else 'distinct':>8} "
              f"{'merged' if fine_same else 'distinct':>11}   "
              f"{' '.join(a)} | {' '.join(b)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300,
                    help="how many vocabulary words to corrupt (candidates are all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--contrast", action="store_true",
                    help="also report what each form keeps distinct")
    args = ap.parse_args()

    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    data = make_dataset(lexicon, align_train=False)
    refs = references_by_word(data.test)
    vocabulary = [(w, list(r[0])) for w, r in sorted(refs.items())]
    rng = random.Random(args.seed)
    sample = rng.sample(vocabulary, min(args.limit, len(vocabulary)))

    print(f"vocabulary {len(vocabulary)} words (all are candidates); "
          f"{len(sample)} corrupted per variant")
    families = {
        "notation": corrupt_notation,
        "mishearing": lambda p, r: corrupt_mishearing(p, r, edits=1),
        "both": lambda p, r: corrupt_mishearing(corrupt_notation(p, r), r, edits=1),
    }
    out: dict = {"candidates": len(vocabulary), "sample": len(sample), "results": {}}
    for family, corrupt in families.items():
        print(f"\n--- {family}")
        print(f"{'variant':14} {'top1':>7} {'top10':>7} {'mean d':>7} {'margin':>7}")
        out["results"][family] = {}
        for name, (prepare, distance) in variants().items():
            result = evaluate(prepare, distance, corrupt, sample, vocabulary,
                              random.Random(args.seed))
            out["results"][family][name] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in result.items() if k != "examples"}
            print(f"{name:14} {result['top1'] * 100:6.1f}% "
                  f"{result['top_k'] * 100:6.1f}% {result['mean_distance']:7.3f} "
                  f"{result['mean_margin']:7.3f}")
            for example in result["examples"][:3]:
                print(f"     missed {example['word']:16} "
                      f"heard {' '.join(example['heard'])}  rank {example['rank']}")

    if args.contrast:
        contrast_report()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0




# --------------------------------------------------------------------------
# the proposal, take two: phonemicise, then *align* the notation away
# --------------------------------------------------------------------------
#
# The coarse form above projects both sides into one alphabet, which is
# irreversible: it merges `żabie`/`żabę` (a real contrast), `masą`/`mason`
# (a nasal vowel and a vowel+nasal, not the same thing) and `żyła`/`żywa`.
# The alternative keeps the expressive form -- only genuinely lossless
# rewrites -- and moves the tolerance into the alignment, where a rule can be
# conditional on position and can match one token against two.

#: Lossless only: marks nobody contrasts, and allophones written as phonemes.
EXPRESSIVE_STEPS: list[tuple[str, object]] = [
    ("dental marks t̪→t", strip_dentals),
    ("allophones ç→x, j̃→ɲ, c→k", alias_allophones),
]


def expressive(phones: list[str]) -> list[str]:
    out = list(phones)
    for _, step in EXPRESSIVE_STEPS:
        out = step(out)
    return out


#: One token on one side, two on the other: the same sound, tokenized
#: differently. Cost is near-zero and it is *symmetric*.
_NOTATION_PAIRS: dict[tuple[str, ...], float] = {}


def _register(single: str, pair: tuple[str, str], cost: float = 0.05) -> None:
    _NOTATION_PAIRS[(single, pair[0], pair[1])] = cost


for _t, _f in (("t\u0282", "\u0282"), ("d\u0290", "\u0290"),
               ("ts", "s"), ("dz", "z"), ("t\u0255", "\u0255"),
               ("d\u0291", "\u0291")):
    _register(_t, ("t" if _t[0] == "t" else "d", _f))
    _register(_t, (("t" if _t[0] == "t" else "d"), _f))
for _nasal, _oral in (("\u0254\u0303", "\u0254"), ("\u025b\u0303", "\u025b")):
    for _n in ("m", "n", "\u0272", "\u014b", "n\u032a"):
        _register(_nasal, (_oral, _n))


def pair_cost(a: list[str], b: list[str], i: int, j: int) -> float | None:
    """Cost of matching one token against two, in either direction."""
    if i + 1 < len(a) and j < len(b):
        cost = _NOTATION_PAIRS.get((a[i], a[i + 1], b[j]))
        if cost is None:
            cost = _NOTATION_PAIRS.get((b[j], a[i], a[i + 1]))
        if cost is not None:
            return cost
    return None


#: A palatalised consonant and the same consonant plus a glide.
def _palatalized_pair(a: list[str], b: list[str], i: int, j: int) -> float | None:
    if i + 1 < len(a) and j < len(b):
        stripped = a[i].replace(PALATALIZATION, "")
        if stripped != a[i] and b[j] == stripped and b[j] != a[i + 1]:
            return 0.05
    return None


def align_distance(a: list[str], b: list[str], *, indel: float = 1.0,
                   devoicing: float = 0.15) -> float:
    """Levenshtein that can match one token against two and knows position.

    Two things a phone-wise projection cannot express:

    * **tokenisation** -- ``ɔ̃`` against ``ɔ n`` is one sound written twice, so
      the DP is allowed a 1-vs-2 move at near-zero cost instead of an indel;
    * **position** -- a voicing difference is nearly free where Polish
      neutralises it (word-final, or against an obstruent: ``dróg`` [druk]),
      and full price in the onset (``kody``/``koty``).
    """
    n, m = len(a), len(b)
    # prev2/prev1 hold the two rows needed for 1-vs-2 and 2-vs-1 moves.
    rows = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        rows[i][0] = i * indel
    for j in range(1, m + 1):
        rows[0][j] = j * indel
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(
                rows[i - 1][j] + geminate_indel(a, i - 1, indel),  # delete a[i-1]
                rows[i][j - 1] + geminate_indel(b, j - 1, indel),  # insert b[j-1]
                rows[i - 1][j - 1] + substitution_cost(a, b, i - 1, j - 1,
                                                       devoicing=devoicing),
            )
            # 1 vs 2: a[i-1] matches b[j-2:j]
            if j >= 2:
                cost = pair_cost(a, b, i - 1, j - 2)
                if cost is not None:
                    best = min(best, rows[i - 1][j - 2] + cost)
            # 2 vs 1: a[i-2:i] matches b[j-1]
            if i >= 2:
                cost = pair_cost(b, a, j - 1, i - 2)
                if cost is not None:
                    best = min(best, rows[i - 2][j - 1] + cost)
                cost = _palatalized_pair(a, b, i - 2, j - 1)
                if cost is not None:
                    best = min(best, rows[i - 2][j - 1] + cost)
            rows[i][j] = best
    return rows[n][m]


#: A doubled phone costs nearly nothing to lose: MFA drops 95.7% of the
#: lexicon's written geminates and ASR drops them too, so the *difference* is
#: real (``panny`` [pannɨ] against ``pany`` [panɨ]) but frequently absent from
#: either side. Cheap, not free: when the transcript does have the geminate,
#: the word with it still wins.
GEMINATE_INDEL = 0.2


def geminate_indel(seq: list[str], i: int, indel: float) -> float:
    """Indel cost, discounted when the token is half of a geminate."""
    neighbour = seq[i - 1] if i else (seq[i + 1] if i + 1 < len(seq) else None)
    return GEMINATE_INDEL if neighbour == seq[i] else indel


def substitution_cost(a: list[str], b: list[str], i: int, j: int, *,
                      devoicing: float = 0.15) -> float:
    """Feature distance, discounted where Polish neutralises the difference."""
    pa, pb = a[i], b[j]
    if pa == pb:
        return 0.0
    # `ł`/`w` against `v`: one phoneme for most speakers, a real one for some.
    if {pa, pb} <= {"w", "v"}:
        return 0.05
    base = feature_distance(pa, pb)
    voicing_only = (VOICELESS_OF.get(pa) == pb or VOICELESS_OF.get(pb) == pa)
    if voicing_only and (in_neutralising_position(a, i) or in_neutralising_position(b, j)):
        return base * devoicing
    return base


def in_neutralising_position(seq: list[str], i: int) -> bool:
    """Word-final, or against an obstruent: where Polish devoices obligatorily.

    No syllabifier: Polish voicing assimilation crosses syllable boundaries
    (``wstrząs``, ``krzywd``), so position here means the obstruent
    neighbourhood, not the coda. Morpheme boundaries are where it is *blocked*
    (``bezradny``), which is the next thing to thread through.
    """
    follower = seq[i + 1] if i + 1 < len(seq) else None
    previous = seq[i - 1] if i else None
    return (follower is None or follower in OBSTRUENTS
            or (previous is not None and previous in OBSTRUENTS))

if __name__ == "__main__":
    raise SystemExit(main())
