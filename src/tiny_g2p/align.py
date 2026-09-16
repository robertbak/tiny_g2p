"""Monotonic grapheme->phone alignment: the teacher that labels characters.

The per-character model needs one label (a phone or blank) per character, but
the lexicon gives unaligned pairs. This module aligns them with a shortest-path
DP over five operations:

* ``SUB``      one char emits one phone (cost: articulatory feature distance
  between the char's canonical phone and the target -- near-misses are cheap);
* ``SKIP``     one char emits nothing (cheap for ``i``, the glide; pricier
  otherwise, so double letters reduce but little else vanishes);
* ``DIGRAPH``  ``cz sz rz dz dż dź ch ci si zi ni`` emit one phone jointly;
* ``TRIGRAPH`` ``dzi`` + vowel emits ``dʑ`` (``dzień``);
* ``NASAL``    ą/ę emit oral vowel + nasal consonant, labelled with the merged
  nasal vowel (``ɔ̃``/``ɛ̃``) that :mod:`tiny_g2p.phones` restores.

Every phone must be consumed and phones are never skipped, so acronyms spelled
letter-by-letter (``abc`` -> ``a b ɛ t̪s̪ ɛ``, more phones than characters)
have no valid path: :func:`align` returns None and the example is excluded
from training (422 lexicon rows, 0.31%). A mean-cost gate drops forced
garbage alignments on top of that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from phoneme_lab.features import feature_distance


#: Character -> its canonical raw MFA phone, for substitution costs.
SEED: dict[str, str] = {
    "a": "a", "ą": "ɔ̃", "b": "b", "c": "t̪s̪", "ć": "tɕ",
    "d": "d̪", "e": "ɛ", "ę": "ɛ̃", "f": "f", "g": "ɡ",
    "h": "x", "i": "i", "j": "j", "k": "k", "l": "l",
    "ł": "w", "m": "m", "n": "n̪", "ń": "ɲ", "o": "ɔ",
    "ó": "u", "p": "p", "r": "r", "s": "s̪", "ś": "ɕ",
    "t": "t̪", "u": "u", "w": "v", "y": "ɨ", "z": "z̪",
    "ź": "ʑ", "ż": "ʐ",
    # ASCII letters MFA keeps in loanwords.
    "q": "k", "v": "v", "x": "k",
}

#: Two characters emitting one phone jointly.
DIGRAPHS: dict[str, str] = {
    "cz": "tʂ", "sz": "ʂ", "rz": "ʐ",
    "dz": "d̪z̪", "dż": "dʐ", "dź": "dʑ",
    "ch": "x",
    "ci": "tɕ", "si": "ɕ", "zi": "ʑ", "ni": "ɲ",
}

#: The one three-character emission: ``dzi`` + vowel.
TRIGRAPH = "dzi"
TRIGRAPH_PHONE = "dʑ"

#: ą/ę -> the oral vowel a NASAL op expects, and the merged label it emits.
NASAL_ORAL: dict[str, str] = {"ą": "ɔ", "ę": "ɛ"}
NASAL_LABEL: dict[str, str] = {"ą": "ɔ̃", "ę": "ɛ̃"}

#: Second phone of a NASAL op: a nasal consonant.
NASAL_CONSONANTS = frozenset({"m", "mʲ", "n̪", "ɲ", "ŋ"})

#: SKIP cost. ``i`` is near-free (the palatalising glide); anything else
#: costs more than a same-class substitution but less than a garbage one,
#: so double letters (``anna`` -> ``a n a``) reduce cleanly.
SKIP_COSTS: dict[str, float] = {"i": 0.05}
DEFAULT_SKIP_COST = 0.6

#: Mean cost per character above which an alignment is judged forced noise.
DEFAULT_MAX_MEAN_COST = 0.3


@dataclass(frozen=True)
class Alignment:
    """One word aligned to one pronunciation."""

    word: str
    phones: tuple[str, ...]
    labels: tuple[str | None, ...]
    cost: float

    @property
    def mean_cost(self) -> float:
        return self.cost / len(self.word) if self.word else 0.0


@dataclass
class _State:
    cost: float = math.inf
    prev: tuple[int, int] | None = None
    op: str = ""
    label: str | None = None  # label for chars[i:ni] head; tail is blank


def _sub_cost(char: str, phone: str) -> float:
    """Articulatory distance between the char's canonical phone and target."""
    seed = SEED.get(char, char)
    if seed == phone:
        return 0.0
    return feature_distance(seed, phone)


def align(word: str, phones: list[str] | tuple[str, ...]) -> Alignment | None:
    """Align *word* to *phones*, or None when no monotonic path exists."""
    chars = list(word)
    phones = list(phones)
    n, m = len(chars), len(phones)
    if n == 0:
        return None if m else Alignment(word, (), (), 0.0)

    grid = [[_State() for _ in range(m + 1)] for _ in range(n + 1)]
    grid[0][0].cost = 0.0

    def relax(i: int, j: int, ni: int, nj: int, op: str,
              extra: float, label: str | None) -> None:
        if ni > n or nj > m:
            return
        here, there = grid[i][j], grid[ni][nj]
        if here.cost + extra < there.cost:
            there.cost = here.cost + extra
            there.prev = (i, j)
            there.op = op
            there.label = label

    for i in range(n + 1):
        for j in range(m + 1):
            if math.isinf(grid[i][j].cost):
                continue
            if i < n and j < m:
                relax(i, j, i + 1, j + 1, "SUB",
                      _sub_cost(chars[i], phones[j]), phones[j])
            if i < n:
                relax(i, j, i + 1, j, "SKIP",
                      SKIP_COSTS.get(chars[i], DEFAULT_SKIP_COST), None)
            if i + 1 < n and j < m and chars[i] + chars[i + 1] in DIGRAPHS:
                canonical = DIGRAPHS[chars[i] + chars[i + 1]]
                cost = 0.0 if canonical == phones[j] else feature_distance(
                    canonical, phones[j])
                relax(i, j, i + 2, j + 1, "DIGRAPH", cost, phones[j])
            if (i + 2 < n and j < m
                    and chars[i] + chars[i + 1] + chars[i + 2] == TRIGRAPH):
                cost = (0.0 if TRIGRAPH_PHONE == phones[j]
                        else feature_distance(TRIGRAPH_PHONE, phones[j]))
                relax(i, j, i + 3, j + 1, "TRIGRAPH", cost, phones[j])
            if (i < n and j + 1 < m and chars[i] in NASAL_ORAL
                    and phones[j] == NASAL_ORAL[chars[i]]
                    and phones[j + 1] in NASAL_CONSONANTS):
                relax(i, j, i + 1, j + 2, "NASAL", 0.0,
                      NASAL_LABEL[chars[i]])

    final = grid[n][m]
    if math.isinf(final.cost):
        return None

    labels: list[str | None] = [None] * n
    i, j = n, m
    while (i, j) != (0, 0):
        state = grid[i][j]
        assert state.prev is not None
        pi, pj = state.prev
        labels[pi] = state.label  # tail chars of a multi-char op stay blank
        i, j = pi, pj
    return Alignment(word, tuple(phones), tuple(labels), final.cost)


def align_entry(word: str, phones: list[str] | tuple[str, ...], *,
                max_mean_cost: float = DEFAULT_MAX_MEAN_COST,
                ) -> Alignment | None:
    """Align one lexicon row, rejecting unalignable rows and forced noise."""
    alignment = align(word, phones)
    if alignment is None or alignment.mean_cost > max_mean_cost:
        return None
    return alignment


def label_inventory(phones_inventory) -> set[str]:
    """Labels the model can emit: raw phones (nasals merged) plus blank.

    Takes the observed phone inventory; kept as a function (not a constant)
    so the vocabulary is always derived from data actually present.
    """
    return set(phones_inventory) | {None}
