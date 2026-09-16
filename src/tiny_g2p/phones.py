"""Phone inventory knowledge for the tiny per-character model.

Two facts, both verified against the lexicon (see ``experiments/``):

1. **Labels keep raw MFA phones** (dentals, ʲ and all), with one exception:
   ą/ę before a stop surface in MFA as oral vowel + nasal consonant
   (``kąt`` = ``k ɔ n̪ t̪``), which is two phones for one character. The
   aligner merges those to the nasal vowel (``ɔ̃``/``ɛ̃``) so every label
   fits its character, and :func:`restore` splits them back for scoring.
   The merge is word-aware (only ą/ę merge), because genuine clusters like
   ``mantra`` (``m a n̪ t̪ ...``) must be left alone -- 5,498 lexicon words
   have V+nasal+stop on plain vowels.

2. **The split is deterministic given the word and the following phone.**
   Raw MFA never puts a nasal vowel directly before a stop, and the nasal
   is always homorganic with the stop (``n̪`` before dentals, ``ɲ`` before
   alveolo-palatals, ``ŋ`` before velars, ``m`` before labials). Dental
   restoration needs no rule at all: raw MFA *always* marks the alveolar
   series (no plain ``t d n s z`` occur), and labels keep the marks.
"""

from __future__ import annotations

#: Raw MFA stops and affricates (dental marks kept). A nasal vowel before one
#: of these, emitted by ą/ę, is always a merged oral+nasal pair.
STOPS = frozenset({
    "p", "b", "pʲ", "bʲ",
    "t̪", "d̪", "tʲ", "t̪s̪", "d̪z̪",
    "tʂ", "dʐ", "tɕ", "dʑ",
    "c", "ɟ",
    "k", "ɡ",
    "ʔ",
})

#: Follower stop -> the homorganic nasal MFA writes. Verified: every
#: (nasal, stop) pair after an oral vowel in ą/ę-words follows this table.
HOMORGANIC: dict[str, str] = {
    "p": "m", "b": "m", "pʲ": "m", "bʲ": "m",
    "t̪": "n̪", "d̪": "n̪", "tʲ": "n̪", "t̪s̪": "n̪", "d̪z̪": "n̪",
    "tʂ": "n̪", "dʐ": "n̪",
    "tɕ": "ɲ", "dʑ": "ɲ",
    # Palatal stops pattern with the velar nasal (brzęki -> b ʐ ɛ ŋ c i),
    # unlike the alveolo-palatal affricates. Found by the restore round-trip,
    # which failed all 197 ą/ę + c/ɟ rows before this fix.
    "c": "ŋ", "ɟ": "ŋ",
    "k": "ŋ", "ɡ": "ŋ",
    "ʔ": "n̪",
}

#: Merged nasal vowel -> its oral half.
ORAL: dict[str, str] = {"ɔ̃": "ɔ", "ɛ̃": "ɛ"}

#: Characters whose emission may be a merged nasal vowel.
NASAL_LETTERS = frozenset({"ą", "ę"})


def is_stop(phone: str | None) -> bool:
    """True for stops and affricates (the nasal-split context)."""
    return phone in STOPS


#: Plain alveolar -> MFA's dental spelling. Deterministic: raw MFA always
#: marks the series (verified: no plain t/d/n/s/z/ts/dz occur), so this
#: exactly inverts the dental collapse for mapped prg2p output.
REDENTALIZE: dict[str, str] = {
    "t": "t̪", "d": "d̪", "n": "n̪", "s": "s̪", "z": "z̪",
    "ts": "t̪s̪", "dz": "d̪z̪",
}


def redentalize(phones: list[str]) -> list[str]:
    """Restore MFA's dental marks after a dental-collapsing mapping.

    Used for prg2p teacher labels in fine-tuning: prg2p output mapped to
    MFA phones comes out dental-collapsed, and the collapse is invertible.
    """
    return [REDENTALIZE.get(p, p) for p in phones]


def restore(word: str, labels: list[str | None]) -> list[str]:
    """Split merged nasal vowels back to MFA's oral+nasal form.

    ``labels`` is one entry per character (a phone or None for blank), as
    emitted by the model. Only ą/ę positions whose label is a nasal vowel
    followed by an emitted stop are split; everything else -- including
    model errors such as a Ṽ on a plain vowel -- passes through untouched,
    so scoring stays faithful to what the model actually said.
    """
    emitted = [(i, phone) for i, phone in enumerate(labels) if phone is not None]
    out: list[str] = []
    for k, (i, phone) in enumerate(emitted):
        if word[i] in NASAL_LETTERS and phone in ORAL:
            follower = emitted[k + 1][1] if k + 1 < len(emitted) else None
            if follower is not None and is_stop(follower):
                out.extend([ORAL[phone], HOMORGANIC[follower]])
                continue
        out.append(phone)
    return out
