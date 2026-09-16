"""Cross-lexicon gold: MFA versus the Common Voice / Epitran dictionary.

``polish_mfa`` (134K words) is the training lexicon; ``polish_cv`` v2.0.0
(53K words, CC-0, Vox Communis, Epitran phone set) is the independent second
opinion. Same word in both, same pronunciation modulo notation, means the
gold is trustworthy; disagreement means genuine ambiguity or a lexicon bug --
either way, not something to score as plain right/wrong.

The two phone sets differ systematically, so both sides are canonicalized
before comparison:

* dentals: MFA ``t̪`` -> ``t`` (MFA always marks, CV never does);
* tie bars: CV ``t͡s`` -> ``ts`` (both sides meet at the plain digraph);
* CV two-phone retroflex affricates merge (``t ʂ`` -> ``tʂ``) -- but only
  where the spelling says ``cz``/``dż``: CV spells ``trz``/``drz`` the same
  way, and MFA splits those, so the merge is word-aware;
* palatalization: MFA ``pʲ`` -> ``p``; then post-consonant pre-vowel ``j``
  drops on *both* sides (CV ``p j ɔ`` and MFA ``pʲ ɔ`` are the same sound
  written twice -- and ``objazd``'s ``b j`` survives identically mangled);
* ``ʎ`` -> ``l``, ``ç`` -> ``x``, ``c`` -> ``k``, ``ɟ`` -> ``ɡ``, ``ʔ``
  dropped, ``w`` -> ``v`` (ł and /v/ share one phone -- ``euro``/``paweł``),
  ``ɲ`` -> ``j̃`` (MFA's palatal nasal is CV's nasal glide before a
  consonant: ``chińsku`` is ``ç i j̃ s̪ k u`` against ``x i ɲ s k u``);
* word-final -ę/-ą: MFA's denasalized ``ɛ``/``ɔ`` bridged to CV's
  ``ɛ̃``/``ɔ̃`` (surface-vs-underlying transcription choice);
* post-vocalic ``i`` -> ``j`` before a consonant or the end (the diphthong
  off-glide both lexicons spell differently);
* CV ``qu`` -> ``k w``, CV ``x`` -> ``k s`` (CV keeps them atomic; MFA
  expands -- ``brexit``'s ``x`` -> ``s`` then correctly disagrees).

The canonical form is a *comparison* device, not a pronunciation: both sides
mangled identically still agree. What it cannot do is invent information --
``bezsilne`` (CV ``s ɕ``, MFA ``s``) stays disputed, as it should.
"""

from __future__ import annotations

import hashlib
import shutil
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

CV_DICT_URL = (
    "https://github.com/MontrealCorpusTools/mfa-models/releases/download/"
    "dictionary-polish_cv-v2.0.0/polish_cv.dict"
)
CV_DICT_SHA256 = "226bce1e7f86c5a6e85a41c1c2b983056164b3fbc5d13c0d985c7c26e37a1cb0"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CV_PATH = REPO_ROOT / "data" / "lexicons" / "polish_cv.dict"

#: Same letter filter as the MFA parse: G2P targets only.
_CV_LETTERS = set("abcdefghijklmnopqrstuvwxyząćęłńóśźż")

DENTAL = "\u032a"
TIE = "\u0361"  # combining double inverted breve: t͡s
PALATALIZATION = "\u02b2"  # ʲ

#: CV two-phone sequences that are one MFA phone.
CV_MERGES: dict[tuple[str, str], str] = {
    ("t", "ʂ"): "tʂ",
    ("d", "ʐ"): "dʐ",
}

#: Canonical vowels, for the post-consonant pre-vowel j-drop.
VOWELS = frozenset({"a", "ɛ", "ɔ", "i", "ɨ", "u", "ɔ̃", "ɛ̃"})

#: Single-phone rewrites applied to both sides after source rules.
#:
#: ``c``/``ɟ`` -> ``k``/``ɡ`` folds MFA's palatal stops into the velars CV
#: writes (``kʲ``/``ɡʲ``, already ʲ-stripped, or plain). Deliberate: "ki" is
#: one sound in two notations. It also folds the cases where CV forgets
#: palatalization entirely (``alergię``: CV plain ``ɡ``) -- counted by the
#: experiment, but not disputed.
#:
#: ``ɲ`` -> ``j̃`` unifies ń before a consonant: MFA writes the palatal nasal,
#: CV the nasal glide (180 words, all -ński/-ństwo/-eńskie shapes). CV's own
#: ``j̃`` lands in the same place, so it needs no rewrite of its own.
#:
#: ``w`` -> ``v`` unifies ł and /v/, which share one phone in Polish and which
#: the two lexicons spell by turn (``euro``: MFA ``ɛ w r ɔ``, CV ``ɛ v r ɔ``;
#: 64 words, all au/eu glides).
SHARED_REWRITES: dict[str, str] = {"ʎ": "l", "ç": "x", "ɲ": "j̃",
                                   "c": "k", "ɟ": "ɡ", "w": "v"}

#: Phones deleted on both sides (hiatus filler MFA writes, CV ignores).
DROPPED = frozenset({"ʔ"})


class XlexError(RuntimeError):
    """The CV dictionary is missing, corrupt, or unparseable."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_cv(path: Path | str = DEFAULT_CV_PATH, *,
             url: str = CV_DICT_URL, sha256: str = CV_DICT_SHA256,
             force: bool = False) -> Path:
    """Download the pinned CV dictionary, verifying its SHA-256.

    Idempotent: an existing file with the right checksum is left alone.
    """
    path = Path(path)
    if path.is_file() and not force:
        if sha256_of(path) == sha256:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise XlexError(f"failed to download {url}: {exc}") from exc
    if sha256_of(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise XlexError(f"checksum mismatch for {url}")
    tmp.replace(path)
    return path


def clean_cv_word(word: str) -> str:
    """Lowercase and strip the quotes/punctuation CV rows carry."""
    word = unicodedata.normalize("NFC", word).strip().lower()
    return word.strip("\"'„”’…—–- .,;:!?()[]{}")


@dataclass
class CvLexicon:
    """Parsed CV dictionary: word -> distinct pronunciations."""

    entries: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)
    raw_lines: int = 0
    kept_rows: int = 0

    @property
    def words(self) -> set[str]:
        return set(self.entries)

    def inventory(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for readings in self.entries.values():
            for reading in readings:
                counts.update(reading)
        return counts


def parse_cv(path: Path | str = DEFAULT_CV_PATH) -> CvLexicon:
    """Parse the two-column (word, phones) CV dictionary.

    Rows whose word is not pure Polish letters are dropped (punctuation,
    digits, stray quotes); case/punctuation variants collapsing onto one
    word keep each distinct pronunciation.
    """
    path = Path(path)
    if not path.is_file():
        raise XlexError(f"CV dictionary not found at {path}. "
                        "Run `tiny-g2p fetch-cv`.")
    lexicon = CvLexicon()
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            lexicon.raw_lines += 1
            fields = line.split("\t")
            word = clean_cv_word(fields[0])
            phones = tuple(fields[-1].split())
            if not word or not phones or not set(word) <= _CV_LETTERS:
                continue
            lexicon.kept_rows += 1
            readings = lexicon.entries.setdefault(word, [])
            if phones not in readings:
                readings.append(phones)
    return lexicon


def _expand_cv(phones: list[str], word: str | None) -> list[str]:
    """Atomic CV phones MFA expands: ``x`` -> ``k s``, ``qu`` -> ``k w``.

    The ``x`` expansion is word-aware: CV also writes the fricative [x]
    (from ``ch``/``h``) as ``x``, and only letter-``x`` expands. Words
    containing both (vanishingly rare) keep ``x`` and risk a dispute.
    """
    out: list[str] = []
    expand_x = word is not None and "x" in word and "ch" not in word \
        and "h" not in word
    i = 0
    while i < len(phones):
        phone = phones[i]
        if phone == "x" and expand_x:
            out.extend(["k", "s"])
        elif phone == "q":
            out.append("k")
            if i + 1 < len(phones) and phones[i + 1] == "u":
                out.append("w")
                i += 1
        else:
            out.append(phone)
        i += 1
    return out


def _merge_cv(phones: list[str], word: str | None) -> list[str]:
    """Join CV's two-phone retroflex affricates (``t ʂ`` -> ``tʂ``).

    CV spells the affricates ``cz``/``dż`` and the clusters ``trz``/``drz`` and
    ``tsz``/``dsz`` alike, always as two phones; MFA merges the first and
    splits the rest. The phone string cannot tell them apart, so the word
    does: those clusters keep their ``t ʂ``/``d ʐ`` pair (matching MFA),
    everything else merges. A word carrying both spellings (``trzcina`` with
    ``cz``) is rare enough to leave split and risk a dispute rather than
    guess positions.
    """
    merge = word is None or not any(cluster in word for cluster in
                                    ("trz", "drz", "tsz", "dsz"))
    out: list[str] = []
    i = 0
    while i < len(phones):
        pair = (phones[i], phones[i + 1]) if i + 1 < len(phones) else None
        if pair in CV_MERGES and merge:
            out.append(CV_MERGES[pair])
            i += 2
        else:
            out.append(phones[i])
            i += 1
    return out


def _drop_post_consonant_j(phones: list[str]) -> list[str]:
    """Drop ``j`` between a consonant and a vowel (both sides, uniformly).

    This is the palatal-glide notation difference: MFA ``pʲ ɔ`` (after ʲ
    stripping, ``p ɔ``) against CV ``p j ɔ``. Intervocalic and initial
    ``j`` (``jajko``) are untouched.
    """
    out: list[str] = []
    for i, phone in enumerate(phones):
        if (phone == "j" and out and out[-1] not in VOWELS
                and i + 1 < len(phones) and phones[i + 1] in VOWELS):
            continue
        out.append(phone)
    return out


def _fold_diphthong_glide(phones: list[str]) -> list[str]:
    """Rewrite post-vocalic ``i`` to ``j`` before a consonant or the end.

    The off-glide in ``beikocące``/``airbusem`` is ``i`` for MFA and ``j``
    for CV -- one sound (Polish has no i/j contrast there), two letters.
    Pre-vowel ``i``/``j`` (``dania``/``moje``) are untouched.
    """
    out = list(phones)
    for i, phone in enumerate(out):
        if (phone == "i" and i > 0 and out[i - 1] in VOWELS
                and (i + 1 == len(out) or out[i + 1] not in VOWELS)):
            out[i] = "j"
    return out


def canonicalize(phones: list[str] | tuple[str, ...], *,
                 source: str, word: str | None = None) -> tuple[str, ...]:
    """Reduce MFA or CV phones to the shared comparison form.

    ``source`` selects the source-side rules (CV merges/expansions never
    touch MFA phones and vice versa); the remainder is uniform. ``word``
    (the lowercased orthographic form) disambiguates CV's ``x``.
    """
    if source not in ("mfa", "cv"):
        raise ValueError(f"source must be 'mfa' or 'cv', got {source!r}")
    seq = list(phones)
    if source == "cv":
        seq = _merge_cv(_expand_cv(seq, word), word)
        seq = [p.replace(TIE, "") for p in seq]
    else:
        seq = [p.replace(DENTAL, "") for p in seq]
    seq = [SHARED_REWRITES.get(p, p) for p in seq]
    seq = [p.replace(PALATALIZATION, "") for p in seq if p not in DROPPED]
    seq = [p for p in seq if p]
    # Word-final -ę/-ą: MFA denasalizes (surface ɛ/ɔ), CV keeps the nasal
    # vowel. Bridge MFA's side to CV's; genuine -e/-o words never match
    # the letter condition, so they are untouched.
    if word is not None and seq:
        if word.endswith("ę") and seq[-1] == "ɛ":
            seq[-1] = "ɛ̃"
        elif word.endswith("ą") and seq[-1] == "ɔ":
            seq[-1] = "ɔ̃"
    return tuple(_drop_post_consonant_j(_fold_diphthong_glide(seq)))


@dataclass
class Agreement:
    """MFA/CV comparison over their shared words."""

    shared: int = 0
    agreed: int = 0
    disagreed: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.agreed / self.shared if self.shared else 0.0


def readings_agree(mfa: list[tuple[str, ...]],
                   cv: list[tuple[str, ...]], *,
                   word: str | None = None) -> bool:
    """True when some MFA reading canonicalizes to some CV reading."""
    mfa_canon = {canonicalize(r, source="mfa", word=word) for r in mfa}
    cv_canon = {canonicalize(r, source="cv", word=word) for r in cv}
    return not mfa_canon.isdisjoint(cv_canon)


def compare(mfa_entries: dict[str, list[tuple[str, ...]]],
            cv: CvLexicon) -> Agreement:
    """Compare the lexicons on every shared word."""
    result = Agreement()
    for word in sorted(set(mfa_entries) & cv.words):
        result.shared += 1
        if readings_agree(mfa_entries[word], cv.entries[word], word=word):
            result.agreed += 1
        else:
            result.disagreed.append(word)
    return result
