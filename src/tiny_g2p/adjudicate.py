"""Adjudicating MFA gold against the CV second opinion, word by word.

The cross-lexicon comparison (`xlex`) says *which* words the two dictionaries
disagree on. It cannot say who is right: ``bezsilne`` (CV ``s ɕ``, MFA ``s``)
is a real ambiguity, ``biura`` (CV ``b i v r a``) is a CV bug, and ``mass``
(CV ``m a s s``) is a gemination convention neither lexicon owns. Deciding
those is a judgement call, so the calls live in ``data/adjudication_verdicts
.json`` -- one entry per word the evidence does not settle, with a note.

Six verdicts, all about the *gold*:

``mfa``
    MFA's reading is correct. Default everywhere: an uncurated word is
    trusted rather than excused.
``both``
    MFA and the other reading are the same sound written differently
    (geminates, surface vs underlying voicing, nasal place, MFA splitting an
    affricate). Either side earns a model credit.
``open``
    The spelling is genuinely ambiguous (loans, names, respellings): MFA's
    reading and every ``correct`` reading are equally acceptable, and there is
    no way to call one of them wrong.
``cv``
    MFA is wrong and CV is right (dropped segments, wrong vowel). A model
    that reproduces MFA's error is an error -- corrected can fall below
    measured, which is the point.
``none``
    Neither lexicon is right; ``correct`` carries the accepted reading(s),
    in MFA notation.
``convention``
    Not a fair item: acronyms and letter names (one label per character
    cannot spell ``agd``), typos and ASCII respellings the lexicon reads
    literally. Excluded from the phonetic denominator, reported separately.

``correct`` (a list of readings, MFA notation) may widen *any* verdict.
Predictions are compared raw first, then canonically -- a prediction that
canonicalizes to MFA's or CV's reading counts as matching it, so dental
marks, tie bars and the ʎ/``l j`` spelling never cost a model a word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from pl_g2p.metrics import edit_distance

from .xlex import canonicalize

#: Allowed verdict values, in the order the report lists them.
VERDICTS = ("mfa", "both", "open", "cv", "none", "convention")

#: Verdicts that accept MFA's own reading as correct.
_MFA_OK = frozenset({"mfa", "both", "open"})

#: Verdicts that accept a prediction matching CV's reading.
_CV_OK = frozenset({"both", "cv"})


class AdjudicationError(ValueError):
    """A verdicts file is malformed."""


@dataclass(frozen=True)
class Verdict:
    """One row's curated call: a verdict, optional extra readings, a note."""

    verdict: str = "mfa"
    correct: tuple[tuple[str, ...], ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise AdjudicationError(
                f"verdict must be one of {VERDICTS}, got {self.verdict!r}")

    @classmethod
    def from_json(cls, payload: dict | str | None) -> "Verdict":
        if payload is None:
            return cls()
        if isinstance(payload, str):
            return cls(verdict=payload)
        extra = tuple(tuple(reading.split()) for reading in
                      payload.get("correct", ()))
        return cls(verdict=payload.get("verdict", "mfa"), correct=extra,
                   note=payload.get("note", ""))

    @property
    def counted(self) -> bool:
        """False for rows excluded from the phonetic denominator."""
        return self.verdict != "convention"


@dataclass
class Judgment:
    """One word's scoring under its verdict."""

    word: str
    verdict: str
    counted: bool
    measured_exact: bool
    corrected_exact: bool
    #: ``None`` when the corrected call agrees with the measured one, else
    #: ``"extra"``/``"notation"``/``"cv"`` (excused) or ``"gold-bad"`` (the
    #: model matched MFA where MFA is judged wrong).
    change: str | None
    distance: int
    reference_length: int
    measured_distance: int = 0
    measured_reference_length: int = 0


@dataclass
class ModelReport:
    """Aggregate over judged words for one model.

    ``measured_*`` is always against MFA's own references (comparable to the
    README's numbers); ``distance``/``reference_length`` follow the verdicts.
    """

    n: int = 0
    measured_exact: int = 0
    corrected_exact: int = 0
    measured_distance: int = 0
    measured_reference_length: int = 0
    distance: int = 0
    reference_length: int = 0
    excused: dict[str, int] = field(default_factory=dict)
    penalized: int = 0

    @property
    def measured_accuracy(self) -> float:
        return self.measured_exact / self.n if self.n else 0.0

    @property
    def corrected_accuracy(self) -> float:
        return self.corrected_exact / self.n if self.n else 0.0

    @property
    def measured_per(self) -> float:
        return (self.measured_distance / self.measured_reference_length
                if self.measured_reference_length else 0.0)

    @property
    def per(self) -> float:
        return self.distance / self.reference_length if self.reference_length else 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "measured_exact": self.measured_exact,
            "corrected_exact": self.corrected_exact,
            "measured_accuracy": round(self.measured_accuracy, 6),
            "corrected_accuracy": round(self.corrected_accuracy, 6),
            "measured_per": round(self.measured_per, 6),
            "per": round(self.per, 6),
            "measured_distance": self.measured_distance,
            "measured_reference_length": self.measured_reference_length,
            "distance": self.distance,
            "reference_length": self.reference_length,
            "excused": dict(sorted(self.excused.items())),
            "penalized": self.penalized,
        }


def _matches(pred: Sequence[str], readings: Iterable[tuple[str, ...]]) -> bool:
    """Raw equality against any reading."""
    return any(tuple(pred) == tuple(r) for r in readings)


def _canonical(pred: Sequence[str], word: str) -> tuple[str, ...]:
    return canonicalize(list(pred), source="mfa", word=word)


def judge(word: str, pred: Sequence[str], mfa_refs: Sequence[tuple[str, ...]],
          cv_refs: Sequence[tuple[str, ...]], verdict: Verdict) -> Judgment:
    """Score one prediction under one word's verdict.

    ``mfa_refs``/``cv_refs`` are the lexicon readings in their own notation;
    ``pred`` is in MFA notation (both models decode that way).
    """
    measured = _matches(pred, mfa_refs)

    accepted: list[tuple[str, ...]] = []
    if verdict.verdict in _MFA_OK:
        accepted.extend(mfa_refs)
    accepted.extend(verdict.correct)

    canon_pred = _canonical(pred, word) if pred else ()
    canon_mfa = ({canonicalize(list(r), source="mfa", word=word)
                  for r in mfa_refs} if verdict.verdict in _MFA_OK else set())
    canon_cv = ({canonicalize(list(r), source="cv", word=word) for r in cv_refs}
                if verdict.verdict in _CV_OK else set())

    # A raw hit on the reference needs no excuse; anything else is checked
    # against the extras, then canonically against MFA (same reading, other
    # notation: ʎ for ``l j``, a dropped glide) and against CV.
    change: str | None = None
    corrected = measured and verdict.verdict in _MFA_OK
    if not corrected:
        if _matches(pred, verdict.correct):
            change, corrected = "extra", True
        elif canon_pred and canon_pred in canon_mfa:
            change, corrected = "notation", True
        elif canon_pred and canon_pred in canon_cv:
            change, corrected = "cv", True
        if measured and not corrected:
            # Reproducing a gold reading the verdict judges wrong.
            change = "gold-bad"

    # Distance is always to the closest accepted reading; an excused word is
    # exact by construction, so it contributes distance 0 and the prediction's
    # own length as the reference length.
    if corrected:
        distance, reference_length = 0, len(pred)
    else:
        candidates = accepted or list(mfa_refs)
        distances = [edit_distance(list(pred), list(r)) for r in candidates]
        distance = min(distances) if distances else len(pred)
        reference_length = len(candidates[distances.index(distance)]) \
            if candidates else 0

    # The measured side never moves: always the closest MFA reference.
    measured_distances = [edit_distance(list(pred), list(r)) for r in mfa_refs]
    measured_distance = min(measured_distances) if measured_distances else len(pred)
    measured_reference_length = len(mfa_refs[measured_distances.index(
        measured_distance)]) if measured_distances else 0

    return Judgment(word=word, verdict=verdict.verdict,
                    counted=verdict.counted, measured_exact=measured,
                    corrected_exact=corrected, change=change,
                    distance=distance, reference_length=reference_length,
                    measured_distance=measured_distance,
                    measured_reference_length=measured_reference_length)


def summarize(judgments: Iterable[Judgment]) -> ModelReport:
    """Fold judgments into a report (convention rows dropped from the math)."""
    report = ModelReport()
    for item in judgments:
        if not item.counted:
            continue
        report.n += 1
        report.measured_exact += int(item.measured_exact)
        report.corrected_exact += int(item.corrected_exact)
        report.distance += item.distance
        report.reference_length += item.reference_length
        report.measured_distance += item.measured_distance
        report.measured_reference_length += item.measured_reference_length
        if item.change == "gold-bad":
            report.penalized += 1
        elif item.change is not None:
            report.excused[item.change] = report.excused.get(item.change, 0) + 1
    return report
