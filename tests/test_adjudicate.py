"""Adjudication semantics: what each verdict does to a model's score."""

from __future__ import annotations

import pytest

from tiny_g2p.adjudicate import (
    VERDICTS,
    AdjudicationError,
    Verdict,
    judge,
    summarize,
)

MFA = "mfa"
CV = "cv"


def test_unknown_verdict_is_rejected():
    with pytest.raises(AdjudicationError, match="verdict must be one of"):
        Verdict(verdict="maybe")


def test_from_json_accepts_shorthand_dict_and_none():
    assert Verdict.from_json(None).verdict == MFA
    assert Verdict.from_json("cv").verdict == CV
    parsed = Verdict.from_json({"verdict": "none", "correct": ["a x m a"],
                                "note": "why"})
    assert parsed.correct == (("a", "x", "m", "a"),)
    assert parsed.note == "why"


def test_default_trusts_mfa():
    refs = [("k", "ɔ", "t")]
    assert judge("kot", ["k", "ɔ", "t"], refs, [], Verdict()).measured_exact
    wrong = judge("kot", ["k", "ɔ", "d"], refs, [], Verdict())
    assert not wrong.measured_exact and not wrong.corrected_exact
    assert wrong.distance == 1 and wrong.change is None


def test_both_accepts_a_canonical_cv_match():
    # Gemination: CV doubles the l, MFA does not. Either reading counts.
    mfa_refs = [("a", "l", "l", "a")]
    cv_refs = [("a", "l", "a")]
    excused = judge("alla", ["a", "l", "a"], mfa_refs, cv_refs,
                    Verdict(verdict="both"))
    assert not excused.measured_exact and excused.corrected_exact
    assert excused.change == CV
    # The measured side still reports the MFA distance, for an honest PER.
    assert excused.measured_distance == 1
    assert excused.distance == 0
    # A prediction matching MFA is simply correct, with nothing excused.
    plain = judge("alla", ["a", "l", "l", "a"], mfa_refs, cv_refs,
                  Verdict(verdict="both"))
    assert plain.measured_exact and plain.corrected_exact
    assert plain.change is None


def test_both_does_not_excuse_a_third_reading():
    result = judge("alla", ["a", "w", "a"], [("a", "l", "l", "a")],
                   [("a", "l", "a")], Verdict(verdict="both"))
    assert not result.corrected_exact and result.change is None


def test_cv_verdict_penalizes_a_model_matching_mfa():
    # MFA drops a sound; a model reproducing the gold is a corrected error.
    result = judge("odessie", ["k", "ɔ", "t"], [("k", "ɔ", "t")],
                   [("k", "ɔ", "t", "ɛ")], Verdict(verdict=CV))
    assert result.measured_exact and not result.corrected_exact
    assert result.change == "gold-bad"
    assert judge("odessie", ["k", "ɔ", "t", "ɛ"], [("k", "ɔ", "t")],
                 [("k", "ɔ", "t", "ɛ")], Verdict(verdict=CV)).corrected_exact


def test_none_reads_the_correct_reading_from_correct():
    verdict = Verdict(verdict="none", correct=(("a", "x", "m", "a"),))
    excused = judge("achmeda", ["a", "x", "m", "a"], [("tʂ", "m", "a")], [],
                    verdict)
    assert not excused.measured_exact and excused.corrected_exact
    assert excused.change == "extra"
    # The gold reading itself no longer counts.
    stale = judge("achmeda", ["tʂ", "m", "a"], [("tʂ", "m", "a")], [], verdict)
    assert stale.measured_exact and not stale.corrected_exact


def test_open_keeps_mfa_and_adds_readings():
    verdict = Verdict(verdict="open", correct=(("b", "l", "a", "j", "r"),))
    assert judge("blair", ["b", "l", "ɛ", "r"], [("b", "l", "ɛ", "r")], [],
                 verdict).corrected_exact
    assert judge("blair", ["b", "l", "a", "j", "r"], [("b", "l", "ɛ", "r")],
                 [], verdict).corrected_exact
    assert not judge("blair", ["b", "l", "a", "r"], [("b", "l", "ɛ", "r")], [],
                     verdict).corrected_exact


def test_convention_rows_leave_the_denominator():
    result = judge("agd", ["a", "ɡ", "t"], [("a", "ɟ", "ɛ", "d", "ɛ")], [],
                   Verdict(verdict="convention"))
    assert not result.counted
    assert summarize([result]).n == 0


def test_summarize_counts_excuses_and_penalties():
    judgments = [
        judge("a", ["k"], [("k",)], [], Verdict()),                       # ok
        judge("b", ["k"], [("k",)], [], Verdict(verdict=CV)),             # penalty
        judge("c", ["l"], [("l", "l")], [("l",)],
              Verdict(verdict="both")),                                   # excused
        judge("d", ["x"], [("k",)], [], Verdict()),                       # error
    ]
    report = summarize(judgments)
    assert (report.n, report.measured_exact, report.corrected_exact) == (4, 2, 2)
    assert report.excused == {CV: 1}
    assert report.penalized == 1
    assert report.measured_accuracy == pytest.approx(0.5)
    assert report.corrected_accuracy == pytest.approx(0.5)


def test_verdict_list_matches_the_documented_set():
    assert set(VERDICTS) == {"mfa", "both", "open", "cv", "none", "convention"}
