"""Cross-lexicon canonicalization and agreement (pins from surveyed rows)."""

from __future__ import annotations

import pytest

from tiny_g2p.xlex import (
    canonicalize,
    clean_cv_word,
    parse_cv,
    readings_agree,
)

MFA = "mfa"
CV = "cv"


@pytest.mark.parametrize(
    "mfa,cv",
    [
        # dentals + retroflex merge + palatal strip agree
        (["a", "b", "r", "a", "m", "ɔ", "vʲ", "i", "tʂ"],
         ["a", "b", "r", "a", "m", "ɔ", "v", "i", "t", "ʂ"]),
        (["b", "u", "dʐ", "ɛ", "t̪"], ["b", "u", "d", "ʐ", "ɛ", "t"]),
        # tie bars + palatal strip agree
        (["a", "k", "a", "pʲ", "i", "tɕ"], ["a", "k", "a", "p", "i", "t͡ɕ"]),
        # CV's glide j is MFA's palatalization
        (["a", "n̪", "t̪", "ɨ", "ʂ", "tʂ", "ɛ", "pʲ", "ɔ"],
         ["a", "n", "t", "ɨ", "ʂ", "t", "ʂ", "ɛ", "p", "j", "ɔ"]),
        # ʎ is CV's l+i
        (["a", "k", "t̪", "u", "a", "ʎ", "i"],
         ["a", "k", "t", "u", "a", "l", "i"]),
        # qu expands
        (["k", "w", "a", "d̪", "ɛ", "m"], ["q", "u", "a", "d", "ɛ", "m"]),
        # identical rows agree
        (["a", "ɕ", "a"], ["a", "ɕ", "a"]),
    ],
)
def test_surveyed_agreements(mfa, cv):
    assert readings_agree([tuple(mfa)], [tuple(cv)])


def test_letter_x_expands_to_ks():
    mfa = ["fʲ", "i", "r", "ɛ", "f", "ɔ", "k", "s̪"]
    cv = ["f", "i", "r", "ɛ", "f", "ɔ", "x"]
    assert readings_agree([tuple(mfa)], [tuple(cv)], word="firefox")


def test_fricative_x_from_ch_does_not_expand():
    assert readings_agree([("x", "a")], [("x", "a")], word="ha")


def test_brexit_disputes_x_as_s_vs_ks():
    mfa = ["b", "r", "ɛ", "s̪", "i", "t̪"]
    cv = ["b", "r", "ɛ", "x", "i", "t"]
    assert not readings_agree([tuple(mfa)], [tuple(cv)], word="brexit")


def test_trz_spelling_keeps_cv_pair_split():
    # CV writes both cz and trz as ``t ʂ``; MFA merges cz but splits trz.
    # The word decides: trzecie must not merge, or every trz word disputes.
    assert canonicalize(["t", "ʂ", "ɛ", "t͡ɕ", "ɛ"], source=CV,
                        word="trzecie") == ("t", "ʂ", "ɛ", "tɕ", "ɛ")
    assert readings_agree([("t̪", "ʂ", "ɛ", "tɕ", "ɛ")],
                          [("t", "ʂ", "ɛ", "t͡ɕ", "ɛ")], word="trzecie")
    assert readings_agree([("s̪", "t̪", "ʂ", "a", "l", "ɛ")],
                          [("s", "t", "ʂ", "a", "l", "ɛ")], word="strzale")


def test_cz_spelling_still_merges_cv_pair():
    assert canonicalize(["t", "ʂ", "ɛ"], source=CV,
                        word="czeczenii") == ("tʂ", "ɛ")


def test_drzenie_disputes_dropped_r():
    # CV drops the r of drz- (it treats the spelling as an affricate), MFA
    # keeps it. Not a notation difference: CV is wrong, so leave it disputed.
    mfa = [("d̪", "r", "ʐ", "ɛ", "ɲ", "ɛ")]
    assert not readings_agree(mfa, [("d", "ʐ", "ɛ", "ɲ", "ɛ")],
                              word="drżenie")


def test_final_e_bridges_mfa_oral_to_cv_nasal():
    mfa = ["a", "ɡ", "r", "ɛ", "s", "ɛ"]
    cv = ["a", "ɡ", "r", "ɛ", "s", "ɛ̃"]
    assert readings_agree([tuple(mfa)], [tuple(cv)], word="agresję")


def test_final_e_untouched_when_word_ends_in_e():
    # Same phone pair, but the letter condition fails: still disputed.
    mfa = ["a", "l", "ɛ"]
    cv = ["a", "l", "ɛ̃"]
    assert not readings_agree([tuple(mfa)], [tuple(cv)], word="ale")


def test_palatal_stops_fold_into_velars():
    # alinki: MFA c, CV kʲ -- one sound, two notations.
    assert readings_agree([("a", "l", "i", "n̪", "c", "i")],
                          [("a", "l", "i", "n", "kʲ", "i")], word="alinki")
    # alergię: CV forgets palatalization (plain ɡ); folded, not disputed.
    assert readings_agree([("a", "l", "ɛ", "r", "ɟ", "ɛ")],
                          [("a", "l", "ɛ", "r", "ɡ", "ɛ")], word="alergię")


@pytest.mark.parametrize(
    "mfa,cv",
    [
        # bezsilne: MFA drops the ɕ CV keeps
        (["b", "ɛ", "s̪", "i", "l", "n̪", "ɛ"],
         ["b", "ɛ", "s", "ɕ", "i", "l", "n", "ɛ"]),
    ],
)
def test_surveyed_disagreements(mfa, cv):
    assert not readings_agree([tuple(mfa)], [tuple(cv)])


def test_polyphony_agrees_when_any_reading_matches():
    mfa = [("d̪", "a", "ɲ", "a"), ("d̪", "a", "ɲ", "j", "a")]
    assert readings_agree(mfa, [("d", "a", "ɲ", "a")])


def test_diphthong_glide_folds_i_to_j():
    # beikocące: MFA ɛi, CV ɛj before a consonant.
    assert readings_agree([("b", "ɛ", "i", "k")], [("b", "ɛ", "j", "k")],
                          word="beik")
    # Pre-vowel i/j stay distinct (dania vs moje keep their shapes).
    assert canonicalize(["d", "a", "ɲ", "i", "a"], source=MFA) == \
        ("d", "a", "j̃", "i", "a")
    assert canonicalize(["m", "ɔ", "j", "ɛ"], source=CV) == ("m", "ɔ", "j", "ɛ")


def test_palatal_nasal_folds_to_nasal_glide():
    # chińsku: MFA writes ń as ɲ, CV as j̃ (ń before a consonant).
    assert canonicalize(["ç", "i", "j̃", "s̪", "k", "u"], source=MFA) == \
        ("x", "i", "j̃", "s", "k", "u")
    assert readings_agree([("x", "i", "ɲ", "s̪", "k", "u")],
                          [("x", "i", "j̃", "s", "k", "u")], word="chińsku")
    # The ńdź/ndź cluster difference survives: it is a real dispute.
    assert not readings_agree([("ʐ", "ɔ", "ɲ", "dʑ", "ɛ")],
                              [("ʐ", "ɔ", "n", "d͡ʑ", "ɛ")], word="rządzie")


def test_glide_w_folds_to_v():
    # euro: MFA ɛ w r ɔ, CV ɛ v r ɔ -- ł and /v/ share one phone.
    assert readings_agree([("ɛ", "w", "r", "ɔ")], [("ɛ", "v", "r", "ɔ")],
                          word="euro")
    assert canonicalize(["m", "a", "w", "ʐ", "ɛ"], source=MFA) == \
        ("m", "a", "v", "ʐ", "ɛ")


def test_j_drop_keeps_intervocalic_j():
    assert canonicalize(["j", "a", "j", "k", "ɔ"], source=CV) == \
        ("j", "a", "j", "k", "ɔ")
    assert canonicalize(["ɔ", "b", "j", "a"], source=CV) == ("ɔ", "b", "a")
    assert canonicalize(["ɔ", "b", "j", "a"], source=MFA) == ("ɔ", "b", "a")


def test_glottal_stop_drops():
    assert canonicalize(["k", "ɔ", "ʔ", "ɔ"], source=MFA) == ("k", "ɔ", "ɔ")


def test_canonicalize_rejects_unknown_source():
    with pytest.raises(ValueError, match="source must be"):
        canonicalize(["a"], source="espeak")


def test_clean_cv_word_strips_quotes_and_punctuation():
    assert clean_cv_word('"Aboż') == "aboż"
    assert clean_cv_word('"...Poco') == "poco"
    assert clean_cv_word("Kot.") == "kot"


def test_parse_cv_funnel_and_dedup(tmp_path):
    path = tmp_path / "cv.dict"
    path.write_text(
        "# comment\n"
        '"Kot\tk ɔ t\n'
        "kot\tk ɔ t\n"       # case variant, same reading -> deduped
        "kot\tk ɔ d\n"         # second reading kept
        "1920\tn ɔ\n"           # digits dropped
        "coś.\t...\n",         # punct stripped, ... kept? no phones check below
        encoding="utf-8")
    lexicon = parse_cv(path)
    assert lexicon.raw_lines == 5
    assert lexicon.entries["kot"] == [("k", "ɔ", "t"), ("k", "ɔ", "d")]
    assert "1920" not in lexicon.entries
