"""Monotonic grapheme->phone alignment: labels for the per-char model."""

from __future__ import annotations

import pytest

from tiny_g2p.align import (
    DIGRAPHS,
    SEED,
    align,
    align_entry,
)


def labels_of(word, phones):
    alignment = align(word, phones)
    assert alignment is not None
    return list(alignment.labels), alignment.cost


def test_plain_word_aligns_one_to_one():
    labels, cost = labels_of("kot", ["k", "ɔ", "t̪"])
    assert labels == ["k", "ɔ", "t̪"]
    assert cost == 0.0


def test_digraph_emits_on_its_first_character():
    labels, cost = labels_of("nie", ["ɲ", "ɛ"])
    assert labels == ["ɲ", None, "ɛ"]
    assert cost == 0.0


def test_digraph_not_taken_when_phones_forbid_it():
    # dania (meals): ni IS the digraph; dania (Denmark): the glide needs i.
    assert labels_of("dania", ["d̪", "a", "ɲ", "a"])[0] == ["d̪", "a", "ɲ", None, "a"]
    assert labels_of("dania", ["d̪", "a", "ɲ", "j", "a"])[0] == ["d̪", "a", "ɲ", "j", "a"]


def test_trigraph_dzi():
    labels, cost = labels_of("dzień", ["dʑ", "ɛ", "ɲ"])
    assert labels == ["dʑ", None, None, "ɛ", "ɲ"]
    assert cost == 0.0


def test_nasal_op_merges_a_before_a_stop():
    labels, cost = labels_of("kąt", ["k", "ɔ", "n̪", "t̪"])
    assert labels == ["k", "ɔ̃", "t̪"]
    assert cost == 0.0


def test_nasal_op_with_palatalised_onset_and_glide():
    labels, _ = labels_of("pięć", ["pʲ", "ɛ", "ɲ", "tɕ"])
    assert labels == ["pʲ", None, "ɛ̃", "tɕ"]


def test_double_letter_reduces():
    # Which of the two n's emits is an arbitrary DP tie-break (both cost the
    # same); pin the invariant, not the choice.
    labels, cost = labels_of("anna", ["a", "n̪", "a"])
    assert labels[0] == "a" and labels[3] == "a"
    assert sorted(p for p in labels[1:3] if p) == ["n̪"]
    assert cost == pytest.approx(0.6)


def test_glide_i_aligns_to_j_when_present():
    labels, _ = labels_of("radio", ["r", "a", "d̪", "j", "ɔ"])
    assert labels == ["r", "a", "d̪", "j", "ɔ"]


def test_unalignable_acronym_returns_none():
    assert align("abc", ["a", "b", "ɛ", "t̪s̪", "ɛ"]) is None
    assert align("ex", ["ɛ", "k", "s̪"]) is None


def test_quality_gate_rejects_forced_noise():
    assert align("kot", ["ʂ", "ʐ", "ɕ"]) is not None  # a path exists ...
    assert align_entry("kot", ["ʂ", "ʐ", "ɕ"]) is None  # ... but it is garbage


def test_every_polish_letter_has_a_seed():
    for char in "aąbcćdeęfghijklłmnńoóprsśtuwyzźż":
        assert char in SEED, f"no seed phone for {char!r}"


def test_digraph_table_covers_the_polish_digraphs():
    for digraph in ["cz", "sz", "rz", "dz", "dż", "dź", "ch",
                    "ci", "si", "zi", "ni"]:
        assert digraph in DIGRAPHS
