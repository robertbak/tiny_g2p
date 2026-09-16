"""Nasal-merge restoration: merged labels back to raw MFA phones."""

from __future__ import annotations

import pytest

from tiny_g2p.phones import HOMORGANIC, STOPS, is_stop, redentalize, restore


def test_every_stop_has_a_homorganic_nasal():
    assert set(HOMORGANIC) == set(STOPS)
    assert set(HOMORGANIC.values()) <= {"m", "n̪", "ɲ", "ŋ"}


@pytest.mark.parametrize(
    "follower,nasal",
    [
        ("p", "m"), ("b", "m"), ("pʲ", "m"), ("bʲ", "m"),
        ("t̪", "n̪"), ("d̪", "n̪"), ("t̪s̪", "n̪"), ("d̪z̪", "n̪"),
        ("tʂ", "n̪"), ("dʐ", "n̪"),
        ("tɕ", "ɲ"), ("dʑ", "ɲ"), ("c", "ŋ"), ("ɟ", "ŋ"),
        ("k", "ŋ"), ("ɡ", "ŋ"),
    ],
)
def test_homorganic_nasal_pins(follower, nasal):
    assert HOMORGANIC[follower] == nasal


@pytest.mark.parametrize(
    "word,labels,expected",
    [
        # ą/ę + stop: split (kąt, ręka, pięć).
        ("kąt", ["k", "ɔ̃", "t̪"], ["k", "ɔ", "n̪", "t̪"]),
        ("ręka", ["r", "ɛ̃", "k", "a"], ["r", "ɛ", "ŋ", "k", "a"]),
        ("pięć", ["pʲ", None, "ɛ̃", "tɕ"], ["pʲ", "ɛ", "ɲ", "tɕ"]),
        ("gęba", ["ɡ", "ɛ̃", "b", "a"], ["ɡ", "ɛ", "m", "b", "a"]),
        # ą/ę + fricative or word end: keep the nasal vowel.
        ("wąż", ["v", "ɔ̃", "ʂ"], ["v", "ɔ̃", "ʂ"]),
        ("gęś", ["ɡ", "ɛ̃", "ɕ"], ["ɡ", "ɛ̃", "ɕ"]),
        ("idą", ["i", "d̪", "ɔ̃"], ["i", "d̪", "ɔ̃"]),
        # plain vowels never split, even before V+nasal+stop (mantra case).
        ("pan", ["p", "a", "n̪"], ["p", "a", "n̪"]),
        ("manta", ["m", "a", "n̪", "t̪", "a"], ["m", "a", "n̪", "t̪", "a"]),
        # blanks vanish; model errors pass through unscathed.
        ("kot", ["k", "ɔ", None], ["k", "ɔ"]),
        ("pan", ["p", "ɔ̃", "n̪"], ["p", "ɔ̃", "n̪"]),
    ],
)
def test_restore(word, labels, expected):
    assert restore(word, labels) == expected


def test_is_stop_covers_affricates_and_nothing_else():
    assert is_stop("tʂ") and is_stop("d̪z̪") and is_stop("pʲ")
    assert not is_stop("ʂ") and not is_stop("ɔ̃") and not is_stop(None)


def test_redentalize_restores_the_alveolar_series():
    assert redentalize(["k", "ɔ", "t"]) == ["k", "ɔ", "t̪"]
    assert redentalize(["t", "s", "ts", "dz", "x"]) == \
        ["t̪", "s̪", "t̪s̪", "d̪z̪", "x"]
