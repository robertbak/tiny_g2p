"""CLI helpers (no torch, no checkpoint needed)."""

from __future__ import annotations

from tiny_g2p.cli import split_cli_words


def test_split_cli_words_splits_quoted_sentences():
    assert split_cli_words(["Wrocław Szczebrzeszyn", "morze"]) == [
        "Wrocław", "Szczebrzeszyn", "morze"]


def test_split_cli_words_drops_empties():
    assert split_cli_words(["  ", "kot"]) == ["kot"]
