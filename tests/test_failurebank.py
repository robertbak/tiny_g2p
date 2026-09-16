"""Failure-bank IO and the verification-leakage guard (no torch)."""

from __future__ import annotations

import tiny_g2p.failurebank as failurebank_module
from tiny_g2p.failurebank import add_words, check_leakage, read_bank


def test_bank_round_trip_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(failurebank_module, "FAILURE_BANK_DIR", tmp_path)
    assert read_bank() == []
    assert add_words(["kot", "ząb", "kot"]) == ["kot", "ząb"]
    assert read_bank() == ["kot", "ząb"]


def test_bank_skips_blanks_and_comments(tmp_path, monkeypatch):
    monkeypatch.setattr(failurebank_module, "FAILURE_BANK_DIR", tmp_path)
    (tmp_path / "words.txt").write_text("kot\n\n# oops\nząb\n",
                                        encoding="utf-8")
    assert read_bank() == ["kot", "ząb"]


def test_check_leakage_flags_verification_words():
    val = {"kot", "nie"}
    test = {"ząb"}
    assert check_leakage(["kot", "mama", "ząb"], val, test) == ["kot", "ząb"]
    assert check_leakage(["mama"], val, test) == []
