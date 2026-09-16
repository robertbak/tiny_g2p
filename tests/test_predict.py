"""Decoding, checkpoint loading, and guard groups."""

from __future__ import annotations

import pytest
import torch

from tiny_g2p.data import EvalRow
from tiny_g2p.model import build
from tiny_g2p.predict import (
    TinyPhonemizer,
    evaluate_guards,
    evaluate_words,
    guard_groups,
    normalize_input,
)
from tiny_g2p.vocab import Vocab

CHARS = ["k", "o", "t", "ą", "n", "i", "e"]
PHONES = ["k", "ɔ", "t̪", "ɔ̃", "n̪", "ɲ", "ɛ"]


def make_phonemizer() -> TinyPhonemizer:
    src = Vocab.build_chars(["".join(CHARS)])
    tgt = Vocab.build_phones([PHONES])
    model = build(len(src), len(tgt), 10)
    model.eval()
    return TinyPhonemizer(model, src, tgt, torch.device("cpu"))


def test_normalize_input_lowercases():
    assert normalize_input("Wrocław ") == "wrocław"


def test_predict_word_returns_phones():
    phonemizer = make_phonemizer()
    out = phonemizer.predict_word("kot")
    assert isinstance(out, list)
    assert all(isinstance(p, str) for p in out)


def test_predict_batch_handles_mixed_lengths():
    phonemizer = make_phonemizer()
    out = phonemizer.predict_batch(["kot", "nie"])
    assert len(out) == 2


def test_predict_empty_batch():
    assert make_phonemizer().predict_batch([]) == []


def test_save_and_load_float_round_trip(tmp_path):
    phonemizer = make_phonemizer()
    torch.save({"state_dict": phonemizer.model.state_dict()},
               tmp_path / "model.pt")
    phonemizer.src_vocab.to_json(tmp_path / "src_vocab.json")
    phonemizer.tgt_vocab.to_json(tmp_path / "tgt_vocab.json")
    loaded = TinyPhonemizer.from_dir(tmp_path, quantized=False)
    before = phonemizer.predict_word("kot")
    after = loaded.predict_word("kot")
    assert before == after


def test_from_active_fails_loudly_when_absent(tmp_path, monkeypatch):
    import tiny_g2p.predict as predict_module

    monkeypatch.setattr(predict_module, "ACTIVE_DIR", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="no promoted model"):
        TinyPhonemizer.from_active()


def test_evaluate_words_scores_polyphony():
    phonemizer = make_phonemizer()
    rows = [EvalRow("kot", ("k", "ɔ", "t̪")),
            EvalRow("nie", ("ɲ", "ɛ"))]
    result = evaluate_words(phonemizer, rows)
    assert result.n == 2
    assert 0.0 <= result.word_accuracy <= 1.0


def test_guard_groups_split_phenomena():
    rows = [EvalRow("kąt", ("k", "ɔ", "n̪", "t̪")),
            EvalRow("nie", ("ɲ", "ɛ")),
            EvalRow("kot", ("k", "ɔ", "t̪"))]
    groups = guard_groups(rows, rare={"t̪"})
    assert [r.word for r in groups["nasal"]] == ["kąt"]
    assert [r.word for r in groups["digraph"]] == ["nie"]
    assert [r.word for r in groups["palatal"]] == []
    assert sorted(r.word for r in groups["rare"]) == ["kot", "kąt"]
    assert len(groups["all"]) == 3


def test_evaluate_guards_scores_every_group():
    phonemizer = make_phonemizer()
    rows = [EvalRow("kąt", ("k", "ɔ", "n̪", "t̪")),
            EvalRow("kot", ("k", "ɔ", "t̪"))]
    guards = evaluate_guards(phonemizer, rows, rare=set())
    assert set(guards) == {"all", "nasal", "digraph", "palatal", "rare"}
    assert guards["all"].n == 2
