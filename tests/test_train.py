"""Training loop end to end on toy data (CPU, two epochs, QAT covered)."""

from __future__ import annotations

import json

import pytest
import torch

from pl_g2p.lexicon import Entry, Lexicon
from tiny_g2p.data import make_dataset
from tiny_g2p.model import build
from tiny_g2p.train import TrainConfig, train

import tiny_g2p.train as train_module

TOY = [
    ("kot", ("k", "ɔ", "t̪")),
    ("kod", ("k", "ɔ", "t̪")),
    ("lud", ("l", "u", "t̪")),
    ("lód", ("l", "u", "t̪")),
    ("kąt", ("k", "ɔ", "n̪", "t̪")),
    ("ręka", ("r", "ɛ", "ŋ", "k", "a")),
    ("wąż", ("v", "ɔ̃", "ʂ")),
    ("gęś", ("ɡ", "ɛ̃", "ɕ")),
    ("nie", ("ɲ", "ɛ")),
    ("ani", ("a", "ɲ", "i")),
    ("czapka", ("tʂ", "a", "p", "k", "a")),
    ("morze", ("m", "ɔ", "ʐ", "ɛ")),
    ("może", ("m", "ɔ", "ʐ", "ɛ")),
    ("pięć", ("pʲ", "ɛ", "ɲ", "tɕ")),
    ("dania", ("d̪", "a", "ɲ", "a")),
    ("dania", ("d̪", "a", "ɲ", "j", "a")),
    ("pan", ("p", "a", "n̪")),
    ("pana", ("p", "a", "n̪", "a")),
    ("mama", ("m", "a", "m", "a")),
    ("tama", ("t̪", "a", "m", "a")),
    ("szafa", ("ʂ", "a", "f", "a")),
    ("mucha", ("m", "u", "x", "a")),
    ("dzwon", ("d̪z̪", "v", "ɔ", "n̪")),
    ("dzień", ("dʑ", "ɛ", "ɲ")),
]


def make_data():
    return make_dataset(Lexicon([Entry(w, p) for w, p in TOY]),
                        seed=3, ratios=(0.7, 0.15, 0.15))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_train_runs_and_selects(tmp_path):
    data = make_data()
    assert data.val and data.test
    config = TrainConfig(epochs=2, batch_size=8, qat_epochs=1, device="cpu",
                         promote=False, seed=1)
    result = train(data, tmp_path / "runs" / "t0", config)
    assert result.best_val_accuracy is not None
    assert 0.0 <= result.best_val_accuracy <= 1.0
    assert result.best_kind in {"random-init", "float", "qat"}
    assert not result.promoted
    history = json.loads((result.run_dir / "history.json").read_text())
    assert len(history) == 3  # epoch zero + two epochs
    assert any(h["phase"] == "qat" for h in history)
    for name in ("best_float.pt", "best_quantized.pt", "src_vocab.json",
                 "tgt_vocab.json", "meta.json", "config.json"):
        assert (result.run_dir / name).is_file(), name
    meta = json.loads((result.run_dir / "meta.json").read_text())
    assert set(meta["guards"]) == {"all", "nasal", "digraph", "palatal",
                                   "rare"}
    assert meta["params"] < 35_000


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_train_warm_starts_matching_active(tmp_path, monkeypatch):
    data = make_data()
    active = tmp_path / "active"
    active.mkdir()
    model = build(len(data.src_vocab), len(data.tgt_vocab), 10)
    torch.save({"state_dict": model.state_dict()}, active / "model.pt")
    data.src_vocab.to_json(active / "src_vocab.json")
    data.tgt_vocab.to_json(active / "tgt_vocab.json")
    monkeypatch.setattr(train_module, "ACTIVE_DIR", active)
    config = TrainConfig(epochs=1, batch_size=8, qat_epochs=0, device="cpu",
                         promote=False, seed=1)
    result = train(data, tmp_path / "runs" / "t1", config)
    meta = json.loads((result.run_dir / "meta.json").read_text())
    assert meta["warmed"] is True
    assert result.best_kind in {"promoted", "float"}


def test_polish_refuses_without_matching_active(tmp_path, monkeypatch):
    data = make_data()
    monkeypatch.setattr(train_module, "ACTIVE_DIR", tmp_path / "missing")
    with pytest.raises(ValueError, match="--polish needs"):
        train(data, tmp_path / "runs" / "t2",
              TrainConfig(epochs=6, polish=True, device="cpu", promote=False))


def test_train_requires_verification_rows(tmp_path):
    data = make_data()
    data.val.clear()
    with pytest.raises(ValueError, match="verification"):
        train(data, tmp_path / "runs" / "t3",
              TrainConfig(epochs=1, device="cpu", promote=False))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_finetune_repairs_bank_words(tmp_path, monkeypatch):
    from tiny_g2p.train import FineTuneConfig, finetune

    data = make_data()
    refs = {w: list(p) for w, p in TOY}
    bank = sorted({e.word for e in data.train})[:2]
    assert bank, "need train words for the bank"

    active = tmp_path / "active"
    active.mkdir()
    model = build(len(data.src_vocab), len(data.tgt_vocab), 10)
    torch.save({"state_dict": model.state_dict()}, active / "model.pt")
    data.src_vocab.to_json(active / "src_vocab.json")
    data.tgt_vocab.to_json(active / "tgt_vocab.json")
    (active / "meta.json").write_text(
        json.dumps({"val_word_accuracy": 0.0, "guards": {}}))
    monkeypatch.setattr(train_module, "ACTIVE_DIR", active)

    # The active dir also needs a quantized twin for the baseline probe.
    tiny = TinyPhonemizer_for_test(data)
    torch.save(tiny, active / "model_quantized.pt")

    result = finetune(
        data, bank, tmp_path / "runs" / "f0",
        FineTuneConfig(steps=4, batch_size=4, rehearsal_epochs=0,
                       rollback_cap=1.0, device="cpu"),
        teacher=lambda words: {w: refs[w] for w in words})
    assert result.saved
    assert 0.0 <= result.bank_accuracy <= 1.0
    assert (result.run_dir / "best_float.pt").is_file()
    meta = json.loads((result.run_dir / "meta.json").read_text())
    assert meta["kind"] == "finetune"


def TinyPhonemizer_for_test(data):
    """A whole-module float model standing in for a quantized twin.

    The baseline probe only decodes with it; a float module decodes
    identically, so the rollback comparison stays meaningful in miniature.
    """
    model = build(len(data.src_vocab), len(data.tgt_vocab), 10)
    model.eval()
    return model


def test_finetune_refuses_verification_words(tmp_path, monkeypatch):
    from tiny_g2p.train import FineTuneConfig, finetune

    data = make_data()
    monkeypatch.setattr(train_module, "ACTIVE_DIR", tmp_path / "active")
    leaked = [data.val[0].word]
    with pytest.raises(ValueError, match="verification words"):
        finetune(data, leaked, tmp_path / "runs" / "f1",
                 FineTuneConfig(device="cpu"),
                 teacher=lambda words: {w: ["a"] for w in words})
