"""Promotion eligibility and the atomic active/ refresh."""

from __future__ import annotations

import json

import pytest

import tiny_g2p.promote as promote_module
from tiny_g2p.promote import check_eligible, promote_run


def meta(acc, guards=None, taxonomy="deadbeef1234"):
    return {"val_word_accuracy": acc,
            "guards": guards or {"all": acc, "nasal": acc},
            "taxonomy": taxonomy}


def test_first_run_is_eligible():
    eligible, reasons = check_eligible(meta(0.5), None)
    assert eligible
    assert "seed" in reasons[0]


def test_strict_improvement_required():
    assert check_eligible(meta(0.9), meta(0.8))[0]
    assert not check_eligible(meta(0.8), meta(0.8))[0]  # equal is not better
    assert not check_eligible(meta(0.7), meta(0.8))[0]


def test_guard_regression_blocks_promotion():
    run = meta(0.9, {"all": 0.9, "nasal": 0.70})
    base = meta(0.8, {"all": 0.8, "nasal": 0.85})
    eligible, reasons = check_eligible(run, base)
    assert not eligible
    assert any("nasal" in r for r in reasons)


def test_guard_within_cap_passes():
    run = meta(0.9, {"all": 0.9, "nasal": 0.849})
    base = meta(0.8, {"all": 0.8, "nasal": 0.85})
    assert check_eligible(run, base)[0]


def make_run(tmp_path, acc=0.9, taxonomy="deadbeef1234"):
    run = tmp_path / "runs" / "ts"
    run.mkdir(parents=True)
    for name in ("best_float.pt", "best_quantized.pt", "src_vocab.json",
                 "tgt_vocab.json"):
        (run / name).write_text("dummy", encoding="utf-8")
    (run / "meta.json").write_text(
        json.dumps(meta(acc, taxonomy=taxonomy)), encoding="utf-8")
    return run


def test_promote_copies_artifacts_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(promote_module, "ACTIVE_DIR", tmp_path / "active")
    run = make_run(tmp_path)
    active, reasons = promote_run(run)
    assert (active / "model.pt").is_file()
    assert (active / "model_quantized.pt").is_file()
    assert (active / "meta.json").is_file()
    assert not (tmp_path / ".active_staging_ts").exists()
    assert reasons  # the seed explanation


def test_promote_refuses_ineligible_run(tmp_path, monkeypatch):
    monkeypatch.setattr(promote_module, "ACTIVE_DIR", tmp_path / "active")
    promote_run(make_run(tmp_path, acc=0.9))
    with pytest.raises(ValueError, match="not eligible"):
        promote_run(make_run(tmp_path / "second", acc=0.8))


def test_promote_force_overrides_eligibility(tmp_path, monkeypatch):
    monkeypatch.setattr(promote_module, "ACTIVE_DIR", tmp_path / "active")
    promote_run(make_run(tmp_path, acc=0.9))
    active, _ = promote_run(make_run(tmp_path / "second", acc=0.5),
                            force=True)
    saved = json.loads((active / "meta.json").read_text(encoding="utf-8"))
    assert saved["provenance"]["forced"] is True


def test_promote_refuses_taxonomy_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(promote_module, "ACTIVE_DIR", tmp_path / "active")
    promote_run(make_run(tmp_path, acc=0.9, taxonomy="aaaa"))
    with pytest.raises(ValueError, match="taxonomy mismatch"):
        promote_run(make_run(tmp_path / "second", acc=0.95, taxonomy="bbbb"))
