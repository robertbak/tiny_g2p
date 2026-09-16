"""The ONNX export agrees with the torch model it came from.

`onnxruntime` is a dev-only dependency: the export itself needs only torch.
The test is deliberately end-to-end -- export a graph, run it, compare argmaxes
against `WindowG2P.forward` -- because the failure mode of an export is silent
numerical drift, not a crash.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tiny_g2p.data import MANNERS
from tiny_g2p.export import export_onnx
from tiny_g2p.model import build
from tiny_g2p.vocab import Vocab

ORT = pytest.importorskip("onnxruntime")

WORDS = [
    "kot", "morze", "szczebrzeszyn", "wroc\u0142aw", "\u017ad\u017ab\u0142o",
    "\u0105", "\u0119", "bezwzgl\u0119dny", "najd\u0142u\u017cszy",
    "chrz\u0105szcz", "gdzie\u017cby", "pi\u0119\u0107set", "kotlet",
    "a", "ab", "abc", "q", "v", "x", "zzz", "\u017bd\u017ab\u0142o",
]


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    return export_onnx(tmp_path_factory.mktemp("onnx") / "tiny_g2p.onnx")


def _torch_model() -> tuple[torch.nn.Module, Vocab, Vocab]:
    src = Vocab.from_json("active/src_vocab.json")
    tgt = Vocab.from_json("active/tgt_vocab.json")
    state = torch.load("active/model.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
    model = build(len(src), len(tgt), len(MANNERS))
    model.load_state_dict(state)
    model.eval()
    return model, src, tgt


def test_onnx_argmax_matches_torch(exported: Path):
    model, src, _ = _torch_model()
    session = ORT.InferenceSession(str(exported),
                                   providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["chars"]
    assert [o.name for o in session.get_outputs()] == ["phones"]

    width = max(len(w) for w in WORDS)
    ids = np.full((len(WORDS), width), src.pad_id, dtype=np.int64)
    for row, word in enumerate(WORDS):
        encoded = src.encode(word)
        ids[row, : len(encoded)] = encoded

    logits = session.run(["phones"], {"chars": ids})[0]
    with torch.no_grad():
        reference = model(torch.tensor(ids))[0].numpy()

    assert logits.shape == reference.shape
    # Float math in two runtimes: allow noise, require the same decisions.
    assert np.abs(logits - reference).max() < 1e-3
    for row, word in enumerate(WORDS):
        length = len(word)
        assert logits[row, :length].argmax() == reference[row, :length].argmax(), \
            f"argmax differs for {word!r}"


def test_onnx_export_is_deterministic(exported: Path, tmp_path: Path):
    again = export_onnx(tmp_path / "again.onnx")
    assert again.read_bytes() == exported.read_bytes()
