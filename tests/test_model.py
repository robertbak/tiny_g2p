"""Window model shapes, size budget, and the QAT round-trip."""

from __future__ import annotations

import pytest
import torch

from tiny_g2p.model import (
    TAP_RADIUS,
    TAPS,
    build,
    convert_to_quantized,
    count_params,
    prepare_for_qat,
)

FGBEMM = "fbgemm" in torch.backends.quantized.supported_engines


def test_forward_shapes():
    model = build(src_size=40, tgt_size=53, n_manners=10)
    chars = torch.randint(0, 40, (4, 9))
    phones, manner = model(chars)
    assert phones.shape == (4, 9, 53)
    assert manner.shape == (4, 9, 10)


def test_model_is_tiny():
    # Real-ish vocab sizes must stay far below the transformer's 2.6M --
    # and below gpu-lexer's own 41K, since words need less than sources.
    model = build(src_size=40, tgt_size=53, n_manners=10)
    assert count_params(model) < 35_000


def test_taps_cover_the_longest_word():
    assert sorted(TAPS) == [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
    assert TAP_RADIUS * 2 + 1 >= 32  # longest lexicon word


def test_pad_embedding_is_zero():
    model = build(src_size=40, tgt_size=53, n_manners=10, pad_id=0)
    assert model.embedding.weight[0].abs().sum().item() == 0.0


def test_single_character_word_runs():
    model = build(src_size=40, tgt_size=53, n_manners=10)
    phones, _ = model(torch.tensor([[5]]))
    assert phones.shape == (1, 1, 53)


@pytest.mark.skipif(not FGBEMM, reason="needs the fbgemm quant backend")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_qat_prepare_convert_round_trip():
    model = build(src_size=40, tgt_size=53, n_manners=10)
    prepare_for_qat(model)
    chars = torch.randint(0, 40, (2, 7))
    model(chars)  # training-mode forward populates observers
    quantised = convert_to_quantized(model)
    phones, manner = quantised(chars)
    assert phones.shape == (2, 7, 53)
    assert manner.shape == (2, 7, 10)


@pytest.mark.skipif(not FGBEMM, reason="needs the fbgemm quant backend")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_qat_convert_leaves_a_cuda_training_model_alone():
    # Regression: convert used to run on the live (CUDA) model and crash.
    model = build(src_size=40, tgt_size=53, n_manners=10)
    prepare_for_qat(model)
    model.cuda().train()
    model(torch.randint(0, 40, (2, 7), device="cuda"))
    quantised = convert_to_quantized(model)
    assert next(model.parameters()).device.type == "cuda"
    assert model.training
    phones, _ = quantised(torch.randint(0, 40, (2, 7)))
    assert phones.shape == (2, 7, 53)
