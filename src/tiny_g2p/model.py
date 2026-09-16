"""The tiny window model: per-character phone-or-blank classification.

The gpu-lexer mapping, in code:

* sparse part features -> 32 channels   ==>  char embedding, 32 channels
* shared five-part local window         ==>  dense taps at offsets -2..+2
* bidirectional scans + tree mixing     ==>  dilated taps at +-4, +-8, +-16
  (receptive field 33 covers the longest lexicon word, 32 chars)
* small classifier per part             ==>  phone head + auxiliary manner head

No recurrence and no attention: every position is classified in parallel from
its fixed taps, so the whole model is Embedding + Linear + ReLU -- which is
exactly what static quantization supports. QAT helpers live here too.
"""

from __future__ import annotations

import torch
from torch import nn

EMBED_DIM = 32
HIDDEN_DIM = 64
CTX_DIM = 32

#: Tap offsets: a dense five-window plus dilated long-range taps.
TAPS = (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
TAP_RADIUS = max(abs(t) for t in TAPS)


class WindowG2P(nn.Module):
    """Embed -> gather taps -> MLP -> per-character phone/manner heads."""

    def __init__(self, src_size: int, tgt_size: int, n_manners: int,
                 *, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(src_size, EMBED_DIM, padding_idx=pad_id)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM * len(TAPS), HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, CTX_DIM),
            nn.ReLU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(CTX_DIM, CTX_DIM),
            nn.ReLU(),
        )
        # Quantized tensors have no plain `add`; FloatFunctional is swapped
        # in by convert and handles the rescaling.
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()
        self.phone_head = nn.Linear(CTX_DIM, tgt_size)
        self.manner_head = nn.Linear(CTX_DIM, n_manners)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, chars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, T] ids -> [B, T, E], padded so edge taps read PAD.
        vectors = self.embedding(chars)
        pad = torch.zeros(1, TAP_RADIUS, EMBED_DIM, device=vectors.device,
                          dtype=vectors.dtype)
        padded = torch.cat([pad.expand(vectors.size(0), -1, -1), vectors,
                            pad.expand(vectors.size(0), -1, -1)], dim=1)
        taps = torch.stack(
            [padded[:, TAP_RADIUS + t: TAP_RADIUS + t + chars.size(1), :]
             for t in TAPS], dim=-1)
        flat = taps.reshape(chars.size(0), chars.size(1), -1)
        flat = self.quant(flat)
        hidden = self.mlp(flat)
        hidden = self.skip_add.add(hidden, self.residual(hidden))
        phones = self.dequant(self.phone_head(hidden))
        manner = self.dequant(self.manner_head(hidden))
        return phones, manner


def count_params(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(src_size: int, tgt_size: int, n_manners: int,
          *, pad_id: int = 0) -> WindowG2P:
    """Construct the model (one place, so train/eval/convert agree)."""
    return WindowG2P(src_size, tgt_size, n_manners, pad_id=pad_id)


def prepare_for_qat(model: WindowG2P) -> WindowG2P:
    """Insert QAT observers (fbgemm backend); model must be training on CPU.

    The embedding stays float: quantized Embedding rejects convert here, and
    1.3K float params out of 28K are not worth fighting torch internals over.
    Everything downstream of the tap gather is int8.
    """
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    model.embedding.qconfig = None  # type: ignore[assignment]
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_quantized(model: WindowG2P) -> nn.Module:
    """Freeze observers into a static int8 model (eval mode, CPU).

    Operates on a CPU copy: conversion must run on CPU, and the live
    training model (possibly CUDA, possibly mid-schedule) is left alone.
    """
    import copy

    clone = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.convert(clone, inplace=False)
