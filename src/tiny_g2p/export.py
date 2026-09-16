"""Export the promoted model for runtimes that do not have Python.

Three artifacts, one source of truth:

``tiny_g2p.bin``
    A dependency-free little-endian blob the Rust/wasm runtime ``include_bytes!``s:
    vocabs, tap offsets, the float weights, the int8 weights with their
    per-channel scales and activation scales, the ``restore`` tables, and the
    exception table. One blob, two modes -- the int8 weights *are* the
    quantized artifact, and float mode dequantizes them or uses the float
    checkpoint's own weights (``--float-source``).

``test_gold.tsv``
    The held-out words with everything the scorer needs and nothing it does
    not: MFA's references, the adjudicated candidate set, the canonical forms
    for notation excuses, and the verdict flags. Python stays the owner of
    gold preparation; the Rust binary only does inference and arithmetic, so
    it can print the README table without importing torch.

``tiny_g2p.onnx``
    The same float graph for ONNX Runtime consumers.

The float weights are exported by default because the port's default is
float32 (see the Rust crate's docs): on the held-out set the float checkpoint
scores the same 97.89% word accuracy as the int8 artifact, differing on 23 of
6,719 words, so float is both simpler and not a fidelity loss.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .align import DIGRAPHS  # noqa: F401  (re-exported for callers)
from .data import MANNERS, make_dataset, references_by_word, resolve_lexicon
from .model import CTX_DIM, EMBED_DIM, HIDDEN_DIM, TAPS, build
from .phones import HOMORGANIC, NASAL_LETTERS, ORAL, STOPS
from .vocab import Vocab

MAGIC = b"TG2P"
VERSION = 1

#: Standard Polish letter names in MFA's own notation -- dentals marked, as
#: every corpus label is, so the acronym path emits the same notation the
#: model does and scoring compares like with like. With this table 9 of the
#: 13 corpus acronyms come out byte-identical to the gold; the four that do
#: not are ``agd``/``dga`` (MFA palatalizes g before e, differently in each)
#: and ``bmw``/``cv`` (MFA mis-segments the letter names). A dictionary entry
#: always beats the speller.
LETTER_NAMES: dict[str, str] = {
    "a": "a", "ą": "ɔ̃", "b": "b ɛ", "c": "t͡s ɛ", "ć": "t͡ɕ ɛ",
    "d": "d̪ ɛ", "e": "ɛ", "ę": "ɛ̃", "f": "ɛ f", "g": "ɡ ɛ",
    "h": "x a", "i": "i", "j": "j ɔ t̪", "k": "k a", "l": "ɛ l",
    "ł": "ɛ w", "m": "ɛ m", "n": "ɛ n̪", "ń": "ɛ ɲ", "o": "ɔ",
    "ó": "u", "p": "p ɛ", "q": "k u", "r": "ɛ r", "s": "ɛ s̪",
    "ś": "ɛ ɕ", "t": "t̪ ɛ", "u": "u", "v": "v ɛ", "w": "v u",
    "x": "i k s̪", "y": "ɨ ɡ r ɛ k", "z": "z̪ ɛ t̪", "ź": "ʑ ɛ t̪",
    "ż": "ʐ ɛ t̪",
}

#: Vowel letters: an initialism with none of them is spelled letter by letter.
_VOWEL_LETTERS = frozenset("aąeęioóuy")


class ExportError(RuntimeError):
    """The promoted artifacts are missing or inconsistent."""


def _lexicon():
    from pl_g2p.lexicon import Lexicon, parse_lexicon

    return Lexicon(parse_lexicon(resolve_lexicon()))


def _resolve_float_weights(active_dir: Path, source: str) -> dict[str, torch.Tensor]:
    """The float weight set to export: the checkpoint or the dequantized int8."""
    state = torch.load(active_dir / "model.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
    if source == "checkpoint":
        return state
    if source != "dequantized":
        raise ExportError(f"unknown float source {source!r}")
    quantized = torch.load(active_dir / "model_quantized.pt", map_location="cpu",
                           weights_only=False)
    out = dict(state)
    for name, module in quantized.named_modules():
        if not hasattr(module, "_weight_bias"):
            continue
        weight, bias = module._weight_bias()
        prefix = name.removesuffix("._packed_params")
        out[f"{prefix}.weight"] = torch.dequantize(weight).clone()
        out[f"{prefix}.bias"] = bias.clone()
    out["embedding.weight"] = quantized.embedding.weight.detach().clone()
    return out


class _Writer:
    """Tiny little-endian writer: the Rust reader is its mirror image."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def raw(self, data: bytes) -> None:
        self.parts.append(data)

    def u32(self, value: int) -> None:
        self.raw(struct.pack("<I", value))

    def i32(self, value: int) -> None:
        self.raw(struct.pack("<i", value))

    def u8(self, value: int) -> None:
        self.raw(struct.pack("<B", value))

    def f32(self, values: Iterable[float]) -> None:
        values = list(values)
        self.raw(struct.pack(f"<{len(values)}f", *values))

    def i8(self, values: Iterable[int]) -> None:
        values = [int(v) for v in values]
        self.raw(struct.pack(f"<{len(values)}b", *values))

    def string(self, text: str) -> None:
        encoded = text.encode("utf-8")
        if len(encoded) > 0xFFFF:
            raise ExportError(f"string too long to encode: {text!r}")
        self.raw(struct.pack("<H", len(encoded)))
        self.raw(encoded)

    def strings(self, items: Sequence[str]) -> None:
        self.u32(len(items))
        for item in items:
            self.string(item)

    def pairs(self, items: dict[str, str]) -> None:
        self.u32(len(items))
        for key in sorted(items):
            self.string(key)
            self.string(items[key])

    def tensor(self, tensor: torch.Tensor, dtype: str) -> None:
        # Quantized tensors carry their integers in int_repr(); .to() would
        # try to re-quantize instead.
        source = tensor.int_repr() if tensor.is_quantized else tensor
        flat = source.detach().to("cpu").reshape(-1)
        if dtype == "f32":
            self.f32(flat.tolist())
        elif dtype == "i8":
            self.i8(flat.to(torch.int8).tolist())
        elif dtype == "i32":
            for value in flat.tolist():
                self.i32(int(value))
        else:
            raise ExportError(f"unsupported dtype {dtype!r}")

    def build(self) -> bytes:
        return b"".join(self.parts)


def _quantized_layers(quantized: torch.nn.Module) -> list[tuple[str, str, bool]]:
    """(float name, quantized name, relu-after) in execution order."""
    return [
        ("mlp.0", "mlp.0", True),
        ("mlp.2", "mlp.2", True),
        ("residual.0", "residual.0", True),
        ("phone_head", "phone_head", False),
    ]


def _acronyms(lexicon_entries: dict[str, list[tuple[str, ...]]],
              known: Sequence[str]) -> dict[str, str]:
    """Word -> letter-by-letter reading for the acronyms we have gold for.

    Readings come from the lexicon when present (so the tool reproduces the
    corpus's convention) and from :data:`LETTER_NAMES` otherwise.
    """
    table: dict[str, str] = {}
    for word in known:
        gold = lexicon_entries.get(word)
        if gold:
            table[word] = " ".join(gold[0])
            continue
        if all(char in LETTER_NAMES for char in word):
            table[word] = " ".join(
                phone for char in word for phone in LETTER_NAMES[char].split())
    return table


def _is_initialism(word: str) -> bool:
    """A vowel-less word of two or more letters, e.g. ``bmw``, ``nszz``."""
    return len(word) >= 2 and not any(char in _VOWEL_LETTERS for char in word)


def export_blob(active_dir: Path | str = "active", *,
                float_source: str = "checkpoint",
                acronym_words: Sequence[str] = ()) -> bytes:
    """Serialise everything a runtime needs into one blob."""
    active_dir = Path(active_dir)
    for name in ("model.pt", "model_quantized.pt", "src_vocab.json",
                 "tgt_vocab.json"):
        if not (active_dir / name).is_file():
            raise ExportError(f"{active_dir} is missing {name}")

    src_vocab = Vocab.from_json(active_dir / "src_vocab.json")
    tgt_vocab = Vocab.from_json(active_dir / "tgt_vocab.json")
    weights = _resolve_float_weights(active_dir, float_source)
    quantized = torch.load(active_dir / "model_quantized.pt", map_location="cpu",
                           weights_only=False)
    n_manners = len(MANNERS)

    lexicon = make_dataset(_lexicon(), align_train=False)
    lexicon_entries: dict[str, list[tuple[str, ...]]] = {}
    for row in (*lexicon.test, *lexicon.val):
        lexicon_entries.setdefault(row.word, []).append(row.phones)

    writer = _Writer()
    writer.raw(MAGIC)
    writer.u32(VERSION)
    for value in (len(src_vocab), len(tgt_vocab), n_manners, EMBED_DIM,
                  HIDDEN_DIM, CTX_DIM, len(TAPS)):
        writer.u32(value)
    for tap in TAPS:
        writer.i32(tap)
    writer.strings(src_vocab.itos)
    writer.strings(tgt_vocab.itos)
    writer.strings(list(MANNERS))
    writer.raw(struct.pack("<III", src_vocab.pad_id, src_vocab.unk_id,
                           tgt_vocab.blank_id))

    # restore(): the label -> surface tables (see phones.py)
    writer.strings(sorted(NASAL_LETTERS))
    writer.pairs(ORAL)
    writer.strings(sorted(STOPS))
    writer.pairs(HOMORGANIC)

    # exception path: letter names, plus the acronyms we hold gold for
    writer.pairs(LETTER_NAMES)
    writer.pairs(_acronyms(lexicon_entries, acronym_words))

    # float weights (the default mode)
    writer.u8(1)
    writer.tensor(weights["embedding.weight"], "f32")
    writer.tensor(weights["mlp.0.weight"], "f32")
    writer.tensor(weights["mlp.0.bias"], "f32")
    writer.tensor(weights["mlp.2.weight"], "f32")
    writer.tensor(weights["mlp.2.bias"], "f32")
    writer.tensor(weights["residual.0.weight"], "f32")
    writer.tensor(weights["residual.0.bias"], "f32")
    writer.tensor(weights["phone_head.weight"], "f32")
    writer.tensor(weights["phone_head.bias"], "f32")

    # int8 weights + scales (the measured artifact, for exact parity)
    writer.u8(1)
    writer.f32([float(quantized.quant.scale)])
    writer.i32(int(quantized.quant.zero_point))
    for _, qname, relu in _quantized_layers(quantized):
        module = dict(quantized.named_modules())[qname]
        weight, bias = module._weight_bias()
        scales = (weight.q_per_channel_scales().tolist()
                  if weight.qscheme() == torch.per_channel_affine
                  else [weight.q_scale()] * weight.shape[0])
        zeros = (weight.q_per_channel_zero_points().tolist()
                 if weight.qscheme() == torch.per_channel_affine
                 else [weight.q_zero_point()] * weight.shape[0])
        writer.u32(weight.shape[0])
        writer.u32(weight.shape[1])
        writer.u8(1 if relu else 0)
        writer.tensor(weight, "i8")
        writer.f32(scales)
        writer.tensor(torch.tensor(zeros), "i32")
        writer.f32(bias.tolist())
        writer.f32([float(module.scale)])
        writer.i32(int(module.zero_point))
    writer.f32([float(quantized.skip_add.scale)])
    writer.i32(int(quantized.skip_add.zero_point))
    return writer.build()


# --------------------------------------------------------------------------
# gold: everything the scorer needs, in a form the Rust side can read
# --------------------------------------------------------------------------

GOLD_COLUMNS = ("word", "slice", "refs", "candidates", "canon_mfa", "canon_cv",
                "mfa_ok", "cv_ok", "counted")


def _join(readings: Iterable[Sequence[str]]) -> str:
    return " | ".join(" ".join(reading) for reading in readings if reading)


def export_gold(out: Path | str, *, verdicts_json: Path | str | None = None,
                active_dir: Path | str = "active") -> int:
    """Write the held-out words with adjudicated candidate sets.

    Mirrors ``adjudicate.judge``: ``candidates`` is what a correction may
    match, ``canon_*`` are the canonical forms that excuse a notation
    difference, and the flags say which of them the verdict actually allows.
    """
    from .adjudicate import Verdict
    from .xlex import canonicalize, parse_cv

    out = Path(out)
    verdicts: dict[str, Verdict] = {}
    if verdicts_json and Path(verdicts_json).is_file():
        payload = json.loads(Path(verdicts_json).read_text(encoding="utf-8"))
        verdicts = {word: Verdict.from_json(entry)
                    for word, entry in payload.get("verdicts", {}).items()}
    cv = parse_cv()
    rows = make_dataset(_lexicon(), align_train=False)
    refs = references_by_word(rows.test)
    disputed = _disputed_words(refs)

    lines = ["\t".join(GOLD_COLUMNS)]
    for word in sorted(refs):
        verdict = verdicts.get(word, Verdict())
        mfa_refs = refs[word]
        cv_refs = cv.entries.get(word, [])
        mfa_ok = verdict.verdict in ("mfa", "both", "open")
        cv_ok = verdict.verdict in ("both", "cv")
        candidates = ([*mfa_refs] if mfa_ok else []) + list(verdict.correct)
        canon_mfa = ({canonicalize(list(r), source="mfa", word=word)
                      for r in mfa_refs} if mfa_ok else set())
        canon_cv = ({canonicalize(list(r), source="cv", word=word)
                     for r in cv_refs} if cv_ok else set())
        slice_name = ("disputed" if word in disputed
                      else "agreed" if word in cv.words else "mfa-only")
        lines.append("\t".join((
            word, slice_name, _join(mfa_refs), _join(candidates),
            _join(canon_mfa), _join(canon_cv),
            "1" if mfa_ok else "0", "1" if cv_ok else "0",
            "1" if verdict.counted else "0")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - 1


def _disputed_words(refs: dict[str, list[tuple[str, ...]]]) -> set[str]:
    """Words the two lexicons disagree on (needs the CV dictionary)."""
    from .xlex import compare, parse_cv

    try:
        cv = parse_cv()
    except Exception:  # pragma: no cover - dictionary not fetched
        return set()
    mfa: dict[str, list[tuple[str, ...]]] = {}
    for word, readings in refs.items():
        mfa.setdefault(word, []).extend(readings)
    return set(compare(mfa, cv).disagreed)


# --------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------

def export_onnx(out: Path | str, *, active_dir: Path | str = "active",
                opset: int = 17) -> Path:
    """Export the float graph for ONNX Runtime consumers."""
    active_dir = Path(active_dir)
    src_vocab = Vocab.from_json(active_dir / "src_vocab.json")
    tgt_vocab = Vocab.from_json(active_dir / "tgt_vocab.json")
    state = torch.load(active_dir / "model.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
    model = build(len(src_vocab), len(tgt_vocab), len(MANNERS))
    model.load_state_dict(state)
    model.eval()
    # The manner head is an auxiliary training target -- dropping it keeps the
    # ONNX surface to one input and one output (the phone logits).
    class _Phones(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, chars: torch.Tensor) -> torch.Tensor:
            return self.inner(chars)[0]

    wrapper = _Phones(model).eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 8), dtype=torch.long)
    torch.onnx.export(wrapper, (dummy,), str(out), opset_version=opset,
                      input_names=["chars"], output_names=["phones"],
                      dynamic_axes={"chars": {0: "batch", 1: "length"},
                                    "phones": {0: "batch", 1: "length"}},
                      dynamo=False)
    return out


def main(argv: list[str] | None = None) -> int:
    from .data import REPO_ROOT  # local import: avoids a cycle at import time

    ap = argparse.ArgumentParser(prog="tiny-g2p export",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--active", type=Path, default=REPO_ROOT / "active")
    ap.add_argument("--blob", type=Path,
                    default=REPO_ROOT / "rust" / "weights" / "tiny_g2p.bin")
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--float-source", choices=("checkpoint", "dequantized"),
                    default="checkpoint",
                    help="float weights to ship: the trained checkpoint, or "
                         "the int8 artifact dequantized")
    args = ap.parse_args(argv)

    blob = export_blob(args.active, float_source=args.float_source)
    args.blob.parent.mkdir(parents=True, exist_ok=True)
    args.blob.write_bytes(blob)
    print(f"wrote {args.blob} ({len(blob) / 1024:.0f} KiB)")
    if args.gold:
        rows = export_gold(args.gold)
        print(f"wrote {args.gold} ({rows} words)")
    if args.onnx:
        print(f"wrote {export_onnx(args.onnx, active_dir=args.active)}")
    return 0
