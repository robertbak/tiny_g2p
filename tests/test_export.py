"""The exported blob, the gold TSV and the exception table.

These tests are the spec in executable form: ``Reader`` below is the reference
reader, and the Rust crate's reader must agree with it field for field.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from pl_g2p.lexicon import Lexicon, parse_lexicon

from tiny_g2p.data import resolve_lexicon
from tiny_g2p.export import (
    LETTER_NAMES,
    MAGIC,
    VERSION,
    _is_initialism,
    _join,
    export_blob,
    export_gold,
)


class Reader:
    """Reference reader: the blob layout, read back."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def raw(self, n: int) -> bytes:
        out = self.data[self.pos:self.pos + n]
        assert len(out) == n, "blob ended early"
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.raw(4))[0]

    def u8(self) -> int:
        return struct.unpack("<B", self.raw(1))[0]

    def f32(self, n: int) -> list[float]:
        return list(struct.unpack(f"<{n}f", self.raw(4 * n)))

    def i8(self, n: int) -> list[int]:
        return list(struct.unpack(f"<{n}b", self.raw(n)))

    def i32s(self, n: int) -> list[int]:
        return list(struct.unpack(f"<{n}i", self.raw(4 * n)))

    def string(self) -> str:
        return self.raw(struct.unpack("<H", self.raw(2))[0]).decode("utf-8")

    def strings(self) -> list[str]:
        return [self.string() for _ in range(self.u32())]

    def pairs(self) -> dict[str, str]:
        return {self.string(): self.string() for _ in range(self.u32())}

    # -- section walkers, so each test can position itself ------------------

    def prefix(self) -> tuple:
        """Header, dims, taps, vocabularies, special-token ids."""
        assert self.raw(4) == MAGIC
        assert self.u32() == VERSION
        dims = tuple(self.u32() for _ in range(7))
        taps = [self.i32() for _ in range(dims[6])]
        src = self.strings()
        tgt = self.strings()
        manners = self.strings()
        specials = (self.u32(), self.u32(), self.u32())
        return dims, taps, src, tgt, manners, specials

    def restore_and_exceptions(self) -> tuple:
        nasal = self.strings()
        oral = self.pairs()
        stops = self.strings()
        homorganic = self.pairs()
        letters = self.pairs()
        acronyms = self.pairs()
        return nasal, oral, stops, homorganic, letters, acronyms

    def to_float_section(self) -> None:
        self.prefix()
        self.restore_and_exceptions()

    def float_weights(self) -> None:
        assert self.u8() == 1
        for count in FLOAT_COUNTS:
            self.f32(count)


#: (name, size) of the float tensors, in blob order.
FLOAT_COUNTS = (37 * 32, 64 * 352, 64, 32 * 64, 32, 32 * 32, 32, 53 * 32, 53)


@pytest.fixture(scope="module")
def blob_bytes() -> bytes:
    return export_blob(acronym_words=["agd", "bmw", "cv"])


def test_blob_header_dims_and_vocabs(blob_bytes: bytes):
    reader = Reader(blob_bytes)
    dims, taps, src, tgt, manners, specials = reader.prefix()
    n_src, n_tgt, n_manners, embed, hidden, ctx, n_taps = dims
    assert (n_src, n_tgt) == (37, 53)
    assert (embed, hidden, ctx, n_taps) == (32, 64, 32, 11)
    assert taps == [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
    assert src[:3] == ["<pad>", "<unk>", "a"]
    assert tgt[:4] == ["<pad>", "<unk>", "<blank>", "a"]
    assert len(manners) == n_manners == 10
    assert specials == (0, 1, 2)          # pad, unk, blank


def test_blob_carries_the_restore_tables_and_exceptions(blob_bytes: bytes):
    reader = Reader(blob_bytes)
    reader.prefix()
    nasal, oral, stops, homorganic, letters, acronyms = \
        reader.restore_and_exceptions()
    assert nasal == ["ą", "ę"]
    assert oral == {"ɔ̃": "ɔ", "ɛ̃": "ɛ"}
    assert "t̪" in stops and "tʂ" in stops
    assert homorganic["t̪"] == "n̪" and homorganic["k"] == "ŋ"
    assert letters["w"] == "v u"
    assert letters["t"] == "t̪ ɛ"          # corpus notation: dentals marked
    # Acronyms come from the lexicon when it has them, so they match the gold.
    assert acronyms["bmw"] == "b ɛ m ɛ v u"
    assert acronyms["agd"].split()[0] == "a" and len(acronyms["agd"].split()) == 5


def test_float_section_round_trips_the_checkpoint(blob_bytes: bytes):
    reader = Reader(blob_bytes)
    reader.to_float_section()
    assert reader.u8() == 1
    state = torch.load("active/model.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
    names = ("embedding.weight", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight",
             "mlp.2.bias", "residual.0.weight", "residual.0.bias",
             "phone_head.weight", "phone_head.bias")
    for name, count in zip(names, FLOAT_COUNTS):
        got = torch.tensor(reader.f32(count))
        want = state[name].detach().reshape(-1)
        assert torch.equal(got, want), name


def test_int8_section_is_per_channel(blob_bytes: bytes):
    reader = Reader(blob_bytes)
    reader.to_float_section()
    reader.float_weights()
    assert reader.u8() == 1
    in_scale, in_zp = reader.f32(1)[0], reader.i32()
    assert 0 < in_scale < 1 and in_zp == 59
    for out_dim, in_dim, relu in ((64, 352, 1), (32, 64, 1), (32, 32, 1),
                                  (53, 32, 0)):
        assert (reader.u32(), reader.u32(), reader.u8()) == (out_dim, in_dim, relu)
        reader.i8(out_dim * in_dim)
        assert len(reader.f32(out_dim)) == out_dim          # per-channel scales
        assert set(reader.i32s(out_dim)) == {0}             # zero weight offsets
        assert len(reader.f32(out_dim)) == out_dim          # float bias
        out_scale, out_zp = reader.f32(1)[0], reader.i32()
        assert out_scale > 0 and 0 <= out_zp <= 255
    assert reader.f32(1)[0] > 0 and reader.i32() == 0       # the residual add
    assert reader.pos == len(reader.data), "blob has trailing bytes"


def test_letter_names_match_the_lexicon_except_the_known_noise():
    """The tool spells letters; MFA's own acronym rows are the convention.

    Nine of the corpus's thirteen acronyms come out byte-identical to the
    gold. The four that do not are MFA's own noise: ``g`` is palatalized
    before e differently in ``agd`` and ``dga`` ([ɡɛ] both times), and
    ``bmw``/``cv`` are mis-segmented -- ``cv`` is read s̪i vʲ i, which is not
    [t͡sɛ vɛ].
    """
    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    gold = {entry.word: list(entry.phonemes) for entry in lexicon.entries}
    words = ("agd", "bmw", "cv", "dga", "mps", "nszz", "ntv", "pzpr", "rpo",
             "rtv", "tpn", "wku", "wtw")
    mismatches = {}
    for word in words:
        spelled = [p for ch in word for p in LETTER_NAMES[ch].split()]
        if spelled != gold[word]:
            mismatches[word] = (spelled, gold[word])
    assert set(mismatches) == {"agd", "bmw", "cv", "dga"}, mismatches
    assert gold["agd"][1] == "ɟ"                  # MFA palatalizes before ɛ
    assert gold["dga"][2:5] == ["ɡ", "j", "ɛ"]    # and does it differently


def test_initialism_detector():
    assert _is_initialism("bmw") and _is_initialism("nszz")
    assert not _is_initialism("agd")   # has a vowel; needs the table
    assert not _is_initialism("kot")
    assert not _is_initialism("a")


def test_gold_tsv_schema_and_verdict_flags(tmp_path: Path):
    out = tmp_path / "gold.tsv"
    rows = export_gold(out, verdicts_json=Path("data/adjudication_verdicts.json"))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert rows == len(lines) - 1 == 6719
    assert lines[0].split("\t") == [
        "word", "slice", "refs", "candidates", "canon_mfa", "canon_cv",
        "mfa_ok", "cv_ok", "counted"]
    by_word = {line.split("\t")[0]: line.split("\t") for line in lines[1:]}
    assert by_word["agd"][-1] == "0"                       # excluded (convention)
    assert by_word["odessie"][6] == "0"                    # a `cv` verdict...
    assert by_word["odessie"][7] == "1"                    # ...accepts CV instead
    assert by_word["wstydzę"][5] != ""                     # canonical CV forms
    assert all(row[2] for row in by_word.values())         # every ref non-empty
    assert by_word["abakus"][6] == "1" and by_word["abakus"][-1] == "1"


def test_join_helper_skips_empty_readings():
    assert _join([("a", "b"), (), ("c",)]) == "a b | c"
    assert _join([]) == ""
