"""Vocabularies and dataset construction."""

from __future__ import annotations

import torch

from pl_g2p.data import split_words as pl_split_words
from pl_g2p.lexicon import Entry, Lexicon
from tiny_g2p.data import (
    MANNERS,
    CharDataset,
    collate,
    make_dataset,
    references_by_word,
    resolve_lexicon,
)
from tiny_g2p.vocab import BLANK, PAD, UNK, Vocab


def test_char_vocab_specials_and_round_trip():
    vocab = Vocab.build_chars(["kot", "ąka"])
    assert vocab.itos[:2] == [PAD, UNK]
    assert vocab.pad_id == 0
    assert vocab.decode(vocab.encode("kot")) == ["k", "o", "t"]
    assert vocab.encode("Q") == [vocab.unk_id]


def test_phone_vocab_has_blank_and_stable_order():
    vocab = Vocab.build_phones([["k", "ɔ", None]])
    assert vocab.itos[:3] == [PAD, UNK, BLANK]
    assert Vocab.build_phones([["ɔ", "k", None]]).itos == vocab.itos


def test_taxonomy_changes_when_the_inventory_changes():
    a = Vocab.build_phones([["k", "ɔ"]])
    b = Vocab.build_phones([["k", "ɔ", "ʔ"]])
    assert a.taxonomy() != b.taxonomy()
    assert a.taxonomy() == Vocab.build_phones([["ɔ", "k"]]).taxonomy()


def test_vocab_json_round_trip(tmp_path):
    vocab = Vocab.build_chars(["kot"])
    path = tmp_path / "vocab.json"
    vocab.to_json(path)
    assert Vocab.from_json(path).itos == vocab.itos


def make_lexicon() -> Lexicon:
    return Lexicon([
        Entry("kot", ("k", "ɔ", "t̪")),
        Entry("kąt", ("k", "ɔ", "n̪", "t̪")),
        Entry("nie", ("ɲ", "ɛ")),
        Entry("abc", ("a", "b", "ɛ", "t̪s̪", "ɛ")),  # unalignable acronym
    ])


def test_make_dataset_labels_train_and_excludes_unalignable():
    data = make_dataset(make_lexicon(), seed=1, ratios=(0.5, 0.25, 0.25))
    by_word = {e.word: e for e in data.train}
    if "kot" in by_word:
        assert tuple(by_word["kot"].labels) == ("k", "ɔ", "t̪")
    if "kąt" in by_word:
        assert tuple(by_word["kąt"].labels) == ("k", "ɔ̃", "t̪")
    # The acronym is either excluded from train or in val/test -- never
    # labelled in train.
    assert "abc" not in by_word


def test_make_dataset_partitions_match_pl_g2p():
    lexicon = make_lexicon()
    words = [e.word for e in lexicon.entries]
    expected = pl_split_words(words, seed=123, ratios=(0.5, 0.25, 0.25))
    data = make_dataset(lexicon, seed=123, ratios=(0.5, 0.25, 0.25))
    got = {
        "train": sorted(e.word for e in data.train),
        "val": sorted(r.word for r in data.val),
        "test": sorted(r.word for r in data.test),
    }
    # Train may drop unalignable rows; val/test must match exactly.
    assert got["val"] == sorted(expected["val"])
    assert got["test"] == sorted(expected["test"])
    assert set(got["train"]) <= set(expected["train"])


def test_nasal_and_rare_examples_are_upweighted():
    # Enough rows that ordinary phones clear the rarity threshold; the ç row
    # stays rare and the nasal row stays non-rare, isolating each rule. (ç via
    # "hi", as in "historia" -- cheap to align, unlike the glottal-stop case
    # the cost gate would drop.)
    lexicon = Lexicon(
        [Entry("kot", ("k", "ɔ", "t̪"))] * 600
        + [Entry("kąt", ("k", "ɔ", "n̪", "t̪"))] * 600
        + [Entry("hi", ("ç", "i"))]
    )
    data = make_dataset(lexicon, seed=1, ratios=(1.0, 0.0, 0.0))
    weights = {e.word: e.weight for e in data.train}
    assert weights["kot"] == 1.0
    assert weights["kąt"] == 2.0
    assert weights["hi"] == 3.0


def test_boundary_weights_mark_label_changes():
    data = make_dataset(make_lexicon(), seed=1, ratios=(1.0, 0.0, 0.0))
    by_word = {e.word: e for e in data.train}
    assert tuple(by_word["nie"].boundary) == (1.0, 2.0, 2.0)


def test_manner_table_covers_blank_and_vowels():
    data = make_dataset(make_lexicon(), seed=1, ratios=(1.0, 0.0, 0.0))
    assert data.manner_of[BLANK] == MANNERS.index("blank")
    assert data.manner_of["ɔ"] == MANNERS.index("vowel")


def test_references_by_word_keeps_polyphony():
    rows = make_dataset(
        Lexicon([Entry("dania", ("d̪", "a", "ɲ", "a")),
                 Entry("dania", ("d̪", "a", "ɲ", "j", "a"))]),
        seed=1, ratios=(0.0, 0.0, 1.0)).test
    refs = references_by_word(rows)
    assert len(refs["dania"]) == 2


def test_chardataset_and_collate_shapes():
    data = make_dataset(make_lexicon(), seed=1, ratios=(1.0, 0.0, 0.0))
    ds = CharDataset(data.train, data.src_vocab, data.tgt_vocab,
                     data.manner_of)
    assert len(ds) == len(data.train)
    batch = collate([ds[0], ds[1]])
    width = max(len(data.train[0].word), len(data.train[1].word))
    assert batch["chars"].shape == (2, width)
    assert batch["labels"].shape == (2, width)
    assert batch["manner"].shape == (2, width)
    assert batch["weight"].shape == (2,)
    assert batch["chars"].dtype == torch.long


def test_resolve_lexicon_finds_the_sibling_copy():
    path = resolve_lexicon(auto_fetch=False)
    assert path.is_file() and path.stat().st_size > 1_000_000
