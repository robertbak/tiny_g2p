# tiny_g2p

A tiny parallel per-character Polish G2P, following the experiments in
[vercel-labs/gpu-lexer](https://github.com/vercel-labs/gpu-lexer): one small
learned model that labels every unit of a sequence in parallel, trained under a
guarded regime (warm-start continuation, quantization-aware epochs, failure-bank
fine-tuning, fixed verification, promotion only on strict improvement).

The sibling [`../pl_g2p`](../pl_g2p) already does Polish G2P with a 2.6M-param
transformer at 98.50% word accuracy. This package asks the gpu-lexer question:
how far does a ~15K-param parallel classifier with local context get, and what
does the disciplined regime buy on top?

```
word ──▶ [embed 32] ──▶ [taps ±2,±4,±8,±16 + MLP] ──▶ phone-or-blank per char
kot           3×32              RF = 33 chars               k ɔ t
```

## The mapping, honestly

| gpu-lexer | tiny_g2p | deviation, if any |
|---|---|---|
| parts (word runs, symbols) | characters | G2P units are chars |
| sparse hand features → 32 ch | char embedding, 32 ch | learned, not hashed |
| five-part window | fixed taps at offsets -2..+2 | literal |
| bidirectional affine scans | dilated taps at ±4, ±8, ±16 (RF 33) | no recurrence: fixed taps cover any word, and all-Linear is QAT-friendly |
| butterfly tree over blocks | folded into the taps + MLP | word-scale needs no tree |
| 9-class head per part | phone-or-blank head per char | |
| Shiki teacher labels | monotonic DP aligner labels | per-char supervision without autoregression |
| boundary-weighted loss | position weights at phone-change points | |
| auxiliary lexical states | auxiliary manner head (from `phoneme_lab.features`) | |
| 32-epoch schedule, lr 0.002 | same | |
| QAT final 6 epochs (int6) | QAT final 6 epochs (int8, float embedding) | theirs is browser int6; ours is PyTorch int8; the 1.3K-param embedding stays float |
| fixed verification, never trained | val selects, test is final, neither trained | |
| failure bank + fine-tune | same (`tiny-g2p fine-tune`) | |
| guarded promotion | guards: polyphony + digraph probes, strict improvement | |

## Quick start

```bash
cd tiny_g2p
make setup                     # uv sync + lexicon (reuses pl_g2p's copy)
make probe                     # lexicon stats that decided the design
make train                     # 32 epochs, warm-starts active/, auto-promotes
make eval                      # verification + held-out test numbers
make test                      # 87 tests
```

```bash
uv run tiny-g2p predict "Wrocław Szczebrzeszyn morze może"
uv run tiny-g2p fine-tune słowo   # repair failures (never auto-promotes)
uv run tiny-g2p promote runs/<ts> # manual promotion under the guards
```

## Where the labels come from

The lexicon gives unaligned pairs, so a monotonic DP aligner (`align.py`)
teaches each character its phone: 1:1 substitutions cost articulatory
feature distance (`phoneme_lab.features`), digraphs and `dzi` emit jointly,
`i` skips near-free, and ą/ę absorb oral+nasal pairs before stops into a
merged nasal vowel. Coverage is 99.68%; the 437 unalignable rows are
acronyms spelled letter-by-letter (`abc` -> `a b ɛ t̪s̪ ɛ`), excluded from
training but kept in eval. A mean-cost gate drops 150 more forced-noise
rows (heavy loans like `utah`, `wow`).

Labels keep raw MFA phones (dentals, ʲ) except the nasal merge, which
`phones.restore` splits back for scoring. The split is deterministic given
the word and the following phone (raw MFA never puts Ṽ before a stop; the
nasal is homorganic) -- verified by a full-lexicon round-trip
(`experiments/align_coverage.py`) with 21 known residues (0.016%):
pre-cluster splits (`księżnik`), assimilation-masked followers (`pięćset`),
blocked splits (`zalękniona`), a compound boundary, and the polyphonic
single letter `ą`. Chasing those would be family-specific overfitting.

## Layout

| Path | Purpose |
|---|---|
| `src/tiny_g2p/phones.py` | nasal-merge restoration, homorganic table |
| `src/tiny_g2p/align.py` | DP grapheme->phone aligner (the teacher) |
| `src/tiny_g2p/data.py` | same splits as pl_g2p, vocabs, weak-group weights |
| `src/tiny_g2p/model.py` | window MLP + QAT helpers |
| `src/tiny_g2p/train.py` | training loop + fine-tuning with rollback |
| `src/tiny_g2p/promote.py` | guarded promotion into `active/` |
| `src/tiny_g2p/predict.py` | decoding, guard-group scoring |
| `src/tiny_g2p/failurebank.py` | local failure bank (leakage-guarded) |
| `src/tiny_g2p/xlex.py` | MFA<->CV canonicalization and agreement (the second opinion) |
| `src/tiny_g2p/adjudicate.py` | verdict semantics: measured vs gold-corrected scoring |
| `experiments/probe_lexicon.py` | length/inventory/nasal-context stats |
| `experiments/align_coverage.py` | aligner coverage + restore round-trip proof |
| `experiments/xlex_agreement.py` | agreement rate and the dispute groups |
| `experiments/xlex_rescore.py` | both models on the agreed/disputed/MFA-only slices |
| `experiments/build_adjudication.py` | the TSV: every contested or missed test word |
| `experiments/score_adjudication.py` | joins the verdicts; measured vs corrected |
| `data/adjudication_verdicts.json` | the curated verdicts (one line of reason each) |
| `src/tiny_g2p/export.py` | packages weights for the Python-free runtimes |
| `rust/` | the same model as a static binary, a `.wasm` and ONNX (see `rust/README.md`) |
| `active/` | promoted float + quantized models, vocabs, metadata |

## Design notes

- **Same test words as pl_g2p.** Splitting reuses `pl_g2p`'s function and
  seed (verified identical: 6,719 words), so numbers compare directly with
  the transformer's 98.50%.
- **Selection is by deployed accuracy.** Every candidate -- epoch zero and
  each epoch -- is scored quantized on the fixed verification split; the
  float numbers are diagnostic only.
- **Verification is never replayed.** The replay buffer draws train errors
  only, PTQ calibrates on train words, and fine-tuning refuses val/test
  words outright.
- **Guards pin phenomena, not just accuracy.** Nasal, digraph, palatal and
  rare groups must stay within cap of the baseline for promotion.
- **Checkpoints carry their taxonomy.** A run only warm-starts data with the
  same label inventory; `--polish` refuses otherwise.

## Limitations

- **Acronyms and loans go around the model, not through it.** One label
  per character cannot spell `agd` as five phones, and no context window
  recovers `blair`. Both classes are routed to an exception path instead
  (see `## Out-of-scope items`), which is also how they are scored -- they
  are excluded from the phonetic numbers rather than charged to the model.
- **No BERT distillation (yet).** Supervision is the lexicon via the
  aligner; distilling the phonetic BERT's soft targets is the obvious
  follow-up the architecture already supports (per-position KL).
- **Static group weights.** The weak-group multipliers are fixed (x2 nasal,
  x3 rare), not adapted from per-group errors during training.
- **No EMA candidate.** gpu-lexer also scores an EMA average; skipped here.
- **`torch.ao.quantization` is deprecated** in favour of torchao; it still
  works on this torch (2.14) and keeps the dependency list unchanged.
- **21 restore residues** (0.016%) as documented above.

## Results

Held-out test words -- the same 6,719 as `pl_g2p` -- decoded with the
promoted **quantized** model (28,991 params, int8 + float embedding):

| model | word accuracy | PER |
|---|---|---|
| **this model** (window MLP, int8) | **97.89%** (6577/6719) | **0.44%** (230/51879) |
| `pl_g2p` transformer (2.6M, float) | 98.50% (6618/6719) | 0.44% (230/51879) |

A 90x smaller parallel classifier lands 0.6 points behind the transformer
on words. The identical PER numerator (230 edits) is largely shared hard
words -- 80 of our 142 error words are transformer errors too -- rather
than a deep fact; the errors distribute differently (ours: more words,
fewer edits each).

### What it gets wrong

The error list is honest about the architecture's limits:

- **Acronyms** (`agd`, `bmw`): unrepresentable by construction -- one label
  per character cannot spell five phones from three letters.
- **English loans** (`authorising`, `heritage`, `beyer`, `blair`): genuinely
  ambiguous orthography, and the aligner gate kept most of their kind out
  of training.
- **Domestic oddities** (`brudz`, `conajmniej`, `franciszek`): the
  long-tail residue a bigger model smooths over.

But how much of the 142 error words is actually this model's fault? That is
what the next section measures -- and the answer moves the numbers.

The `rare` group is best read as a note, not a verdict. "Rare" means a phone
with fewer than 500 training occurrences, which here is `{dʐ, tʲ, ʔ}` (260,
272 and 26 rows) -- *not* the `ʔ/ç/j̃` an earlier draft claimed: `ç` and `j̃`
were folded into `x`/`ɲ` by the cross-lexicon work above. 23 verification
rows contain one and the model gets 15 right (65.2%). That number is mostly
about the sample: 9 of the 23 are already `digraph` rows (`dż`) and 15 are
already `palatal` rows (`tʲ`), so the only rows it adds are three borrowings
(`george`, `jiangа`, `energizerem`), while the native mappings do pass
(`dżbik`, `dżin`, `radża`, `odjeżdżam`, `odmóżdżyć`; 11 of 12 `tʲ` rows).
The phones are rare *because* they mark borrowings -- i.e. this group is
largely the out-of-scope class. It is kept as a recorded comparison number
(with `guards_n` for `{dʐ, tʲ, ʔ}`: 23 rows), not as something to act on.


### What the regime bought

- **Selection picked float epoch 21** (PTQ-quantized for the artifact);
  the six QAT epochs scored within noise of it. int8 costs this model
  essentially nothing -- the q/f gap never exceeded 0.3 points -- so QAT
  was insurance rather than a win.
- **Promotion fired once**, seeding `active/` (there was no baseline).
  The guards, caps and rollback paths are exercised by tests, not yet by
  a real contested run -- the next training run will be the first one.
- **Fine-tuning is untested against real failures** (unit-tested only);
  `tiny-g2p fine-tune` on the acronym/loan errors above is the natural
  next experiment, though acronyms will stay unfixable by construction.

## Running it without Python

The model is a window MLP -- embedding, eleven taps, three small matmuls -- so
it ports cleanly to anything. `rust/` carries the same weights as a **static
binary with no dependencies**, a **`.wasm`** module for the browser, and an
**ONNX** graph; all three are bit-exact against the Python model on every one
of the 6,725 golden words, and ONNX agrees with torch on every argmax.

```bash
cd rust && cargo build --release
target/release/tinyg2p predict "Szczebrzeszyn bmw"    # whitespace splits; one word per line
# ʂ tʂ ɛ b ʐ ɛ ʂ ɨ n̪
# b ɛ m ɛ v u
```

| artifact | size | notes |
|---|---|---|
| `tinyg2p` binary | 680 KiB (592 stripped) | static, no runtime, weights embedded |
| `.wasm` | 222 KiB (167 gzipped) | float + int8 + exception path |
| ONNX | 131 KiB | for ONNX Runtime consumers |

That binary also carries the **eval** path: `tinyg2p eval --gold
data/test_gold.tsv` reproduces the table above -- measured *and*
gold-corrected, per slice -- from the weights alone, in 0.5 s on one core,
with no torch anywhere. It takes words, not sentences: whitespace splits, so a
quoted run of text, stdin or a whole file all work, and nothing is refused for
being long -- the 33-character tap window is a training fact, not an input
limit. The acronyms and unresolvable borrowings go to an exception path (dictionary, acronym table, letter-name spelling) rather than
through the model, which is where the measured 62 PER edits of out-of-scope
items stop being charged to it. `rust/README.md` has the build commands, the
parity table and the regeneration steps; `make export rust wasm` runs them.

## Is the MFA gold right?

Both models are scored against one lexicon, so its errors are charged to
them. The independent second opinion is `polish_cv` v2.0.0 (Common Voice /
Vox Communis, 53K words, Epitran phone set, CC-0). 37,108 words are in both
dictionaries; after canonicalizing the notation differences -- dentals, tie
bars, ʲ, ń before a consonant (`ɲ` against `j̃`), ł and /v/ (`w` against
`v`), CV's two-phone `t ʂ` for `cz`, MFA's word-final denasalization --
**94.78% agree outright** (`xlex.py`, `experiments/xlex_agreement.py`).

The residue is not noise. It is devoicing (surface `f` against underlying
`v`), geminates, nasal splitting and nasal place, plus real lexicon bugs on
both sides: MFA drops the `r` of `drżenie` and reads `microsoft` as
`m a j k r ɔ s ɔ f t`; CV renders post-vocalic `u` as `v`/`f` (`biura` ->
`b i v r a`) and drops the `r` of `skarżyć`.

So every test word that a model missed, plus every contested one, is
adjudicated by hand. `experiments/build_adjudication.py` writes
`data/adjudication.tsv` (248 rows: the disputed test words, the error union,
the polyphonic and the unalignable ones), `data/adjudication_verdicts.json`
carries a verdict and a one-line reason for the 148 that are not a plain
`mfa`, and `experiments/score_adjudication.py` joins the two:

| model | measured | gold-corrected | genuine errors |
|---|---|---|---|
| tiny (int8) | 97.89% (6577/6719), PER 0.44% | **98.93%** (6627/6699), PER 0.18% | 72 |
| transformer | 98.50% (6618/6719), PER 0.44% | **99.22%** (6647/6699), PER 0.15% | 52 |

Corrected means: judged on the 6,699 phonetic items (20 excluded, below),
with three kinds of excuse. 51 of this model's 142 error words are not its
fault -- 24 where MFA is wrong or no reading is settled (`microsoft`,
`volenti`, `blair`), 7 where the prediction matches CV's reading instead
(`odessie`, `wstydzę`), and 20 that differ from the reference only in
notation (`eliasz`: `ʎ` for `l j`; `spotkania`: the `j` of `ɲ j a`). For
the transformer it is 34 of 101 (and 3 words where it faithfully reproduces
a reading the adjudication judges wrong). So the two models are 0.29
points apart on real errors, not the 0.61 the measured table shows.

Two limits keep this honest. The verdicts are mine, not a second annotator's
-- each carries its reason and `--show`/`--audit` reprint the evidence, so
they can be argued with. And CV only covers 37K of MFA's 134K words: for the
other 97K (`mfa-only` in the slices) there is no second opinion at all, only
what the adjudication could see internally.

## Out-of-scope items

Acronyms, letter names and unresolvable borrowings are *not* the model's
job, so they are routed around it rather than trained into it. The detector
already exists: the DP aligner fails on them (`align_entry(...) is None`),
which is exactly how the adjudication set finds its 24 unalignable rows.
From there:

- **acronyms and letter names** (`agd`, `bmw`, `nszz`) spell their letters --
  `a ɡ ɛ d ɛ`, a table lookup, not a pronunciation problem;
- **borrowings and names** (`blair`, `guacamole`, `illinois`) go to a
  dictionary lookup; MFA already has them, and CV's 37K words is the second
  opinion to check the entry against.

This is why `convention` rows leave the phonetic denominator in the table
above. It is worth more than tidiness: the 20 excluded items carry **62 of
this model's 230 PER edits and 100 of the transformer's** -- the transformer
tries harder to spell them phonetically and is wrong more often. A model
that never sees them does not spend capacity on them.

### Does this need a retrain?

No, and the reason is worth writing down: **the model was never trained on
them.** `align_entry` returns `None` for structurally unalignable rows and
for alignments whose mean feature cost exceeds 0.3, and `make_dataset` drops
those from train while keeping them in val/test -- 529 train rows. None of
the 20 `convention` words and none of the 4 `open` words is in the training
split. The supervision that *does* train is clean: 99.2% of alignable train
rows align at mean cost <= 0.2, and only 975 rows (0.81%) sit in the dubious
0.2-0.3 band. There is no training-side defect for this experiment to point
at, so a retrain would re-roll the seed and little else.

The asymmetry is on the eval side, and it is deliberate. Training refuses 24
test words as unusable supervision (16 unalignable, 8 over the cost gate);
they are still scored, because the lexicon's reading may well be right even
when the spelling does not predict it. All 8 gated words are errors for the
tiny model (7 for the transformer) -- about 6% of the error list -- and 5 of
them are excused by verdicts (`georgia`, `trance`, `krzywd`, `wczesniej`,
`coo`). The other two, `curry` and `makes`, stay scored: the models mangle
the `c` the lexicon reads as `k`, which is a real miss whatever the gate
says.

## Data and licences

- The **code** is MIT (`rust/Cargo.toml`, `rust/core/Cargo.toml`).
- The **model weights** are trained on the `polish_mfa` v2.0.0 lexicon —
  McAuliffe & Sonderegger (2022), **CC BY 4.0**. Fetched from a pinned GitHub
  release and verified against SHA-256 `0c9cc5c0…cad713`; not vendored into the
  repo (`make lexicon`, which reuses `../pl_g2p`'s copy).

The two are not the same licence, and it matters because the weights are a
redistributable artifact: `rust/core/weights/tiny_g2p.bin` (145 KB) is embedded in
the Rust binary and the wasm build, so anyone shipping those is redistributing
something derived from CC BY 4.0 data and inherits the attribution. The code
licence alone does not say so, which is why this section exists — `../pl_g2p`
has had one for the lexicon it downloads, and this project had none for the
weights it produces.

`prg2p` (MIT) is used by `../phoneme_lab` for its baseline comparison only.
