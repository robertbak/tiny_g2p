# tiny-g2p (Rust)

The same model, with no Python in the loop: one static binary, one `.wasm`,
or an ONNX graph. The core has **no native dependencies** — the weights are
`include_bytes!`d and the blob is parsed by hand — so `cargo build --release`
produces a single executable and `wasm-pack` produces a single `.wasm`.

```
$ tinyg2p predict Szczebrzeszyn Wrocław bmw
ʂ tʂ ɛ b ʐ ɛ ʂ ɨ n̪
v r ɔ t̪s̪ w a f
b ɛ m ɛ v u

$ tinyg2p eval --gold ../data/test_gold.tsv
weights: float   words: 6699   excluded: 20
slice            n   measured   corrected      PER  PERcorr
agreed        1756     99.89%      99.94%    0.01%    0.01%
disputed        96     85.42%      96.88%    2.09%    0.55%
mfa-only      4847     97.81%      98.64%    0.39%    0.23%
measured   98.18% (6577/6699)  PER 0.32% (165/51746)
corrected  98.96% (6629/6699)  PER 0.18% (93/51750)
```

## Build

```bash
cd rust

cargo build --release                      # target/release/tinyg2p (680 KiB, no deps)
cargo test --release --workspace           # 26 unit + 5 parity + 1 doc test

wasm-pack build --release --target nodejs --out-name tiny_g2p -d ../pkg-nodejs wasm
node smoke.mjs                             # the wasm surface, exercised

cd .. && uv run tiny-g2p export --onnx rust/core/weights/tiny_g2p.onnx
```

Three crates, one workspace. They are separate because a single crate cannot
be both a `bin` and a `cdylib`: the binary links against the cdylib's metadata
and `cargo test --release` fails with `undefined symbol` (tried `lto = false`,
tried crate-type ordering -- it is the mixing itself).

| crate | what | why it exists |
|---|---|---|
| `tiny-g2p` (root) | the `tinyg2p` binary | no `[lib]`, so nothing collides |
| `tiny-g2p-core` | the model, **no dependencies** | the whole implementation |
| `tiny-g2p-wasm` | `cdylib` + wasm-bindgen | the browser surface |

## What ships

| artifact | size | notes |
|---|---|---|
| `tinyg2p` binary | 680 KiB (592 KiB stripped) | static; the model is 142 KiB of that |
| `pkg-nodejs/tiny_g2p_bg.wasm` | 222 KiB (167 KiB gzipped) | float + int8 + exception path |
| `tiny_g2p.onnx` | 131 KiB | the float graph, one input, one output |

Both weight sets live in the one blob, so float and int8 modes cost nothing
extra to carry. If a deployment ever needs to be smaller, dropping the float
set (or storing it as f16) is an export-side change, not a code change.

## The three targets agree

Parity is *bit-exact*, not approximate, and it is tested:

| check | result |
|---|---|
| Rust float vs Python float, all 6,725 test words | identical (`tests/parity.rs`) |
| Rust int8 vs Python int8, same words | identical |
| ONNX Runtime vs torch, argmax over the same words | 0 mismatches (`tests/test_onnx.py`) |
| `tinyg2p eval` measured accuracy | 6577/6699 float, 6576/6699 int8 — the Python numbers |

Why the two modes: the published 97.89% is the **int8** artifact, while the
**float** checkpoint — the default here — is one word ahead on the same 6,699
phonetic items (6577 against 6576) with three fewer edit operations (165
against 168 on 51,746 reference phones), and it is the simpler arithmetic. The
two differ on 23 of 6,719 words. `--int8` exists so a deployment can *prove*
it replays the measured artifact.

Speed, for scale: `tinyg2p eval` scores all 6,699 words in **0.5 s** on one
CPU core (the Python path needs torch and ~14 s).

## Several words at once

Input is words, not sentences: whitespace separates them, and every word gets
one transcription, in order. So arguments, a quoted line, stdin and a whole
file all work, and the batch entry point is one call rather than a loop over
an FFI boundary:

```rust
let phones = g2p.phonemize_words("Wrocław morze\nmoże  kot"); // 4 entries
let text = g2p.phonemize_text("Wrocław morze");               // one line each
```

```js
g2p.phonemizeText("Wrocław morze");   // "v r ɔ t̪s̪ w a f\nm ɔ ʐ ɛ"
```

**Nothing is refused for being long.** The model's taps cover a 33-character
window -- `2 x max|tap| + 1`, sized to the lexicon's longest word
(`pięćdziesięcioprocentowąprowizję`, 32) -- but that is a fact about training,
not a limit on input. A longer token still computes: the taps are fixed
offsets, so it is one character at a time with the same arithmetic, and the
model is simply extrapolating. `window_len()` / `in_window()` exist so a
caller *can* care, and `info` reports it; the transcribe path never fails.
That matters in practice, because failing would mean one URL or chemical name
aborting a batch run.

## The exception path

Acronyms and unresolvable borrowings are routed *around* the model rather than
trained into it — one label per character cannot spell `agd` as five phones,
and no context window recovers `blair`. In order:

1. **`--lexicon FILE`**: `word<TAB>phones`, the reliable route for borrowings
   and names;
2. **the acronym table** embedded in the blob (13 words we hold gold for);
3. **the initialism rule**: a vowel-less word of ≥2 letters whose letters all
   have names is spelled letter by letter (`bmw` → `b ɛ m ɛ v u`, `nszz` → …);
4. otherwise the model.

`--explain` says which path answered. The rule is a heuristic and the table is
authoritative: `agd` and `rpo` contain a vowel letter, so the rule misses them
and the table answers. Nine of the corpus's thirteen acronyms come out
byte-identical to the gold; the four that do not are MFA's own noise
(`g` palatalized before `e` inconsistently in `agd`/`dga`; `bmw` and `cv`
mis-segmented, the latter read `s̪i vʲ i` rather than `t͡sɛ vɛ`).

A dictionary is a lookup, not a pronunciation, so it is exact — see
`tests/test_export.py::test_letter_names_match_the_lexicon_except_the_known_noise`.

## Regenerating the weights

The binary embeds `weights/tiny_g2p.bin`, so **the export comes first**:

```bash
uv run tiny-g2p export \
  --blob rust/core/weights/tiny_g2p.bin \
  --gold data/test_gold.tsv \
  --acronym agd --acronym bmw --acronym cv --acronym dga --acronym mps \
  --acronym nszz --acronym ntv --acronym pzpr --acronym rpo --acronym rtv \
  --acronym tpn --acronym wku --acronym wtw
```

`make export` in the parent directory runs exactly that. The blob is versioned
(`TG2P`/1) and self-describing; a mismatched version is rejected rather than
misread, and `tiny_g2p/tests/test_export.py` holds the reference reader, so the
layout has one definition in each language and a test on each side.

The gold TSV is generated, not curated: it carries MFA's references, the
adjudicated candidate sets, the canonical forms that excuse notation
differences, and the verdict flags — everything `eval` needs to print both the
measured and the gold-corrected numbers, and nothing it would have to guess.

## Layout

| path | what |
|---|---|
| `src/blob.rs` | the blob reader (mirror of `export.py`'s writer) |
| `src/model.rs` | float and int8 forward pass |
| `src/text.rs` | Polish NFC-lite normalisation, vocabularies |
| `src/phones.rs` | `restore`: the nasal merge split back |
| `src/canon.rs` | the MFA-side canonicalization used for scoring |
| `src/exceptions.rs` | dictionary, acronym table, initialism rule |
| `src/gold.rs` | gold TSV reader and the scorer |
| `core/src/*` | the model: blob, forward, text, restore, canon, exceptions, gold |
| `core/tests/parity.rs` | bit-exactness against `golden_predictions.json` |
| `wasm/src/lib.rs` | the browser surface |
| `src/main.rs` | `predict`, `eval`, `info` |
| `smoke.mjs` | the wasm build, exercised under node |
| `weights/` | generated: the blob and the ONNX graph |
