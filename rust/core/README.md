# tiny-g2p-core

A tiny Polish **grapheme-to-phoneme** model, with no Python in the loop.

One small learned model labels every character of a word in parallel and emits
its phones. The whole model is 142 KiB of embedded weights and arithmetic, so
there is nothing to download, no service to call, no file to find, and **no
dependencies at all** — `cargo build` produces a static binary and
`wasm-pack` a single 222 KiB `.wasm`.

```rust
let g2p = tiny_g2p::G2P::embedded(tiny_g2p::Mode::Float)?;

g2p.phonemize_str("Szczebrzeszyn");
// "ʂ tʂ ɛ b ʐ ɛ ʂ ɨ n̪"

// Several words at once: whitespace splits, one transcription per line.
g2p.phonemize_text("Wrocław morze może");
// "v r ɔ t̪s̪ w a f\nm ɔ ʐ ɛ\nm ɔ ʐ ɛ"
```

Measured against the `polish_mfa` lexicon: **98.18% word accuracy**, 0.32%
phoneme error rate. The crate carries its own bit-exactness test against the
Python reference, for both the float and the int8 weights.

## What it does and does not do

- **Input is words, not sentences.** Whitespace separates them and every word
  gets one transcription, in order — arguments, a quoted line, stdin or a whole
  file all work.
- **Nothing is refused for being long.** The model's taps cover a 33-character
  window, which is a fact about training, not a limit on input: a longer token
  still computes, the model is simply extrapolating. `G2P::in_window` lets a
  caller decide whether that is acceptable for them.
- **Inference is infallible.** Any string, any length, any Unicode state, gets
  phones. The only failure in the crate is a corrupt weights blob, checked once
  at construction, which is why `G2P::embedded` is the only fallible call in
  the common path.
- **Acronyms and unresolvable borrowings take a separate route** — a
  `word<TAB>phones` dictionary, an embedded table, then an initialism rule —
  because one label per character cannot spell `agd` and no context window
  recovers `blair`. `G2P::explain` reports which path answered.

## The exception path

```rust
let mut g2p = tiny_g2p::G2P::embedded(tiny_g2p::Mode::Float)?;
g2p.load_lexicon("blair\tb l E r\n");   // word<TAB>phones, one per line
assert_eq!(g2p.explain("blair"), Some("dictionary"));
```

`load_lexicon` takes the dictionary's *text*, not a path — this crate does no
I/O and takes no dependency to do it, so the caller decides where the file
comes from.

## Licences — two of them

The **code** is MIT (see `LICENSE`).

The **model weights** are trained on the `polish_mfa` v2.0.0 pronunciation
lexicon — McAuliffe & Sonderegger (2022), **CC BY 4.0** — so redistributing
this crate redistributes something derived from that data, and the attribution
travels with it. The code's licence does not say that, which is why it is said
here.

## Where this sits

`tiny-g2p-core` is the Rust core of the `tiny_g2p` project: the Python side
trains and evaluates the model, this crate runs it. It follows the design of
[vercel-labs/gpu-lexer](https://github.com/vercel-labs/gpu-lexer) — one small
model labelling a whole sequence in parallel — and is the smaller of two Polish
G2P models in that project, the other being a 2.6M-parameter transformer in the
sibling `pl_g2p`.
