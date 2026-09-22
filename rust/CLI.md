# tiny-g2p

Polish **grapheme-to-phoneme** on the command line, with no Python in the loop.

```console
$ tinyg2p predict "Wrocław Szczebrzeszyn"
v r ɔ t̪s̪ w a f
ʂ tʂ ɛ b ʐ ɛ ʂ ɨ n̪
```

One small model (~15K parameters, 142 KiB of weights) is compiled into the
binary, along with the arithmetic to run it. There is nothing to download, no
model file to find, and no runtime to install.

## Install

```sh
cargo install tiny-g2p        # from crates.io; installs the `tinyg2p` binary
```

Or from a checkout, which is what you want before the first release:

```sh
cargo build --release         # ./target/release/tinyg2p
```

The binary is named `tinyg2p` — the package keeps the hyphen, the executable
does not.

## What it does

```
tinyg2p predict [words...]      transcribe (reads stdin when no words are given)
tinyg2p eval --gold FILE        score against an exported gold TSV
tinyg2p info                    show what is compiled in
```

| option | effect |
|---|---|
| `--int8` | replay the quantized artifact instead of the float weights |
| `--lexicon FILE` | a dictionary for borrowings and names — `word<TAB>phones`, or MFA's dictionary exactly as it comes |
| `--blob FILE` | load weights from a file instead of the embedded ones |
| `--explain` | *(predict)* say which path answered each word |
| `--model-only` | *(predict)* bypass the exception path |
| `--json FILE` | *(eval)* also write the report as JSON |

Input is words, not sentences: whitespace separates them and every word gets one
transcription, in order. Arguments, a quoted line, stdin and a whole file all
work:

```console
$ echo "morze może" | tinyg2p predict
m ɔ ʐ ɛ
m ɔ ʐ ɛ
$ tinyg2p predict --explain bmw blair
b ɛ m ɛ v u        initialism
```

`--explain` prints where each reading came from — `model`, `dictionary`,
`acronym table` or `initialism`. Borrowings and names do not follow Polish
graphemics, so they are looked **up** rather than guessed; supply them with
`--lexicon`, and the project's own dictionary is MFA's lexicon with Common Voice
as a second opinion where the two readings differ. MFA's dictionary is accepted
as it is — its per-reading probability columns are recognised and dropped, so
`--lexicon polish_mfa.dict` works with no conversion step. Acronyms are the structural
case: one label per character cannot spell `agd`, so there is a small embedded
table and a vowel-less initialism rule beneath it.

`eval` scores a prediction against a gold TSV, reporting word accuracy and
phoneme error rate — and, separately, the *corrected* figures, so a notation
difference is visible rather than silently counted as an error.

## Speed and size

`tinyg2p eval` scores 6,699 words in about half a second on one CPU core. The
release binary is ~680 KiB static (592 KiB stripped), the model being 142 KiB of
that.

## Licences — two of them

The **code** is MIT (see `LICENSE`).

The **model weights** are trained on the `polish_mfa` v2.0.0 pronunciation
lexicon — McAuliffe & Sonderegger (2022), **CC BY 4.0**. They are compiled into
this binary, so installing it redistributes something derived from that data,
and the attribution travels with it. The code's licence does not say that, which
is why it is said here.

## The library

This is the CLI for [`tiny-g2p-core`](https://crates.io/crates/tiny-g2p-core),
which is the same model as a zero-dependency library if you want it in a
program rather than a shell:

```rust
let g2p = tiny_g2p::G2P::embedded(tiny_g2p::Mode::Float)?;
g2p.phonemize_str("Szczebrzeszyn");
```
