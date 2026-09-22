# Examples on real speech

Eight utterances, their phonemes, and one asserted name — so you can see what
the tool does to Polish it did not write, and check it yourself.

```sh
cd rust && cargo build --release
target/release/tinyg2p predict --explain --lexicon ../examples/names.dict \
    < ../examples/utterances.txt
```

That is exactly what produced `expected.txt`, so running it should reproduce the
file byte for byte. Each line is `word ⇥ phones ⇥ (where the answer came from)`.

## Where the utterances come from

Seven are references from the [BIGOS](https://github.com/goodmike31/bigos)
validation splits — real speech with the transcript the corpus ships, which is
unpunctuated and carries the speaker's disfluencies, so it looks like what an
ASR hands you rather than like a written sentence.

| utterance | corpus | recording |
|---|---|---|
| ja chcę herbatki w filiżance | pwr-shortwords-unk | `…dev-0002-00001` |
| czyżby go o coś podejrzewał | pwr-shortwords-unk | `…dev-0002-00007` |
| nie możesz patrzeć na nie z zewnątrz | pwr-shortwords-unk | `…dev-0002-00011` |
| a może to po prostu sprawa zdrowego klimatu w warszawie | pwr-shortwords-unk | `…dev-0002-00013` |
| mruknął z satysfakcją jan uszek | pwr-shortwords-unk | `…dev-0002-00009` |
| żeby jakoś yyy połączyć te dwa zupełnie | pwr-azon_spont-20 | `…dev-0002-00012` |
| Dzień dobry zgubiłem swoją kartę kredytową proszę o jej zablokowanie | polyai-minds14-21 | `…dev-0002-00001` |

The eighth is not from BIGOS. It is line 2 of `live_stt`'s test script, and it is
here because it is the one case in this file where the dictionary changes the
answer.

## What to look for

**`Dziemianowicz-Bąk` is the only `(dictionary)` line.** Everything else the
model answered by itself. Without `--lexicon` the model reads the hyphen as a
break and inserts a vowel — `dʑ ɛ mʲ a n̪ ɔ vʲ i tʂ ɛ b ɔ ŋ k` — and MFA's
lexicon does not have the word at all, so nothing but a person saying how it is
said can fix it. That is what the exception path is for:

```
dʑ ɛ mʲ a n ɔ vʲ i tʂ b ɔ ŋ k      with names.dict
dʑ ɛ mʲ a n̪ ɔ vʲ i tʂ ɛ b ɔ ŋ k    without
```

**Final `ę` denasalises but final `ą` does not.** `kartę` → `k a r t̪ ɛ` and
`proszę` → `p r ɔ ʂ ɛ`, while `satysfakcją` → `… j ɔ̃` and `kredytową` →
`… v ɔ̃`. That is Polish, not a bug: word-final `ę` is [ɛ] for most speakers
now, and `ą` stays nasal.

**`mruknął` → `m r u k n̪ ɔ w`.** `ą` before `ł` resolves to an oral vowel plus
the glide, which is what the model does rather than emitting a nasal it would
have to take back.

**`yyy` → `ɨ ɨ ɨ`.** A filler is a sequence of characters like any other, so the
G2P transcribes it. Dropping fillers is the post-processor's job, not this
crate's — and a reader of `expected.txt` can see exactly why that separation is
worth having.

**`może` → `m ɔ ʐ ɛ`.** Identical to `morze`. Polish really does not
distinguish them, so nothing downstream of phonemes can either; a matcher that
wants to tell "maybe" from "sea" has to keep the spelling.

## Caveats

The model is about 98% accurate on held-out words, so a word here may not match
how you would say it — these are real outputs, not corrected ones. `expected.txt`
is a snapshot for a regression test, not a claim that every line is right; the
[gold split](../data/test_gold.tsv) is where accuracy is actually measured.

`names.dict` is an *asserted* dictionary — one person saying how a word is said.
It is deliberately not the same file as `data/base.dict`, which is generated and
loaded with `--suggest`; see the project README for why those are two levels.
