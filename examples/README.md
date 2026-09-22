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

## What happens in each utterance

One entry per line of `utterances.txt`, in order and numbered the same way. The
words named are the ones where something happened; the rest is the model doing
its job quietly.

**1. ja chcę herbatki w filiżance**

- `chcę` → `x t̪s̪ ɛ` — `ch` is one phone and `c` another, and the final `ę`
  denasalises.
- `filiżance` → `fʲ i ʎ i ʐ a n̪ t̪s̪ ɛ` — `fi` and `li` palatalise before `i`
  (`fʲ`, `ʎ`), and `ż` is the retroflex `ʐ`.

**2. czyżby go o coś podejrzewał**

- `czyżby` → `tʂ ɨ ʐ b ɨ`, `coś` → `t̪s̪ ɔ ɕ`, `podejrzewał` →
  `p ɔ d̪ ɛ j ʐ ɛ v a w` — the digraphs, all in five words: `cz` → `tʂ`,
  `ż`/`rz` → `ʐ`, `ś` → `ɕ`, `ł` → `w`.

**3. nie możesz patrzeć na nie z zewnątrz**

- `możesz` → `m ɔ ʐ ɛ ʂ` — `ż`, then `sz`.
- `patrzeć` → `p a t̪ ʂ ɛ tɕ` — `trz` is the retroflex affricate, `ć` is `tɕ`.
- `zewnątrz` → `z̪ ɛ v n̪ ɔ n̪ t̪ ʂ` — `ą` before `t` resolves to an oral vowel
  plus the homorganic nasal, rather than staying a nasal vowel.

**4. a może to po prostu sprawa zdrowego klimatu w warszawie**

- `może` → `m ɔ ʐ ɛ` — the homophone. Identical to `morze`; see below.
- `klimatu` → `k ʎ i m a t̪ u` and `warszawie` → `v a r ʂ a vʲ ɛ` — `li`
  palatalises, `sz` → `ʂ`, and `wi` is `vʲ`: the `i` marks the palatalisation
  rather than sounding as a vowel of its own.

**5. mruknął z satysfakcją jan uszek**

- `mruknął` → `m r u k n̪ ɔ w` — `ął` comes out as an oral vowel plus the glide,
  because the nasal does not survive into `ł`.
- `satysfakcją` → `s̪ a t̪ ɨ s̪ f a k t̪s̪ j ɔ̃` — and here the final `ą` *does*
  stay nasal. Same letter, two outcomes, decided by what follows it.
- `jan` → `j a n̪` and `uszek` → `u ʂ ɛ k` — a name in lower case reads like any
  other word: nothing in the text marks it as a name, which is exactly why
  names have to be asserted rather than inferred.

**6. żeby jakoś yyy połączyć te dwa zupełnie**

- `yyy` → `ɨ ɨ ɨ` — a filler is a sequence of characters like any other, so the
  G2P transcribes it. Dropping fillers is a post-processor's job, and this line
  is the argument for keeping those two things apart.
- `połączyć` → `p ɔ w ɔ n̪ tʂ ɨ tɕ` — `łą` is `w ɔ`, `ą` before `cz` resolves
  against it, `cz` → `tʂ`, `ć` → `tɕ`.

**7. Dzień dobry zgubiłem swoją kartę kredytową proszę o jej zablokowanie**

- `Dzień` → `dʑ ɛ ɲ` — `ń` is a phone of its own.
- `zgubiłem` → `z̪ ɡ u bʲ i w ɛ m` — `bi` palatalises, `ł` is `w`.
- `kartę` → `k a r t̪ ɛ` and `proszę` → `p r ɔ ʂ ɛ`: final `ę` denasalises, while
  `swoją` → `s̪ f ɔ j ɔ̃` and `kredytową` → `k r ɛ d̪ ɨ t̪ ɔ v ɔ̃` keep final `ą`
  nasal. That is Polish, not a defect, and it is the thing most likely to look
  like one from outside.

**8. Posłanka Dziemianowicz-Bąk przedstawiła poprawkę do ustawy o ochronie zdrowia.**

- `Dziemianowicz-Bąk` → `dʑ ɛ mʲ a n ɔ vʲ i tʂ b ɔ ŋ k` — **the only
  `(dictionary)` line in the file.** The rest of this section is about that one
  word.
- `Posłanka` → `p ɔ s̪ w a n̪ k a` and `przedstawiła` → `p ʂ ɛ t̪ s̪ t̪ a vʲ i w a`
  — `ł` is `w`, twice.
- `zdrowia.` → `z̪ d̪ r ɔ vʲ a j` — the full stop rides along in the token.
  Nothing strips it and nothing needed to.

### The one dictionary line

Without `--lexicon`, the model reads the hyphen in the surname as a break and
inserts a vowel into it, and MFA's lexicon does not have the word at all — so
nothing but a person saying how it is said can fix it:

```
dʑ ɛ mʲ a n ɔ vʲ i tʂ b ɔ ŋ k      with examples/names.dict
dʑ ɛ mʲ a n̪ ɔ vʲ i tʂ ɛ b ɔ ŋ k    without
```

That is the whole case for an asserted dictionary, in one line of output. It is
also why `expected.txt` has a test of its own asserting that this line still
comes from the dictionary: a snapshot would pass just as happily if the model
phonemised it and the snapshot were regenerated to match.

### Why `może` is not a defect

`może` → `m ɔ ʐ ɛ` is identical to `morze`, because Polish really does not
distinguish them. Nothing downstream of phonemes can separate "maybe" from
"sea" either, and no threshold will: a matcher that needs to tell them apart has
to keep the spelling.

## Caveats

The model is about 98% accurate on held-out words, so a word here may not match
how you would say it — these are real outputs, not corrected ones. `expected.txt`
is a snapshot for a regression test, not a claim that every line is right; the
[gold split](../data/test_gold.tsv) is where accuracy is actually measured.

`names.dict` is an *asserted* dictionary — one person saying how a word is said.
It is deliberately not the same file as `data/base.dict`, which is generated and
loaded with `--suggest`; see the project README for why those are two levels.
