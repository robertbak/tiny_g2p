# Provenance

A tiny parallel per-character Polish G2P, following the experiments in
[vercel-labs/gpu-lexer](https://github.com/vercel-labs/gpu-lexer): one small
learned model (~15K params) that labels every character in parallel, trained
under a guarded regime. 98.18% measured / 98.96% corrected PER.

## Origin

Written in this project's own working sessions (2026-09-16 → 2026-09-17), under
the project owner's direction. Its trained model lives in `active/` and is
committed, so the artifact is in the repository, not only on a machine.

**`git log` understates the work, and this file exists to say so.** The
repository was created *afterwards* — `git init -b main` plus one commit titled
"chore: initial import" — wrapping a directory that was already built. Its
second commit (`feat: build targets with word boundaries, and record per-word
phone counts`) is the only one made in the repository itself. No earlier history
exists to recover (`git fsck` finds nothing unreachable; author and committer
dates are identical). Nothing about the code came from elsewhere.

## Third-party material

| what | licence | used for |
|---|---|---|
| `polish_mfa` v2.0.0 — McAuliffe & Sonderegger (2022) | **CC BY 4.0** | the lexicon the model is trained on. Fetched and pinned by SHA-256 `0c9cc5c0…cad713`; not vendored |
| [prg2p](https://github.com/mdm-code/prg2p) v1.0.1 | MIT | baseline comparison |
| [vercel-labs/gpu-lexer](https://github.com/vercel-labs/gpu-lexer) | — | the design this follows; no code taken |
| [`../pl_g2p`](../pl_g2p) | this project's sibling | the dataset and splits (`from pl_g2p.data import …`), and the lexicon copy |

The licence that matters for redistribution is the lexicon's: the weights in
`rust/weights/tiny_g2p.bin` are derived from `polish_mfa`, so **CC BY 4.0
attribution travels with them**, and the code's own licence does not say so.

## Where the code goes

`rust/core` (`tiny-g2p-core`) is vendored into `stt-router` so that repository
builds without this one: `vendor/tiny-g2p-core/PROVENANCE.md` records the
revision, the blob's SHA-256, both licences, and the two mechanical deltas
applied. That copy is the source of record for anyone who does not have this
tree, since this repository has no remote.
