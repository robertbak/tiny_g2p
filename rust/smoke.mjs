// Smoke test for the wasm build: `node smoke.mjs` after
// `wasm-pack build --release --target nodejs -d pkg-nodejs -- --features wasm`.
//
// Asserts rather than prints, so it fails the way a test should.

import assert from "node:assert/strict";
import { Phonemizer, phonemize } from "./pkg-nodejs/tiny_g2p.js";

const g2p = new Phonemizer();
assert.ok(g2p.weightsKib() > 100, "weights are embedded");
assert.equal(g2p.phonemize("Szczebrzeszyn"), "ʂ tʂ ɛ b ʐ ɛ ʂ ɨ n̪");
assert.equal(g2p.phonemize("Wrocław"), "v r ɔ t̪s̪ w a f");
assert.equal(g2p.phonemize("morze"), g2p.phonemize("może"));
assert.equal(g2p.phonemizeText("kot morze"), "k ɔ t̪\nm ɔ ʐ ɛ");
assert.equal(g2p.windowLen(), 33);
// Beyond the training window it extrapolates rather than refusing.
assert.ok(g2p.phonemize("a".repeat(171)).length > 0);
assert.equal(g2p.explain("bmw"), "acronym table");
assert.equal(g2p.explain("kot"), "model");
assert.equal(phonemize("morze"), "m ɔ ʐ ɛ");

// int8 replays the quantized artifact; the two modes agree on ordinary words.
const int8 = new Phonemizer(true);
assert.equal(int8.phonemize("Szczebrzeszyn"), g2p.phonemize("Szczebrzeszyn"));
assert.equal(int8.phonemize("kot"), "k ɔ t̪");

// The exception path is installed at runtime.
g2p.loadLexicon("# name\treading\nblair\tb l a j r\n");
assert.equal(g2p.explain("blair"), "dictionary");
assert.equal(g2p.phonemize("blair"), "b l a j r");

console.log(
  `wasm smoke test passed: ${g2p.weightsKib()} KiB of weights, ` +
  `float and int8 modes, exception path`,
);
