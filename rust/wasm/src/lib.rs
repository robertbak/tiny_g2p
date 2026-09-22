//! The browser surface: the same core, one `.wasm` file, no dependencies.
//!
//! ```js
//! import init, { Phonemizer } from "./pkg-nodejs/tiny_g2p.js";
//! await init();
//! const g2p = new Phonemizer();          // float weights; pass true for int8
//! g2p.phonemize("Szczebrzeszyn");        // "ʂ tʂ ɛ b ʐ ɛ ʂ ɨ n̪"
//! g2p.phonemizeText("kot morze");        // "k ɔ t̪\nm ɔ ʐ ɛ"
//! ```
//!
//! The weights are compiled in, so there is nothing to fetch and nothing to
//! keep in sync: the module *is* the model. Inference never fails -- the only
//! error is a corrupt blob, which cannot happen in a build that compiled.

use wasm_bindgen::prelude::*;

use tiny_g2p::{G2P, Mode};

/// A loaded model. Construct once; it holds no per-call state.
#[wasm_bindgen]
pub struct Phonemizer {
    inner: G2P,
}

#[wasm_bindgen]
impl Phonemizer {
    /// `new()` uses the float weights; `new(true)` replays the int8 artifact
    /// (one word apart on the held-out set: 98.16% against 98.18%).
    #[wasm_bindgen(constructor)]
    pub fn new(quantized: Option<bool>) -> Result<Phonemizer, JsValue> {
        let mode = if quantized.unwrap_or(false) { Mode::Int8 } else { Mode::Float };
        Ok(Phonemizer {
            inner: G2P::embedded(mode).map_err(|e| JsValue::from_str(&e.to_string()))?,
        })
    }

    /// One word in, a spaced phone string out.
    pub fn phonemize(&self, word: &str) -> String {
        self.inner.phonemize_str(word)
    }

    /// Many words at once: whitespace splits, one transcription per line. The
    /// batch entry point -- one call, one boundary crossing.
    #[wasm_bindgen(js_name = phonemizeText)]
    pub fn phonemize_text(&self, text: &str) -> String {
        self.inner.phonemize_text(text)
    }

    /// Install a `word<TAB>phones` dictionary for borrowings and names.
    #[wasm_bindgen(js_name = loadLexicon)]
    pub fn load_lexicon(&mut self, text: &str) {
        self.inner.load_lexicon(text);
    }

    /// How a word will be answered: `"dictionary"`, `"acronym table"`,
    /// `"initialism"`, or `"model"`.
    pub fn explain(&self, word: &str) -> String {
        self.inner.explain(word).unwrap_or("model").to_string()
    }

    /// The character window the taps cover; longer words are extrapolating.
    #[wasm_bindgen(js_name = windowLen)]
    pub fn window_len(&self) -> usize {
        self.inner.window_len()
    }

    /// Embedded weight size, in KiB -- the whole download.
    #[wasm_bindgen(js_name = weightsKib)]
    pub fn weights_kib(&self) -> usize {
        tiny_g2p::EMBEDDED_WEIGHTS.len() / 1024
    }
}

/// One-off convenience for callers that do not want a constructor.
#[wasm_bindgen]
pub fn phonemize(word: &str) -> Result<String, JsValue> {
    let g2p = G2P::embedded(Mode::Float).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(g2p.phonemize_str(word))
}
