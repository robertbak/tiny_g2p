//! Parity with the Python model, word for word.
//!
//! `golden_predictions.json` is emitted by the Python side (see the header of
//! `src/tiny_g2p/export.py`) and holds every held-out test word plus a few
//! awkward ones -- uppercase, a decomposed diacritic, a 21-character word, an
//! empty string -- with what each mode must produce.
//!
//! Both modes are required to be *bit-exact*, not close: the int8 path replays
//! the quantized artifact, and the float path uses the checkpoint's own
//! weights. If a refactor changes the tap layout or the requantization
//! rounding, this is the test that notices.

use tiny_g2p::{G2P, Mode};

fn golden() -> serde_json::Value {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden_predictions.json"
    ))
    .expect("run the Python export to produce golden_predictions.json");
    serde_json::from_str(&text).expect("golden vectors are valid JSON")
}

fn check(mode: Mode, key: &str) {
    let g2p = G2P::embedded(mode).expect("embedded weights parse");
    let golden = golden();
    let table = golden[key].as_object().expect("mode table");
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for (word, expected) in table {
        let want: Vec<String> = expected
            .as_array()
            .expect("phone list")
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect();
        let got = g2p.phonemize_model_only(word);
        if got != want {
            mismatches.push(format!("{word:?}: want {want:?}, got {got:?}"));
        }
        checked += 1;
    }
    assert!(
        mismatches.is_empty(),
        "{} of {checked} words differ from Python:\n  {}",
        mismatches.len(),
        mismatches.join("\n  ")
    );
    assert!(checked > 6700, "expected the full test set, got {checked}");
}

#[test]
fn float_mode_is_bit_exact() {
    check(Mode::Float, "float");
}

#[test]
fn int8_mode_is_bit_exact() {
    check(Mode::Int8, "int8");
}

#[test]
fn awkward_inputs_survive_the_round_trip() {
    let g2p = G2P::embedded(Mode::Float).unwrap();
    // Case folds to the same ids. Escapes, not literals: `łodź` typed by hand
    // is a homoglyph trap (plain `o` looks exactly like `ó`).
    let upper = "\u{141}\u{f3}d\u{17a}"; // Łódź
    let lower = "\u{142}\u{f3}d\u{17a}"; // łódź
    // The observable form of "case folds": the phones are identical. Asserting
    // `text::normalize` instead only restated an internal step, so it is gone
    // and this crate keeps its normalisation private.
    assert_eq!(g2p.phonemize_model_only(upper), g2p.phonemize_model_only(lower));
    // A decomposed acute normalises onto the letter it modifies.
    assert_eq!(
        g2p.phonemize_model_only("S\u{301}limak"),
        g2p.phonemize_model_only("\u{15b}limak")
    );
    assert!(g2p.phonemize_model_only("").is_empty());
}

#[test]
fn length_is_never_a_reason_to_refuse_input() {
    let g2p = G2P::embedded(Mode::Float).unwrap();
    assert_eq!(g2p.window_len(), 33);
    assert!(g2p.in_window("kot"));
    // Beyond the training window the model extrapolates; it does not fail, so
    // one long token cannot abort a batch run.
    let long = "a".repeat(171);
    assert!(!g2p.in_window(&long));
    assert!(!g2p.phonemize(&long).is_empty());
    let compound = "pi\u{119}\u{107}dziesi\u{119}cioprocentow\u{105}prowizj\u{119}";
    assert!(g2p.in_window(compound), "the lexicon's longest word fits");
    assert!(!g2p.phonemize(compound).is_empty());
}

#[test]
fn several_words_at_once() {
    let g2p = G2P::embedded(Mode::Float).unwrap();
    // Whitespace splits: arguments, a pasted line, a whole file.
    let phones = g2p.phonemize_words("Wroc\u{142}aw morze\nmo\u{17c}e  kot");
    assert_eq!(phones.len(), 4, "one entry per word, order preserved");
    assert_eq!(phones[0], g2p.phonemize("Wroc\u{142}aw"));
    assert_eq!(phones[3], g2p.phonemize("kot"));
    assert_eq!(g2p.phonemize_text("kot morze").lines().count(), 2);
    assert_eq!(g2p.phonemize_text("   "), "");
}
