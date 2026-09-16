//! `restore`: the label -> surface step, ported from `tiny_g2p/phones.py`.
//!
//! The aligner merges ą/ę before a stop into a single nasal-vowel label, which
//! is what lets one character own one label. Scoring happens against MFA's raw
//! phones, so the merge has to be undone: a nasal vowel label on an ą/ę
//! character followed by an emitted stop splits back into the oral vowel plus
//! the homorganic nasal MFA writes.
//!
//! The split is deterministic given the word and the following phone, and only
//! ą/ę positions are eligible -- a nasal vowel the model emits on a plain vowel
//! is a model error and passes through untouched, so scoring stays faithful to
//! what the model actually said.

use crate::blob::Blob;

/// Split merged nasal vowels back to MFA's oral+nasal form.
pub fn restore(blob: &Blob, word: &[char], labels: &[Option<&str>]) -> Vec<String> {
    let emitted: Vec<(usize, &str)> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, label)| label.map(|phone| (i, phone)))
        .collect();
    let mut out = Vec::with_capacity(emitted.len());
    for (k, (i, phone)) in emitted.iter().enumerate() {
        let letter = word.get(*i).copied().unwrap_or('\0');
        let is_nasal_letter = blob
            .nasal_letters
            .iter()
            .any(|l| l.chars().eq(std::iter::once(letter)));
        if is_nasal_letter {
            if let Some(oral) = blob.oral.get(*phone) {
                let follower = emitted.get(k + 1).map(|(_, p)| *p);
                if let Some(stop) = follower {
                    if blob.stops.iter().any(|s| s == stop) {
                        out.push(oral.clone());
                        out.push(
                            blob.homorganic
                                .get(stop)
                                .cloned()
                                .unwrap_or_else(|| stop.to_string()),
                        );
                        continue;
                    }
                }
            }
        }
        out.push((*phone).to_string());
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    fn blob() -> &'static Blob {
        // Leaked once for the test process; the tables are what matter here.
        Box::leak(Box::new(
            crate::G2P::embedded(crate::model::Mode::Float).unwrap().blob().clone(),
        ))
    }

    /// ɔ̃: "ɔ" plus a combining tilde, the merged label the aligner emits.
    const NASAL_O: &str = "\u{254}\u{303}";

    #[test]
    fn splits_a_merged_nasal_before_a_stop() {
        let b = blob();
        // kąt: k, ą (merged ɔ̃), t -> k ɔ n̪ t̪
        let word: Vec<char> = "k\u{105}t".chars().collect();
        let labels = [Some("k"), Some(NASAL_O), Some("t\u{32a}")];
        assert_eq!(
            restore(b, &word, &labels),
            vec!["k", "\u{254}", "n\u{32a}", "t\u{32a}"]
        );
    }

    #[test]
    fn leaves_a_nasal_before_a_fricative_alone() {
        let b = blob();
        let word: Vec<char> = "k\u{105}s".chars().collect();
        let labels = [Some("k"), Some(NASAL_O), Some("s\u{32a}")];
        assert_eq!(restore(b, &word, &labels), vec!["k", NASAL_O, "s\u{32a}"]);
    }

    #[test]
    fn never_splits_on_a_plain_vowel() {
        let b = blob();
        // A nasal vowel label on an `o` is a model error, not a merge.
        let word: Vec<char> = "kot".chars().collect();
        let labels = [Some("k"), Some(NASAL_O), Some("t\u{32a}")];
        assert_eq!(restore(b, &word, &labels), vec!["k", NASAL_O, "t\u{32a}"]);
    }
}
