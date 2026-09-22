//! Turning user text into token ids and ids back into phones.
//!
//! Two jobs, both small:
//!
//! * **Normalisation.** The lexicon is lowercase NFC. Rust's `to_lowercase`
//!   is full Unicode, but NFC is not, and pulling in a normalisation crate to
//!   fold nine Polish letters would be silly -- so this composes exactly the
//!   combining marks Polish uses and documents that it is not general NFC.
//! * **Vocabulary.** The blob carries the id-to-token lists; encoding is a
//!   map lookup with UNK for anything unseen.

use std::collections::HashMap;

use crate::blob::Blob;

/// A decomposed Polish letter: which combining marks are composed, and onto
/// what. Everything else passes through untouched.
fn compose(base: char, mark: char) -> Option<char> {
    match (base, mark) {
        ('a', '\u{328}') => Some('ą'),   // ogonek
        ('e', '\u{328}') => Some('ę'),
        ('c', '\u{301}') => Some('ć'),   // acute
        ('n', '\u{301}') => Some('ń'),
        ('o', '\u{301}') => Some('ó'),
        ('s', '\u{301}') => Some('ś'),
        ('z', '\u{301}') => Some('ź'),
        ('z', '\u{307}') => Some('ż'),   // dot above
        _ => None,
    }
}

fn is_combining(ch: char) -> bool {
    matches!(ch, '\u{301}' | '\u{307}' | '\u{328}')
}

/// Lowercase and compose: enough NFC for Polish words, not general NFC.
pub fn normalize(input: &str) -> String {
    let mut out: Vec<char> = Vec::with_capacity(input.len());
    for ch in input.trim().chars() {
        let lower = ch.to_lowercase().next().unwrap_or(ch);
        if is_combining(lower) {
            match out.pop() {
                Some(base) => match compose(base, lower) {
                    Some(composed) => out.push(composed),
                    None => {
                        out.push(base);
                        out.push(lower);
                    }
                },
                None => out.push(lower),
            }
            continue;
        }
        out.push(lower);
    }
    out.into_iter().collect()
}

/// The character vocabulary: token strings in, ids out.
///
/// Built once from the blob's source vocabulary. Holds only what encoding
/// needs — the special-token ids are read by the decoder, not here.
pub struct Encoder {
    chars: HashMap<char, usize>,
    unk_id: usize,
}

impl Encoder {
    pub fn new(blob: &Blob) -> Self {
        let mut chars = HashMap::with_capacity(blob.src_itos.len());
        for (id, token) in blob.src_itos.iter().enumerate() {
            let mut it = token.chars();
            if let (Some(ch), None) = (it.next(), it.next()) {
                chars.insert(ch, id); // single-character tokens only
            }
        }
        Encoder { chars, unk_id: blob.unk_id }
    }

    /// One id per character of `word` (already normalised), UNK for anything
    /// the vocabulary has never seen.
    pub fn encode(&self, word: &str) -> Vec<usize> {
        word.chars()
            .map(|ch| self.chars.get(&ch).copied().unwrap_or(self.unk_id))
            .collect()
    }
}

/// Drop the special tokens from a per-character argmax, leaving the phones the
/// model actually emitted (one `None` per silent character).
pub fn decode<'a>(blob: &'a Blob, ids: &[usize]) -> Vec<Option<&'a str>> {
    ids.iter()
        .map(|id| {
            if *id == blob.pad_id || *id == blob.unk_id || *id == blob.blank_id {
                None
            } else {
                Some(blob.tgt_itos[*id].as_str())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_case_and_whitespace() {
        assert_eq!(normalize("  Szczebrzeszyn "), "szczebrzeszyn");
        assert_eq!(normalize("KOT"), "kot");
    }

    #[test]
    fn composes_decomposed_polish_letters() {
        // Escapes, not literals: hand-typed Polish is a homoglyph trap.
        assert_eq!(normalize("\u{141}\u{f3}d\u{17a}"), "\u{142}\u{f3}d\u{17a}");
        assert_eq!(normalize("S\u{301}limak"), "\u{15b}limak");
        assert_eq!(normalize("A\u{328}"), "\u{105}");
        assert_eq!(normalize("Z\u{307}"), "\u{17c}");
        // An unhandled mark survives rather than being dropped.
        assert_eq!(normalize("a\u{300}"), "a\u{300}");
    }

    #[test]
    fn encodes_known_characters_and_unks_the_rest() {
        let g2p = crate::G2P::embedded(crate::Mode::Float).unwrap();
        let encoder = Encoder::new(g2p.blob());
        let ids = encoder.encode("kot");
        assert_eq!(ids.len(), 3);
        assert_eq!(g2p.blob().src_itos[ids[0]], "k");
        assert_eq!(g2p.blob().src_itos[ids[1]], "o");
        assert_eq!(g2p.blob().src_itos[ids[2]], "t");
        assert_eq!(encoder.encode("\u{df}")[0], g2p.blob().unk_id);
    }
}
