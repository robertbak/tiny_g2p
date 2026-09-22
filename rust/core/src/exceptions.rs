//! The exception path: the words the model should not own.
//!
//! Two classes, for two different reasons:
//!
//! - **Borrowings and names** -- `blair`, `guacamole`, `illinois` -- do not
//!   follow Polish graphemics, so a dictionary is the right mechanism rather
//!   than a workaround for a weak model. The project holds one (MFA's lexicon,
//!   with Common Voice as a second opinion where the two readings differ), and
//!   this module is where it enters, at 1.
//! - **Acronyms** are structural: one label per character cannot spell `agd` as
//!   five phones, whatever the context window, so they are routed around the
//!   model instead of trained into it, at 2 and 3.
//!
//! In this order:
//!
//! 1. **an explicit dictionary** (`--lexicon word<TAB>phones`), the route for
//!    borrowings and names;
//! 2. **the acronym table** in the blob -- words we hold a gold letter-by-letter
//!    reading for, so the tool reproduces the corpus convention;
//! 3. **the initialism rule**: a vowel-less word of two or more letters whose
//!    letters all have names is spelled out (`bmw` -> b ɛ m ɛ v u);
//! 4. otherwise the model.
//!
//! The initialism rule is a heuristic and says so: `agd` and `rpo` contain a
//! vowel letter and are missed, which is why the table exists. Nothing here
//! guesses at a *pronunciation* -- it either looks one up or spells letters.

use std::collections::HashMap;

use crate::blob::Blob;

/// Vowel letters: a word with none of them is an initialism.
const VOWELS: &[char] = &['a', 'ą', 'e', 'ę', 'i', 'o', 'ó', 'u', 'y'];

pub struct Exceptions {
    /// Optional external dictionary: word -> phones.
    lexicon: HashMap<String, Vec<String>>,
    acronyms: HashMap<String, Vec<String>>,
    letter_names: HashMap<char, Vec<String>>,
}

impl Exceptions {
    pub fn new(blob: &Blob) -> Self {
        Exceptions {
            lexicon: HashMap::new(),
            acronyms: to_readings(&blob.acronyms),
            letter_names: blob
                .letter_names
                .iter()
                .filter_map(|(key, value)| {
                    let mut it = key.chars();
                    match (it.next(), it.next()) {
                        (Some(ch), None) => {
                            Some((ch, value.split_whitespace().map(str::to_string).collect()))
                        }
                        _ => None,
                    }
                })
                .collect(),
        }
    }

    /// Load a `word<TAB>phones` (or space-separated, whitespace-tolerant)
    /// dictionary. Later entries win.
    pub fn load_lexicon(&mut self, text: &str) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(word) = parts.next() else { continue };
            let phones: Vec<String> = parts.map(str::to_string).collect();
            if !phones.is_empty() {
                self.lexicon.insert(word.to_lowercase(), phones);
            }
        }
    }

    /// The reading for `word`, or `None` to let the model decide.
    pub fn lookup(&self, word: &str) -> Option<Vec<String>> {
        if let Some(reading) = self.lexicon.get(word) {
            return Some(reading.clone());
        }
        if let Some(reading) = self.acronyms.get(word) {
            return Some(reading.clone());
        }
        if is_initialism(word, &self.letter_names) {
            return Some(self.spell(word));
        }
        None
    }

    /// Why `lookup` answered the way it did, for `--explain`.
    pub fn reason(&self, word: &str) -> Option<&'static str> {
        if self.lexicon.contains_key(word) {
            Some("dictionary")
        } else if self.acronyms.contains_key(word) {
            Some("acronym table")
        } else if is_initialism(word, &self.letter_names) {
            Some("initialism")
        } else {
            None
        }
    }

    fn spell(&self, word: &str) -> Vec<String> {
        word.chars()
            .filter_map(|ch| self.letter_names.get(&ch))
            .flatten()
            .cloned()
            .collect()
    }
}

fn to_readings(table: &HashMap<String, String>) -> HashMap<String, Vec<String>> {
    table
        .iter()
        .map(|(word, phones)| {
            (word.clone(), phones.split_whitespace().map(str::to_string).collect())
        })
        .collect()
}

/// A vowel-less word of two or more letters, every letter of which has a name.
fn is_initialism(word: &str, letter_names: &HashMap<char, Vec<String>>) -> bool {
    let mut chars = word.chars();
    let has_letter = chars.next().is_some();
    word.chars().count() >= 2
        && has_letter
        && !word.chars().any(|ch| VOWELS.contains(&ch))
        && word.chars().all(|ch| letter_names.contains_key(&ch))
}
#[cfg(test)]
mod tests {
    use super::*;

    fn exceptions() -> Exceptions {
        let g2p = crate::G2P::embedded(crate::model::Mode::Float).unwrap();
        Exceptions::new(g2p.blob())
    }

    #[test]
    fn spells_an_initialism_from_the_letter_table() {
        let ex = exceptions();
        // `bmwx` is not in the table (the corpus words are) but is a vowel-less
        // sequence of named letters, so the rule spells it out.
        assert!(is_initialism("bmwx", &ex.letter_names));
        assert_eq!(ex.reason("bmwx"), Some("initialism"));
        let spelled = ex.lookup("bmwx").unwrap();
        assert_eq!(spelled.len(), 9, "b, m, w are two phones each; x is three");
        assert_eq!(spelled[0], "b");
        // `agd` has a vowel letter, so the rule misses it and the table answers.
        assert!(!is_initialism("agd", &ex.letter_names));
        assert_eq!(ex.reason("agd"), Some("acronym table"));
    }

    #[test]
    fn uses_the_acronym_table_when_it_has_one() {
        let ex = exceptions();
        assert_eq!(ex.reason("cv"), Some("acronym table"));
        assert!(!ex.lookup("cv").unwrap().is_empty());
    }

    #[test]
    fn a_dictionary_beats_everything() {
        let mut ex = exceptions();
        ex.load_lexicon("# comment\nbmw\tB M W\nblair b l E r\n");
        // Both entries loaded and both outrank the acronym table -- asserted
        // through `reason`/`lookup`, so the size accessor is not needed.
        assert_eq!(ex.reason("bmw"), Some("dictionary"));
        assert_eq!(ex.lookup("bmw").unwrap(), ["B", "M", "W"]);
        assert_eq!(ex.lookup("blair").unwrap(), ["b", "l", "E", "r"]);
    }

    #[test]
    fn ordinary_words_go_to_the_model() {
        let ex = exceptions();
        for word in ["kot", "szczebrzeszyn", "wroc\u{142}aw"] {
            assert_eq!(ex.reason(word), None, "{word} should reach the model");
            assert!(ex.lookup(word).is_none());
        }
    }
}
