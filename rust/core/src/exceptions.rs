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
//! 1. **an asserted dictionary** (`--lexicon`): a person wrote down that this
//!    word is called this, and is said like this. Nothing outranks an
//!    assertion, because an assertion is not evidence weighed against
//!    evidence;
//! 2. **the acronym table** in the blob -- words we hold a gold letter-by-letter
//!    reading for, so the tool reproduces the corpus convention;
//! 3. **a suggested dictionary** ([`Exceptions::load_suggestions`], and what
//!    `tinyg2p miss-lexicon` writes): a lexicon's reading for a word the model
//!    disagrees with. Better than a guess, and still a dictionary's opinion
//!    rather than someone saying "this is my name";
//! 4. **the initialism rule**: a vowel-less word of two or more letters whose
//!    letters all have names is spelled out (`bmw` -> b ɛ m ɛ v u);
//! 5. otherwise the model.
//!
//! The split between 1 and 3 is the point of having two levels at all. An
//! acronym table entry and a generated dictionary entry are *alternative
//! spellings* -- a preference among readings, derived from data. A user's own
//! list is a claim about the world, and when the two disagree the claim should
//! win, and `--explain` should say which one answered.
//!
//! The initialism rule is a heuristic and says so: `agd` and `rpo` contain a
//! vowel letter and are missed, which is why the table exists. Nothing here
//! guesses at a *pronunciation* -- it either looks one up or spells letters.

use std::collections::HashMap;

use crate::blob::Blob;

/// Read a dictionary, in any of the shapes [`Exceptions::load_lexicon`]
/// accepts.
///
/// Returned as `(word, phones)` pairs in file order, lowercased, with the lines
/// that name no phones left out. Public because the format is the project's — a
/// caller that wants to scan a lexicon, or convert one, needs the same reading
/// of it that the exception path uses, and a second parser would be a second
/// answer.
#[must_use]
pub fn parse_dictionary(text: &str) -> Vec<(String, Vec<String>)> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(parse_line)
        .collect()
}

/// One dictionary line into `(word, phones)`, lowercased; `None` to skip it.
fn parse_line(line: &str) -> Option<(String, Vec<String>)> {
    let (word, phones) = if line.contains('\t') {
        let mut fields = line.split('\t').map(str::trim);
        let word = fields.next()?.to_lowercase();
        let phones = phones_from_fields(&fields.collect::<Vec<_>>());
        (word, phones)
    } else {
        // No tab: the first token is the word, the rest are its phones.
        let mut tokens = line.split_whitespace();
        let word = tokens.next()?.to_lowercase();
        (word, tokens.map(str::to_string).collect())
    };
    if word.is_empty() || phones.is_empty() {
        return None;
    }
    Some((word, phones))
}

/// The key a dictionary entry is stored under, and looked up by.
///
/// The same function on both sides, which is the point: `load_lexicon` used to
/// store a plain `to_lowercase()` while a lookup went through
/// `text::normalize`, and those differ for decomposed input. A name written
/// with `a` + U+0328 instead of `ą` then matched nothing and fell through to the
/// model -- silently, which is the worst way for a dictionary to fail.
fn normalize_key(word: &str) -> String {
    crate::text::normalize(word)
}

/// The phones among the tab-separated fields that follow the word.
///
/// MFA writes `word<TAB>p1<TAB>p2<TAB>p3<TAB>p4<TAB>phones`, where the four
/// columns are per-reading probabilities. So when everything between the word
/// and the last field parses as a number, that last field is the phone list;
/// otherwise every remaining field is. Getting this wrong is not subtle in
/// effect and is easy to miss in testing: MFA's probabilities became phones and
/// the dictionary route answered `kot` with `0.99 0.49 2.75 1.1 k ɔ t̪`.
fn phones_from_fields(rest: &[&str]) -> Vec<String> {
    let Some((last, middle)) = rest.split_last() else {
        return Vec::new();
    };
    let mfa_shaped =
        !middle.is_empty() && middle.iter().all(|field| field.parse::<f32>().is_ok());
    let tokens: Vec<&str> = if mfa_shaped {
        last.split_whitespace().collect()
    } else {
        rest.iter().flat_map(|field| field.split_whitespace()).collect()
    };
    tokens.into_iter().map(str::to_string).collect()
}

/// Vowel letters: a word with none of them is an initialism.
const VOWELS: &[char] = &['a', 'ą', 'e', 'ę', 'i', 'o', 'ó', 'u', 'y'];

pub struct Exceptions {
    /// A dictionary the caller *asserts*: these words are called this, and are
    /// said like this. Someone wrote it down, so it outranks everything.
    asserted: HashMap<String, Vec<String>>,
    /// A curated table of acronyms from the blob, with gold letter-by-letter
    /// readings. Not user-editable, and authoritative for the words it holds.
    acronyms: HashMap<String, Vec<String>>,
    /// Readings *suggested* from where the model disagrees with a lexicon --
    /// the file `tinyg2p miss-lexicon` writes. A preference, not an assertion:
    /// it is how some dictionary reads the word, which is good evidence and
    /// still not the same as a person saying so.
    suggested: HashMap<String, Vec<String>>,
    letter_names: HashMap<char, Vec<String>>,
}

impl Exceptions {
    pub fn new(blob: &Blob) -> Self {
        Exceptions {
            asserted: HashMap::new(),
            acronyms: to_readings(&blob.acronyms),
            suggested: HashMap::new(),
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
    /// Add words to the exception path from a dictionary, one per line.
    ///
    /// Three shapes work, because the project's own lexicon is the second one
    /// and a parser that only took the first would answer MFA words with
    /// `0.99 0.49 2.75 1.1` in front of their phones:
    ///
    /// ```text
    /// blair<TAB>b l E r                     word, then its phones
    /// blair b l E r                         the same, space-separated
    /// kot<TAB>0.99<TAB>0.49<TAB>2.75<TAB>1.1<TAB>k ɔ t̪     MFA's dictionary
    /// ```
    ///
    /// The rule is on the tab-separated fields after the word: when all of the
    /// ones before the last parse as numbers — which is where MFA writes its
    /// per-reading probabilities — the last field is the phone list, otherwise
    /// every remaining field is. A line with no phones, or an empty word, is
    /// skipped rather than stored, so a malformed file cannot make a word
    /// unpronounceable.
    pub fn load_lexicon(&mut self, text: &str) {
        for (word, phones) in parse_dictionary(text) {
            self.asserted.insert(normalize_key(&word), phones);
        }
    }

    /// Add readings the caller *suggests* rather than asserts.
    ///
    /// Same format and same reader as [`Self::load_lexicon`]; the difference is
    /// authority. A suggestion is used where nothing asserted and no curated
    /// acronym entry has an answer, because a dictionary's reading is good
    /// evidence and a person's statement is not evidence but fact.
    pub fn load_suggestions(&mut self, text: &str) {
        for (word, phones) in parse_dictionary(text) {
            self.suggested.insert(normalize_key(&word), phones);
        }
    }

    /// The reading for `word`, or `None` to let the model decide.
    pub fn lookup(&self, word: &str) -> Option<Vec<String>> {
        if let Some(reading) = self.asserted.get(word) {
            return Some(reading.clone());
        }
        if let Some(reading) = self.acronyms.get(word) {
            return Some(reading.clone());
        }
        if let Some(reading) = self.suggested.get(word) {
            return Some(reading.clone());
        }
        if is_initialism(word, &self.letter_names) {
            return Some(self.spell(word));
        }
        None
    }

    /// Why `lookup` answered the way it did, for `--explain`.
    pub fn reason(&self, word: &str) -> Option<&'static str> {
        if self.asserted.contains_key(word) {
            Some("dictionary")
        } else if self.acronyms.contains_key(word) {
            Some("acronym table")
        } else if self.suggested.contains_key(word) {
            Some("suggestion")
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

    /// MFA's dictionary is `word<TAB>4 probabilities<TAB>phones`. Fed directly,
    /// those probabilities used to load as phones, so the dictionary route
    /// confidently answered `kot` with `0.99 0.49 2.75 1.1 k ɔ t̪`.
    #[test]
    fn an_mfa_line_drops_its_probabilities() {
        let mut ex = exceptions();
        ex.load_lexicon("kot\t0.99\t0.49\t2.75\t1.1\tk ɔ t̪\n");
        assert_eq!(ex.reason("kot"), Some("dictionary"));
        assert_eq!(ex.lookup("kot").unwrap(), ["k", "ɔ", "t̪"]);
    }

    /// The drop is conditional: a middle field that is not a number means the
    /// fields are phones, and none of them are discarded.
    #[test]
    fn a_tab_separated_phone_list_keeps_every_field() {
        let mut ex = exceptions();
        ex.load_lexicon("kot\tk\tɔ\tt̪\n");
        assert_eq!(ex.lookup("kot").unwrap(), ["k", "ɔ", "t̪"]);
    }

    /// A line that names no phones is skipped rather than stored, so a
    /// malformed dictionary cannot make a word look answered but empty.
    #[test]
    fn a_phoneless_line_is_skipped() {
        let mut ex = exceptions();
        ex.load_lexicon("kot\npusty\t\n# comment\n");
        assert_eq!(ex.reason("kot"), None, "a bare word is not an entry");
        assert_eq!(ex.reason("pusty"), None, "nor is a word with an empty field");
    }

    /// The levels, weighed in order: an assertion beats the curated table, the
    /// table beats a suggestion, and a suggestion beats the initialism
    /// heuristic. `cv` is in the embedded acronym table; `krk` has no vowel
    /// letter and is not in the table, so the heuristic would spell it out.
    #[test]
    fn the_levels_are_weighed_in_order() {
        let mut ex = exceptions();
        ex.load_suggestions("cv\tS ɛ\nkrk\tK R K\n");
        assert_eq!(
            ex.reason("cv"),
            Some("acronym table"),
            "a curated entry outranks a suggestion"
        );
        assert_eq!(
            ex.reason("krk"),
            Some("suggestion"),
            "and a suggestion outranks the heuristic"
        );

        ex.load_lexicon("cv\tC V\n");
        assert_eq!(
            ex.reason("cv"),
            Some("dictionary"),
            "an assertion outranks everything"
        );
        assert_eq!(ex.lookup("cv").unwrap(), ["C", "V"]);
    }

    /// A key written with a decomposed diacritic matches the word asked with a
    /// composed one. Storing a plain lowercase key while looking up a
    /// normalised one meant such a name matched nothing and fell through to the
    /// model, silently.
    #[test]
    fn a_decomposed_key_still_matches() {
        let mut g2p = crate::G2P::embedded(crate::Mode::Float).expect("embedded model");
        // `ą` as `a` + combining ogonek, which is how some keyboards emit it.
        g2p.load_lexicon("Ba\u{328}k\tSHOULD WIN\n");
        assert_eq!(g2p.explain("Bąk"), Some("dictionary"));
        assert_eq!(g2p.phonemize("Bąk"), ["SHOULD", "WIN"]);
        assert_eq!(g2p.explain("bąk"), Some("dictionary"), "and case does not matter");
    }
    /// The shared reader, which `load_lexicon` is now a thin loop over -- so a
    /// caller scanning or converting a lexicon sees the same parse.
    #[test]
    fn the_reader_yields_pairs_in_order() {
        let entries = parse_dictionary(
            "# a comment\n\nblair\tb l E r\nkot\t0.99\t0.49\t2.75\t1.1\tk ɔ t̪\n\nmorze m ɔ ʐ ɛ\nignored\n",
        );
        assert_eq!(
            entries,
            vec![
                ("blair".to_string(), vec!["b".to_string(), "l".to_string(), "E".to_string(), "r".to_string()]),
                ("kot".to_string(), vec!["k".to_string(), "ɔ".to_string(), "t̪".to_string()]),
                ("morze".to_string(), vec!["m".to_string(), "ɔ".to_string(), "ʐ".to_string(), "ɛ".to_string()]),
            ],
            "comments and blanks skipped, MFA's probabilities dropped, a phoneless line left out"
        );
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
