//! The MFA side of `xlex.canonicalize`, so the tool can be scored exactly the
//! way the Python side scores it.
//!
//! Only the MFA-source rules live here: the CV-source merges exist to compare
//! two dictionaries, and the tool never emits CV notation. The canonical form
//! is a *comparison* device, not a pronunciation -- it lets a prediction that
//! writes `ʎ` where the reference writes `l j` count as the same reading.

const DENTAL: char = '\u{32a}';        // combining bridge below: t̪
const PALATALIZATION: char = '\u{2b2}'; // ʲ

const VOWELS: &[&str] = &["a", "ɛ", "ɔ", "i", "ɨ", "u", "ɔ̃", "ɛ̃"];

/// Single-phone rewrites applied to both sides in the Python version; here
/// only the MFA side ever needs them.
fn rewrite(phone: &str) -> &str {
    match phone {
        "ʎ" => "l",
        "ç" => "x",
        "ɲ" => "j̃",
        "c" => "k",
        "ɟ" => "ɡ",
        "w" => "v",
        other => other,
    }
}

fn is_vowel(phone: &str) -> bool {
    VOWELS.contains(&phone)
}

/// MFA phones -> the shared comparison form, given the word for the
/// context-dependent rules.
pub fn canonicalize_mfa(phones: &[String], word: &str) -> Vec<String> {
    let mut seq: Vec<String> = phones
        .iter()
        .map(|p| {
            let stripped: String = p.chars().filter(|c| *c != DENTAL).collect();
            rewrite(&stripped).to_string()
        })
        .filter(|p| p != "ʔ")
        .map(|p| p.chars().filter(|c| *c != PALATALIZATION).collect::<String>())
        .filter(|p| !p.is_empty())
        .collect();

    // Word-final -ę/-ą: MFA denasalizes, so bridge the last phone back.
    if let Some(last) = seq.last_mut() {
        if word.ends_with('ę') && last == "ɛ" {
            *last = "ɛ̃".to_string();
        } else if word.ends_with('ą') && last == "ɔ" {
            *last = "ɔ̃".to_string();
        }
    }
    let seq = fold_diphthong_glide(seq);
    drop_post_consonant_j(&seq)
}

/// Post-vocalic `i` -> `j` before a consonant or the end (the off-glide).
fn fold_diphthong_glide(seq: Vec<String>) -> Vec<String> {
    let mut out = seq;
    for i in 0..out.len() {
        let post_vocalic = i > 0 && is_vowel(&out[i - 1]);
        let before_consonant = i + 1 == out.len() || !is_vowel(&out[i + 1]);
        if out[i] == "i" && post_vocalic && before_consonant {
            out[i] = "j".to_string();
        }
    }
    out
}

/// Drop `j` between a consonant and a vowel: MFA's `pʲ ɔ` and CV's `p j ɔ` are
/// one sound written twice.
fn drop_post_consonant_j(seq: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(seq.len());
    for (i, phone) in seq.iter().enumerate() {
        let after_consonant = out.last().map(|p| !is_vowel(p)).unwrap_or(false);
        let before_vowel = seq.get(i + 1).map(|p| is_vowel(p)).unwrap_or(false);
        if phone == "j" && after_consonant && before_vowel {
            continue;
        }
        out.push(phone.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(phones: &[&str], word: &str) -> Vec<String> {
        canonicalize_mfa(&phones.iter().map(|p| p.to_string()).collect::<Vec<_>>(), word)
    }

    #[test]
    fn strips_notational_diacritics() {
        assert_eq!(canon(&["t\u{32a}", "a"], "ta"), ["t", "a"]);
        assert_eq!(canon(&["r\u{2b2}", "i"], "ri"), ["r", "i"]);
        assert_eq!(canon(&["k", "\u{294}", "a"], "koa"), ["k", "a"]);
    }

    #[test]
    fn folds_the_li_spelling_difference() {
        // MFA writes `l j`, the model wrote the single phone.
        assert_eq!(canon(&["\u{28e}", "a"], "la"), canon(&["l", "j", "a"], "la"));
    }

    #[test]
    fn drops_the_glide_but_keeps_it_intervocalically() {
        assert_eq!(canon(&["b", "j", "a"], "bja"), ["b", "a"]);
        assert_eq!(canon(&["j", "a", "j", "k", "o"], "jajko"),
                   ["j", "a", "j", "k", "o"]);
    }

    #[test]
    fn bridges_the_word_final_nasal() {
        assert_eq!(canon(&["a", "g", "r", "s", "\u{25b}"], "agresj\u{119}"),
                   ["a", "g", "r", "s", "\u{25b}\u{303}"]);
        assert_eq!(canon(&["a", "l", "\u{25b}"], "ale"), ["a", "l", "\u{25b}"]);
    }

    #[test]
    fn folds_the_diphthong_off_glide() {
        assert_eq!(canon(&["b", "\u{25b}", "i", "k"], "beik"),
                   ["b", "\u{25b}", "j", "k"]);
    }
}
