//! Reading the gold TSV and scoring against it.
//!
//! Mirrors `tiny_g2p/adjudicate.py` exactly, so the binary can print the
//! README's measured *and* gold-corrected numbers without importing torch:
//!
//! * **measured** -- raw equality against MFA's reference, and the edit
//!   distance to the closest one;
//! * **corrected** -- the same prediction judged against the adjudicated gold:
//!   a verdict may accept MFA's reading, widen the candidate set, or excuse a
//!   difference that is only notation. `canon_*` arrive already filtered by
//!   the verdict flags, so the rules here are the same four lines everywhere.

use crate::canon::canonicalize_mfa;

/// The gold TSV header, in order. `parse_gold` insists on exactly this
/// header, and `GoldRow::parse` reads the body fields positionally.
pub const COLUMNS: [&str; 9] = [
    "word", "slice", "refs", "candidates", "canon_mfa", "canon_cv", "mfa_ok",
    "cv_ok", "counted",
];

/// One gold TSV row: a held-out word, MFA's references, the adjudicated
/// candidate readings, and the verdict flags saying which of them count.
#[derive(Debug, Clone)]
pub struct GoldRow {
    /// The orthographic word this row scores.
    pub word: String,
    /// The held-out slice the word belongs to: `agreed`, `mfa-only` or
    /// `disputed`.
    pub slice: String,
    /// MFA's reference readings. These are the measured denominator, and the
    /// only set `judge`'s measured distance is ever taken against.
    pub refs: Vec<Vec<String>>,
    /// Readings a correction may match: MFA's own when the verdict allows
    /// them, plus any the adjudicator added.
    pub candidates: Vec<Vec<String>>,
    /// MFA readings in the shared canonical form. A match here excuses a
    /// difference of notation -- the same reading written another way.
    pub canon_mfa: Vec<Vec<String>>,
    /// CV readings in canonical form. A match here excuses a dictionary
    /// difference the verdict accepts.
    pub canon_cv: Vec<Vec<String>>,
    /// Whether the verdict accepts MFA's own reading as correct.
    pub mfa_ok: bool,
    /// Whether the verdict accepts a prediction matching CV's reading.
    pub cv_ok: bool,
    /// `false` for rows a verdict excludes from the phonetic denominator (a
    /// "convention" call); `eval` skips them.
    pub counted: bool,
}

impl GoldRow {
    fn parse(line: &str) -> Option<GoldRow> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != COLUMNS.len() {
            return None;
        }
        Some(GoldRow {
            word: fields[0].to_string(),
            slice: fields[1].to_string(),
            refs: split_readings(fields[2]),
            candidates: split_readings(fields[3]),
            canon_mfa: split_readings(fields[4]),
            canon_cv: split_readings(fields[5]),
            mfa_ok: fields[6] == "1",
            cv_ok: fields[7] == "1",
            counted: fields[8] == "1",
        })
    }
}

/// `"a b | c d"` -> two readings. Empty means "none".
fn split_readings(field: &str) -> Vec<Vec<String>> {
    field
        .split(" | ")
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.split_whitespace().map(str::to_string).collect())
        .collect()
}

/// Parse the gold TSV, header included.
///
/// Panics if the header is not [`COLUMNS`]: a foreign TSV is a bug, not an
/// empty run. Body lines with the wrong field count are silently skipped.
pub fn parse_gold(text: &str) -> Vec<GoldRow> {
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default();
    let expected: Vec<&str> = header.split('\t').collect();
    assert!(
        expected == COLUMNS.to_vec(),
        "unexpected gold header: {header:?} (regenerate with `tiny-g2p export --gold`)"
    );
    lines.filter_map(GoldRow::parse).collect()
}

/// One word's outcome under both readings of "correct".
#[derive(Debug, Clone, Copy, Default)]
pub struct Outcome {
    /// Raw equality with one of MFA's references.
    pub measured_exact: bool,
    /// Correct once the verdict is applied: a raw hit, an accepted extra
    /// reading, a notation match, or a CV match.
    pub corrected_exact: bool,
    /// Levenshtein distance to the closest MFA reference.
    pub measured_distance: usize,
    /// Length of that closest MFA reference: the measured PER denominator.
    pub measured_reference_length: usize,
    /// Edit distance under the verdict: `0` when corrected, otherwise to the
    /// closest accepted candidate.
    pub distance: usize,
    /// Length of that candidate: the corrected PER denominator.
    pub reference_length: usize,
    /// `None`, `"extra"`, `"cv"`, `"notation"` or `"gold-bad"`.
    pub change: Option<&'static str>,
}

/// Summed outcomes over a set of words. The measured fields count only
/// against MFA's references; the corrected fields follow the verdicts, so the
/// two can disagree on the same prediction.
#[derive(Debug, Default, Clone)]
pub struct Report {
    /// Counted rows folded in; excluded rows never reach here.
    pub rows: usize,
    /// Rows exactly matching an MFA reference.
    pub measured_exact: usize,
    /// Rows correct under their verdict.
    pub corrected_exact: usize,
    /// Summed edit distance to MFA's references.
    pub measured_distance: usize,
    /// Summed length of the closest MFA references.
    pub measured_reference_length: usize,
    /// Summed edit distance under the verdicts.
    pub distance: usize,
    /// Summed length of the accepted candidates.
    pub reference_length: usize,
    /// Rows excused because they matched an adjudicated extra reading.
    pub excused_extra: usize,
    /// Rows excused by a canonical match against CV's reading.
    pub excused_cv: usize,
    /// Rows excused because the reading differed only in notation.
    pub excused_notation: usize,
    /// Rows that matched a reference the verdict judges wrong ("gold-bad").
    pub penalized: usize,
}

impl Report {
    /// `measured_exact / rows`, or zero for an empty report.
    pub fn measured_accuracy(&self) -> f64 {
        ratio(self.measured_exact, self.rows)
    }

    /// `corrected_exact / rows`, or zero for an empty report.
    pub fn corrected_accuracy(&self) -> f64 {
        ratio(self.corrected_exact, self.rows)
    }

    /// Measured phoneme error rate: `measured_distance /
    /// measured_reference_length`, or zero when the denominator is zero.
    pub fn measured_per(&self) -> f64 {
        ratio(self.measured_distance, self.measured_reference_length)
    }

    /// Corrected phoneme error rate: `distance / reference_length`, or zero
    /// when the denominator is zero.
    pub fn per(&self) -> f64 {
        ratio(self.distance, self.reference_length)
    }

    /// Fold one word's [`Outcome`] into the totals, counting it under the
    /// excuse its `change` names.
    pub fn add(&mut self, outcome: Outcome) {
        self.rows += 1;
        self.measured_exact += outcome.measured_exact as usize;
        self.corrected_exact += outcome.corrected_exact as usize;
        self.measured_distance += outcome.measured_distance;
        self.measured_reference_length += outcome.measured_reference_length;
        self.distance += outcome.distance;
        self.reference_length += outcome.reference_length;
        match outcome.change {
            Some("extra") => self.excused_extra += 1,
            Some("cv") => self.excused_cv += 1,
            Some("notation") => self.excused_notation += 1,
            Some("gold-bad") => self.penalized += 1,
            _ => {}
        }
    }

    /// All excused rows -- extra, CV and notation -- but not
    /// [`Report::penalized`], which is a different kind of disagreement.
    pub fn excused(&self) -> usize {
        self.excused_extra + self.excused_cv + self.excused_notation
    }
}

fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Score one prediction against one gold row.
pub fn judge(row: &GoldRow, prediction: &[String]) -> Outcome {
    let measured = row.refs.iter().any(|r| r == prediction);
    let canon = canonicalize_mfa(prediction, &row.word);

    let mut change: Option<&'static str> = None;
    let mut corrected = measured && row.mfa_ok;
    if !corrected {
        if row.candidates.iter().any(|c| c == prediction) && !row.candidates.is_empty() {
            change = Some("extra");
            corrected = true;
        } else if row.canon_mfa.iter().any(|c| c == &canon) && !row.canon_mfa.is_empty() {
            change = Some("notation");
            corrected = true;
        } else if row.canon_cv.iter().any(|c| c == &canon) && !row.canon_cv.is_empty() {
            change = Some("cv");
            corrected = true;
        }
        if measured && !corrected {
            change = Some("gold-bad");
        }
    }

    // Distances: the measured side never moves; the corrected side is exact
    // when corrected, else the closest accepted candidate.
    let (measured_distance, measured_reference_length) =
        closest(prediction, &row.refs);
    let (distance, reference_length) = if corrected {
        (0, prediction.len())
    } else if !row.candidates.is_empty() {
        closest(prediction, &row.candidates)
    } else {
        closest(prediction, &row.refs)
    };
    Outcome {
        measured_exact: measured,
        corrected_exact: corrected,
        measured_distance,
        measured_reference_length,
        distance,
        reference_length,
        change,
    }
}

/// Whether two readings are the same pronunciation, notation excused.
///
/// This is the comparison scoring applies to a prediction against a reference:
/// both sides through the MFA canonical form, which folds what two dictionaries
/// spell differently -- dental marks, affricates split or joined, `ł` as `w` or
/// `v`, an obligatory devoiced coda, a merged nasal vowel. A caller looking for
/// the words a model disagrees with a lexicon needs exactly this, and needs it
/// to be the rule scoring uses, so it is one definition rather than two.
///
/// `word` is needed because some of the canonicalisation is context-dependent
/// -- whether a nasal vowel has an oral half split out depends on what follows.
#[must_use]
pub fn notation_equal(a: &[String], b: &[String], word: &str) -> bool {
    a == b || canonicalize_mfa(a, word) == canonicalize_mfa(b, word)
}

fn closest(prediction: &[String], readings: &[Vec<String>]) -> (usize, usize) {
    let mut best: Option<(usize, usize)> = None;
    for reading in readings {
        let distance = edit_distance(prediction, reading);
        let candidate = (distance, reading.len());
        best = match best {
            Some(current) if current.0 <= distance => Some(current),
            _ => Some(candidate),
        };
    }
    best.unwrap_or((prediction.len(), 0))
}

/// Levenshtein distance over phone tokens (same DP as `pl_g2p.metrics`).
pub fn edit_distance(a: &[String], b: &[String]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, token_a) in a.iter().enumerate() {
        let mut curr = vec![0usize; b.len() + 1];
        curr[0] = i + 1;
        for (j, token_b) in b.iter().enumerate() {
            let cost = usize::from(token_a != token_b);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        prev = curr;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readings(items: &[&str]) -> Vec<Vec<String>> {
        items.iter().map(|s| s.split_whitespace().map(str::to_string).collect()).collect()
    }

    struct Row {
        refs: Vec<Vec<String>>,
        candidates: Vec<Vec<String>>,
        canon_mfa: Vec<Vec<String>>,
        canon_cv: Vec<Vec<String>>,
        mfa_ok: bool,
    }

    fn judge_row(row: &Row, prediction: &str) -> Outcome {
        let gold = GoldRow {
            word: "test".to_string(),
            slice: "disputed".to_string(),
            refs: row.refs.clone(),
            candidates: row.candidates.clone(),
            canon_mfa: row.canon_mfa.clone(),
            canon_cv: row.canon_cv.clone(),
            mfa_ok: row.mfa_ok,
            counted: true,
            cv_ok: !row.canon_cv.is_empty(),
        };
        let phones: Vec<String> =
            prediction.split_whitespace().map(str::to_string).collect();
        judge(&gold, &phones)
    }

    #[test]
    fn a_correct_prediction_is_correct_both_ways() {
        let row = Row {
            refs: readings(&["k \u{254} t"]),
            candidates: readings(&["k \u{254} t"]),
            canon_mfa: readings(&["k \u{254} t"]),
            canon_cv: vec![],
            mfa_ok: true,
        };
        let outcome = judge_row(&row, "k \u{254} t");
        assert!(outcome.measured_exact && outcome.corrected_exact);
        assert_eq!(outcome.change, None);
        assert_eq!(outcome.measured_distance, 0);
    }

    #[test]
    fn a_notation_difference_is_excused_not_measured() {
        let row = Row {
            refs: readings(&["\u{25b} l j a \u{283}"]),
            candidates: readings(&["\u{25b} l j a \u{283}"]),
            canon_mfa: readings(&["\u{25b} l a \u{283}"]),
            canon_cv: vec![],
            mfa_ok: true,
        };
        let outcome = judge_row(&row, "\u{25b} \u{28e} a \u{283}");
        assert!(!outcome.measured_exact, "raw equality must not hold");
        assert!(outcome.corrected_exact);
        assert_eq!(outcome.change, Some("notation"));
    }

    #[test]
    fn a_cv_verdict_penalises_a_gold_faithful_model() {
        let row = Row {
            refs: readings(&["k \u{254} t"]),
            candidates: readings(&["k \u{254} t \u{25b}"]),
            canon_mfa: vec![],
            canon_cv: readings(&["k \u{254} t \u{25b}"]),
            mfa_ok: false,
        };
        let faithful = judge_row(&row, "k \u{254} t");
        assert!(faithful.measured_exact);
        assert!(!faithful.corrected_exact, "MFA is judged wrong here");
        assert_eq!(faithful.change, Some("gold-bad"));
        let right = judge_row(&row, "k \u{254} t \u{25b}");
        assert!(!right.measured_exact && right.corrected_exact);
    }

    #[test]
    fn a_candidate_set_widens_what_counts() {
        let row = Row {
            refs: readings(&["t\u{282} m a"]),
            candidates: readings(&["a x m a"]),
            canon_mfa: vec![],
            canon_cv: vec![],
            mfa_ok: false,
        };
        let outcome = judge_row(&row, "a x m a");
        assert_eq!(outcome.change, Some("extra"));
        assert!(outcome.corrected_exact && !outcome.measured_exact);
    }

    #[test]
    fn a_wrong_prediction_stays_wrong_with_a_distance() {
        let row = Row {
            refs: readings(&["k \u{254} t"]),
            candidates: readings(&["k \u{254} t"]),
            canon_mfa: readings(&["k \u{254} t"]),
            canon_cv: vec![],
            mfa_ok: true,
        };
        let outcome = judge_row(&row, "k \u{254} d");
        assert!(!outcome.corrected_exact);
        assert_eq!(outcome.change, None);
        assert_eq!(outcome.measured_distance, 1);
        assert_eq!(outcome.reference_length, 3);
    }

    #[test]
    fn edit_distance_is_levenshtein() {
        let a = readings(&["a b c"]).remove(0);
        let b = readings(&["a c"]).remove(0);
        assert_eq!(edit_distance(&a, &b), 1);
        assert_eq!(edit_distance(&a, &a), 0);
        assert_eq!(edit_distance(&[], &b), 2);
    }

    #[test]
    #[should_panic(expected = "unexpected gold header")]
    fn a_foreign_tsv_is_rejected() {
        parse_gold("word\treferences\nkot\tk \u{254} t\n");
    }

    /// The comparison a lexicon scan needs: two readings of the same sound in
    /// different notations are equal, and two different sounds are not.
    #[test]
    fn notation_differences_are_excused() {
        let word = "kot";
        let mfa = vec!["k".to_string(), "\u{254}".to_string(), "t\u{32a}".to_string()];
        let plain = vec!["k".to_string(), "\u{254}".to_string(), "t".to_string()];
        assert!(notation_equal(&mfa, &plain, word), "a dental mark is notation");
        assert!(notation_equal(&mfa, &mfa, word), "identity, without canonicalising");

        let other = vec!["p".to_string(), "\u{254}".to_string(), "t".to_string()];
        assert!(!notation_equal(&mfa, &other, word), "k and p are different sounds");
        assert!(!notation_equal(&mfa, &[], word));
    }
}
