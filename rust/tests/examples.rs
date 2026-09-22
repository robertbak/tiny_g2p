//! The examples are a snapshot, and this is what keeps them true.
//!
//! `examples/expected.txt` is the binary's output on real speech -- seven BIGOS
//! validation references and one line from the sibling project's test script.
//! A change to the model, the lexicon reader or the exception path moves that
//! output, and this makes it move on purpose rather than quietly.
//!
//! Regenerate with the command in `examples/README.md`, after checking that the
//! diff is what you meant.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples")
}

/// Run the binary over the examples, the way `examples/README.md` documents.
fn run_examples(extra: &[&str]) -> String {
    let stdin =
        std::fs::File::open(examples().join("utterances.txt")).expect("examples/utterances.txt");
    let run = Command::new(env!("CARGO_BIN_EXE_tinyg2p"))
        .args(["predict", "--explain"])
        .args(extra)
        .args(["--lexicon"])
        .arg(examples().join("names.dict"))
        .stdin(Stdio::from(stdin))
        .output()
        .expect("running the binary");

    assert!(
        run.status.success(),
        "the binary failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

#[test]
fn the_examples_reproduce_their_expected_output() {
    let expected =
        std::fs::read_to_string(examples().join("expected.txt")).expect("examples/expected.txt");
    assert_eq!(
        run_examples(&[]),
        expected,
        "the examples moved. Regenerate examples/expected.txt if that was intended"
    );
}

/// The one line the exception path answers is still answered by it.
///
/// Asserted separately because the snapshot would pass just as happily if the
/// surname were phonemised by the model and `expected.txt` regenerated to match
/// -- which is the failure that matters here, since the dictionary is the only
/// thing in the examples that proves the exception path runs at all.
#[test]
fn the_surname_is_still_answered_by_the_asserted_dictionary() {
    let stdout = run_examples(&[]);
    let surname = stdout
        .lines()
        .find(|line| line.starts_with("Dziemianowicz-Bąk\t"))
        .expect("the surname should be in the output");

    assert!(
        surname.ends_with("(dictionary)"),
        "the surname came from somewhere other than the dictionary: {surname}"
    );
    // The asserted reading has no vowel between "cz" and "Bąk"; the model's
    // guess inserts one, reading the hyphen as a break.
    assert!(
        surname.contains("tʂ b ɔ ŋ k"),
        "expected the asserted reading, which runs 'cz' straight into 'Bąk': {surname}"
    );
}
