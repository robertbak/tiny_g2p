//! `tinyg2p` -- the command-line surface. No Python, no runtime, one binary.
//!
//! The model is the dependency-free library; this wrapper is the one place a
//! crate is allowed in, for argument parsing (clap), so that `--version`,
//! `--opt=value`, generated help and rejected-argument messages come from one
//! place instead of a hand-maintained usage string.
//!
//! ```text
//! tinyg2p predict Szczebrzeszyn            # any number of words, or stdin
//! tinyg2p predict "Wrocław morze może"     # a quoted run of text splits too
//! tinyg2p predict --explain bmw            # say when the exception path fired
//! tinyg2p eval --gold data/test_gold.tsv   # measured vs gold-corrected
//! tinyg2p miss-lexicon --lexicon x.dict --out base.dict   # a dictionary of the misses
//! tinyg2p info                             # what is compiled in
//! ```
//!
//! Input is words, not sentences: whitespace separates them, and each gets one
//! transcription. Nothing is refused for being long -- words past the model's
//! 33-character training window are transcribed like any other, because the
//! alternative (failing the run) is worse than extrapolating.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use tiny_g2p::gold::{self, Report};
use tiny_g2p::{G2P, Mode};

#[derive(Parser)]
#[command(
    name = "tinyg2p",
    version,
    about = "tiny Polish G2P",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(flatten)]
    globals: Globals,
    #[command(subcommand)]
    command: Command,
}

// Options every command shares, because every command loads the model.
//
// A plain comment, not a doc comment: clap takes a doc comment on a flattened
// `Args` struct as user-facing help text, and this is for whoever maintains the
// parser, not for someone reading `--help`.
//
// `global` so they may be written on either side of the subcommand.
#[derive(Args)]
struct Globals {
    /// Replay the quantized artifact instead of the float weights
    #[arg(long, global = true)]
    int8: bool,
    /// Load weights from a file instead of the embedded ones
    #[arg(long, value_name = "FILE", global = true)]
    blob: Option<PathBuf>,
    /// A dictionary you assert: how these words are said, because you say so
    ///
    /// One word per line, as word<TAB>phones -- or MFA's dictionary exactly as
    /// it comes, probabilities and all. Repeatable; later files override
    /// earlier ones. Nothing outranks an assertion: not a suggested reading,
    /// not the built-in acronym table.
    #[arg(long, value_name = "FILE", global = true)]
    lexicon: Vec<PathBuf>,
    /// A dictionary of suggestions: readings for the words the model gets
    /// wrong
    ///
    /// Same format as --lexicon, and weaker: a suggestion is the reading some
    /// dictionary gives, which is good evidence and not the same as you saying
    /// so. It answers only where nothing asserted has an answer. This is what
    /// miss-lexicon writes.
    #[arg(long, value_name = "FILE", global = true)]
    suggest: Vec<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Transcribe words (whitespace separates them; stdin when none are given)
    Predict {
        /// Say which path answered each word
        #[arg(long)]
        explain: bool,
        /// Bypass the exception path (parity testing)
        #[arg(long)]
        model_only: bool,
        /// The words to transcribe
        words: Vec<String>,
    },
    /// Score against an exported gold TSV
    Eval {
        /// The gold TSV to score against
        #[arg(long, value_name = "FILE")]
        gold: PathBuf,
        /// Also write the report as JSON here
        #[arg(long, value_name = "FILE")]
        json: Option<PathBuf>,
    },
    /// Emit a dictionary of the words the model gets wrong
    ///
    /// Scans a lexicon for words the model disagrees with, and writes them as
    /// word<TAB>phones with the reading the model should have given. Load the
    /// result with --suggest.
    ///
    /// Here --lexicon is the reference to scan, not a dictionary that gets
    /// loaded: a word that could answer itself would match, and the scan would
    /// find nothing. --gold adds adjudicated readings, preferred wherever it
    /// has one.
    #[command(name = "miss-lexicon")]
    MissLexicon {
        /// A gold TSV, whose adjudicated readings are preferred wherever it
        /// covers a word
        #[arg(long, value_name = "FILE")]
        gold: Option<PathBuf>,
        /// Write the dictionary here instead of stdout
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
    },
    /// Show what is compiled in
    Info,
}

fn main() -> ExitCode {
    let Cli { globals, command } = Cli::parse();
    match run(&globals, command) {
        Ok(()) => ExitCode::SUCCESS,
        // A rejected argument never reaches here: clap reports it in its own
        // words and exits 2. This is for the failures the commands find.
        Err(message) => {
            eprintln!("tinyg2p: {message}");
            ExitCode::FAILURE
        }
    }
}

fn mode(globals: &Globals) -> Mode {
    if globals.int8 { Mode::Int8 } else { Mode::Float }
}

fn load(globals: &Globals) -> Result<G2P, String> {
    let mut g2p = match &globals.blob {
        Some(path) => {
            let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            G2P::from_bytes(&data, mode(globals)).map_err(|e| e.to_string())?
        }
        None => G2P::embedded(mode(globals)).map_err(|e| e.to_string())?,
    };
    // Assertions, then suggestions. The order between the two loops is not what
    // decides anything -- they are separate levels of authority, applied by
    // strength rather than by when they arrived.
    for path in &globals.lexicon {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        g2p.load_lexicon(&text);
    }
    for path in &globals.suggest {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        g2p.load_suggestions(&text);
    }
    Ok(g2p)
}

fn run(globals: &Globals, command: Command) -> Result<(), String> {
    match command {
        Command::Predict { explain, model_only, words } => {
            predict(globals, explain, model_only, &words)
        }
        Command::Eval { gold, json } => evaluate(globals, &gold, json.as_deref()),
        Command::MissLexicon { gold, out } => miss_lexicon(globals, gold.as_deref(), out.as_deref()),
        Command::Info => info(globals),
    }
}

fn predict(
    globals: &Globals,
    explain: bool,
    model_only: bool,
    words: &[String],
) -> Result<(), String> {
    let g2p = load(globals)?;
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    // Words come from the arguments, or from stdin when there are none. Either
    // way whitespace separates them, so a quoted sentence behaves like a list.
    let input: Vec<String> = if words.is_empty() {
        let mut text = String::new();
        io::stdin().lock().read_to_string(&mut text).map_err(io_err)?;
        text.split_whitespace().map(str::to_string).collect()
    } else {
        words
            .iter()
            .flat_map(|arg| arg.split_whitespace().map(str::to_string).collect::<Vec<_>>())
            .collect()
    };
    for word in &input {
        let phones = if model_only {
            g2p.phonemize_model_only(word)
        } else {
            g2p.phonemize(word)
        };
        if explain {
            let via = g2p.explain(word).unwrap_or("model");
            writeln!(out, "{word}\t{}\t({via})", phones.join(" ")).map_err(io_err)?;
        } else {
            writeln!(out, "{}", phones.join(" ")).map_err(io_err)?;
        }
    }
    out.flush().map_err(io_err)
}

fn io_err(error: io::Error) -> String {
    error.to_string()
}

fn evaluate(globals: &Globals, path: &Path, json: Option<&Path>) -> Result<(), String> {
    let mut text = String::new();
    std::fs::File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .read_to_string(&mut text)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let rows = gold::parse_gold(&text);
    let g2p = load(globals)?;

    let mut overall = Report::default();
    let mut by_slice: BTreeMap<String, Report> = BTreeMap::new();
    let mut errors: Vec<(String, Vec<String>)> = Vec::new();
    for row in &rows {
        if !row.counted {
            continue; // out-of-scope items leave the phonetic denominator
        }
        let prediction = g2p.phonemize(&row.word);
        let outcome = gold::judge(row, &prediction);
        if !outcome.corrected_exact {
            errors.push((row.word.clone(), prediction.clone()));
        }
        by_slice.entry(row.slice.clone()).or_default().add(outcome);
        overall.add(outcome);
    }

    let mode = match g2p.mode() {
        Mode::Float => "float",
        Mode::Int8 => "int8",
    };
    println!("weights: {mode}   words: {}   excluded: {}",
             overall.rows, rows.len() - overall.rows);
    println!("\n{:<10} {:>6} {:>10} {:>11} {:>8} {:>8}",
             "slice", "n", "measured", "corrected", "PER", "PERcorr");
    for (name, report) in &by_slice {
        println!(
            "{:<10} {:>6} {:>9.2}% {:>10.2}% {:>7.2}% {:>7.2}%",
            name, report.rows,
            report.measured_accuracy() * 100.0,
            report.corrected_accuracy() * 100.0,
            report.measured_per() * 100.0,
            report.per() * 100.0,
        );
    }
    println!(
        "\nmeasured  {:>6.2}% ({}/{})  PER {:.2}% ({}/{})",
        overall.measured_accuracy() * 100.0,
        overall.measured_exact, overall.rows,
        overall.measured_per() * 100.0,
        overall.measured_distance, overall.measured_reference_length,
    );
    println!(
        "corrected {:>6.2}% ({}/{})  PER {:.2}% ({}/{})",
        overall.corrected_accuracy() * 100.0,
        overall.corrected_exact, overall.rows,
        overall.per() * 100.0,
        overall.distance, overall.reference_length,
    );
    println!(
        "errors: {} measured -> {} gold-attributable (extra {}, cv {}, notation {}) \
         -> {} genuine (plus {} matching a reference judged wrong)",
        overall.rows - overall.measured_exact,
        overall.excused(),
        overall.excused_extra, overall.excused_cv, overall.excused_notation,
        overall.rows - overall.corrected_exact,
        overall.penalized,
    );

    if let Some(path) = json {
        let mut slices = String::new();
        for (name, report) in &by_slice {
            slices.push_str(&format!(
                "    {:?}: {{\"n\": {}, \"measured\": {:.6}, \"corrected\": {:.6}, \
                 \"per\": {:.6}, \"per_corrected\": {:.6}}},\n",
                name, report.rows, report.measured_accuracy(),
                report.corrected_accuracy(), report.measured_per(), report.per(),
            ));
        }
        let payload = format!(
            "{{\n  \"mode\": {mode:?},\n  \"rows\": {},\n  \"excluded\": {},\n  \
             \"measured_exact\": {},\n  \"corrected_exact\": {},\n  \
             \"measured_accuracy\": {:.6},\n  \"corrected_accuracy\": {:.6},\n  \
             \"measured_per\": {:.6},\n  \"per\": {:.6},\n  \
             \"excused\": {{\"extra\": {}, \"cv\": {}, \"notation\": {}}},\n  \
             \"penalized\": {},\n  \"slices\": {{\n{slices}  }}\n}}\n",
            overall.rows,
            rows.len() - overall.rows,
            overall.measured_exact,
            overall.corrected_exact,
            overall.measured_accuracy(),
            overall.corrected_accuracy(),
            overall.measured_per(),
            overall.per(),
            overall.excused_extra,
            overall.excused_cv,
            overall.excused_notation,
            overall.penalized,
        );
        std::fs::write(path, payload).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("\nwrote {}", path.display());
    }

    let limit: usize = 10;
    if !errors.is_empty() {
        println!("\nfirst {limit} genuine errors (model output):");
        for (word, prediction) in errors.iter().take(limit) {
            println!("  {word:<20} {}", prediction.join(" "));
        }
    }
    Ok(())
}

/// Emit a dictionary of the words the model gets wrong, so the exception path
/// can fix them without loading a whole lexicon.
///
/// Two sources, because they answer different questions. `--gold` scans the
/// adjudicated held-out split — a few thousand words, with corrected readings
/// where MFA is known to be wrong — which is the curated seed. `--lexicon`
/// scans every word a lexicon knows, so nothing is left unscanned. Given both,
/// the lexicon gives the coverage and the gold supplies the reading to prefer
/// wherever it has adjudicated one.
///
/// The scanned lexicon is *reference* data and is deliberately **not** loaded
/// into the exception path: a word that can answer itself from the dictionary
/// being built would match, and the scan would report nothing. That is also why
/// this does not go through `load()`.
fn miss_lexicon(
    globals: &Globals,
    gold_path: Option<&Path>,
    out_path: Option<&Path>,
) -> Result<(), String> {
    if globals.lexicon.is_empty() && gold_path.is_none() {
        return Err("miss-lexicon needs --lexicon FILE, --gold FILE, or both".to_string());
    }

    let g2p = match &globals.blob {
        Some(path) => {
            let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            G2P::from_bytes(&data, mode(globals)).map_err(|e| e.to_string())?
        }
        None => G2P::embedded(mode(globals)).map_err(|e| e.to_string())?,
    };

    // Adjudicated readings win over the lexicon's own, for the words the gold
    // covers: on 45 of its 148 curated words MFA is known to be wrong.
    let mut preferred: BTreeMap<String, Vec<String>> = BTreeMap::new();
    // Words the gold marks wrong and records no alternative for. A dictionary
    // entry would be a confident guess, so they get none.
    let mut unfixable: BTreeSet<String> = BTreeSet::new();
    let mut gold_rows = Vec::new();
    if let Some(path) = gold_path {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for row in gold::parse_gold(&text) {
            let word = row.word.to_lowercase();
            match accepted_reading(&row) {
                Some(reading) => {
                    preferred.insert(word, reading.clone());
                }
                None => {
                    unfixable.insert(word);
                }
            }
            gold_rows.push(row);
        }
    }

    let mut misses: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut scanned = 0usize;
    let mut unfixable_seen = 0usize;
    let source;

    if !globals.lexicon.is_empty() {
        let mut sources: Vec<String> = Vec::new();
        for path in &globals.lexicon {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            sources.push(path.display().to_string());
            for (word, reading) in tiny_g2p::parse_dictionary(&text) {
                // Special tokens (`<unk>`, `[bracketed]`) are not words and
                // cannot be phonemised; nothing wants them in a dictionary.
                if word.contains(['<', '[', ' ']) {
                    continue;
                }
                scanned += 1;
                let heard = g2p.phonemize(&word);
                if !gold::notation_equal(&reading, &heard, &word) {
                    if unfixable.contains(&word) {
                        unfixable_seen += 1;
                        continue;
                    }
                    let emit = preferred.get(&word).cloned().unwrap_or(reading);
                    misses.insert(word, emit);
                }
            }
        }
        source = sources.join(", ");
    } else {
        source = gold_path.map(|p| p.display().to_string()).unwrap_or_default();
        for row in &gold_rows {
            // An uncounted row is not in the phonetic denominator -- a
            // convention call -- so it needs no entry.
            if !row.counted {
                continue;
            }
            scanned += 1;
            let heard = g2p.phonemize(&row.word);
            if !gold::judge(row, &heard).corrected_exact {
                match accepted_reading(row) {
                    Some(reading) => {
                        misses.insert(row.word.to_lowercase(), reading.clone());
                    }
                    None => unfixable_seen += 1,
                }
            }
        }
    }

    let mut out = String::new();
    out.push_str(&format!(
        "# tiny-g2p exception dictionary: {} of {scanned} words in {source} \
         disagreed with the model, with the reading the model should have given.\n",
        misses.len()
    ));
    if unfixable_seen > 0 {
        out.push_str(&format!(
            "# {unfixable_seen} more disagreed where the gold marks the reference wrong and \
             records no alternative; a dictionary entry would be a guess, so there is none.\n"
        ));
    }
    out.push_str("# word<TAB>phones; pass with --lexicon, and see `tinyg2p miss-lexicon --help`.\n");
    for (word, phones) in &misses {
        out.push_str(&format!("{word}\t{}\n", phones.join(" ")));
    }

    match out_path {
        Some(path) => {
            std::fs::write(path, &out).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!("{} misses in {scanned} words -> {}", misses.len(), path.display());
        }
        None => print!("{out}"),
    }
    Ok(())
}

/// The reading the gold would have the model give, if it has one to offer.
///
/// `candidates` are the accepted readings, already adjudicated; MFA's own refs
/// count only when the verdict accepts them. A row with neither is one the gold
/// marks *wrong* and records no alternative for — so there is nothing to emit,
/// and emitting MFA's reading anyway would make the tool confidently repeat the
/// error the gold just flagged.
fn accepted_reading(row: &gold::GoldRow) -> Option<&Vec<String>> {
    row.candidates
        .first()
        .or_else(|| if row.mfa_ok { row.refs.first() } else { None })
}

fn info(globals: &Globals) -> Result<(), String> {
    let g2p = load(globals)?;
    let blob = g2p.blob();
    println!("tiny-g2p (Rust)");
    println!("  mode            {:?}", g2p.mode());
    println!("  characters      {} in, {} phones out", blob.n_src, blob.n_tgt);
    println!("  taps            {} over a {}-character training window",
             blob.taps.len(), g2p.window_len());
    println!("  layers          4 quantized/float linears + residual");
    println!("  exception table {} acronyms, {} letter names",
             blob.acronyms.len(), blob.letter_names.len());
    println!("  weights         {} KiB embedded", tiny_g2p::EMBEDDED_WEIGHTS.len() / 1024);
    println!("  note            inference never fails; words past the window \
             still transcribe");
    Ok(())
}
