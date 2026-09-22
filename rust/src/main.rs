//! `tinyg2p` -- the command-line surface. No deps, no runtime, one binary.
//!
//! ```text
//! tinyg2p predict Szczebrzeszyn            # any number of words, or stdin
//! tinyg2p predict "Wrocław morze może"     # a quoted run of text splits too
//! tinyg2p predict --explain bmw            # say when the exception path fired
//! tinyg2p eval --gold data/test_gold.tsv   # measured vs gold-corrected, no Python
//! tinyg2p info                             # what is compiled in
//! ```
//!
//! Input is words, not sentences: whitespace separates them, and each gets one
//! transcription. Nothing is refused for being long -- words past the model's
//! 33-character training window are transcribed like any other, because the
//! alternative (failing the run) is worse than extrapolating.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use tiny_g2p::gold::{self, Report};
use tiny_g2p::{G2P, Mode};

const USAGE: &str = "\
tinyg2p -- tiny Polish G2P with no Python in the loop

USAGE:
    tinyg2p predict [words...]      transcribe (stdin when no words are given)
    tinyg2p eval --gold FILE        score against an exported gold TSV
    tinyg2p info                    show what is compiled in

OPTIONS:
    --int8              replay the quantized artifact instead of the float weights
    --lexicon FILE      dictionary for borrowings and names: word<TAB>phones,
                        or MFA's dictionary exactly as it comes
    --blob FILE         load weights from a file instead of the embedded ones
    --explain           (predict) say which path answered each word
    --model-only        (predict) bypass the exception path (parity testing)
    --json FILE         (eval) also write the report as JSON
    -h, --help          this text
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("tinyg2p: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    mode: Mode,
    blob: Option<String>,
    lexicon: Option<String>,
    explain: bool,
    model_only: bool,
    json: Option<String>,
    gold: Option<String>,
    positional: Vec<String>,
}

fn parse(args: &[String]) -> Result<(String, Options), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        std::process::exit(0);
    }
    let command = args[0].clone();
    let mut opts = Options {
        mode: Mode::Float,
        blob: None,
        lexicon: None,
        explain: false,
        model_only: false,
        json: None,
        gold: None,
        positional: Vec::new(),
    };
    let mut i = 1;
    while i < args.len() {
        let arg = args[i].as_str();
        let mut take = |what: &str| -> Result<String, String> {
            i += 1;
            args.get(i).cloned().ok_or_else(|| format!("{what} needs a value"))
        };
        match arg {
            "--int8" => opts.mode = Mode::Int8,
            "--blob" => opts.blob = Some(take("--blob")?),
            "--lexicon" => opts.lexicon = Some(take("--lexicon")?),
            "--gold" => opts.gold = Some(take("--gold")?),
            "--json" => opts.json = Some(take("--json")?),
            "--explain" => opts.explain = true,
            "--model-only" => opts.model_only = true,
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => opts.positional.push(other.to_string()),
        }
        i += 1;
    }
    Ok((command, opts))
}

fn load(opts: &Options) -> Result<G2P, String> {
    let mut g2p = match &opts.blob {
        Some(path) => {
            let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            G2P::from_bytes(&data, opts.mode).map_err(|e| e.to_string())?
        }
        None => G2P::embedded(opts.mode).map_err(|e| e.to_string())?,
    };
    if let Some(path) = &opts.lexicon {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        g2p.load_lexicon(&text);
    }
    Ok(g2p)
}

fn run(args: &[String]) -> Result<(), String> {
    let (command, opts) = parse(args)?;
    match command.as_str() {
        "predict" => predict(&opts),
        "eval" => evaluate(&opts),
        "info" => info(&opts),
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    }
}

fn predict(opts: &Options) -> Result<(), String> {
    let g2p = load(opts)?;
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    // Words come from the arguments, or from stdin when there are none. Either
    // way whitespace separates them, so a quoted sentence behaves like a list.
    let input: Vec<String> = if opts.positional.is_empty() {
        let mut text = String::new();
        io::stdin().lock().read_to_string(&mut text).map_err(io_err)?;
        text.split_whitespace().map(str::to_string).collect()
    } else {
        opts.positional.iter().flat_map(|arg| {
            arg.split_whitespace().map(str::to_string).collect::<Vec<_>>()
        }).collect()
    };
    for word in &input {
        let phones = if opts.model_only {
            g2p.phonemize_model_only(word)
        } else {
            g2p.phonemize(word)
        };
        if opts.explain {
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

fn evaluate(opts: &Options) -> Result<(), String> {
    let path = opts
        .gold
        .as_ref()
        .ok_or("eval needs --gold FILE (write one with `tiny-g2p export --gold`)")?;
    let mut text = String::new();
    std::fs::File::open(path)
        .map_err(|e| format!("{path}: {e}"))?
        .read_to_string(&mut text)
        .map_err(|e| format!("{path}: {e}"))?;
    let rows = gold::parse_gold(&text);
    let g2p = load(opts)?;

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

    if let Some(path) = &opts.json {
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
        std::fs::write(path, payload).map_err(|e| format!("{path}: {e}"))?;
        println!("\nwrote {path}");
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

fn info(opts: &Options) -> Result<(), String> {
    let g2p = load(opts)?;
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
