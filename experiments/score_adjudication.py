#!/usr/bin/env python
"""Score the adjudication set: measured vs gold-corrected accuracy.

Joins ``data/adjudication.tsv`` (built by ``build_adjudication.py``) with the
curated ``data/adjudication_verdicts.json``. Reports, per test slice and per
model:

* **measured** -- word accuracy against MFA's references, the number the
  README quotes;
* **corrected** -- the same predictions judged against the adjudicated gold:
  MFA errors removed and defensible variants accepted, so it answers "how
  much of the error is actually the model's?".

``--draft`` writes a verdicts skeleton with the per-row evidence (both
lexicons, both predictions, what each prediction matches) for curation;
existing entries are preserved. ``--show VERDICT`` lists the rows under one
verdict and is how the curated file is reviewed.

    uv run python experiments/score_adjudication.py
    uv run python experiments/score_adjudication.py --draft
    uv run python experiments/score_adjudication.py --show cv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_g2p.adjudicate import (  # noqa: E402
    VERDICTS,
    Verdict,
    judge,
    summarize,
)
from tiny_g2p.xlex import canonicalize  # noqa: E402

TSV = ROOT / "data" / "adjudication.tsv"
VERDICTS_JSON = ROOT / "data" / "adjudication_verdicts.json"
RESCORE_JSON = ROOT / "runs" / "xlex_rescore.json"
#: Scratch copy for review: every unsettled row with its evidence. Not the
#: curated file -- curation edits the real one (and only the rows that are
#: not a plain ``mfa``).
DRAFT_JSON = ROOT / "data" / "adjudication_verdicts.draft.json"
MODELS = ("tiny", "trans")

NOTES = [
    "Adjudication verdicts: what the MFA reference gets right, word by word.",
    "Every word here is one the evidence did not settle: the lexicons disagree",
    "(slice 'disputed'), a model missed it (slice 'mfa-only'), it is polyphonic,",
    "or it is spelled letter-by-letter. Verdicts are about the GOLD, not the",
    "models -- see src/tiny_g2p/adjudicate.py for the semantics.",
    "mfa = MFA's reading is correct (default: unlisted words are mfa)",
    "both = MFA and the listed/CV reading are the same sound, transcribed",
    "       differently (geminates, surface voicing, nasal place/split)",
    "open = genuinely ambiguous spelling (loans, names): MFA's reading and",
    "       every listed 'correct' reading are acceptable",
    "cv = MFA is wrong, CV is right",
    "none = neither lexicon is right; 'correct' lists the reading(s)",
    "convention = not a fair item: acronyms, letter names, typos/respellings",
    "'correct' optionally widens any verdict with MFA-notation readings;",
    "'note' says why in one line. Regenerate the skeleton with --draft.",
]


def read_rows() -> list[dict]:
    if not TSV.is_file():
        raise SystemExit(f"missing {TSV}; run experiments/build_adjudication.py")
    with TSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for row in rows:
        row["mfa_readings"] = [tuple(r.split()) for r in row["mfa"].split(" | ") if r]
        row["cv_readings"] = [tuple(r.split()) for r in row["cv"].split(" | ") if r]
    return rows


def load_verdicts(path: Path = VERDICTS_JSON) -> dict[str, Verdict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {word: Verdict.from_json(entry)
            for word, entry in payload.get("verdicts", {}).items()}


def _match_note(row: dict, model: str) -> str:
    """What one model's prediction lines up with, for the draft evidence."""
    pred = row[model].split()
    if not pred:
        return "none"
    hits = []
    if any(tuple(pred) == r for r in row["mfa_readings"]):
        hits.append("mfa")
    canon = canonicalize(pred, source="mfa", word=row["word"])
    if row["cv_readings"] and canon in {
            canonicalize(list(r), source="cv", word=row["word"])
            for r in row["cv_readings"]}:
        hits.append("cv")
    return "+".join(hits) if hits else "neither"


def draft_rows(rows: list[dict], existing: dict[str, Verdict],
               out: Path) -> None:
    """Write the review skeleton: unsettled rows + evidence, curated kept.

    Rows already carrying a curated verdict are copied verbatim; the rest
    get the per-row evidence and a placeholder verdict for the curator.
    """
    verdicts: dict[str, dict] = {}
    pending = 0
    for row in rows:
        word = row["word"]
        current = existing.get(word)
        if current is not None:
            entry: dict = {"verdict": current.verdict, "note": current.note}
            if current.correct:
                entry["correct"] = [" ".join(r) for r in current.correct]
            verdicts[word] = entry
            continue
        pending += 1
        verdicts[word] = {
            "verdict": "mfa",
            "note": "",
            "evidence": {
                "slice": row["slice"], "sources": row["sources"],
                "mfa": row["mfa"], "cv": row["cv"],
                "tiny": row["tiny"], "tiny_matches": _match_note(row, "tiny"),
                "trans": row["trans"], "trans_matches": _match_note(row, "trans"),
            },
        }
    out.write_text(json.dumps({"_notes": NOTES, "verdicts": verdicts},
                              ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} ({len(verdicts)} rows, {pending} awaiting a verdict)")


def report(rows: list[dict], verdicts: dict[str, Verdict],
           json_path: Path | None) -> int:
    slices: dict[str, list[dict]] = {}
    for row in rows:
        slices.setdefault(row["slice"], []).append(row)

    out: dict = {"slices": {}, "gold": {}, "uncertain": []}
    for name, subset in sorted(slices.items()):
        out["slices"][name] = {}
        for model in MODELS:
            judgments = [judge(row["word"], row[model].split(),
                               row["mfa_readings"], row["cv_readings"],
                               verdicts.get(row["word"], Verdict()))
                         for row in subset]
            out["slices"][name][model] = summarize(judgments).as_dict()

    # Gold quality: how often is the MFA reference itself in question?
    gold: Counter[str] = Counter()
    for row in rows:
        gold[verdicts.get(row["word"], Verdict()).verdict] += 1
    out["gold"] = dict(gold)

    # Rows that still need a human: uncurated disputed/none-lexicon rows.
    out["uncertain"] = sorted(
        row["word"] for row in rows
        if row["word"] not in verdicts and row["slice"] == "disputed")

    print(f"rows: {len(rows)}  "
          f"curated: {sum(1 for r in rows if r['word'] in verdicts)}")
    print("gold verdicts: " + ", ".join(
        f"{name} {gold.get(name, 0)}" for name in VERDICTS))
    print(f"\n{'slice':10} {'n':>5} {'model':>6} {'measured':>9} "
          f"{'corrected':>10} {'excused':>8} {'PER':>7} {'PERcorr':>8}")
    for name in sorted(slices):
        for model in MODELS:
            detail = out["slices"][name][model]
            excused = sum(detail["excused"].values()) + detail["penalized"]
            print(f"{name:10} {detail['n']:5d} {model:>6} "
                  f"{detail['measured_accuracy'] * 100:8.2f}% "
                  f"{detail['corrected_accuracy'] * 100:9.2f}% "
                  f"{excused:8d} {detail['measured_per'] * 100:6.2f}% "
                  f"{detail['per'] * 100:7.2f}%")

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\nwrote {json_path}")
    return 0


def audit(rows: list[dict], verdicts: dict[str, Verdict]) -> int:
    """Flag rows that look like they need a verdict but have none.

    The default (plain ``mfa``) is right for the majority of the set -- a
    model simply got the word wrong. What it is not right for is a contested
    word, or a word where a model matched CV's reading on a row the two
    lexicons do not already agree on. Uncurated *agreed* rows are skipped:
    they are settled lexically, and the model's notation is excused anyway.
    """
    suspects = []
    for row in rows:
        if row["word"] in verdicts or row["slice"] == "agreed":
            continue
        matches = {_match_note(row, model) for model in MODELS}
        if row["slice"] == "disputed" or "cv" in matches:
            suspects.append((row["word"], row["slice"], ",".join(sorted(matches))))
    for word, slice_name, matches in suspects:
        print(f"  {word:24} {slice_name:9} predictions match: {matches}")
    print(f"{len(suspects)} uncurated rows worth a look "
          f"(of {sum(1 for r in rows if r['word'] not in verdicts)} uncurated)")
    return 0


def show(rows: list[dict], verdicts: dict[str, Verdict], wanted: str) -> int:
    shown = 0
    for row in rows:
        verdict = verdicts.get(row["word"], Verdict()).verdict
        if verdict != wanted:
            continue
        shown += 1
        note = verdicts.get(row["word"], Verdict()).note
        print(f"{row['word']:24} {row['slice']:9} "
              f"mfa: {row['mfa'] or '-':38} cv: {row['cv'] or '-'}")
        for model in MODELS:
            print(f"{'':24} {model:>6}: {row[model]:38} "
                  f"[{_match_note(row, model)}]")
        if note:
            print(f"{'':24} note: {note}")
    print(f"{shown} rows with verdict {wanted!r}")
    return 0


def full_test(verdicts: dict[str, Verdict], json_path: Path | None) -> int:
    """Measured vs gold-corrected accuracy over the whole held-out test set.

    The adjudication TSV only holds the words that went wrong; every other
    test word is a plain ``mfa`` (both lexicons agree or CV is silent and
    nothing suggests an MFA bug), so the correction it reports is a floor.

    Predictions come from ``xlex_rescore.py``; references and CV readings
    from the lexicons, exactly as the models were scored.
    """
    if not RESCORE_JSON.is_file():
        print(f"no {RESCORE_JSON}; run experiments/xlex_rescore.py --json "
              "runs/xlex_rescore.json first", file=sys.stderr)
        return 2
    from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402
    from pl_g2p.metrics import score as pl_score  # noqa: E402

    from tiny_g2p.data import (  # noqa: E402
        make_dataset, references_by_word, resolve_lexicon)
    from tiny_g2p.xlex import parse_cv  # noqa: E402

    payload = json.loads(RESCORE_JSON.read_text(encoding="utf-8"))
    preds = payload["preds"]
    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    refs = references_by_word(make_dataset(lexicon, align_train=False).test)
    cv = parse_cv()
    words = sorted(refs)

    print(f"\nfull test set ({len(words)} words), unlisted = plain mfa:")
    report: dict = {"full": {}, "rows": len(words)}
    for model in MODELS:
        measured = pl_score([preds[w][model] for w in words],
                            [refs[w] for w in words], words=words)
        judgments = [judge(word, preds[word][model], refs[word],
                           cv.entries.get(word, []),
                           verdicts.get(word, Verdict()))
                     for word in words]
        summary = summarize(judgments)
        report["full"][model] = summary.as_dict() | {
            "all_n": measured.n, "all_exact": measured.exact,
            "all_distance": measured.distance,
            "all_reference_length": measured.reference_length}
        excused = sum(summary.excused.values())
        print(f"  {model:>6}: measured {measured.word_accuracy * 100:6.2f}% "
              f"({measured.exact}/{measured.n})  "
              f"PER {measured.per * 100:.2f}% "
              f"({measured.distance}/{measured.reference_length})")
        print(f"          corrected on the {summary.n} phonetic items "
              f"({len(words) - summary.n} excluded): "
              f"{summary.corrected_accuracy * 100:6.2f}% "
              f"({summary.corrected_exact}/{summary.n})  "
              f"PER {summary.per * 100:.2f}% "
              f"({summary.distance}/{summary.reference_length})")
        print(f"          measured errors {measured.n - measured.exact} -> "
              f"gold-attributable {excused} {summary.excused} -> "
              f"genuine {summary.n - summary.corrected_exact} "
              f"(plus {summary.penalized} matching a reference judged wrong)")
    report["excluded"] = {w: verdicts[w].verdict for w in sorted(verdicts)
                          if not verdicts[w].counted}

    if json_path:
        merged = json.loads(json_path.read_text(encoding="utf-8")) \
            if json_path.is_file() else {}
        merged.update(report)
        json_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\nwrote {json_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--full", action="store_true",
                    help="also score the whole test set (needs xlex_rescore.json)")
    ap.add_argument("--draft", action="store_true",
                    help="write/refresh the verdicts skeleton")
    ap.add_argument("--show", choices=VERDICTS, default=None,
                    help="list the rows under one verdict")
    ap.add_argument("--audit", action="store_true",
                    help="list uncurated rows whose predictions look suspicious")
    args = ap.parse_args()

    rows = read_rows()
    verdicts = load_verdicts()
    if args.draft:
        draft_rows(rows, verdicts, DRAFT_JSON)
        return 0
    if args.show:
        return show(rows, verdicts, args.show)
    if args.audit:
        return audit(rows, verdicts)
    report(rows, verdicts, args.json)
    if args.full:
        return full_test(verdicts, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
