#!/usr/bin/env python
"""Build the adjudication set: every word where the gold is in question.

Rows come from four sources:

* disputed test words (lexicons disagree -- all 128);
* error union on agreed/mfa-only slices (a model is wrong where MFA may be
  the one at fault);
* polyphonic test words (ambiguity is inherent);
* unalignable test words (acronyms/letter-names: convention, not phonetics).

Writes ``data/adjudication.tsv`` (predictions, no verdicts). Verdicts live
in ``data/adjudication_verdicts.json`` (curated by hand) and
``experiments/score_adjudication.py`` joins the two.

    uv run python experiments/build_adjudication.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402

from tiny_g2p.align import align_entry  # noqa: E402
from tiny_g2p.data import (  # noqa: E402
    make_dataset,
    references_by_word,
    resolve_lexicon,
)
from tiny_g2p.xlex import compare, parse_cv  # noqa: E402

RESCORE = ROOT / "runs" / "xlex_rescore.json"
OUT = ROOT / "data" / "adjudication.tsv"


def main() -> int:
    if not RESCORE.is_file():
        print("run experiments/xlex_rescore.py --json runs/xlex_rescore.json "
              "first", file=sys.stderr)
        return 2
    rescore = json.loads(RESCORE.read_text(encoding="utf-8"))
    preds = rescore["preds"]

    cv = parse_cv()
    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    mfa_entries: dict[str, list[tuple[str, ...]]] = {}
    for entry in lexicon.entries:
        mfa_entries.setdefault(entry.word, []).append(entry.phonemes)
    disputed = set(compare(mfa_entries, cv).disagreed)

    data = make_dataset(lexicon, align_train=False)
    refs = references_by_word(data.test)
    polyphonic = {w for w, rs in refs.items() if len(rs) > 1}

    rows: dict[str, dict] = {}
    sources: dict[str, set[str]] = {}

    def add(word: str, source: str) -> None:
        sources.setdefault(word, set()).add(source)
        if word in rows:
            return
        tiny = preds[word]["tiny"]
        trans = preds[word]["trans"]
        rows[word] = {
            "word": word,
            "slice": ("disputed" if word in disputed
                      else "agreed" if word in cv.words else "mfa-only"),
            "mfa": [" ".join(r) for r in refs[word]],
            "cv": [" ".join(r) for r in cv.entries.get(word, [])],
            "tiny": " ".join(tiny),
            "tiny_ok_mfa": any(list(tiny) == list(r) for r in refs[word]),
            "trans": " ".join(trans),
            "trans_ok_mfa": any(list(trans) == list(r)
                                for r in refs[word]),
        }

    for word in sorted(disputed & set(refs)):
        add(word, "disputed")
    for word in sorted(refs):
        tiny_ok = any(preds[word]["tiny"] == list(r) for r in refs[word])
        trans_ok = any(preds[word]["trans"] == list(r) for r in refs[word])
        if not tiny_ok or not trans_ok:
            add(word, "error-union")
    for word in sorted(polyphonic):
        add(word, "polyphonic")
    for word in sorted(refs):
        if all(align_entry(word, r) is None for r in refs[word]):
            add(word, "unalignable")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["word", "slice", "sources", "mfa", "cv", "tiny",
                            "tiny_ok_mfa", "trans", "trans_ok_mfa"],
            delimiter="\t")
        writer.writeheader()
        for word in sorted(rows):
            row = rows[word]
            row["sources"] = "+".join(sorted(sources[word]))
            row["mfa"] = " | ".join(row["mfa"])
            row["cv"] = " | ".join(row["cv"])
            writer.writerow(row)

    by_source: dict[str, int] = {}
    for word, srcs in sources.items():
        for source in srcs:
            by_source[source] = by_source.get(source, 0) + 1
    print(f"adjudication rows: {len(rows)}")
    for source, count in sorted(by_source.items()):
        print(f"  {source:12} {count}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
