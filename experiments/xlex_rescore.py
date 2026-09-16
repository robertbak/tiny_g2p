#!/usr/bin/env python
"""Re-score both G2P models on cross-lexicon gold slices.

Test words split three ways by the MFA<->CV comparison:

* **agreed**    both lexicons share a reading -- the trusted gold;
* **disputed**  the lexicons genuinely differ -- models are diagnosed
  (sides with MFA / CV / neither), not scored;
* **mfa-only**  outside CV's vocabulary -- MFA is the only gold available.

Both models decode in MFA notation: the tiny model from ``active/`` and
the transformer from ``../pl_g2p/runs/g2p/best.pt``.

    uv run python experiments/xlex_rescore.py [--json out]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_g2p.lexicon import Lexicon, parse_lexicon  # noqa: E402
from pl_g2p.metrics import score as pl_score  # noqa: E402
from pl_g2p.predict import Phonemizer  # noqa: E402

from tiny_g2p.data import make_dataset, references_by_word, resolve_lexicon  # noqa: E402
from tiny_g2p.predict import TinyPhonemizer  # noqa: E402
from tiny_g2p.xlex import (  # noqa: E402
    canonicalize,
    compare,
    parse_cv,
)

TRANSFORMER_CKPT = ROOT.parent / "pl_g2p" / "runs" / "g2p" / "best.pt"


def _canon_set(readings, source: str, word: str) -> set[tuple[str, ...]]:
    return {canonicalize(r, source=source, word=word) for r in readings}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cv = parse_cv()
    mfa_entries: dict[str, list[tuple[str, ...]]] = {}
    for entry in Lexicon(parse_lexicon(resolve_lexicon())).entries:
        mfa_entries.setdefault(entry.word, []).append(entry.phonemes)
    agreement = compare(mfa_entries, cv)
    disputed = set(agreement.disagreed)
    print(f"lexicons: {agreement.shared} shared words, "
          f"{agreement.rate * 100:.2f}% agree", flush=True)

    data = make_dataset(Lexicon(parse_lexicon(resolve_lexicon())),
                        align_train=False)
    refs = references_by_word(data.test)
    words = sorted(refs)
    slices = {
        "agreed": [w for w in words
                   if w in cv.words and w not in disputed],
        "disputed": [w for w in words if w in disputed],
        "mfa-only": [w for w in words if w not in cv.words],
    }
    print("test slices: " + ", ".join(f"{k} {len(v)}" for k, v in
                                      slices.items()), flush=True)

    print("loading tiny model...", flush=True)
    tiny = TinyPhonemizer.from_dir("active", quantized=True)
    tiny_pred = {}
    for i in range(0, len(words), 512):
        chunk = words[i:i + 512]
        tiny_pred.update(zip(chunk, tiny.predict_batch(chunk)))
        print(f"  tiny {i + len(chunk)}/{len(words)}", flush=True)

    if not TRANSFORMER_CKPT.is_file():
        print(f"no transformer checkpoint at {TRANSFORMER_CKPT}; "
              "run `make train` in ../pl_g2p", file=sys.stderr)
        return 2
    print("loading transformer...", flush=True)
    transformer = Phonemizer.from_checkpoint(TRANSFORMER_CKPT, device="auto")
    trans_pred = dict(zip(words, transformer.best(words)))
    print("  transformer done", flush=True)

    out: dict = {"slices": {}, "disputed_breakdown": {}}
    for name, subset in slices.items():
        if not subset:
            continue
        per_model = {}
        for model_name, preds in (("tiny", tiny_pred),
                                  ("transformer", trans_pred)):
            s = pl_score([preds[w] for w in subset],
                         [refs[w] for w in subset], words=subset)
            per_model[model_name] = {
                "n": s.n, "exact": s.exact,
                "word_accuracy": round(s.word_accuracy, 6),
                "per": round(s.per, 6), "distance": s.distance,
                "reference_length": s.reference_length}
        out["slices"][name] = per_model

    # Disputed words: which lexicon (if any) does each model side with?
    for model_name, preds in (("tiny", tiny_pred),
                              ("transformer", trans_pred)):
        buckets = {"mfa-only": 0, "cv-only": 0, "both": 0, "neither": 0}
        examples: dict[str, list[str]] = {k: [] for k in buckets}
        for word in slices["disputed"]:
            pred_canon = canonicalize(preds[word], source="mfa", word=word)
            in_mfa = pred_canon in _canon_set(mfa_entries[word], "mfa", word)
            in_cv = pred_canon in _canon_set(cv.entries[word], "cv", word)
            bucket = ("both" if in_mfa and in_cv else
                      "mfa-only" if in_mfa else
                      "cv-only" if in_cv else "neither")
            buckets[bucket] += 1
            if len(examples[bucket]) < 5:
                examples[bucket].append(
                    f"{word}: pred {' '.join(pred_canon)}")
        out["disputed_breakdown"][model_name] = {
            "buckets": buckets, "examples": examples}

    print("\nword accuracy vs MFA refs by slice:")
    print(f"  {'slice':10} {'n':>6} {'tiny':>8} {'trans':>8}")
    for name, per_model in out["slices"].items():
        print(f"  {name:10} {per_model['tiny']['n']:6d} "
              f"{per_model['tiny']['word_accuracy'] * 100:7.2f}% "
              f"{per_model['transformer']['word_accuracy'] * 100:7.2f}%")
    out["preds"] = {w: {"tiny": tiny_pred[w], "trans": trans_pred[w]}
                    for w in words}
    print("\ndisputed words: which lexicon does each model match?")
    for model_name, detail in out["disputed_breakdown"].items():
        print(f"  {model_name}: {detail['buckets']}")
        for bucket, words_shown in detail["examples"].items():
            for sample in words_shown[:3]:
                print(f"    [{bucket}] {sample}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
