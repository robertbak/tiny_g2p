#!/usr/bin/env python
"""Write the audio->IPA training corpus: one row per utterance.

Each row carries the phone sequence (the CTC target), per-word provenance, and the
audio path, so a trained model's errors can be attributed back to which reading
they came from. That attribution is the point: 72% of the phones are pl_ref gold,
9% rest on a G2P, and a model that disagrees with the G2P on the gold 72% is
making a claim about the audio, whereas on the 9% it is only disagreeing with the
G2P.

Lives in ``tiny_g2p/experiments`` because that is where the pieces meet: this
project owns the fallback model, its venv has ``phoneme_lab`` and torch, and the
audio is BIGOS, unpacked under ``live_stt/data``.

    .venv/bin/python experiments/build_ipa_targets.py --split train
    .venv/bin/python experiments/build_ipa_targets.py --split validation

The G2P is cached in ``g2p_cache.tsv``, so a rerun after a policy change only pays
for what is new.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phoneme_lab" / "src"))

from phoneme_lab import (  # noqa: E402
    PAUSE,
    SEPARATOR,
    Reference,
    TargetBuilder,
    coverage,
    provenance_per_phone,
)

#: One letter per provenance, for the compact per-word column.
SHORT = {"pl_ref:verified": "V", "pl_ref:unverified": "U"}

COLUMNS = ("id", "audio", "seconds", "phones", "words", "dropped", "text")


def duration_s(path: Path) -> float:
    """Seconds of audio, from the WAV header.

    Read rather than looked up: the released TSVs carry ten columns and no
    duration, although the dataset card lists ``audio_duration_seconds``. The
    header read is cheap and needs no dependency.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate() or 1)
    except Exception:
        return 0.0


def short(provenance: str) -> str:
    return SHORT.get(provenance, "G")


def rows_for(root: Path, split: str) -> list[dict]:
    """Every TSV row of a split, with its audio resolved.

    The two roots lay their metadata out differently: the validation copies nest
    one TSV per subset directory (``bigos/<subset>/validation.tsv``) while the
    train split drops one flat file per subset (``bigos_train/<subset>.tsv``).
    Both are matched, so the caller does not have to know which it has.
    """
    out: list[dict] = []
    candidates = sorted(set(root.glob(f"**/{split}.tsv")) | set(root.glob("*.tsv")))
    for tsv in candidates:
        audio_root = tsv.parent if tsv.parent != root else root / tsv.stem
        if not audio_root.is_dir():
            audio_root = root / "data" / tsv.stem
        by_name: dict[str, Path] = {}
        if audio_root.is_dir():
            by_name = {p.name: p for p in audio_root.rglob("*.wav")}
        with tsv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                text = (row.get("ref_orig") or "").strip()
                if not text:
                    continue
                name = (row.get("audioname") or "").strip()
                path = by_name.get(f"{name}.wav") or by_name.get(
                    Path(row.get("audiopath_bigos") or "").name
                )
                # Read the duration from the real path, then store one relative to
                # the repo so the corpus survives the tree being moved. Relativising
                # first made every lookup fail and every duration zero -- and the
                # exception guard swallowed it.
                seconds = duration_s(path) if path else 0.0
                if path is not None:
                    try:
                        path = path.relative_to(ROOT)
                    except ValueError:
                        pass
                out.append(
                    {
                        "id": name,
                        "subset": row.get("dataset") or tsv.stem,
                        "audio": str(path) if path else "",
                        "seconds": seconds,
                        "text": text,
                    }
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--out", type=Path, default=ROOT / "ipa_targets")
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="stop after N utterances")
    args = ap.parse_args()

    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)

    # The model is imported late so `--help` works without torch.
    from tiny_g2p.predict import TinyPhonemizer

    reference = Reference.load(args.reference) if args.reference else Reference.load()
    phone = TinyPhonemizer.from_active()
    builder = TargetBuilder(
        reference,
        fallback=phone.predict_word,
        batch_fallback=phone.predict_batch,
        fallback_name="tiny_g2p",
        strict_alphabet=True,
        # The sequence carries word boundaries and marks hesitations rather than
        # dropping them: a flat phone string cannot be looked up in a lexicon, and
        # deleting a filler silently absorbs its audio into its neighbours.
        word_separator=SEPARATOR,
        filler_marker=PAUSE,
    )

    cache_path = out / "g2p_cache.tsv"
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8") as handle:
            seeded = {
                word: tuple(phones.split())
                for word, phones in (line.split("\t") for line in handle if line.strip())
            }
        print(f"loaded {builder.load_cache(seeded)} cached words", flush=True)

    source_root = ROOT / "live_stt" / "data" / ("bigos_train" if args.split == "train" else "bigos")
    records = rows_for(source_root, args.split)
    if args.limit:
        records = records[: args.limit]
    missing = sum(1 for r in records if not r["audio"])
    print(f"{len(records)} utterances from {source_root} ({missing} without audio)", flush=True)
    if not records:
        print("nothing to do -- check --split and the data root", file=sys.stderr)
        return 2

    started = time.time()
    targets = builder.build_all([r["text"] for r in records])
    print(f"built targets in {time.time() - started:.0f}s", flush=True)

    targets_path = out / "targets.tsv"
    with targets_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(COLUMNS)
        for record, utterance in zip(records, targets):
            # word:provenance:phones -- the count makes word boundaries derivable
            # from the flat phones column alone, so the artifact is decodable
            # without re-running pl_ref or the G2P.
            words = " ".join(
                f"{t.word}:{short(t.provenance)}:{len(t.phones)}" for t in utterance.targets
            )
            dropped = " ".join(f"{s.word}:{s.reason}" for s in utterance.skipped)
            writer.writerow(
                (
                    record["id"],
                    record["audio"],
                    f"{record['seconds']:.2f}",
                    " ".join(utterance.phones),
                    words,
                    dropped,
                    record["text"],
                )
            )
    print(f"wrote {targets_path} ({targets_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    with cache_path.open("w", encoding="utf-8", newline="") as handle:
        for word, phones in sorted(builder.cache_snapshot().items()):
            handle.write(f"{word}\t{' '.join(phones)}\n")
    print(f"wrote {cache_path} ({len(builder.cache_snapshot())} words)", flush=True)

    by_subset: dict[str, Counter] = {}
    for record, utterance in zip(records, targets):
        counts = by_subset.setdefault(record["subset"], Counter())
        for target in utterance.targets:
            counts[target.provenance] += len(target.phones)
        for skip in utterance.skipped:
            counts[f"skipped:{skip.reason}"] += 1

    summary = {
        "split": args.split,
        "reference": str(args.reference or "default pl_ref"),
        "utterances": len(records),
        "seconds": round(sum(r["seconds"] for r in records), 1),
        "tokens": dict(coverage(targets)),
        "phones": dict(provenance_per_phone(targets)),
        "by_subset_phones": {k: dict(v) for k, v in sorted(by_subset.items())},
        "alphabet_violations": builder.alphabet_violations(),
        "is_gold_utterances": sum(1 for t in targets if t.is_gold),
        # The output alphabet a trainer should use, markers included.
        "inventory": sorted({p for u in targets for p in u.phones}),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    phones = summary["phones"]
    total = sum(phones.values()) or 1
    print(f"\nphones: {total}")
    for key, value in sorted(phones.items(), key=lambda kv: -kv[1]):
        print(f"  {key:22} {value:8} {value / total * 100:5.1f}%")
    skipped = {k: v for k, v in summary["tokens"].items() if k.startswith("skipped:")}
    print(f"skipped tokens: {skipped}")
    print(f"all-gold utterances: {summary['is_gold_utterances']} / {len(records)}")
    print(f"alphabet violations: {summary['alphabet_violations']}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
