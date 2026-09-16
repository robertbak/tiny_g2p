"""Command-line interface.

    tiny-g2p fetch                 pinned dictionary (or the sibling copy)
    tiny-g2p stats --splits        lexicon summary, alignment coverage, splits
    tiny-g2p train --epochs 32     warm-start continuation (auto-promotes)
    tiny-g2p eval                  promoted checkpoint on held-out test words
    tiny-g2p predict Wrocław ...   transcribe words
    tiny-g2p fine-tune słowo ...   repair failure words (never auto-promotes)
    tiny-g2p promote runs/<ts>     promote a run under the usual guards
    tiny-g2p export --gold ...     package weights for the Rust/wasm runtime
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

#: Resolved without importing :mod:`tiny_g2p.data`, which would pull in torch
#: for every CLI invocation including `--help`.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLOB_PATH = REPO_ROOT / "rust" / "weights" / "tiny_g2p.bin"
DEFAULT_ACTIVE_DIR = REPO_ROOT / "active"
DEFAULT_VERDICTS_PATH = REPO_ROOT / "data" / "adjudication_verdicts.json"


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def cmd_fetch(args) -> int:
    from .data import resolve_lexicon

    path = resolve_lexicon()
    print(f"lexicon ready: {path}")
    return 0


def cmd_fetch_cv(args) -> int:
    from .xlex import fetch_cv

    path = fetch_cv(force=args.force)
    print(f"CV dictionary ready: {path}")
    return 0


def cmd_export(args) -> int:
    from .export import export_blob, export_gold, export_onnx

    blob = export_blob(args.active, float_source=args.float_source,
                       acronym_words=args.acronym)
    args.blob.parent.mkdir(parents=True, exist_ok=True)
    args.blob.write_bytes(blob)
    print(f"wrote {args.blob} ({len(blob) / 1024:.0f} KiB)")
    if args.gold:
        rows = export_gold(args.gold, verdicts_json=args.verdicts,
                           active_dir=args.active)
        print(f"wrote {args.gold} ({rows} words)")
    if args.onnx:
        print(f"wrote {export_onnx(args.onnx, active_dir=args.active)}")
    return 0


def cmd_stats(args) -> int:
    from pl_g2p.lexicon import Lexicon, parse_lexicon

    from .data import make_dataset, resolve_lexicon

    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    print(lexicon.summary())
    if args.splits:
        data = make_dataset(lexicon)
        counts = data.counts()
        print(f"\ntrain {counts['train']}  val {counts['val']}  "
              f"test {counts['test']}  excluded {counts['excluded']}")
        print(f"train chars {len(data.src_vocab)}  "
              f"train phones {len(data.tgt_vocab)}  "
              f"rare phones {len(data.rare)}")
    return 0


def _print_progress(row: dict) -> None:
    if row["epoch"] == 0:
        print(f"  epoch zero ({row['phase']}): "
              f"val {row['val_acc_quant'] * 100:.2f}% (quantized)",
              flush=True)
    else:
        print(f"  epoch {row['epoch']:>3} [{row['phase']:5}] "
              f"loss {row['train_loss']:.4f}  "
              f"val {row['val_acc_quant'] * 100:.2f}% (q) / "
              f"{row['val_acc_float'] * 100:.2f}% (f)", flush=True)


def cmd_train(args) -> int:
    from pl_g2p.lexicon import Lexicon, parse_lexicon

    from .data import RUNS_DIR, make_dataset, resolve_lexicon
    from .train import TrainConfig, train

    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    data = make_dataset(lexicon)
    counts = data.counts()
    print(f"train {counts['train']}  val {counts['val']}  "
          f"test {counts['test']}  excluded {counts['excluded']}")
    config = TrainConfig(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        qat_epochs=0 if args.no_qat else args.qat_epochs,
        seed=args.seed, device=args.device, promote=not args.no_promote,
        polish=args.polish, fresh=args.fresh)
    run_dir = RUNS_DIR / (_stamp() if not args.polish else f"{_stamp()}-polish")
    print(f"run {run_dir}  epochs {config.epochs}  lr {config.lr}  "
          f"qat {config.qat_epochs}  device {config.device}")
    result = train(data, run_dir, config, progress=_print_progress)
    print(f"\nbest: {result.best_kind} epoch {result.best_epoch}  "
          f"val {result.best_val_accuracy * 100:.2f}%")
    for group, acc in sorted(result.guards.items()):
        print(f"  guard {group:8} {acc * 100:6.2f}%")
    print("promoted" if result.promoted else "not promoted")
    for reason in result.reasons:
        print(f"  {reason}")
    print(f"wrote {run_dir}")
    return 0


def cmd_eval(args) -> int:
    from pl_g2p.lexicon import Lexicon, parse_lexicon
    from pl_g2p.metrics import format_errors

    from .data import make_dataset, resolve_lexicon
    from .predict import TinyPhonemizer, evaluate_guards, evaluate_words

    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    data = make_dataset(lexicon, align_train=False)
    if args.run:
        phonemizer = TinyPhonemizer.from_dir(args.run, quantized=True)
        label = str(args.run)
    else:
        phonemizer = TinyPhonemizer.from_active(quantized=True)
        label = "active/"
    print(f"checkpoint: {label} (quantized)\n")
    val = evaluate_words(phonemizer, data.val, keep_errors=args.show_errors)
    print(f"verification {val.format()}")
    test = evaluate_words(phonemizer, data.test, keep_errors=args.show_errors)
    print(f"held-out     {test.format()}")
    print("\nverification guards:")
    for name, s in evaluate_guards(phonemizer, data.val,
                                   set(data.rare)).items():
        print(f"  {name:8} {s.format()}")
    if args.show_errors and test.errors:
        print(f"\nheld-out errors (first {args.show_errors}):")
        print(format_errors(test.errors, args.show_errors))
    return 0


def split_cli_words(args: list[str]) -> list[str]:
    """Split CLI word arguments on whitespace (a quoted sentence is many words)."""
    return [w for arg in args for w in arg.split() if w]


def cmd_predict(args) -> int:
    from .predict import TinyPhonemizer

    phonemizer = TinyPhonemizer.from_active(quantized=True)
    for word in split_cli_words(args.words):
        print(f"{word} -> {' '.join(phonemizer.predict_word(word))}")
    return 0


def cmd_finetune(args) -> int:
    from pl_g2p.lexicon import Lexicon, parse_lexicon

    from .data import RUNS_DIR, make_dataset, resolve_lexicon
    from .failurebank import add_words, read_bank
    from .train import FineTuneConfig, finetune

    add_words(split_cli_words(args.words))
    bank = read_bank()
    print(f"bank holds {len(bank)} word(s)")
    lexicon = Lexicon(parse_lexicon(resolve_lexicon()))
    data = make_dataset(lexicon)
    config = FineTuneConfig(steps=args.steps, lr=args.lr,
                            rehearsal_epochs=args.rehearsal,
                            device=args.device)
    run_dir = RUNS_DIR / f"{_stamp()}-finetune"
    try:
        result = finetune(data, bank, run_dir, config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"bank accuracy {result.bank_accuracy * 100:.1f}%  "
          f"val {result.val_accuracy * 100:.2f}% "
          f"(baseline {result.baseline_val * 100:.2f}%)")
    print("saved" if result.saved else "ROLLED BACK")
    for reason in result.reasons:
        print(f"  {reason}")
    return 0


def cmd_promote(args) -> int:
    from .promote import promote_run

    try:
        active, reasons = promote_run(args.run, force=args.force,
                                      cap=args.cap)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"promoted {args.run} -> {active}")
    for reason in reasons:
        print(f"  {reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tiny-g2p",
        description="Tiny parallel per-character Polish G2P.")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="fetch or locate the dictionary")
    f.set_defaults(func=cmd_fetch)

    fc = sub.add_parser("fetch-cv", help="fetch the pinned CV dictionary")
    fc.add_argument("--force", action="store_true")
    fc.set_defaults(func=cmd_fetch_cv)

    x = sub.add_parser("export", help="package the model for a Python-free runtime")
    x.add_argument("--active", type=Path, default=DEFAULT_ACTIVE_DIR)
    x.add_argument("--blob", type=Path,
                   default=DEFAULT_BLOB_PATH)
    x.add_argument("--gold", type=Path, default=None,
                   help="also write the held-out gold TSV")
    x.add_argument("--onnx", type=Path, default=None,
                   help="also write an ONNX graph")
    x.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS_PATH)
    x.add_argument("--float-source", choices=("checkpoint", "dequantized"),
                   default="checkpoint")
    x.add_argument("--acronym", action="append", default=[],
                   help="acronym to spell letter-by-letter in the table")
    x.set_defaults(func=cmd_export)

    s = sub.add_parser("stats", help="lexicon and split summary")
    s.add_argument("--splits", action="store_true",
                   help="align train and report split sizes")
    s.set_defaults(func=cmd_stats)

    t = sub.add_parser("train", help="train (warm-starts active/)")
    t.add_argument("--epochs", type=int, default=32)
    t.add_argument("--lr", type=float, default=0.002)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--qat-epochs", type=int, default=6)
    t.add_argument("--no-qat", action="store_true",
                   help="skip quantization-aware training")
    t.add_argument("--fresh", action="store_true",
                   help="ignore the promoted checkpoint (random baseline)")
    t.add_argument("--polish", action="store_true",
                   help="short continuation; requires a matching checkpoint")
    t.add_argument("--no-promote", action="store_true")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--device", default="auto")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="score a checkpoint on held-out words")
    e.add_argument("--run", type=Path, default=None,
                   help="run directory (default: the promoted checkpoint)")
    e.add_argument("--show-errors", type=int, default=0)
    e.set_defaults(func=cmd_eval)

    pr = sub.add_parser("predict", help="transcribe words")
    pr.add_argument("words", nargs="+")
    pr.set_defaults(func=cmd_predict)

    ft = sub.add_parser("fine-tune", help="repair failure words")
    ft.add_argument("words", nargs="+")
    ft.add_argument("--steps", type=int, default=200)
    ft.add_argument("--lr", type=float, default=0.0005)
    ft.add_argument("--rehearsal", type=int, default=1)
    ft.add_argument("--device", default="auto")
    ft.set_defaults(func=cmd_finetune)

    pm = sub.add_parser("promote", help="promote a run into active/")
    pm.add_argument("run", type=Path)
    pm.add_argument("--force", action="store_true",
                    help="manual override of the eligibility check")
    pm.add_argument("--cap", type=float, default=0.005,
                    help="guard regression cap (default: 0.005)")
    pm.set_defaults(func=cmd_promote)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
