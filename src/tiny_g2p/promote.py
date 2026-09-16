"""Guarded promotion: only strictly-better runs replace ``active/``.

Eligibility mirrors gpu-lexer: the run's deployed (quantized) accuracy on the
fixed verification split must *strictly* improve on the promoted baseline,
and every guard group must stay within cap of it -- a run cannot buy overall
accuracy by sacrificing nasals, digraphs, palatals or the rare tail. The
first run has no baseline and is eligible by definition; it becomes the seed.

``tiny-g2p promote <run>`` is the manual path; training auto-promotes
eligible runs unless told not to. ``force=True`` (the ``--force`` flag)
records a manual override, for the reviewer who has read the errors.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .data import ACTIVE_DIR
from .predict import (
    FLOAT_NAME,
    META_NAME,
    QUANTIZED_NAME,
    SRC_VOCAB_NAME,
    TGT_VOCAB_NAME,
)

#: A guard group may fall this far below baseline before it blocks promotion.
GUARD_CAP = 0.005

BEST_FLOAT_NAME = "best_float.pt"
BEST_QUANTIZED_NAME = "best_quantized.pt"


def load_active_meta() -> dict | None:
    """The promoted run's metadata, or None when nothing is promoted yet."""
    path = ACTIVE_DIR / META_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_eligible(run_meta: dict, baseline_meta: dict | None,
                   *, cap: float = GUARD_CAP) -> tuple[bool, list[str]]:
    """Is the run promotable? Returns (eligible, human-readable reasons)."""
    reasons: list[str] = []
    if baseline_meta is None:
        return True, ["first run: no baseline, becomes the seed"]
    run_acc = run_meta["val_word_accuracy"]
    base_acc = baseline_meta["val_word_accuracy"]
    if run_acc <= base_acc:
        reasons.append(f"val {run_acc:.4f} does not strictly improve on "
                       f"baseline {base_acc:.4f}")
    else:
        reasons.append(f"val {run_acc:.4f} > baseline {base_acc:.4f}")
    eligible = run_acc > base_acc
    run_guards = run_meta.get("guards", {})
    base_guards = baseline_meta.get("guards", {})
    for group in sorted(set(run_guards) | set(base_guards)):
        run_g = run_guards.get(group)
        base_g = base_guards.get(group)
        if run_g is None or base_g is None:
            continue  # groups missing on either side cannot be compared
        if run_g < base_g - cap:
            eligible = False
            reasons.append(f"guard {group}: {run_g:.4f} < {base_g:.4f} "
                           f"(cap {cap})")
    if eligible:
        reasons.append("all guards within cap")
    return eligible, reasons


def promote_run(run_dir: Path | str, *, force: bool = False,
                cap: float = GUARD_CAP) -> tuple[Path, list[str]]:
    """Copy a run's best artifacts into ``active/``, atomically.

    Returns (active_dir, reasons). Raises when the run is ineligible and
    ``force`` is not set, or when the run directory is incomplete.
    """
    run_dir = Path(run_dir)
    for name in (BEST_FLOAT_NAME, BEST_QUANTIZED_NAME, SRC_VOCAB_NAME,
                 TGT_VOCAB_NAME, META_NAME):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"{run_dir} is missing {name}")
    run_meta = json.loads((run_dir / META_NAME).read_text(encoding="utf-8"))
    baseline = load_active_meta()
    if run_meta.get("taxonomy") != (baseline or {}).get("taxonomy") \
            and baseline is not None:
        raise ValueError("taxonomy mismatch: the run's label inventory "
                         "differs from active/; retrain from scratch")
    eligible, reasons = check_eligible(run_meta, baseline, cap=cap)
    if not eligible and not force:
        raise ValueError("run is not eligible for promotion:\n  "
                         + "\n  ".join(reasons))

    staged = ACTIVE_DIR.parent / f".active_staging_{run_dir.name}"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    shutil.copy(run_dir / BEST_FLOAT_NAME, staged / FLOAT_NAME)
    shutil.copy(run_dir / BEST_QUANTIZED_NAME, staged / QUANTIZED_NAME)
    shutil.copy(run_dir / SRC_VOCAB_NAME, staged / SRC_VOCAB_NAME)
    shutil.copy(run_dir / TGT_VOCAB_NAME, staged / TGT_VOCAB_NAME)
    run_meta["provenance"] = {
        "run": str(run_dir),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "forced": force,
    }
    (staged / META_NAME).write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if ACTIVE_DIR.exists():
        shutil.rmtree(ACTIVE_DIR)
    staged.rename(ACTIVE_DIR)
    return ACTIVE_DIR, reasons
