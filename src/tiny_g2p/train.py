"""Training: warm-start continuation, QAT finale, quantized selection.

The regime, point by point:

* **Epoch zero is the promoted checkpoint** when its label taxonomy matches;
  otherwise training starts fresh (``--fresh`` forces the random baseline).
  ``--polish`` runs a short 6-epoch continuation and refuses without a
  taxonomy-matching checkpoint.
* **Loss** is boundary-times-example-weighted phone cross-entropy plus the
  auxiliary manner head at 0.3.
* **Replay** mixes 5% recent train-error words into every batch (train errors
  only -- verification is never replayed into training).
* **QAT** runs the final 6 epochs; the embedding stays float (see model.py).
* **Selection** scores *every* candidate -- epoch zero and each epoch -- by
  *quantized* held-out word accuracy, converting float candidates via PTQ
  calibrated on train words. Ties keep the earlier candidate.
* **Auto-promotion** follows when the run is eligible, unless disabled.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from pl_g2p.data import length_bucketed_batches
from torch.utils.data import Dataset

from .data import (
    ACTIVE_DIR,
    MANNERS,
    CharDataset,
    DataSet,
    EvalRow,
    collate,
)
from .model import (
    build,
    convert_to_quantized,
    count_params,
    prepare_for_qat,
)
from .predict import (
    TinyPhonemizer,
    evaluate_guards,
    evaluate_words,
    resolve_device,
    write_meta,
)
from .promote import (
    BEST_FLOAT_NAME,
    BEST_QUANTIZED_NAME,
    promote_run,
)

POLISH_EPOCHS = 6
CALIB_WORDS = 512


@dataclass
class TrainConfig:
    """Training hyperparameters (saved with every run)."""

    epochs: int = 32
    lr: float = 0.002
    min_lr: float = 0.0002
    batch_size: int = 1024
    aux_weight: float = 0.3
    qat_epochs: int = 6
    replay_fraction: float = 0.05
    seed: int = 7
    device: str = "auto"
    promote: bool = True
    polish: bool = False
    fresh: bool = False


@dataclass
class RunResult:
    """What a training run produced and whether it promoted."""

    run_dir: Path
    best_val_accuracy: float
    best_epoch: int
    best_kind: str
    guards: dict = field(default_factory=dict)
    promoted: bool = False
    reasons: list[str] = field(default_factory=list)


def _taxonomy_of(directory: Path) -> dict | None:
    from .vocab import Vocab

    src, tgt = directory / "src_vocab.json", directory / "tgt_vocab.json"
    if not src.is_file() or not tgt.is_file():
        return None
    return {"src": Vocab.from_json(src).taxonomy(),
            "tgt": Vocab.from_json(tgt).taxonomy()}


def _losses(phone_logits: torch.Tensor, manner_logits: torch.Tensor,
            batch: dict, *, aux_weight: float,
            pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    valid = batch["chars"] != pad_id
    phone_ce = F.cross_entropy(phone_logits[valid], batch["labels"][valid],
                               reduction="none")
    weights = (batch["boundary"][valid]
               * batch["weight"].unsqueeze(1).expand_as(batch["boundary"])[valid])
    phone_loss = (phone_ce * weights).sum() / weights.sum().clamp_min(1e-6)
    manner_loss = F.cross_entropy(manner_logits[valid], batch["manner"][valid])
    return phone_loss + aux_weight * manner_loss, phone_loss.detach()


@torch.no_grad()
def _mismatched(batch_indices: list[int], phone_logits: torch.Tensor,
                batch: dict, *, pad_id: int) -> list[int]:
    """Train indices in the batch with any label mismatch (for replay)."""
    best = phone_logits.argmax(dim=-1)
    wrong = (best != batch["labels"]) & (batch["chars"] != pad_id)
    return [idx for idx, row in zip(batch_indices, wrong) if row.any()]


def _quantize_float_copy(model: torch.nn.Module,
                         calib: list[dict]) -> torch.nn.Module:
    """PTQ a float model: prepare, calibrate on train words, convert (CPU)."""
    import copy

    clone = copy.deepcopy(model).cpu().eval()
    clone.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
    clone.embedding.qconfig = None  # type: ignore[assignment]
    torch.ao.quantization.prepare(clone, inplace=True)
    with torch.no_grad():
        for batch in calib:
            clone(batch["chars"])
    return torch.ao.quantization.convert(clone, inplace=False)


@torch.no_grad()
def _val_accuracy(phonemizer: TinyPhonemizer,
                  val_rows: list[EvalRow]) -> float:
    return evaluate_words(phonemizer, val_rows).word_accuracy


def train(data: DataSet, run_dir: Path | str,
          config: TrainConfig | None = None, *,
          progress=None) -> RunResult:
    """Run training per ``config``; returns the run result (never raises on
    ineligibility -- a run that fails promotion is still a complete run).

    ``progress``, when given, is called with each history row (epoch zero
    first) as the run proceeds.
    """
    config = config or TrainConfig()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not data.val:
        raise ValueError("no verification rows: selection needs held-out data")
    if config.polish:
        config.epochs = POLISH_EPOCHS

    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = resolve_device(config.device)
    model = build(len(data.src_vocab), len(data.tgt_vocab), len(MANNERS))
    model.to(device)

    # --- epoch zero: warm start when the taxonomy matches -------------------
    taxonomy = {"src": data.src_vocab.taxonomy(),
                "tgt": data.tgt_vocab.taxonomy()}
    active_taxonomy = _taxonomy_of(ACTIVE_DIR)
    warmed = False
    if not config.fresh and active_taxonomy == taxonomy \
            and (ACTIVE_DIR / "model.pt").is_file():
        payload = torch.load(ACTIVE_DIR / "model.pt", map_location=device,
                             weights_only=True)
        model.load_state_dict(payload["state_dict"], strict=False)
        warmed = True
    elif config.polish:
        raise ValueError("--polish needs a taxonomy-matching promoted "
                         "checkpoint; run the full schedule instead")

    dataset = CharDataset(data.train, data.src_vocab, data.tgt_vocab,
                          data.manner_of)
    pad_id = data.src_vocab.pad_id
    calib_idx = list(range(0, len(dataset), max(1, len(dataset) // CALIB_WORDS)))
    calib = [collate([dataset[i]]) for i in calib_idx[:CALIB_WORDS]]

    def phonemizer_for(candidate: torch.nn.Module,
                       quantized: bool) -> TinyPhonemizer:
        where = torch.device("cpu") if quantized else device
        return TinyPhonemizer(candidate, data.src_vocab, data.tgt_vocab,
                              where, quantized=quantized)

    def score_quantized(candidate: torch.nn.Module, *, qat: bool) -> float:
        if qat:
            quantised = convert_to_quantized(candidate)
        else:
            quantised = _quantize_float_copy(candidate, calib)
        return _val_accuracy(phonemizer_for(quantised, True), data.val)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=config.min_lr)
    replay: deque[int] = deque(maxlen=4096)

    history: list[dict] = []
    best = {"acc": -1.0, "epoch": -1, "kind": "none",
            "float_state": None, "quantized": None, "float_acc": 0.0}

    def consider(epoch: int, kind: str, acc: float, acc_float: float,
                 float_state: dict, quantized: torch.nn.Module) -> None:
        if acc > best["acc"]:
            best.update(acc=acc, epoch=epoch, kind=kind,
                        float_state={k: v.cpu().clone()
                                     for k, v in float_state.items()},
                        quantized=quantized, float_acc=acc_float)
            torch.save({"state_dict": best["float_state"]},
                       run_dir / BEST_FLOAT_NAME)
            torch.save(best["quantized"], run_dir / BEST_QUANTIZED_NAME)

    # Epoch zero counts as a candidate (ties keep it: it is evaluated first).
    model.eval()
    epoch0_float = _val_accuracy(phonemizer_for(model, False), data.val)
    epoch0_quant = score_quantized(model, qat=False)
    consider(-1 if warmed else -2, "promoted" if warmed else "random-init",
             epoch0_quant, epoch0_float, model.state_dict(),
             _quantize_float_copy(model, calib))
    history.append({"epoch": 0, "phase": "epoch-zero",
                    "val_acc_float": round(epoch0_float, 6),
                    "val_acc_quant": round(epoch0_quant, 6)})
    if progress is not None:
        progress(history[-1])

    qat_start = max(0, config.epochs - config.qat_epochs)
    qat_active = False
    for epoch in range(config.epochs):
        if epoch >= qat_start and config.qat_epochs > 0 and not qat_active:
            model.cpu()
            model.train()  # prepare_qat requires training mode
            prepare_for_qat(model)
            model.to(device)
            qat_active = True
        model.train()
        total, weight_sum = 0.0, 0.0
        batches = length_bucketed_batches(data.train,
                                          batch_size=config.batch_size,
                                          seed=config.seed + epoch)
        for indices in batches:
            fresh = list(indices)
            take = min(len(replay), int(len(fresh) * config.replay_fraction))
            if take:
                picks = rng.sample(list(replay), take)
                mixed = fresh[: len(fresh) - take] + picks
            else:
                mixed = fresh
            batch = {k: v.to(device) for k, v in collate([dataset[i] for i in mixed]).items()
                     if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            phone_logits, manner_logits = model(batch["chars"])
            loss, _ = _losses(phone_logits, manner_logits, batch,
                              aux_weight=config.aux_weight, pad_id=pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            replay.extend(_mismatched(fresh, phone_logits.detach(), batch,
                                      pad_id=pad_id))
            total += float(loss.detach()) * len(mixed)
            weight_sum += len(mixed)
        scheduler.step()

        model.eval()
        val_float = _val_accuracy(phonemizer_for(model, False), data.val)
        if qat_active:
            quantised = convert_to_quantized(model)
            val_quant = _val_accuracy(phonemizer_for(quantised, True),
                                      data.val)
        else:
            val_quant = score_quantized(model, qat=False)
            quantised = _quantize_float_copy(model, calib)
        consider(epoch, "qat" if qat_active else "float", val_quant,
                 val_float, model.state_dict(), quantised)
        history.append({"epoch": epoch + 1,
                        "phase": "qat" if qat_active else "float",
                        "train_loss": round(total / max(1, weight_sum), 6),
                        "lr": round(scheduler.get_last_lr()[0], 6),
                        "val_acc_float": round(val_float, 6),
                        "val_acc_quant": round(val_quant, 6)})
        if progress is not None:
            progress(history[-1])

    # --- guards, metadata, promotion -----------------------------------------
    data.src_vocab.to_json(run_dir / "src_vocab.json")
    data.tgt_vocab.to_json(run_dir / "tgt_vocab.json")
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    best_phonemizer = phonemizer_for(best["quantized"], True)
    guard_scores = evaluate_guards(best_phonemizer, data.val, set(data.rare))
    guards = {name: round(s.word_accuracy, 6)
              for name, s in guard_scores.items()}
    # Group sizes are recorded alongside the rates: 65% of 23 rows and 65% of
    # 1,186 are not the same statement, and a run comparison needs to see it.
    guards_n = {name: s.n for name, s in guard_scores.items()}
    meta = {
        "val_word_accuracy": round(best["acc"], 6),
        "val_per": round(evaluate_words(best_phonemizer, data.val).per, 6),
        "val_acc_float": round(best["float_acc"], 6),
        "best_epoch": best["epoch"],
        "best_kind": best["kind"],
        "guards": guards,
        "guards_n": guards_n,
        "taxonomy": taxonomy,
        "params": count_params(model),
        "warmed": warmed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_meta(run_dir, meta)

    result = RunResult(run_dir, meta["val_word_accuracy"], best["epoch"],
                       best["kind"], guards)
    if config.promote:
        try:
            _, result.reasons = promote_run(run_dir)
            result.promoted = True
        except ValueError as exc:
            result.reasons = [str(exc)]
    return result


# --- fine-tuning -------------------------------------------------------------


@dataclass
class FineTuneConfig:
    """Targeted correction on failure-bank words (never auto-promotes)."""

    steps: int = 200
    bank_fraction: float = 0.5
    batch_size: int = 64
    rehearsal_epochs: int = 1
    lr: float = 0.0005
    rollback_cap: float = 0.01
    seed: int = 11
    device: str = "auto"


@dataclass
class FineTuneResult:
    """What a fine-tuning run did (``saved`` is False after a rollback)."""

    run_dir: Path
    bank_accuracy: float
    val_accuracy: float
    baseline_val: float
    saved: bool
    reasons: list[str] = field(default_factory=list)


def prg2p_teacher(words: list[str]) -> dict[str, list[str]]:
    """Label words with the rule-based prg2p engine (the Shiki analogue).

    prg2p output is mapped to MFA phones and re-dentalized, so labels match
    the training inventory. Raises listing words prg2p refused.
    """
    from phoneme_lab.g2p import Prg2pG2P
    from pl_g2p.baseline import map_prg2p

    from .phones import redentalize

    engine = Prg2pG2P()
    transcripts = engine.transcribe(words)
    missing = [w for w in words if w not in transcripts]
    if missing:
        raise ValueError(f"prg2p refused {len(missing)} word(s): "
                         f"{missing[:10]}")
    out = {}
    for word in words:
        prg2p_phones = list(transcripts[word][0].phonemes)
        out[word] = redentalize(map_prg2p(prg2p_phones, strict=True))
    return out


def finetune(data: DataSet, words: list[str], run_dir: Path | str,
             config: FineTuneConfig | None = None, *,
             teacher=prg2p_teacher) -> FineTuneResult:
    """Repair bank words: balanced failure-heavy updates, then rehearsal.

    A snippet match alone is not sufficient: the run is *rolled back*
    (weights discarded) unless val stays within ``rollback_cap`` of the
    promoted baseline, and it never auto-promotes -- ``tiny-g2p promote``
    decides that, under the usual guards.
    """
    from .align import align_entry
    from .data import Example
    from .failurebank import check_leakage
    from .predict import normalize_input

    config = config or FineTuneConfig()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    words = [normalize_input(w) for w in words if normalize_input(w)]
    if not words:
        raise ValueError("no bank words to fine-tune on")

    leaked = check_leakage(words, {r.word for r in data.val},
                           {r.word for r in data.test})
    if leaked:
        raise ValueError(f"refusing to train on verification words: {leaked}")

    teacher_phones = teacher(words)
    bank: list[Example] = []
    unlabelled = []
    for word in words:
        alignment = align_entry(word, teacher_phones[word])
        if alignment is None:
            unlabelled.append(word)
            continue
        bank.append(Example(word, tuple(teacher_phones[word]),
                            alignment.labels))
    if unlabelled:
        raise ValueError(f"could not label {len(unlabelled)} bank word(s): "
                         f"{unlabelled[:10]}")

    taxonomy = {"src": data.src_vocab.taxonomy(),
                "tgt": data.tgt_vocab.taxonomy()}
    if _taxonomy_of(ACTIVE_DIR) != taxonomy \
            or not (ACTIVE_DIR / "model.pt").is_file():
        raise ValueError("fine-tuning needs a taxonomy-matching promoted "
                         "model; train one first")
    device = resolve_device(config.device)
    model = build(len(data.src_vocab), len(data.tgt_vocab), len(MANNERS))
    payload = torch.load(ACTIVE_DIR / "model.pt", map_location=device,
                         weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device)

    baseline = TinyPhonemizer.from_dir(ACTIVE_DIR, quantized=True)
    baseline_val = _val_accuracy(baseline, data.val)

    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    pad_id = data.src_vocab.pad_id
    train_ds = CharDataset(data.train, data.src_vocab, data.tgt_vocab,
                           data.manner_of)

    def step_on(examples: list[int], dataset: Dataset) -> float:
        batch = {k: v.to(device) for k, v in collate(
            [dataset[i] for i in examples]).items()
            if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        phone_logits, manner_logits = model(batch["chars"])
        loss, _ = _losses(phone_logits, manner_logits, batch,
                          aux_weight=0.3, pad_id=pad_id)
        loss.backward()
        optimizer.step()
        return float(loss)

    model.train()
    combined = CharDataset(bank + data.train, data.src_vocab, data.tgt_vocab,
                           data.manner_of)
    for _ in range(config.steps):
        take_bank = int(config.batch_size * config.bank_fraction)
        indices = (rng.sample(range(len(bank)), min(take_bank, len(bank)))
                   + rng.sample(range(len(bank), len(combined)),
                                min(config.batch_size - take_bank,
                                    len(data.train))))
        step_on(indices, combined)
    for epoch in range(config.rehearsal_epochs):
        for indices in length_bucketed_batches(
                data.train, batch_size=config.batch_size,
                seed=config.seed + epoch):
            step_on(indices, train_ds)

    model.eval()
    current = TinyPhonemizer(model, data.src_vocab, data.tgt_vocab, device)
    bank_hits = sum(1 for w in words
                    if current.predict_word(w) == teacher_phones[w])
    bank_accuracy = bank_hits / len(words)
    quantised = _quantize_float_copy(
        model, [collate([train_ds[i]])
                for i in range(0, len(train_ds),
                               max(1, len(train_ds) // CALIB_WORDS))][:CALIB_WORDS])
    val_accuracy = _val_accuracy(
        TinyPhonemizer(quantised, data.src_vocab, data.tgt_vocab,
                       torch.device("cpu"), quantized=True), data.val)

    reasons = [f"bank {bank_hits}/{len(words)}",
               f"val {val_accuracy:.4f} vs baseline {baseline_val:.4f}"]
    saved = val_accuracy >= baseline_val - config.rollback_cap
    if not saved:
        reasons.append(f"val regressed past cap {config.rollback_cap}: "
                       "rolled back, weights discarded")
        write_meta(run_dir, {"kind": "finetune-rolled-back",
                             "bank_accuracy": round(bank_accuracy, 6),
                             "val_accuracy": round(val_accuracy, 6),
                             "baseline_val": round(baseline_val, 6),
                             "taxonomy": taxonomy})
        return FineTuneResult(run_dir, bank_accuracy, val_accuracy,
                              baseline_val, False, reasons)

    torch.save({"state_dict": {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}},
               run_dir / BEST_FLOAT_NAME)
    torch.save(quantised, run_dir / BEST_QUANTIZED_NAME)
    data.src_vocab.to_json(run_dir / "src_vocab.json")
    data.tgt_vocab.to_json(run_dir / "tgt_vocab.json")
    finetuned = TinyPhonemizer(quantised, data.src_vocab, data.tgt_vocab,
                               torch.device("cpu"), quantized=True)
    guard_scores = evaluate_guards(finetuned, data.val, set(data.rare))
    write_meta(run_dir, {
        "kind": "finetune",
        "val_word_accuracy": round(val_accuracy, 6),
        "val_per": round(evaluate_words(finetuned, data.val).per, 6),
        "guards": {name: round(s.word_accuracy, 6)
                   for name, s in guard_scores.items()},
        "guards_n": {name: s.n for name, s in guard_scores.items()},
        "bank_accuracy": round(bank_accuracy, 6),
        "bank_words": words,
        "taxonomy": taxonomy,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    reasons.append("saved (promotion is a separate, manual decision)")
    return FineTuneResult(run_dir, bank_accuracy, val_accuracy, baseline_val,
                          True, reasons)
