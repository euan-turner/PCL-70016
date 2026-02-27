#!/usr/bin/env python3
"""
train.py — Fine-tune DeBERTa-v3-large with LoRA for binary PCL detection.

Supports:
  • LoRA rank sweep over configurable set (default: 8, 16, 32, 64)
  • Keyword-stratified K-fold cross-validation on the official train split
  • Class-weighted loss OR minority oversampling for 9.5:1 imbalance
  • Early stopping on validation F1 (positive class)
  • After sweep: retrain best rank on full train, evaluate on dev, save adapter

Usage examples
--------------
  python train.py                                  # full sweep, 5-fold CV
  python train.py --ranks 16 --folds 3             # quick test
  python train.py --folds 0 --ranks 8 16           # no CV: train on full train, eval on dev
  python train.py --balance oversample             # oversample PCL instead of weighted loss
  python train.py --dry-run                        # print config and exit
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

# DeBERTa-v2's disentangled attention triggers CUBLAS_STATUS_INVALID_VALUE
# with the default cuBLAS on Ada Lovelace GPUs (PTX 2.10 + CUDA 12.8).
# Switching to cuBLASLt resolves the issue.
torch.backends.cuda.preferred_blas_library("cublaslt")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

# Defaults
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 128          # p95 ≈ 128 tokens from EDA
LORA_RANKS = [8, 16, 32, 64]
LORA_ALPHA_MULT = 2       # = alpha / rank
LORA_DROPOUT = 0.1
TARGET_MODULES = ["query_proj", "key_proj", "value_proj"]  # DeBERTa-v2 attention projections
N_FOLDS = 2
SEED = 42
EPOCHS = 15
BATCH_SIZE = 8
GRAD_ACCUM = 4            # effective batch = BATCH_SIZE × GRAD_ACCUM = 32
LR = 1e-4                 # Standard LoRA LR (adapters start at zero, need higher LR)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 7              # early-stopping patience (epochs)

DATA_DIR = Path("preprocessed")
OUTPUT_DIR = Path("outputs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

class PCLFinetuneDataset(Dataset):
    """Tokenised dataset for binary PCL classification."""

    def __init__(self, records: List[Dict], tokenizer, max_length: int = MAX_LENGTH):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(rec["label_binary"], dtype=torch.long),
        }

class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weighted cross-entropy."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self._class_weights = class_weights  # kept on CPU; moved lazily

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self._class_weights is not None:
            w = self._class_weights.to(logits.device)
            loss = nn.functional.cross_entropy(logits, labels, weight=w)
        else:
            loss = nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
    }

def load_json(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def compute_class_weights(records: List[Dict]) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N / (C × n_c)."""
    labels = [r["label_binary"] for r in records]
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = len(labels) / (2.0 * counts)
    log.info(f"Class weights  NO_PCL={weights[0]:.3f}  PCL={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def make_strat_key(records: List[Dict]) -> np.ndarray:
    """Stratification key = keyword × label for keyword-stratified CV."""
    return np.array([f"{r['keyword']}_{r['label_binary']}" for r in records])


def oversample_minority(records: List[Dict], seed: int = SEED) -> List[Dict]:
    """Duplicate PCL samples until roughly balanced with NO_PCL."""
    pos_idx = [i for i, r in enumerate(records) if r["label_binary"] == 1]
    neg_idx = [i for i, r in enumerate(records) if r["label_binary"] == 0]
    if len(pos_idx) >= len(neg_idx):
        return list(records)

    rng = np.random.default_rng(seed)
    n_needed = len(neg_idx) - len(pos_idx)
    extra_idx = rng.choice(pos_idx, size=n_needed, replace=True).tolist()
    all_idx = neg_idx + pos_idx + extra_idx
    rng.shuffle(all_idx)
    log.info(f"  Oversampled PCL: {len(pos_idx)} → {len(pos_idx) + n_needed} "
             f"(total {len(all_idx)})")
    return [records[i] for i in all_idx]


def build_lora_model(rank: Optional[int]):
    """Load pretrained DeBERTa-v3-large and attach LoRA adapters.

    If `rank` is None the base model is returned and no PEFT/LoRA is applied.
    """
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, torch_dtype=torch.float32,
    )

    if rank is None:
        log.info("  No LoRA/PEFT requested — using base model")
        return base

    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank * LORA_ALPHA_MULT,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base, cfg)
    
    # Fix classifier head initialisation
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == 2:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"  LoRA r={rank}  trainable {trainable:,} / {total:,} "
             f"({100 * trainable / total:.2f}%)")
    return model


def train_fold(
    *,
    fold,
    rank: int,
    train_records: List[Dict],
    val_records: List[Dict],
    tokenizer,
    class_weights: Optional[torch.Tensor],
    output_dir: Path,
    args: argparse.Namespace,
    model: Optional[nn.Module] = None,
    freeze_backbone: bool = False,
) -> Dict[str, float]:
    """Train one (fold, rank) combination. Returns val metrics dict."""
    log.info(f"  Fold {fold}  train={len(train_records)}  val={len(val_records)}")

    # Optionally oversample
    if args.balance in ("oversample", "both"):
        train_records = oversample_minority(train_records, seed=args.seed)

    train_ds = PCLFinetuneDataset(train_records, tokenizer, args.max_length)
    val_ds = PCLFinetuneDataset(val_records, tokenizer, args.max_length)

    # If a model was provided (e.g. baseline), use it; otherwise build/apply LoRA
    if model is None:
        model = build_lora_model(rank)
    else:
        log.info("  Using provided base model (no LoRA applied)")

    # Optionally freeze the backbone and leave only the classification head trainable
    if freeze_backbone:
        for n, p in model.named_parameters():
            p.requires_grad = False

        # Try common classifier attribute names, otherwise fall back to name matching
        if hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "score"):
            for p in model.score.parameters():
                p.requires_grad = True
        else:
            for n, p in model.named_parameters():
                if "classifier" in n or "pooler" in n or "out_proj" in n:
                    p.requires_grad = True

        # Log count of trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info(f"  Frozen backbone — trainable {trainable:,} / {total:,} "
                 f"({100 * trainable / total:.2f}%)")

    run_dir = output_dir / f"rank{rank}_fold{fold}"
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        bf16=torch.cuda.is_available(),   # bf16 instead of fp16 — DeBERTa-v2's disentangled attention breaks under fp16 autocast
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
    )

    # Use class-weighted loss if set; otherwise vanilla cross-entropy
    cw = class_weights if args.balance in ("weight", "both") else None
    trainer = WeightedTrainer(
        class_weights=cw,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Optionally save adapter for final model
    if fold == "final":
        save_dir = output_dir / f"best_rank{rank}_final"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        log.info(f"  Saved final adapter → {save_dir}")

    # Free GPU + CPU memory aggressively to avoid OOM killer
    del model, trainer, train_ds, val_ds
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return {
        "f1": metrics["eval_f1"],
        "precision": metrics["eval_precision"],
        "recall": metrics["eval_recall"],
    }


def _jsonify(obj):
    """Convert numpy scalars for json.dump."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning sweep for PCL detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ranks", nargs="+", type=int, default=LORA_RANKS,
                        help="LoRA ranks to sweep")
    parser.add_argument("--folds", type=int, default=N_FOLDS,
                        help="K-fold CV splits (0 = no CV, train on full train → eval on dev)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--balance", choices=["weight", "oversample", "both", "none"],
                        default="both",
                        help="How to handle class imbalance")
    parser.add_argument("--masked", action="store_true",
                        help="Use entity-masked text variants")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without training")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluation with frozen backbone")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_masked" if args.masked else ""
    train_data = load_json(args.data_dir / f"train{suffix}.json")
    dev_data = load_json(args.data_dir / f"dev{suffix}.json")
    log.info(f"Data: train={len(train_data)} (PCL={sum(r['label_binary'] for r in train_data)})  "
             f"dev={len(dev_data)} (PCL={sum(r['label_binary'] for r in dev_data)})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    class_weights = compute_class_weights(train_data)

    if args.dry_run:
        log.info("DRY RUN — Configuration:")
        for k, v in vars(args).items():
            log.info(f"  {k}: {v}")
        build_lora_model(args.ranks[0])  # show param counts
        return

    if args.baseline:
        log.info("Running baseline evaluation with frozen backbone")
        train_data = load_json(args.data_dir / f"train{suffix}.json")
        dev_data = load_json(args.data_dir / f"dev{suffix}.json")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        class_weights = compute_class_weights(train_data)

        # Load the base model directly
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.requires_grad_(False)  # Freeze the backbone
        model.classifier.requires_grad_(True)  # Unfreeze the classifier head

        m = train_fold(
            fold=0,
            rank=None,  # Skip LoRA or any PEFT application
            train_records=train_data,
            val_records=dev_data,
            tokenizer=tokenizer,
            class_weights=class_weights,
            output_dir=args.output_dir,
            args=args,
            model=model,
            freeze_backbone=True,  # Ensure backbone is frozen
        )
        log.info(f"Baseline evaluation →  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
        return

    # Rank sweep
    results = {}

    for rank in args.ranks:
        log.info(f"\n{'═' * 60}")
        log.info(f"  LoRA rank = {rank}")
        log.info(f"{'═' * 60}")

        if args.folds > 0:
            # Keyword-stratified K-fold CV
            strat_keys = make_strat_key(train_data)

            # Fall back to label-only if keyword×label groups are too small
            try:
                skf = StratifiedKFold(
                    n_splits=args.folds, shuffle=True, random_state=args.seed
                )
                list(skf.split(train_data, strat_keys))  # dry-run to check
            except ValueError:
                log.warning(
                    "Keyword×label groups too small for %d-fold CV — "
                    "falling back to label-only stratification", args.folds
                )
                strat_keys = np.array([r["label_binary"] for r in train_data])
                skf = StratifiedKFold(
                    n_splits=args.folds, shuffle=True, random_state=args.seed
                )

            fold_metrics = []
            for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(train_data, strat_keys)
            ):
                fold_train = [train_data[i] for i in train_idx]
                fold_val = [train_data[i] for i in val_idx]

                m = train_fold(
                    fold=fold_idx,
                    rank=rank,
                    train_records=fold_train,
                    val_records=fold_val,
                    tokenizer=tokenizer,
                    class_weights=class_weights,
                    output_dir=args.output_dir,
                    args=args,
                )
                fold_metrics.append(m)
                log.info(f"  Fold {fold_idx} →  F1={m['f1']:.4f}  "
                         f"P={m['precision']:.4f}  R={m['recall']:.4f}")

            mean_f1 = np.mean([m["f1"] for m in fold_metrics])
            std_f1 = np.std([m["f1"] for m in fold_metrics])
            mean_p = np.mean([m["precision"] for m in fold_metrics])
            mean_r = np.mean([m["recall"] for m in fold_metrics])
            log.info(
                f"\n  rank {rank} CV:  F1={mean_f1:.4f}±{std_f1:.4f}  "
                f"P={mean_p:.4f}  R={mean_r:.4f}"
            )
            results[rank] = {
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_precision": mean_p,
                "mean_recall": mean_r,
                "folds": fold_metrics,
            }
        else:
            # No CV — train on full train, evaluate on dev
            m = train_fold(
                fold=0,
                rank=rank,
                train_records=train_data,
                val_records=dev_data,
                tokenizer=tokenizer,
                class_weights=class_weights,
                output_dir=args.output_dir,
                args=args,
            )
            log.info(f"  rank {rank} dev →  F1={m['f1']:.4f}  "
                     f"P={m['precision']:.4f}  R={m['recall']:.4f}")
            results[rank] = {
                "dev_f1": m["f1"],
                "dev_precision": m["precision"],
                "dev_recall": m["recall"],
            }

    # Sweep summary
    log.info(f"\n{'═' * 60}")
    log.info("  SWEEP SUMMARY")
    log.info(f"{'═' * 60}")

    best_rank, best_f1 = None, -1.0
    for rank, res in results.items():
        f1_key = "mean_f1" if "mean_f1" in res else "dev_f1"
        f1_val = res[f1_key]
        std_tag = f" ± {res['std_f1']:.4f}" if "std_f1" in res else ""
        log.info(f"  rank {rank:>2d}:  F1 = {f1_val:.4f}{std_tag}")
        if f1_val > best_f1:
            best_f1 = f1_val
            best_rank = rank

    log.info(f"\n  ★ Best rank = {best_rank}  (F1 = {best_f1:.4f})")

    # Save results
    results_path = args.output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(_jsonify({"config": vars(args), "results": {str(k): v for k, v in results.items()}}), f, indent=2, default=str)
    log.info(f"  Results → {results_path}")

    # Retrain best rank on full train, then eval on dev
    if args.folds > 0:
        log.info(f"\n  Retraining best rank={best_rank} on full train → eval on dev …")
        final = train_fold(
            fold="final",
            rank=best_rank,
            train_records=train_data,
            val_records=dev_data,
            tokenizer=tokenizer,
            class_weights=class_weights,
            output_dir=args.output_dir,
            args=args,
        )
        log.info(
            f"  Final dev →  F1={final['f1']:.4f}  "
            f"P={final['precision']:.4f}  R={final['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
