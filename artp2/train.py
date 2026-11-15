"""Experiment runner for models 1 and 2: configuration -> data -> model -> training -> evaluation.

    python train.py --pretrained true                               # model 2: pretrained roberta-base
    python train.py --pretrained false --lr 1e-4 --max_epochs 30    # model 1: same architecture, random init
    python train.py --subset 64 --wandb_mode disabled               # smoke test on 64 examples per split

Every run writes ``runs/<run_name>/`` with config.json, results.json (accuracy, loss and confusion
matrix on train, validation and test for the best checkpoint), predictions_{validation,test}.csv and
metrics/version_0/metrics.csv (local copy of the curves), and logs to Weights & Biases.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import Collator, encode_dataset, load_art
from .lit import ARTClassifier
from .models import ARTModel, count_parameters


@dataclass
class Config:
    # model
    checkpoint: str = "roberta-base"  # architecture and tokenizer of both models
    pretrained: bool = True  # True = model 2 (pretrained weights), False = model 1 (random init)
    pooling: str = "first"  # first (hidden state of <s>) | mean
    dropout: float = 0.1  # dropout in the classifier
    # input
    seq_format: str = "o1_h_o2"  # o1_h_o2 | o1_o2_h | h_only
    max_length: int = 128  # truncation limit in tokens (the longest sequence has 65)
    train_size: int = 0  # train on a random subset of N training examples (drawn with the seed); 0 = all
    # training
    lr: float = 2e-5
    classifier_lr: float = 0.0  # 0 = same as lr
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # share of all steps used for the linear warmup
    batch_size: int = 32
    max_epochs: int = 5
    patience: int = 2  # early stopping: epochs without a better validation accuracy
    grad_clip: float = 1.0
    seed: int = 42
    # tracking
    wandb_project: str = "nlp-project2-art"
    wandb_mode: str = "online"  # online | offline | disabled
    run_name: str = ""  # "" = built from the configuration
    group: str = ""  # "" = run name without the seed (groups the seeds of one configuration)
    tags: str = ""  # comma-separated
    output_dir: str = "runs"
    keep_checkpoint: bool = False  # the best checkpoint (0.5 GB) is deleted after the evaluation
    # debugging
    subset: int = 0  # use only the first N examples of every split; 0 = all


def default_group(cfg: Config) -> str:
    """Short name of a configuration: model, learning rate and every value that differs from the default."""
    default = Config()
    parts = ["pre" if cfg.pretrained else "rand", f"lr{cfg.lr:g}"]
    if cfg.classifier_lr:
        parts.append(f"clr{cfg.classifier_lr:g}")
    if cfg.seq_format != default.seq_format:
        parts.append(cfg.seq_format.replace("_", ""))
    if cfg.pooling != default.pooling:
        parts.append(cfg.pooling)
    if cfg.dropout != default.dropout:
        parts.append(f"do{cfg.dropout:g}")
    if cfg.weight_decay != default.weight_decay:
        parts.append(f"wd{cfg.weight_decay:g}")
    if cfg.batch_size != default.batch_size:
        parts.append(f"bs{cfg.batch_size}")
    if cfg.max_epochs != default.max_epochs:
        parts.append(f"ep{cfg.max_epochs}")
    if cfg.train_size:
        parts.append(f"n{cfg.train_size}")
    if cfg.subset:
        parts.append(f"sub{cfg.subset}")
    return "-".join(parts)


def str2bool(value: str) -> bool:
    if value.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in fields(Config):
        kind = type(field.default)
        parser.add_argument(f"--{field.name}", type=str2bool if kind is bool else kind, default=field.default)
    return parser


def evaluate(trainer: L.Trainer, lit: ARTClassifier, loader: DataLoader, labels: list[int],
             ckpt_path: str | None = None) -> tuple[dict, torch.Tensor]:
    """Accuracy, loss and confusion matrix (rows = true label, columns = prediction) of one split."""
    probs = torch.cat(trainer.predict(lit, dataloaders=loader, ckpt_path=ckpt_path)).float().cpu()
    target = torch.tensor(labels)
    pred = probs.argmax(dim=1)
    metrics = {
        "acc": (pred == target).float().mean().item(),
        "loss": F.nll_loss(torch.log(probs.clamp_min(1e-12)), target).item(),
        "confusion": [[int(((target == t) & (pred == p)).sum()) for p in (0, 1)] for t in (0, 1)],
    }
    return metrics, probs


def save_predictions(path: Path, labels: list[int], probs: torch.Tensor) -> None:
    """One row per example in the order of the split: label, both probabilities, prediction."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "label", "p_h1", "p_h2", "pred", "correct"])
        for index, (label, (p_h1, p_h2)) in enumerate(zip(labels, probs.tolist())):
            pred = int(p_h2 > p_h1)
            writer.writerow([index, label, f"{p_h1:.6f}", f"{p_h2:.6f}", pred, int(pred == label)])


def run(cfg: Config) -> dict:
    group = cfg.group or default_group(cfg)
    run_name = cfg.run_name or f"{group}-s{cfg.seed}"
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    start = time.time()

    # data: both models use the tokenizer of the checkpoint
    tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
    ds = load_art()
    if cfg.subset:
        ds = ds.filter(lambda example, index: index < cfg.subset, with_indices=True)
    encoded = encode_dataset(ds, tokenizer, cfg.seq_format, cfg.max_length)
    splits = {split: encoded[split] for split in ("train", "validation", "test")}
    if cfg.train_size:
        splits["train"] = splits["train"].shuffle(seed=cfg.seed).select(range(cfg.train_size))
    collate = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(splits["train"], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    eval_loaders = {split: DataLoader(splits[split], batch_size=2 * cfg.batch_size, collate_fn=collate)
                    for split in splits}

    # model
    model = ARTModel(cfg.checkpoint, cfg.pretrained, cfg.pooling, cfg.dropout)
    lit = ARTClassifier(model, cfg.lr, cfg.classifier_lr, cfg.weight_decay, cfg.warmup_ratio)
    n_params = count_parameters(model)

    # training
    logger = WandbLogger(
        project=cfg.wandb_project, name=run_name, group=group, tags=[tag for tag in cfg.tags.split(",") if tag],
        save_dir=str(run_dir), mode=cfg.wandb_mode, log_model=False,
        config={**asdict(cfg), "n_params": n_params["total"], "n_train": len(splits["train"])},
    )
    csv_logger = CSVLogger(save_dir=str(run_dir), name="metrics", version=0)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", monitor="val/acc", mode="max", save_top_k=1, save_weights_only=True,
        filename="best-epoch{epoch}", auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(monitor="val/acc", mode="max", patience=cfg.patience)
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=cfg.grad_clip,
        callbacks=[checkpoint, early_stopping],
        logger=[logger, csv_logger],
        log_every_n_steps=10,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=eval_loaders["validation"])
    train_seconds = time.time() - start

    # evaluation of the best checkpoint on all three splits (dropout off)
    best_path = checkpoint.best_model_path
    results = {
        "run_name": run_name,
        "group": group,
        "n_params": n_params,
        "epochs_trained": trainer.current_epoch,
        "best_epoch": int(re.search(r"best-epoch(\d+)", best_path).group(1)),
        "train_seconds": train_seconds,
    }
    for index, split in enumerate(("validation", "test", "train")):
        labels = splits[split]["label"]
        metrics, probs = evaluate(trainer, lit, eval_loaders[split], list(labels), best_path if index == 0 else None)
        results[split] = metrics
        if split != "train":
            save_predictions(run_dir / f"predictions_{split}.csv", list(labels), probs)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    if not cfg.keep_checkpoint:
        shutil.rmtree(run_dir / "checkpoints")

    summary = logger.experiment.summary
    summary["best/epoch"] = results["best_epoch"]
    for split, short in (("train", "train"), ("validation", "val"), ("test", "test")):
        summary[f"best/{short}_acc"] = results[split]["acc"]
        summary[f"best/{short}_loss"] = results[split]["loss"]
    logger.experiment.finish()
    print(f"[{run_name}] best epoch {results['best_epoch']} | train {results['train']['acc']:.4f} | "
          f"val {results['validation']['acc']:.4f} | test {results['test']['acc']:.4f} | {train_seconds:.0f}s")
    return results


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return run(Config(**vars(args)))


if __name__ == "__main__":
    main()
