"""Collect the results of all runs into results/summary.csv and print them, with mean and standard deviation per configuration.

    python scripts/collect_results.py
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = ("pretrained", "lr", "classifier_lr", "seq_format", "pooling", "train_size", "batch_size", "max_epochs", "patience",
               "seed", "tags")
SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))

rows = []
for results_file in sorted((ROOT / "runs").glob("*/results.json")):
    results = json.loads(results_file.read_text())
    config = json.loads((results_file.parent / "config.json").read_text())
    row = {"run": results["run_name"], "group": results["group"]}
    row.update({key: config.get(key, 0) for key in CONFIG_KEYS})  # runs before phase 2 have no train_size
    row.update({"best_epoch": results["best_epoch"], "epochs_trained": results["epochs_trained"]})
    for split, short in SPLITS:
        row[f"{short}_acc"] = round(results[split]["acc"], 4)
        row[f"{short}_loss"] = round(results[split]["loss"], 4)
    row["train_seconds"] = round(results["train_seconds"])
    rows.append(row)

(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "summary.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows)} runs -> results/summary.csv\n")
print(f"{'run':34s} {'best ep':>7s} {'train':>7s} {'val':>7s} {'test':>7s} {'val loss':>9s}")
for row in rows:
    print(f"{row['run']:34s} {row['best_epoch']:7d} {row['train_acc']:7.3f} {row['val_acc']:7.3f} {row['test_acc']:7.3f} {row['val_loss']:9.3f}")

groups = defaultdict(list)
for row in rows:
    groups[row["group"]].append(row)
print(f"\n{'configuration':30s} {'seeds':>5s} {'val mean':>9s} {'val sd':>7s} {'test mean':>10s} {'test sd':>8s}")
for group, members in groups.items():
    val = [member["val_acc"] for member in members]
    test = [member["test_acc"] for member in members]
    val_sd = statistics.stdev(val) if len(val) > 1 else float("nan")
    test_sd = statistics.stdev(test) if len(test) > 1 else float("nan")
    print(f"{group:30s} {len(members):5d} {statistics.mean(val):9.3f} {val_sd:7.3f} {statistics.mean(test):10.3f} {test_sd:8.3f}")
