"""Collect the results of all runs and print them.

    python scripts/collect_results.py

Models 1 and 2 -> results/summary.csv, with mean and standard deviation per configuration.
Model 3 (LLM)  -> results/summary_llm.csv
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = ("pretrained", "lr", "classifier_lr", "seq_format", "pooling", "train_size", "batch_size", "max_epochs", "patience",
               "seed", "tags")
LLM_KEYS = ("model", "provider", "prompt", "shots", "swap", "reasoning_effort", "max_tokens", "tags")
SPLITS = (("train", "train"), ("validation", "val"), ("test", "test"))

rows, llm_rows = [], []
for results_file in sorted((ROOT / "runs").glob("*/results.json")):
    results = json.loads(results_file.read_text())
    config = json.loads((results_file.parent / "config.json").read_text())
    if "prompt" in config:  # model 3: no training, not every split is evaluated in every run
        row = {"run": results["run_name"]}
        row.update({key: config[key] for key in LLM_KEYS})
        row["shots"] = config["shots"] if config["prompt"] == "few" else 0
        for split, short in SPLITS:
            result = results.get(split, {})
            row[f"{short}_n"] = result.get("n", "")
            row[f"{short}_acc"] = round(result["acc"], 4) if result else ""
            row[f"{short}_invalid"] = result.get("invalid", "")
        row["prompt_tokens"] = sum(results[split]["prompt_tokens"] for split, _ in SPLITS if split in results)
        row["completion_tokens"] = sum(results[split]["completion_tokens"] for split, _ in SPLITS if split in results)
        llm_rows.append(row)
        continue
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
    writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
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

if llm_rows:
    with open(ROOT / "results" / "summary_llm.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(llm_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(llm_rows)
    print(f"\n{len(llm_rows)} LLM runs -> results/summary_llm.csv\n")
    print(f"{'run':50s} {'val n':>6s} {'val':>7s} {'invalid':>7s} {'test n':>6s} {'test':>7s} {'invalid':>7s} {'tokens in':>10s} {'out':>9s}")
    for row in llm_rows:
        cells = [f"{row[key]:{width}}" for key, width in (("val_n", 6), ("val_acc", 7), ("val_invalid", 7), ("test_n", 6),
                                                         ("test_acc", 7), ("test_invalid", 7))]
        print(f"{row['run']:50s} " + " ".join(cells) + f" {row['prompt_tokens']:10d} {row['completion_tokens']:9d}")
