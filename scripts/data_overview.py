"""Print what we need to know about the splits before building anything.

    python scripts/data_overview.py
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artp2.data import TEXT_FIELDS, load_art  # noqa: E402

ds = load_art()
print(ds, "\n")

print("== sizes and label balance (after mapping 1/2 -> 0/1)")
for split in ds:
    counts = collections.Counter(ds[split]["label"])
    n = len(ds[split])
    print(f"{split:11s} n={n:6d}  label 0: {counts[0]:5d}  label 1: {counts[1]:5d}  "
          f"majority class: {max(counts.values()) / n:.3f}")

print("\n== are the splits ordered? (blocks in which observation_1 is sorted alphabetically)")
for split in ds:
    o1 = ds[split]["observation_1"]
    breaks = sum(1 for a, b in zip(o1, o1[1:]) if a > b)
    print(f"{split:11s} {breaks + 1:5d} sorted blocks in {len(o1)} rows")

print("\n== overlap of observation pairs between splits")
pairs = {split: set(zip(ds[split]["observation_1"], ds[split]["observation_2"])) for split in ds}
print(f"validation and train: {len(pairs['validation'] & pairs['train'])}   "
      f"test and train: {len(pairs['test'] & pairs['train'])}   "
      f"test and validation: {len(pairs['test'] & pairs['validation'])}")

print("\n== rows with two identical hypotheses")
for split in ds:
    same = sum(1 for h1, h2 in zip(ds[split]["hypothesis_1"], ds[split]["hypothesis_2"]) if h1 == h2)
    print(f"{split:11s} {same}")

print("\n== one example")
example = ds["train"][0]
for field in TEXT_FIELDS:
    print(f"{field:14s} {example[field]}")
print(f"{'label':14s} {example['label']}")
