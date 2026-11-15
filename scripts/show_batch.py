"""Show the input format: token lengths, one encoded example and the tensors of one batch.

    python scripts/show_batch.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artp2.data import SEQ_FORMATS, Collator, encode_dataset, load_art  # noqa: E402

CHECKPOINT = "roberta-base"
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
ds = load_art()

print("== tokens per sequence, special tokens included, no truncation")
for seq_format in SEQ_FORMATS:
    measured = encode_dataset(ds, tokenizer, seq_format, max_length=tokenizer.model_max_length)
    for split in measured:
        lengths = torch.tensor([len(ids) for pair in measured[split]["input_ids"] for ids in pair], dtype=torch.float)
        print(f"{seq_format}  {split:11s} mean {lengths.mean():5.1f}   99 %: {lengths.quantile(0.99):4.0f}   "
              f"max {lengths.max():4.0f}   longer than {MAX_LENGTH}: {int((lengths > MAX_LENGTH).sum())}")

encoded = encode_dataset(ds, tokenizer, "o1_h_o2", MAX_LENGTH)
example = encoded["train"][0]
print(f"\n== one example (o1_h_o2), label {example['label']}")
for ids in example["input_ids"]:
    print(tokenizer.decode(ids))

print("\n== one batch of 4 examples")
loader = DataLoader(encoded["train"], batch_size=4, collate_fn=Collator(tokenizer.pad_token_id))
batch = next(iter(loader))
for name, tensor in batch.items():
    print(f"{name:15s} {tuple(tensor.shape)}")
print("input_ids[0]:\n", batch["input_ids"][0])
print("attention_mask[0]:\n", batch["attention_mask"][0])
