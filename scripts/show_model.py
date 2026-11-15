"""Show models 1 and 2: configuration, parameter counts and the output for one batch.

    python scripts/show_model.py
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artp2.data import Collator, encode_dataset, load_art  # noqa: E402
from artp2.models import ARTModel, count_parameters  # noqa: E402

CHECKPOINT = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
encoded = encode_dataset(load_art(), tokenizer)
batch = next(iter(DataLoader(encoded["validation"], batch_size=4, collate_fn=Collator(tokenizer.pad_token_id))))

models = {}
for name, pretrained in (("model 1 (random init)", False), ("model 2 (pretrained)", True)):
    torch.manual_seed(0)
    model = ARTModel(CHECKPOINT, pretrained=pretrained).eval()
    models[name] = model
    with torch.no_grad():
        logits = model(batch)
    print(f"\n== {name}")
    print("parameters:", {key: f"{value:,}" for key, value in count_parameters(model).items()})
    print("logits shape:", tuple(logits.shape))
    print("logits:\n", logits)

config = model.encoder.config
print("\n== model hyperparameters (identical for both models)")
for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size", "hidden_act",
            "vocab_size", "max_position_embeddings", "hidden_dropout_prob", "attention_probs_dropout_prob"):
    print(f"{key:30s} {getattr(config, key)}")

random_init, pretrained = (dict(m.encoder.named_parameters()) for m in models.values())
same_shapes = random_init.keys() == pretrained.keys() and all(
    random_init[key].shape == pretrained[key].shape for key in random_init)
same_values = sum(torch.equal(random_init[key], pretrained[key]) for key in random_init)
print(f"\nencoder tensors: {len(random_init)}   same names and shapes: {same_shapes}   "
      f"with identical values: {same_values}")
