"""ART (αNLI) data: loading, input format and batching."""
from __future__ import annotations

import torch
from datasets import DatasetDict, Value, load_dataset

DATASET_ID = "amarfurt/art-nlp-hs26"
TEXT_FIELDS = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")
SEQ_FORMATS = ("o1_h_o2", "o1_o2_h")


def load_art() -> DatasetDict:
    """Load the train / validation / test splits.

    The hub dataset stores the label as the number of the correct hypothesis (1 or 2; the class
    name "0" is never used). It is mapped to 0 / 1 = index of the correct hypothesis, so that it
    can be used directly as the target of a 2-way cross-entropy.
    """
    ds = load_dataset(DATASET_ID)
    labels = {label for split in ds.values() for label in split["label"]}
    if labels != {1, 2}:
        raise ValueError(f"unexpected label values {labels}")
    ds = ds.cast_column("label", Value("int64"))
    return ds.map(lambda example: {"label": example["label"] - 1}, desc="labels 1/2 -> 0/1")


def text_pairs(example: dict, seq_format: str = "o1_h_o2") -> list[tuple[str, str]]:
    """The two text segments of the model input, once per hypothesis.

    * ``o1_h_o2``: (o1 + h, o2) - narrative order, the hypothesis stands between the observations
    * ``o1_o2_h``: (o1 + o2, h) - both observations as context, then the hypothesis

    The tokenizer joins the two segments with its special tokens.
    """
    o1, o2 = example["observation_1"], example["observation_2"]
    pairs = []
    for hypothesis in (example["hypothesis_1"], example["hypothesis_2"]):
        if seq_format == "o1_h_o2":
            pairs.append((f"{o1} {hypothesis}", o2))
        elif seq_format == "o1_o2_h":
            pairs.append((f"{o1} {o2}", hypothesis))
        else:
            raise ValueError(f"unknown seq_format {seq_format!r}")
    return pairs


def encode_dataset(ds: DatasetDict, tokenizer, seq_format: str = "o1_h_o2", max_length: int = 128) -> DatasetDict:
    """Tokenize every example into two sequences, one per hypothesis, without padding.

    Adds the column ``input_ids``: two lists of token ids. Sequences with more than ``max_length``
    tokens are truncated.
    """

    def encode(example: dict) -> dict:
        firsts, seconds = zip(*text_pairs(example, seq_format))
        encoded = tokenizer(list(firsts), list(seconds), truncation=True, max_length=max_length)
        return {"input_ids": encoded["input_ids"]}

    return ds.map(encode, desc=f"tokenize ({seq_format})")


class Collator:
    """Pad the 2 * B sequences of a batch to the longest one and build the attention mask.

    Returns ``input_ids`` and ``attention_mask`` with shape [B, 2, L] and ``label`` with shape [B].
    The attention mask is 1 for tokens and 0 for padding.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        length = max(len(ids) for example in examples for ids in example["input_ids"])
        input_ids = torch.full((len(examples), 2, length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), 2, length), dtype=torch.long)
        for i, example in enumerate(examples):
            for j, ids in enumerate(example["input_ids"]):
                input_ids[i, j, : len(ids)] = torch.tensor(ids)
                attention_mask[i, j, : len(ids)] = 1
        labels = torch.tensor([example["label"] for example in examples])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}
