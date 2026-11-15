"""ART (αNLI) data: loading the course version from the Hugging Face hub."""
from __future__ import annotations

from datasets import DatasetDict, Value, load_dataset

DATASET_ID = "amarfurt/art-nlp-hs26"
TEXT_FIELDS = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")


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
