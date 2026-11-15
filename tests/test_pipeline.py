"""Sanity checks for the data, the input format, the models and the learning rate schedule.

    python -m pytest -q tests

The model tests use a tiny RoBERTa (5 layers, hidden size 32), so they run in seconds on a CPU.
The first run downloads the dataset, two tokenizers and the tiny model from the Hugging Face hub.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artp2.data import SEQ_FORMATS, Collator, encode_dataset, load_art, text_pairs  # noqa: E402
from artp2.lit import lr_factor  # noqa: E402
from artp2.models import POOLINGS, ARTModel  # noqa: E402
from artp2.train import Config, default_group  # noqa: E402

TINY = "hf-internal-testing/tiny-random-roberta"
EXAMPLE = {
    "observation_1": "Ann lost her keys.",
    "observation_2": "She was late for work.",
    "hypothesis_1": "She found them at once.",
    "hypothesis_2": "She searched the whole flat.",
    "label": 1,
}
LONGER = {**EXAMPLE, "hypothesis_1": EXAMPLE["hypothesis_1"] + " Then she drove to the office as fast as she could."}


@pytest.fixture(scope="module")
def ds():
    return load_art()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base")


@pytest.fixture(scope="module")
def tiny_tokenizer():
    return AutoTokenizer.from_pretrained(TINY)


def make_batch(examples, tokenizer):
    rows = list(encode_dataset(Dataset.from_list(examples), tokenizer))
    return Collator(tokenizer.pad_token_id)(rows), rows


# ---- data -------------------------------------------------------------------------------
def test_split_sizes_and_labels(ds):
    assert {split: len(ds[split]) for split in ds} == {"train": 16801, "validation": 1000, "test": 1532}
    for split in ds:
        assert set(ds[split]["label"]) == {0, 1}


# ---- input format -----------------------------------------------------------------------
def test_text_pairs():
    assert text_pairs(EXAMPLE, "o1_h_o2")[0] == ("Ann lost her keys. She found them at once.", "She was late for work.")
    assert text_pairs(EXAMPLE, "o1_o2_h")[0] == ("Ann lost her keys. She was late for work.", "She found them at once.")
    for seq_format in SEQ_FORMATS:
        texts = [" ".join(pair) for pair in text_pairs(EXAMPLE, seq_format)]
        assert len(texts) == 2
        assert EXAMPLE["hypothesis_1"] in texts[0] and EXAMPLE["hypothesis_2"] not in texts[0]
        assert EXAMPLE["hypothesis_2"] in texts[1] and EXAMPLE["hypothesis_1"] not in texts[1]
        for text in texts:  # both sequences contain both observations
            assert EXAMPLE["observation_1"] in text and EXAMPLE["observation_2"] in text


def test_encoding_adds_special_tokens_but_no_padding(tokenizer):
    _, rows = make_batch([EXAMPLE], tokenizer)
    assert len(rows[0]["input_ids"]) == 2
    for ids in rows[0]["input_ids"]:
        assert ids[0] == tokenizer.bos_token_id and ids[-1] == tokenizer.eos_token_id
        assert tokenizer.pad_token_id not in ids
        assert tokenizer.decode(ids).count("</s></s>") == 1  # exactly one boundary between the two segments


def test_collator_pads_to_the_longest_sequence_and_masks_the_padding(tokenizer):
    batch, rows = make_batch([EXAMPLE, LONGER], tokenizer)
    lengths = torch.tensor([[len(ids) for ids in row["input_ids"]] for row in rows])
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (2, 2, int(lengths.max()))
    assert batch["label"].tolist() == [1, 1]
    assert torch.equal(batch["attention_mask"].sum(dim=-1), lengths)
    assert torch.equal(batch["input_ids"] == tokenizer.pad_token_id, batch["attention_mask"] == 0)


# ---- models -----------------------------------------------------------------------------
def test_models_1_and_2_have_the_same_architecture_and_both_train_with_dropout():
    random_init, pretrained = ARTModel(TINY, pretrained=False), ARTModel(TINY, pretrained=True)
    first, second = dict(random_init.named_parameters()), dict(pretrained.named_parameters())
    assert first.keys() == second.keys()
    assert all(first[name].shape == second[name].shape for name in first)
    matrices = [name for name in first if name.startswith("encoder.") and first[name].dim() > 1]
    assert not any(torch.equal(first[name], second[name]) for name in matrices)  # different weights
    for model in (random_init, pretrained):
        assert all(module.training for module in model.modules())  # nothing left in evaluation mode


@pytest.mark.parametrize("pooling", POOLINGS)
def test_swapping_the_hypotheses_swaps_the_scores(tiny_tokenizer, pooling):
    torch.manual_seed(0)
    model = ARTModel(TINY, pretrained=True, pooling=pooling).eval()
    swapped = {**EXAMPLE, "hypothesis_1": EXAMPLE["hypothesis_2"], "hypothesis_2": EXAMPLE["hypothesis_1"]}
    with torch.no_grad():
        logits = model(make_batch([EXAMPLE], tiny_tokenizer)[0])
        logits_swapped = model(make_batch([swapped], tiny_tokenizer)[0])
    assert logits.shape == (1, 2)
    assert torch.allclose(logits, logits_swapped.flip(dims=[1]), atol=1e-5)


@pytest.mark.parametrize("pooling", POOLINGS)
def test_padding_does_not_change_the_scores(tiny_tokenizer, pooling):
    torch.manual_seed(0)
    model = ARTModel(TINY, pretrained=True, pooling=pooling).eval()
    with torch.no_grad():
        alone = model(make_batch([EXAMPLE], tiny_tokenizer)[0])
        padded = model(make_batch([EXAMPLE, LONGER], tiny_tokenizer)[0])[:1]
    assert torch.allclose(alone, padded, atol=1e-5)


def test_model_can_memorize_eight_examples(ds, tiny_tokenizer):
    """With a wrong loss, wrong labels or broken gradients the model could not learn 8 examples."""
    torch.manual_seed(0)
    batch, _ = make_batch([ds["train"][index] for index in range(8)], tiny_tokenizer)
    model = ARTModel(TINY, pretrained=False).eval()  # evaluation mode = no dropout, gradients still flow
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(100):
        loss = F.cross_entropy(model(batch), batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = (model(batch).argmax(dim=-1) == batch["label"]).float().mean().item()
    assert accuracy == 1.0 and loss.item() < 0.1


# ---- training ---------------------------------------------------------------------------
def test_learning_rate_schedule():
    total_steps = 1000
    factors = [lr_factor(step, total_steps, warmup_ratio=0.1) for step in range(total_steps)]
    assert factors[0] == pytest.approx(0.01)  # starts at lr / 100
    assert factors.index(max(factors)) == 100 and max(factors) == pytest.approx(1.0)  # peak after the warmup
    assert all(before < after for before, after in zip(factors[:100], factors[1:101]))
    assert all(before > after for before, after in zip(factors[100:-1], factors[101:]))
    assert 1e-4 <= factors[-1] < 0.01  # ends far below the start
    for few_steps in (1, 2, 3, 10):  # very short runs must not fail
        assert all(0 < lr_factor(step, few_steps, 0.1) <= 1 for step in range(few_steps))


def test_run_names_are_short_and_name_what_differs_from_the_default():
    assert default_group(Config()) == "pre-lr2e-05"
    assert default_group(Config(pretrained=False, lr=1e-4, max_epochs=30)) == "rand-lr0.0001-ep30"
    assert default_group(Config(seq_format="o1_o2_h", pooling="mean")) == "pre-lr2e-05-o1o2h-mean"
