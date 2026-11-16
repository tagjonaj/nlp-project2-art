"""Checks for the LLM part: prompt, parsing of the answer, sampling and the run with a fake client.

    python -m pytest -q tests

No request is sent: the fake client answers without a network connection.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artp2.data import load_art  # noqa: E402
from artp2.llm import Config, build_messages, default_run_name, parse_answer, pick_shots, run, sample_indices  # noqa: E402

EXAMPLE = {
    "observation_1": "Ann lost her keys.",
    "observation_2": "She was late for work.",
    "hypothesis_1": "She found them at once.",
    "hypothesis_2": "She searched the whole flat.",
    "label": 1,
}


class FakeClient:
    """Always answers with the same text and counts the requests."""

    def __init__(self, text: str) -> None:
        self.text, self.calls = text, 0

    def chat_completion(self, messages, **kwargs):
        self.calls += 1
        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=1))


@pytest.fixture(scope="module")
def ds():
    return load_art()


def test_zero_shot_prompt_shows_both_hypotheses_in_order():
    messages = build_messages(EXAMPLE, "zero", shots=[])
    assert [message["role"] for message in messages] == ["system", "user"]
    text = messages[-1]["content"]
    assert "Beginning: Ann lost her keys." in text and "Ending: She was late for work." in text
    assert "Middle 1: She found them at once." in text and "Middle 2: She searched the whole flat." in text
    assert text.endswith("Answer with 1 or 2 only.")


def test_swap_changes_the_order_and_the_parsed_answer():
    text = build_messages(EXAMPLE, "zero", shots=[], swap=True)[-1]["content"]
    assert "Middle 1: She searched the whole flat." in text and "Middle 2: She found them at once." in text
    assert parse_answer("1", "zero", swap=True) == 1  # shown as number 1 = hypothesis with index 1
    assert parse_answer("2", "zero", swap=True) == 0


def test_few_shot_prompt_contains_solved_examples():
    messages = build_messages(EXAMPLE, "few", shots=[EXAMPLE, {**EXAMPLE, "label": 0}])
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [messages[2]["content"], messages[4]["content"]] == ["2", "1"]  # label 1 -> "2", label 0 -> "1"


def test_parse_answer():
    assert parse_answer("2", "zero") == 1
    assert parse_answer(" 1.", "zero") == 0
    assert parse_answer("Middle 2 is more plausible.", "zero") == 1
    assert parse_answer("I cannot decide.", "zero") is None
    assert parse_answer("", "zero") is None
    assert parse_answer("Middle 1 does not fit because ... so middle 2.\nAnswer: 2", "cot") == 1
    assert parse_answer("Step 1 ... step 2 ...\n**Answer: 1**", "cot") == 0
    assert parse_answer("Both are possible, but 2 fits the ending better than 1. So it is 2", "cot") == 1
    assert parse_answer("<think>Middle 1 could ... but 2 ...</think>\n1", "zero") == 0  # reasoning is not the answer
    assert parse_answer("<think>Middle 1 could", "zero") is None  # cut off before the answer


def test_shots_are_balanced_and_fixed_by_the_seed(ds):
    shots = pick_shots(ds["train"], 4, seed=42)
    assert len(set(shots)) == 4 and shots == pick_shots(ds["train"], 4, seed=42)
    assert sorted(ds["train"][index]["label"] for index in shots) == [0, 0, 1, 1]


def test_sample_indices():
    assert sample_indices(5, limit=0, seed=42) == [0, 1, 2, 3, 4]
    sample = sample_indices(1000, limit=50, seed=42, exclude={3, 4})
    assert len(sample) == 50 and sample == sorted(sample) and not {3, 4} & set(sample)
    assert sample == sample_indices(1000, limit=50, seed=42, exclude={3, 4})


def test_run_names():
    assert default_run_name(Config()) == "llm-Llama-3.3-70B-Instruct-zero"
    assert default_run_name(Config(model="openai/gpt-oss-120b", prompt="few", shots=8, limit=200)) == "llm-gpt-oss-120b-few8-n200"


def test_run_with_fake_client_writes_results_and_uses_the_cache(ds, tmp_path):
    cfg = Config(splits="validation,train", limit=30, train_limit=20, prompt="few", wandb_mode="disabled",
                 output_dir=str(tmp_path))
    client = FakeClient("2")
    results = run(cfg, client)
    assert client.calls == 50
    assert results["validation"]["n"] == 30 and results["train"]["n"] == 20
    labels = [ds["validation"][index]["label"] for index in sample_indices(1000, 30, 42)]
    assert results["validation"]["acc"] == pytest.approx(sum(labels) / 30)  # "2" is right whenever the label is 1
    assert results["validation"]["invalid"] == 0
    assert not set(results["shot_indices"]) & set(sample_indices(len(ds["train"]), 20, 42, set(results["shot_indices"])))
    saved = json.loads((tmp_path / results["run_name"] / "results.json").read_text())
    assert saved["validation"]["confusion"][0][0] == 0  # nothing predicted as hypothesis 1
    run(cfg, client)  # second run: everything comes from the cache
    assert client.calls == 50


def test_run_folder_is_not_continued_with_other_settings(tmp_path):
    cfg = Config(limit=3, wandb_mode="disabled", output_dir=str(tmp_path), run_name="pilot")
    run(cfg, FakeClient("1"))
    saved = json.loads((tmp_path / "pilot" / "config.json").read_text())
    assert saved["max_tokens"] == 8  # 0 is replaced by the limit that was used
    with pytest.raises(SystemExit):
        run(Config(limit=3, wandb_mode="disabled", output_dir=str(tmp_path), run_name="pilot", max_tokens=1024), FakeClient("1"))


def test_wrong_model_or_provider_stops_at_the_first_request(tmp_path):
    class BrokenClient(FakeClient):
        def chat_completion(self, messages, **kwargs):
            self.calls += 1
            raise ValueError("model is not supported by provider")

    client = BrokenClient("")
    with pytest.raises(ValueError):
        run(Config(limit=20, wandb_mode="disabled", output_dir=str(tmp_path)), client)
    assert client.calls == 1
