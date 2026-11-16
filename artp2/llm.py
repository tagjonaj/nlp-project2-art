"""Model 3: an LLM with 8B+ parameters behind the Hugging Face Inference Providers API.

The LLM is not trained. Every run evaluates one prompt format:

    python llm.py --model meta-llama/Llama-3.3-70B-Instruct --provider novita --prompt zero --limit 200

The access token comes from an earlier ``huggingface_hub.login()`` or from the environment variable HF_TOKEN.
Every answer is stored in runs/<run_name>/responses_<split>.jsonl, so an interrupted run continues where it
stopped and no request is paid twice.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import wandb
from huggingface_hub import InferenceClient

from .data import load_art

SYSTEM = ("You read the beginning and the ending of a short story and two possible middle parts. "
          "Decide which middle part makes the story more plausible.")
QUESTION = "Beginning: {o1}\nEnding: {o2}\nMiddle 1: {h1}\nMiddle 2: {h2}\n"
DIRECT = "Which middle is more plausible? Answer with 1 or 2 only."
STEP_BY_STEP = ("Which middle is more plausible? Think step by step about what happened between the beginning "
                "and the ending. Then write your final answer on the last line as 'Answer: 1' or 'Answer: 2'.")
PROMPTS = ("zero", "few", "cot")
SHORT = {"train": "train", "validation": "val", "test": "test"}
# settings that change the answers: a run folder is only continued if they are unchanged
REQUEST_KEYS = ("model", "provider", "prompt", "shots", "swap", "max_tokens", "reasoning_effort", "temperature", "seed")


@dataclass
class Config:
    # model
    model: str = "meta-llama/Llama-3.3-70B-Instruct"  # repository on the Hugging Face hub
    provider: str = "auto"  # inference provider that serves the model, e.g. novita, groq, deepinfra
    # prompt
    prompt: str = "zero"  # zero = instruction only | few = with solved examples | cot = step by step
    shots: int = 4  # few: number of solved examples from the training split, half per label
    swap: bool = False  # show the two hypotheses in swapped order (check for a position bias)
    max_tokens: int = 0  # limit for the answer; 0 = 8 (zero, few) or 512 (cot). Reasoning models need more.
    reasoning_effort: str = ""  # low | medium | high for reasoning models; "" = default of the provider
    temperature: float = 0.0
    # data
    splits: str = "validation"  # comma-separated: train,validation,test
    limit: int = 0  # validation/test: random sample of N examples (drawn with the seed); 0 = all
    train_limit: int = 1000  # train: random sample of N examples that are not used as solved examples
    seed: int = 42
    # requests
    workers: int = 8  # parallel requests
    # tracking
    wandb_project: str = "nlp-project2-art"
    wandb_mode: str = "online"  # online | offline | disabled
    run_name: str = ""  # "" = built from the configuration
    tags: str = "llm"  # comma-separated
    output_dir: str = "runs"


def default_run_name(cfg: Config) -> str:
    parts = ["llm", cfg.model.split("/")[-1], cfg.prompt + (str(cfg.shots) if cfg.prompt == "few" else "")]
    if cfg.swap:
        parts.append("swap")
    if cfg.reasoning_effort:
        parts.append(cfg.reasoning_effort)
    if cfg.limit:
        parts.append(f"n{cfg.limit}")
    return "-".join(parts)


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------
def question(example: dict, swap: bool) -> str:
    h1, h2 = example["hypothesis_1"], example["hypothesis_2"]
    if swap:
        h1, h2 = h2, h1
    return QUESTION.format(o1=example["observation_1"], o2=example["observation_2"], h1=h1, h2=h2)


def shown_number(label: int, swap: bool) -> str:
    """Number (1 or 2) under which the hypothesis with index ``label`` is shown."""
    return str((1 - label if swap else label) + 1)


def build_messages(example: dict, prompt: str, shots: list[dict], swap: bool = False) -> list[dict]:
    """Chat messages for one example. ``shots`` are solved training examples for the few-shot prompt."""
    instruction = STEP_BY_STEP if prompt == "cot" else DIRECT
    messages = [{"role": "system", "content": SYSTEM}]
    for shot in shots if prompt == "few" else []:
        messages.append({"role": "user", "content": question(shot, swap) + instruction})
        messages.append({"role": "assistant", "content": shown_number(shot["label"], swap)})
    messages.append({"role": "user", "content": question(example, swap) + instruction})
    return messages


def parse_answer(text: str, prompt: str, swap: bool = False) -> int | None:
    """Index (0 or 1) of the chosen hypothesis, or None if the answer contains no usable number.

    Direct prompts: the first number 1 or 2 counts. Step by step: the number after 'Answer:' counts,
    without it the last number. Reasoning inside <think> tags is not part of the answer.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:  # the reasoning was cut off by the token limit before the answer
        return None
    numbers = re.findall(r"Answer:\W*([12])\b", text) if prompt == "cot" else []
    if not numbers:
        numbers = re.findall(r"\b([12])\b", text)
        numbers = numbers[-1:] if prompt == "cot" else numbers[:1]
    if not numbers:
        return None
    chosen = int(numbers[-1]) - 1
    return 1 - chosen if swap else chosen


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def pick_shots(train, shots: int, seed: int) -> list[int]:
    """Indices of ``shots`` training examples, half per label, in random order."""
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    labels = train["label"]
    wanted = {0: shots // 2, 1: shots - shots // 2}
    chosen = []
    for index in order:
        if wanted[labels[index]] > 0:
            wanted[labels[index]] -= 1
            chosen.append(index)
        if len(chosen) == shots:
            break
    return chosen


def sample_indices(size: int, limit: int, seed: int, exclude: set[int] = frozenset()) -> list[int]:
    """All indices, or a random sample of ``limit`` indices, without the excluded ones."""
    indices = [index for index in range(size) if index not in exclude]
    if limit and limit < len(indices):
        indices = sorted(random.Random(seed).sample(indices, limit))
    return indices


# --------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------
def ask(client, cfg: Config, messages: list[dict]) -> dict:
    """One request. Rate limits, timeouts and server errors are retried. Returns the answer and the token counts."""
    options = {"extra_body": {"reasoning_effort": cfg.reasoning_effort}} if cfg.reasoning_effort else {}
    for attempt in range(6):
        try:
            output = client.chat_completion(messages=messages, model=cfg.model, max_tokens=cfg.max_tokens,
                                            temperature=cfg.temperature, **options)
            usage = output.usage
            return {"text": output.choices[0].message.content or "",
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}
        except (ValueError, TypeError):  # wrong model, provider or argument: a retry does not help
            raise
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            retry = status is None or status in (408, 429) or status >= 500
            if not retry or attempt == 5:
                raise
            time.sleep(2 ** attempt)


def answer_split(client, cfg: Config, name: str, split, indices: list[int], shots: list[dict], cache_file: Path) -> dict[int, dict]:
    """Ask the LLM for every example that is not in the cache yet. Returns index -> response."""
    cache = {}
    if cache_file.exists():
        cache = {row["index"]: row for row in map(json.loads, cache_file.read_text().splitlines())}
    todo = [index for index in indices if index not in cache]
    lock = threading.Lock()
    done = 0

    def work(index: int) -> dict:
        nonlocal done
        row = {"index": index, **ask(client, cfg, build_messages(split[index], cfg.prompt, shots, cfg.swap))}
        with lock:
            with open(cache_file, "a") as file:
                file.write(json.dumps(row) + "\n")
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"[{name}] {done}/{len(todo)} answers", flush=True)
        return row

    for index in todo[:1]:  # the first request alone: a wrong token, model or provider fails here
        cache[index] = work(index)
    pool = ThreadPoolExecutor(max_workers=cfg.workers)
    try:
        for row in pool.map(work, todo[1:]):
            cache[row["index"]] = row
    finally:
        pool.shutdown(cancel_futures=True)
    return {index: cache[index] for index in indices}


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------
def run(cfg: Config, client=None) -> dict:
    if not cfg.max_tokens:
        cfg = replace(cfg, max_tokens=512 if cfg.prompt == "cot" else 8)
    run_name = cfg.run_name or default_run_name(cfg)
    run_dir = Path(cfg.output_dir) / run_name
    config_file = run_dir / "config.json"
    if config_file.exists():
        old = json.loads(config_file.read_text())
        changed = [key for key in REQUEST_KEYS if old.get(key) != getattr(cfg, key)]
        if changed:
            raise SystemExit(f"{run_dir} holds answers for other settings ({', '.join(changed)}): "
                             "choose another --run_name or delete the folder")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(asdict(cfg), indent=2))
    client = client or InferenceClient(provider=cfg.provider, timeout=120)
    ds = load_art()
    start = time.time()

    shot_indices = pick_shots(ds["train"], cfg.shots, cfg.seed) if cfg.prompt == "few" else []
    shots = [ds["train"][index] for index in shot_indices]
    results = {"run_name": run_name, "group": run_name, "shot_indices": shot_indices}
    for split in cfg.splits.split(","):
        limit = cfg.train_limit if split == "train" else cfg.limit
        indices = sample_indices(len(ds[split]), limit, cfg.seed, exclude=set(shot_indices) if split == "train" else set())
        responses = answer_split(client, cfg, split, ds[split], indices, shots, run_dir / f"responses_{split}.jsonl")
        labels = ds[split]["label"]
        confusion = [[0, 0], [0, 0]]  # rows = true label, columns = prediction (valid answers only)
        correct = invalid = 0
        with open(run_dir / f"predictions_{split}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["index", "label", "pred", "correct", "answer"])
            for index in indices:
                label, text = labels[index], responses[index]["text"]
                pred = parse_answer(text, cfg.prompt, cfg.swap)
                if pred is None:
                    invalid += 1
                else:
                    confusion[label][pred] += 1
                    correct += int(pred == label)
                writer.writerow([index, label, -1 if pred is None else pred, int(pred == label), text.strip()[-40:]])
        results[split] = {
            "n": len(indices),
            "acc": correct / len(indices),  # an answer without a usable number counts as wrong
            "invalid": invalid,
            "confusion": confusion,
            "prompt_tokens": sum(responses[index]["prompt_tokens"] for index in indices),
            "completion_tokens": sum(responses[index]["completion_tokens"] for index in indices),
        }
    results["seconds"] = time.time() - start
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    tracking = wandb.init(project=cfg.wandb_project, name=run_name, group=run_name, config=asdict(cfg),
                          tags=[tag for tag in cfg.tags.split(",") if tag], mode=cfg.wandb_mode, dir=str(run_dir))
    for split in cfg.splits.split(","):
        tracking.summary[f"best/{SHORT[split]}_acc"] = results[split]["acc"]
        tracking.summary[f"invalid/{SHORT[split]}"] = results[split]["invalid"]
    tracking.finish()
    for split in cfg.splits.split(","):
        result = results[split]
        print(f"[{run_name}] {SHORT[split]} {result['acc']:.4f} | n {result['n']} | invalid {result['invalid']} | "
              f"tokens in {result['prompt_tokens']} out {result['completion_tokens']}")
    return results


def str2bool(value: str) -> bool:
    if value.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {value!r}")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in fields(Config):
        kind = type(field.default)
        parser.add_argument(f"--{field.name}", type=str2bool if kind is bool else kind, default=field.default)
    cfg = Config(**vars(parser.parse_args(argv)))
    if cfg.prompt not in PROMPTS:
        parser.error(f"unknown prompt {cfg.prompt!r}")
    return run(cfg)


if __name__ == "__main__":
    main()
