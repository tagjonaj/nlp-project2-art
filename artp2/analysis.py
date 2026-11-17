"""Evaluation of the three models from the files in runs/.

Used by notebooks/analysis.ipynb. Every function reads the saved results and predictions, so all numbers can
be recomputed without training a model or calling the LLM. The plot functions save their figure to results/figures/.

Files of a run that are read here:
* results.json - accuracy, loss and confusion matrix per split
* predictions_<split>.csv - one row per example: index, label, prediction, correct (models 1 and 2 also p_h1, p_h2)
* metrics/version_0/metrics.csv - loss and accuracy per epoch (models 1 and 2)
"""
from __future__ import annotations

import csv
import functools
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGURES = ROOT / "results" / "figures"

M1, M2, M3 = "Model 1 · random weights", "Model 2 · pretrained", "Model 3 · LLM"
SEEDS = (42, 1, 2, 3)
MODELS = {  # final configuration of every model, selected on the validation split
    M1: [f"rand-lr2e-05-ep30-s{seed}" for seed in SEEDS],
    M2: [f"pre-lr1e-05-s{seed}" for seed in SEEDS],
    M3: ["llm-qwen3-235b-few8"],
}
LR_RUNS = {  # learning rate search, seed 42
    M2: ["pre-lr1e-05-s42", "pre-lr2e-05-s42", "pre-lr3e-05-s42", "pre-lr5e-05-s42"],
    M1: ["rand-lr1e-05-ep30-s42", "rand-lr2e-05-ep30-s42", "rand-lr5e-05-ep30-s42", "rand-lr0.0001-ep30-s42"],
}
ONE = (42,)
ABLATIONS = {  # variant -> for model 1 and for model 2: (name of the runs without the seed, seeds); None = not run
    "Final configuration": (("rand-lr2e-05-ep30", SEEDS), ("pre-lr1e-05", SEEDS)),
    "Hypothesis only": (("rand-lr2e-05-honly-ep30", SEEDS), ("pre-lr1e-05-honly", SEEDS)),
    "4,000 training examples": (("rand-lr2e-05-ep30-n4000", SEEDS), ("pre-lr1e-05-n4000", SEEDS)),
    "1,000 training examples": (("rand-lr2e-05-ep30-n1000", SEEDS), ("pre-lr1e-05-n1000", SEEDS)),
    "Segment order: O1 O2, then hypothesis": (("rand-lr2e-05-o1o2h-ep30", ONE), ("pre-lr1e-05-o1o2h", SEEDS)),
    "Mean pooling instead of the <s> vector": (("rand-lr2e-05-mean-ep30", ONE), ("pre-lr1e-05-mean", ONE)),
    "Classifier learning rate 1e-3": (None, ("pre-lr1e-05-clr0.001", ONE)),
    "10 instead of 5 epochs": (None, ("pre-lr1e-05-ep10", ONE)),
}
PILOT_RUNS = ["llm-Llama-3.3-70B-Instruct-zero-n200", "llm-DeepSeek-V4-Pro-zero-n200", "llm-gpt-oss-120b-zero-n200",
              "llm-Qwen3-235B-A22B-Instruct-2507-zero-n200", "llm-GLM-5.2-zero-n200"]
PROMPT_RUNS = ["llm-qwen3-235b-zero", "llm-qwen3-235b-few4", "llm-qwen3-235b-few8", "llm-qwen3-235b-cot",
               "llm-qwen3-235b-zero-swap", "llm-gpt-oss-120b-zero"]
SPLITS = ("train", "validation", "test")
HUMAN_TEST_ACCURACY = 91.4  # Bhagavatula et al. 2020

# not, no, never, nothing, nobody, cannot and every contraction that ends in n't
NEGATION = re.compile(r"\b(?:not|no|never|nothing|nobody|cannot)\b|n['’]t\b", re.IGNORECASE)
NEGATION_GROUPS = ("only in the wrong hypothesis", "only in the correct hypothesis", "in neither hypothesis", "in both hypotheses")

COLORS = {M1: "#2a78d6", M2: "#eb6834", M3: "#1baf7a"}  # one fixed colour per model in every figure
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"


# --------------------------------------------------------------------------------------
# Reading the saved runs
# --------------------------------------------------------------------------------------
def results(run: str) -> dict:
    return json.loads((RUNS / run / "results.json").read_text(encoding="utf-8"))


def config(run: str) -> dict:
    return json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def predictions(run: str, split: str) -> tuple[dict, ...]:
    """Rows of predictions_<split>.csv. Models 1 and 2 store p_h1 and p_h2, the LLM only its prediction."""
    with open(RUNS / run / f"predictions_{split}.csv", newline="", encoding="utf-8") as file:
        return tuple(csv.DictReader(file))


def correct(run: str, split: str) -> dict[int, int]:
    """Index of the example -> 1 if the prediction is right, else 0."""
    return {int(row["index"]): int(row["correct"]) for row in predictions(run, split)}


def p_correct(run: str, split: str) -> dict[int, float]:
    """Index of the example -> probability that a run of model 1 or 2 gives to the correct hypothesis."""
    return {int(row["index"]): float(row["p_h2"] if row["label"] == "1" else row["p_h1"]) for row in predictions(run, split)}


def curves(run: str) -> dict[str, list[float]]:
    """Training curves of a run of model 1 or 2: one value per epoch for every logged metric."""
    out: dict[str, list[float]] = {}
    with open(RUNS / run / "metrics" / "version_0" / "metrics.csv", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for key, value in row.items():
                if key not in ("epoch", "step") and value:
                    out.setdefault(key, []).append(float(value))
    return out


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def accuracy(values) -> float:
    """Accuracy in percent of a list of 0 / 1 values."""
    values = list(values)
    return 100 * sum(values) / len(values)


def interval(accuracy_percent: float, n: int) -> float:
    """Half width of the 95 % interval of an accuracy measured on n examples, in points."""
    p = accuracy_percent / 100
    return 196 * math.sqrt(p * (1 - p) / n)


def mean_sd(runs: list[str], split: str) -> tuple[float, float]:
    """Mean and standard deviation of the accuracy over some runs (sd = 0 for a single run)."""
    values = [100 * results(run)[split]["acc"] for run in runs]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def accuracy_table(ds) -> list[dict]:
    """Accuracy on train, validation and test: mean and standard deviation over the seeds of each model."""
    row = {"model": "Majority class", "runs": 0}
    for split in SPLITS:
        labels = ds[split]["label"]
        row[split], row[f"{split} sd"] = 100 * max(sum(labels), len(labels) - sum(labels)) / len(labels), 0.0
    rows = [row]
    for name, runs in MODELS.items():
        row = {"model": name, "runs": len(runs)}
        for split in SPLITS:
            row[split], row[f"{split} sd"] = mean_sd(runs, split)
        rows.append(row)
    return rows


def gap_table(table: list[dict]) -> list[dict]:
    """Difference between validation and test accuracy of every model, in points."""
    return [{"model": row["model"], "validation": row["validation"], "test": row["test"], "gap": row["validation"] - row["test"]}
            for row in table]


def confusion(run: str, split: str) -> list[list[int]]:
    """Rows = true hypothesis, columns = predicted hypothesis. Invalid LLM answers are not in the matrix."""
    return results(run)[split]["confusion"]


def lr_table() -> list[dict]:
    """Learning rate search (seed 42): one run per model and learning rate."""
    rows = []
    for name, runs in LR_RUNS.items():
        for run in runs:
            result = results(run)
            rows.append({"model": name, "learning rate": f"{config(run)['lr']:.0e}", "best epoch": result["best_epoch"],
                         "train": 100 * result["train"]["acc"], "validation": 100 * result["validation"]["acc"],
                         "test": 100 * result["test"]["acc"], "validation loss": result["validation"]["loss"]})
    return rows


def ablation_runs(variant: str, model: str) -> list[str]:
    """The runs of an ablation for one model, one per seed (empty if the variant was not run for this model)."""
    entry = ABLATIONS[variant][(M1, M2).index(model)]
    return [f"{entry[0]}-s{seed}" for seed in entry[1]] if entry else []


def ablation_table() -> list[dict]:
    """Accuracy of every ablation: mean and standard deviation over its seeds. Each variant changes one factor.

    The last column is the difference to the final configuration of the same model on validation.
    """
    rows = []
    for model in (M1, M2):
        reference, _ = mean_sd(MODELS[model], "validation")
        for variant in ABLATIONS:
            runs = ablation_runs(variant, model)
            if not runs:
                continue
            row = {"model": model, "variant": variant, "seeds": len(runs)}
            for split in ("validation", "test"):
                row[split], row[f"{split} sd"] = mean_sd(runs, split)
            row["validation - final"] = round(row["validation"] - reference, 2) + 0.0  # + 0.0 turns -0.0 into 0.0
            rows.append(row)
    return rows


def negation_group(example: dict) -> str:
    """Where a negation word occurs in the two hypotheses of an example."""
    hypotheses = (example["hypothesis_1"], example["hypothesis_2"])
    in_correct = bool(NEGATION.search(hypotheses[example["label"]]))
    in_wrong = bool(NEGATION.search(hypotheses[1 - example["label"]]))
    if in_wrong and not in_correct:
        return NEGATION_GROUPS[0]
    if in_correct and not in_wrong:
        return NEGATION_GROUPS[1]
    return NEGATION_GROUPS[3] if in_correct else NEGATION_GROUPS[2]


def negation_table(ds, splits=("validation", "test")) -> list[dict]:
    """Accuracy of every model by the place of the negation word; models 1 and 2 as the mean over their seeds."""
    rows = []
    for split in splits:
        groups = [negation_group(example) for example in ds[split]]
        for group in NEGATION_GROUPS:
            indices = [index for index, name in enumerate(groups) if name == group]
            row = {"split": split, "negation word": group, "pairs": len(indices)}
            for name, runs in MODELS.items():
                row[name] = statistics.mean(accuracy(correct(run, split)[index] for index in indices) for run in runs)
            rows.append(row)
    return rows


def confidence_table(runs: list[str], split: str, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    """Share of the predictions and their accuracy per confidence bin, pooled over the runs.

    Confidence = probability that the model gives to the hypothesis it predicts.
    """
    pairs = [(max(float(row["p_h1"]), float(row["p_h2"])), int(row["correct"])) for run in runs for row in predictions(run, split)]
    table = []
    for low, high in zip(edges, edges[1:]):
        members = [right for confidence, right in pairs if low <= confidence and (confidence < high or high == edges[-1])]
        table.append({"confidence": f"{low:.1f} to {high:.1f}", "middle": 50 * (low + high), "share": 100 * len(members) / len(pairs),
                      "accuracy": accuracy(members) if members else float("nan")})
    return table


def overlap(run_a: str, run_b: str, split: str) -> dict[str, int]:
    """How many examples both runs, only one of them or neither solves."""
    a, b = correct(run_a, split), correct(run_b, split)
    indices = sorted(set(a) & set(b))
    return {"both right": sum(a[i] and b[i] for i in indices), "only the first right": sum(a[i] and not b[i] for i in indices),
            "only the second right": sum(b[i] and not a[i] for i in indices), "both wrong": sum(not a[i] and not b[i] for i in indices)}


def mcnemar(first: dict[int, int], second: dict[int, int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test for two sets of answers to the same examples.

    Only the examples that exactly one of the two solves count. Returns how many only the second solves,
    how many only the first solves, and the p-value: the probability of a split at least this uneven
    if both were equally good.
    """
    indices = sorted(set(first) & set(second))
    only_second = sum(second[i] and not first[i] for i in indices)
    only_first = sum(first[i] and not second[i] for i in indices)
    discordant = only_first + only_second
    if not discordant:
        return only_second, only_first, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1)) / 2 ** discordant
    return only_second, only_first, min(1.0, 2 * tail)


def pilot_table() -> list[dict]:
    """Step 1 of the LLM selection: five models, zero-shot, 200 validation examples."""
    rows = []
    for run in PILOT_RUNS:
        result = results(run)["validation"]
        accuracy_percent = 100 * result["acc"]
        rows.append({"model": config(run)["model"], "examples": result["n"], "validation": accuracy_percent,
                     "interval": interval(accuracy_percent, result["n"]), "invalid": result["invalid"],
                     "tokens out": result["completion_tokens"]})
    return rows


def prompt_table(reference: str = "llm-qwen3-235b-zero") -> list[dict]:
    """Step 2 of the LLM selection: prompt formats on the whole validation split, compared with zero-shot."""
    base = correct(reference, "validation")
    base_predictions = {int(row["index"]): row["pred"] for row in predictions(reference, "validation")}
    rows = []
    for run in PROMPT_RUNS:
        result = results(run)["validation"]
        accuracy_percent = 100 * result["acc"]
        only_run, only_reference, p = mcnemar(base, correct(run, "validation"))
        same = [base_predictions[int(row["index"])] == row["pred"] for row in predictions(run, "validation")]
        rows.append({"run": run.removeprefix("llm-"), "validation": accuracy_percent, "interval": interval(accuracy_percent, result["n"]),
                     "invalid": result["invalid"], "tokens out": result["completion_tokens"],
                     "only this run right": only_run, "only zero-shot right": only_reference, "p": p,
                     "same answer as zero-shot": 100 * sum(same) / len(same)})
    return rows


def invalid_answers(run: str, split: str) -> list[dict]:
    """LLM answers without a readable 1 or 2 (they count as wrong). The file keeps the last 40 characters."""
    return [{"index": int(row["index"]), "end of the answer": row["answer"]} for row in predictions(run, split) if row["pred"] == "-1"]


def confident_errors(run: str, split: str, count: int = 5) -> list[int]:
    """Indices of the examples where a run of model 1 or 2 gives the lowest probability to the correct hypothesis."""
    scores = p_correct(run, split)
    return sorted(scores, key=scores.get)[:count]


def examples(ds, split: str, indices: list[int]) -> list[dict]:
    """The texts of some examples with the answers of the three models (models 1 and 2: seed 42).

    For models 1 and 2 the value is the probability of the correct hypothesis, for the LLM right or wrong.
    """
    rows = []
    for index in indices:
        example = ds[split][index]
        hypotheses = (example["hypothesis_1"], example["hypothesis_2"])
        row = {"index": index, "observation 1": example["observation_1"], "correct hypothesis": hypotheses[example["label"]],
               "wrong hypothesis": hypotheses[1 - example["label"]], "observation 2": example["observation_2"]}
        for name in (M1, M2):
            row[f"{name} · p(correct)"] = round(p_correct(MODELS[name][0], split)[index], 3)
        row[M3] = "right" if correct(MODELS[M3][0], split)[index] else "wrong"
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _axes(ax, ylabel: str = "", ylim=(0, 100), grid: str = "y") -> None:
    """Quiet axes: no frame, thin grid lines behind the data."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)


def _save(figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return path


def _columns(ax, positions, values, color, errors=None, label=None, width=0.14) -> None:
    """Thin columns with the value above the column (and above the error bar, if there is one)."""
    ax.bar(positions, values, width=width, color=color, label=label, yerr=errors, error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2})
    for x, value, error in zip(positions, values, errors or [0] * len(values)):
        ax.text(x, value + error + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color=INK)


def _legend(figure, ax, columns: int) -> None:
    figure.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, ncols=columns, loc="outside upper center",
                  handlelength=1, handleheight=1)


def plot_accuracy(table: list[dict]):
    """Accuracy of the three models on train, validation and test."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    models = [row for row in table if row["model"] in MODELS]
    for offset, row in zip((-0.24, 0, 0.24), models):
        _columns(ax, [x + offset for x in range(3)], [row[split] for split in SPLITS], COLORS[row["model"]],
                 errors=[row[f"{split} sd"] for split in SPLITS], label=row["model"])
    majority = next(row for row in table if row["model"] == "Majority class")
    ax.axhline(majority["test"], color=MUTED, linewidth=0.8)
    ax.text(2.97, majority["test"] + 1, "majority class", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.plot([2.42, 2.56], [HUMAN_TEST_ACCURACY] * 2, color=INK, linewidth=1.2)
    ax.text(2.59, HUMAN_TEST_ACCURACY, f"human {HUMAN_TEST_ACCURACY}", ha="left", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), SPLITS)
    ax.set_xlim(-0.5, 3.0)
    _axes(ax, "accuracy (%)", ylim=(0, 108))
    _legend(figure, ax, 3)
    return _save(figure, "accuracy.png")


def plot_confusion(split: str = "test"):
    """Confusion matrix of every model (models 1 and 2: seed 42) with the share of each row."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("ramp", ["#f1f0ec", "#3c3b38"])
    figure, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), layout="constrained")
    for ax, (name, runs) in zip(axes, MODELS.items()):
        matrix = confusion(runs[0], split)
        shares = [[value / sum(row) for value in row] for row in matrix]
        ax.imshow(shares, cmap=ramp, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i][j]}\n{100 * shares[i][j]:.0f} %", ha="center", va="center", fontsize=10,
                        color="white" if shares[i][j] > 0.6 else INK)
        ax.set_xticks(range(2), ["h1", "h2"])
        ax.set_yticks(range(2), ["h1", "h2"])
        ax.set_xlabel("predicted", color=MUTED, fontsize=9)
        ax.set_title(name, fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("true", color=MUTED, fontsize=9)
    return _save(figure, f"confusion_{split}.png")


def plot_curves():
    """Loss and accuracy per epoch for the final configurations of models 1 and 2, one line per seed."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(7.5, 5), sharex="col", layout="constrained")
    for column, name in enumerate((M1, M2)):
        for row, metric in enumerate(("loss", "acc")):
            ax = axes[row][column]
            for run in MODELS[name]:
                data = curves(run)
                ax.plot(data[f"train/{metric}"], color=MUTED, linewidth=1, label="train")
                ax.plot(data[f"val/{metric}"], color=COLORS[name], linewidth=1.8, label="validation")
            _axes(ax, "loss" if metric == "loss" else "accuracy", ylim=(0.4, 1.02) if metric == "acc" else (0, None))
            ax.xaxis.get_major_locator().set_params(integer=True)
        handles, labels = axes[0][column].get_legend_handles_labels()
        axes[0][column].legend(handles[:2], labels[:2], frameon=False, fontsize=8, loc="lower left", ncols=2)
        axes[0][column].set_title(name, fontsize=9, color=INK)
        axes[1][column].set_xlabel("epoch (counted from 0)", color=MUTED, fontsize=9)
    return _save(figure, "curves.png")


def plot_ablations():
    """Validation accuracy without the observations and with fewer training examples (mean and sd of 4 seeds)."""
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.3), layout="constrained", sharey=True)
    inputs = ("Final configuration", "Hypothesis only")
    sizes = ("1,000 training examples", "4,000 training examples", "Final configuration")
    for offset, name in zip((-0.1, 0.1), (M1, M2)):
        means, sds = zip(*(mean_sd(ablation_runs(variant, name), "validation") for variant in inputs))
        _columns(left, [offset, 1 + offset], means, COLORS[name], errors=sds, label=name, width=0.1)
        means, sds = zip(*(mean_sd(ablation_runs(variant, name), "validation") for variant in sizes))
        _columns(right, [x + offset for x in range(3)], means, COLORS[name], errors=sds)
    left.set_xticks(range(2), ["full input", "hypothesis only"])
    left.set_xlim(-0.5, 1.5)
    right.set_xticks(range(3), ["1,000", "4,000", "16,801"])
    right.set_xlabel("training examples", color=MUTED, fontsize=9)
    for ax in (left, right):
        _axes(ax, "validation accuracy (%)" if ax is left else "", ylim=(0, 108))
        ax.axhline(50, color=MUTED, linewidth=0.8)
    left.text(1.47, 51, "chance", ha="right", va="bottom", fontsize=8, color=MUTED)
    _legend(figure, left, 2)
    return _save(figure, "ablations.png")


def plot_negation(table: list[dict]):
    """Accuracy of the three models by the place of the negation word."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("validation", "test")):
        rows = [row for row in table if row["split"] == split][:3]
        for offset, name in zip((-0.26, 0, 0.26), MODELS):
            _columns(ax, [x + offset for x in range(3)], [row[name] for row in rows], COLORS[name], label=name, width=0.18)
        ax.set_xticks(range(3), [f"{label}\n{row['pairs']:,} pairs" for label, row in zip(("only wrong", "only correct", "neither"), rows)])
        ax.axhline(50, color=MUTED, linewidth=0.8)
        ax.set_title(split, fontsize=9, color=INK)
        ax.set_xlabel("hypothesis with a negation word", color=MUTED, fontsize=9)
        _axes(ax, "accuracy (%)" if split == "validation" else "", ylim=(0, 108))
    _legend(figure, axes[0], 3)
    return _save(figure, "negation.png")


def plot_confidence():
    """Accuracy by confidence for models 1 and 2 on validation and test, pooled over the seeds."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("validation", "test")):
        ax.plot([50, 100], [50, 100], color=MUTED, linewidth=0.8)
        for name in (M1, M2):
            table = confidence_table(MODELS[name], split)
            ax.plot([row["middle"] for row in table], [row["accuracy"] for row in table], color=COLORS[name], linewidth=1.8,
                    marker="o", markersize=6, markeredgecolor="white", label=name)
        ax.set_title(split, fontsize=9, color=INK)
        ax.set_xlabel("confidence of the prediction (%)", color=MUTED, fontsize=9)
        ax.set_xlim(50, 100)
        ax.set_xticks([row["middle"] for row in table], [f"{row['middle'] - 5:.0f}-{row['middle'] + 5:.0f}" for row in table])
        _axes(ax, "accuracy (%)" if split == "validation" else "", ylim=(40, 102))
    axes[1].text(72, 73.5, "accuracy = confidence", ha="right", va="bottom", fontsize=8, color=MUTED)
    _legend(figure, axes[0], 2)
    return _save(figure, "confidence.png")


def plot_prompts(table: list[dict]):
    """Validation accuracy of the LLM prompt formats with their 95 % intervals."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(7.5, 2.6), layout="constrained")
    positions = list(range(len(table), 0, -1))
    for y, row in zip(positions, table):
        ax.plot([row["validation"] - row["interval"], row["validation"] + row["interval"]], [y, y], color=COLORS[M3], linewidth=1.8)
        ax.plot(row["validation"], y, "o", color=COLORS[M3], markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(row["validation"] + row["interval"] + 0.12, y, f"{row['validation']:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(positions, [row["run"] for row in table])
    ax.set_xlim(92, 99)
    ax.set_ylim(0.4, len(table) + 0.6)
    ax.set_xlabel("validation accuracy (%) with 95 % interval, 1,000 examples", color=MUTED, fontsize=9)
    _axes(ax, ylim=None, grid="x")
    return _save(figure, "llm_prompts.png")
