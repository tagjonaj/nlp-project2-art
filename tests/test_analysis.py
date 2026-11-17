"""Checks for the evaluation code in artp2/analysis.py.

    python -m pytest -q tests

The first tests use small made-up inputs. The last ones read the committed files in runs/ and check
that the saved predictions give the same accuracy and confusion matrix as results.json.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artp2 import analysis  # noqa: E402


def example(correct: str, wrong: str) -> dict:
    return {"hypothesis_1": correct, "hypothesis_2": wrong, "label": 0}


@pytest.mark.parametrize("correct, wrong, group", [
    ("She found them.", "She did not look for them.", "only in the wrong hypothesis"),
    ("She didn't look for them.", "She found them.", "only in the correct hypothesis"),
    ("She found them.", "She searched the flat.", "in neither hypothesis"),
    ("Nobody helped her.", "She never asked.", "in both hypotheses"),
    ("She couldn’t sleep.", "She slept.", "only in the correct hypothesis"),  # typographic apostrophe
    ("I know another note.", "No one came.", "only in the wrong hypothesis"),  # know, another, note are no negations
])
def test_negation_group(correct, wrong, group):
    assert analysis.negation_group(example(correct, wrong)) == group


def test_negation_group_uses_the_label():
    swapped = {"hypothesis_1": "She did not look.", "hypothesis_2": "She found them.", "label": 1}
    assert analysis.negation_group(swapped) == "only in the wrong hypothesis"


def test_accuracy_and_interval():
    assert analysis.accuracy([1, 1, 0, 1]) == 75.0
    assert analysis.interval(50.0, 100) == pytest.approx(9.8)
    assert analysis.interval(100.0, 100) == 0.0


def test_mcnemar():
    first = {i: int(i < 60) for i in range(100)}  # solves examples 0 to 59
    second = {i: int(i < 50 or i >= 90) for i in range(100)}  # loses 10 of them and solves 10 others
    assert analysis.mcnemar(first, second) == (10, 10, 1.0)
    assert analysis.mcnemar(first, first) == (0, 0, 1.0)
    better = {i: int(i < 68) for i in range(100)}  # solves 8 more and loses none: 2 * 0.5 ** 8
    only_second, only_first, p = analysis.mcnemar(first, better)
    assert (only_second, only_first) == (8, 0)
    assert p == pytest.approx(2 * 0.5 ** 8)


def write_predictions(folder: Path, rows: list[tuple]) -> None:
    folder.mkdir(parents=True)
    lines = ["index,label,p_h1,p_h2,pred,correct"] + [",".join(map(str, row)) for row in rows]
    (folder / "predictions_test.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tables_from_prediction_files(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "RUNS", tmp_path)
    analysis.predictions.cache_clear()
    write_predictions(tmp_path / "a", [(0, 0, 0.95, 0.05, 0, 1), (1, 1, 0.92, 0.08, 0, 0), (2, 1, 0.45, 0.55, 1, 1), (3, 0, 0.3, 0.7, 1, 0)])
    write_predictions(tmp_path / "b", [(0, 0, 0.6, 0.4, 0, 1), (1, 1, 0.2, 0.8, 1, 1), (2, 1, 0.9, 0.1, 0, 0), (3, 0, 0.1, 0.9, 1, 0)])

    assert analysis.correct("a", "test") == {0: 1, 1: 0, 2: 1, 3: 0}
    assert analysis.p_correct("a", "test") == {0: 0.95, 1: 0.08, 2: 0.55, 3: 0.3}
    assert analysis.confident_errors("a", "test", 2) == [1, 3]
    assert analysis.overlap("a", "b", "test") == {"both right": 1, "only the first right": 1, "only the second right": 1, "both wrong": 1}

    table = analysis.confidence_table(["a"], "test")
    assert [row["confidence"] for row in table] == ["0.5 to 0.6", "0.6 to 0.7", "0.7 to 0.8", "0.8 to 0.9", "0.9 to 1.0"]
    assert [row["share"] for row in table] == [25.0, 0.0, 25.0, 0.0, 50.0]
    assert table[0]["accuracy"] == 100.0 and table[2]["accuracy"] == 0.0 and table[4]["accuracy"] == 50.0
    analysis.predictions.cache_clear()


FINAL_RUNS = [run for runs in analysis.MODELS.values() for run in runs]


@pytest.mark.parametrize("run", FINAL_RUNS)
def test_saved_predictions_match_results(run):
    """The accuracy and the confusion matrix in results.json can be recomputed from the prediction files."""
    if not (analysis.RUNS / run).exists():
        pytest.skip(f"runs/{run} is not available")
    analysis.predictions.cache_clear()
    for split in ("validation", "test"):
        result = analysis.results(run)[split]
        rows = analysis.predictions(run, split)
        assert analysis.accuracy(int(row["correct"]) for row in rows) == pytest.approx(100 * result["acc"], abs=1e-4)
        matrix = [[0, 0], [0, 0]]
        for row in rows:
            if row["pred"] != "-1":  # invalid LLM answers are counted as wrong but are not in the matrix
                matrix[int(row["label"])][int(row["pred"])] += 1
        assert matrix == result["confusion"]


def test_ablation_runs():
    assert analysis.ablation_runs("Final configuration", analysis.M1) == analysis.MODELS[analysis.M1]
    assert analysis.ablation_runs("Final configuration", analysis.M2) == analysis.MODELS[analysis.M2]
    assert analysis.ablation_runs("Hypothesis only", analysis.M2) == [f"pre-lr1e-05-honly-s{seed}" for seed in (42, 1, 2, 3)]
    assert analysis.ablation_runs("Segment order: O1 O2, then hypothesis", analysis.M1) == ["rand-lr2e-05-o1o2h-ep30-s42"]
    assert analysis.ablation_runs("10 instead of 5 epochs", analysis.M1) == []


def test_every_listed_run_exists():
    listed = FINAL_RUNS + analysis.PILOT_RUNS + analysis.PROMPT_RUNS
    listed += [run for runs in analysis.LR_RUNS.values() for run in runs]
    listed += [run for variant in analysis.ABLATIONS for model in (analysis.M1, analysis.M2) for run in analysis.ablation_runs(variant, model)]
    if not analysis.RUNS.exists():
        pytest.skip("runs/ is not available")
    missing = [run for run in listed if not (analysis.RUNS / run / "results.json").exists()]
    assert not missing
