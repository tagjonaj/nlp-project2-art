# NLP HS26 - Course Project 2: Transformers on ART (αNLI)

**Author:** Gabriel Gjonaj · HSLU, Natural Language Processing (HS26, A. Marfurt)

**Task:** given two observations *O₁* (before) and *O₂* (after), pick the more plausible of two
hypotheses *h₁*, *h₂* (ART dataset, Bhagavatula et al., 2020; course version `amarfurt/art-nlp-hs26`).

**Models (from the task sheet)**
1. A randomly initialized Transformer.
2. A pretrained Transformer with the same model hyperparameters as 1.
3. An LLM with 8B+ parameters.

Models 1 and 2 are finetuned on ART, the LLM is prompt engineered.

`EXPECTATIONS.md` contains the expected accuracies, written on 2026-11-15 before the first experiment.

## Tools and sources (declaration)

* Libraries: PyTorch, Lightning, Weights & Biases, Hugging Face `transformers` and `datasets`.
* **Claude (Anthropic) is used as a coding assistant** for writing and reviewing code, tests and
  documentation, and as a discussion partner for design decisions. Every decision in this repository
  is mine and I can explain it.
* Course material (lecture slides and notebooks of NLP HS26) as reference.

## Setup

```bash
git clone https://github.com/tagjonaj/nlp-project2-art.git && cd nlp-project2-art
pip install -r requirements.txt
```
