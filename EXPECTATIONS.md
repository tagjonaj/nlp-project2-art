# EXPECTATIONS.md

Written **before** running any experiment (2026-11-15).

**Task:** ART / αNLI (`amarfurt/art-nlp-hs26`): choose the more plausible of two hypotheses given
two observations. Metric: accuracy on the test split (1532 examples).

**Reference points**
* Bhagavatula et al. (2020), adversarially filtered test set: majority 50.8 %, BERT-large 68.9 %,
  human 91.4 %.
* My Project 1 on the same splits (no pretrained Transformer): about 72 % validation and
  52-53 % test accuracy.
* Llama 3.1 70B with a zero-shot prompt: 86.2 % on αNLI (arXiv:2507.08264).

| model | expected best **val** accuracy | expected best **test** accuracy |
|---|---|---|
| 1. Randomly initialized Transformer (`roberta-base` configuration) | 55-68 % | **50-54 %** |
| 2. Pretrained Transformer (`roberta-base`), finetuned | 80-88 % | **68-75 %** |
| 3. LLM with 8B+ parameters (hosted open-weight model, 70B class or larger), prompt engineering | 88-94 % | **84-90 %** |

**Reasoning**
* *Validation above test for all models.* The test pairs were filtered with BERT as the adversary,
  the validation split is a held-out part of the training pairs. Project 1 showed this gap clearly.
* *Model 1.* Same architecture as model 2 but no pretraining. 16801 examples are too few to learn
  language from scratch with 125 M parameters, so I expect results close to Project 1 and near
  chance on the test split.
* *Model 2.* Pretraining supplies the language knowledge that model 1 lacks. BERT-large reached
  68.9 % on the test split although the filter was built against it. RoBERTa was not the adversary,
  so I expect `roberta-base` at or slightly above that value despite being the smaller model.
* *Model 3.* A 70B model is reported at 86.2 % without examples. Prompt engineering (few-shot
  examples, reasoning) should add a few points; human performance (91.4 %) is the practical ceiling.
  The public ART data may be part of the LLM's pretraining data, so this number can overestimate
  its reasoning ability.

The LLM is selected among hosted open-weight models with a pilot on a validation subset; the
expectation refers to that model class.
