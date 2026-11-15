"""Models 1 and 2: Transformer encoder + classifier that gives one score per hypothesis."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

POOLINGS = ("first", "mean")


class ARTModel(nn.Module):
    """Encoder -> one vector per sequence -> 2-layer classifier -> one score per hypothesis.

    ``pretrained=True`` loads the weights of ``checkpoint`` (model 2). ``pretrained=False`` builds
    the same architecture from the configuration of ``checkpoint`` and initializes it randomly
    (model 1). Both models therefore share all model hyperparameters and the tokenizer.

    The classifier is the head RoBERTa uses for sequence classification: dropout, linear, tanh,
    dropout, linear. It is randomly initialized in both models.
    """

    def __init__(self, checkpoint: str = "roberta-base", pretrained: bool = True, pooling: str = "first",
                 dropout: float = 0.1) -> None:
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}")
        if pretrained:
            self.encoder = AutoModel.from_pretrained(checkpoint, add_pooling_layer=False)
        else:
            config = AutoConfig.from_pretrained(checkpoint)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        self.pooling = pooling
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]  # [B, 2, L]
        batch_size, num_hypotheses, length = input_ids.shape
        input_ids = input_ids.view(-1, length)  # [2B, L]: every sequence is encoded on its own
        attention_mask = attention_mask.view(-1, length)
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [2B, L, H]
        if self.pooling == "first":
            pooled = hidden[:, 0]  # hidden state of the first token <s>
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # padding does not count
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        scores = self.classifier(pooled)  # [2B, 1]
        return scores.view(batch_size, num_hypotheses)  # [B, 2] logits, one per hypothesis


def count_parameters(model: ARTModel) -> dict[str, int]:
    """Number of parameters of the encoder and the classifier; all of them are trained."""
    encoder = sum(p.numel() for p in model.encoder.parameters())
    classifier = sum(p.numel() for p in model.classifier.parameters())
    return {"encoder": encoder, "classifier": classifier, "total": encoder + classifier}
