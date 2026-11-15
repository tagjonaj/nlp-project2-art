"""LightningModule: loss, metrics, optimizer and learning rate schedule."""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from .models import ARTModel


def lr_factor(step: int, total_steps: int, warmup_ratio: float) -> float:
    """Factor on the learning rate at ``step``: linear warmup from 1/100 to 1, then linear decay to 1/10 000."""
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)
    start, end = 1e-2, 1e-4
    if step < warmup_steps:
        return start + (1 - start) * step / warmup_steps
    return max(end, 1 - (1 - end) * (step - warmup_steps) / decay_steps)


class ARTClassifier(L.LightningModule):
    """Trains an ``ARTModel`` that maps a batch to [B, 2] logits, one per hypothesis.

    * Loss: cross-entropy over the two hypotheses (softmax over the two scores).
    * Optimizer: AdamW on all parameters. The classifier may get its own learning rate, because it
      is randomly initialized also in the pretrained model.
    * Schedule: linear warmup from lr / 100 to lr during the first ``warmup_ratio`` of all steps,
      then linear decay to lr / 10 000 (stepped after every batch).
    * Logged once per epoch: ``train/loss``, ``train/acc``, ``val/loss``, ``val/acc``.
    """

    def __init__(self, model: ARTModel, lr: float = 2e-5, classifier_lr: float = 0.0, weight_decay: float = 0.01,
                 warmup_ratio: float = 0.1) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.classifier_lr = classifier_lr or lr  # 0 = same learning rate as the encoder
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch)
        loss = F.cross_entropy(logits, batch["label"])
        acc = (logits.argmax(dim=-1) == batch["label"]).float().mean()
        batch_size = batch["label"].size(0)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def predict_step(self, batch, batch_idx):
        return torch.softmax(self(batch), dim=-1)  # [B, 2] probabilities of the two hypotheses

    def configure_optimizers(self):
        groups = [
            {"params": self.model.encoder.parameters(), "lr": self.lr},
            {"params": self.model.classifier.parameters(), "lr": self.classifier_lr},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor(step, total_steps, self.warmup_ratio))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
