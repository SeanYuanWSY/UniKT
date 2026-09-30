"""BenchmarkTarget: the efficiency framework's only contact surface with a model.

A narrow protocol over a built trainer (or any model-bearing object) so the
benchmark never reaches into trainer internals (``opt``/``loss``/
``_compute_loss``/``max_clip_grad_norm``). :class:`TrainerBenchmarkAdapter`
adapts a single-stage ``BaseTrainer`` subclass; its ``compute_train_step``
routes through ``trainer.compute_train_step``, the same path the real training
loop uses, so the benchmark and training never drift apart.
"""

from typing import Any, Protocol

import torch


class BenchmarkTarget(Protocol):
    """The surface efficiency stages measure against."""

    @property
    def model(self) -> torch.nn.Module:
        """The underlying module (params/FLOPs/dtype inspection)."""
        ...

    @property
    def device(self) -> torch.device:
        """The device the model is measured on."""
        ...

    @property
    def train_data(self) -> Any:
        """The training DataLoader (one representative batch is prefetched)."""
        ...

    @property
    def inference_data(self) -> Any:
        """The validation DataLoader used by the trainer's eval forward path."""
        ...

    @property
    def test_data(self) -> Any:
        """The test DataLoader, or ``None`` when the target has no test split."""
        ...

    def forward(self, batch: Any) -> Any:
        """Evaluation forward; the stage selects model mode and grad context."""
        ...

    def test_forward(self, batch: Any) -> Any:
        """Test forward; the stage selects model mode and grad context."""
        ...

    def compute_train_step(self, batch: Any) -> tuple[dict, torch.Tensor]:
        """One training step's pure computation, returning ``(output, loss)``."""
        ...

    def prepare(self, device: torch.device) -> None:
        """Move model/loss onto the device before measurement."""
        ...


class TrainerBenchmarkAdapter:
    """Wrap a built single-stage trainer as a :class:`BenchmarkTarget`."""

    def __init__(self, trainer: Any) -> None:
        """Store the trainer; all access goes through its public surface.

        Duck-typed: the :class:`BenchmarkTarget` protocol is the contract, so
        the concrete trainer type is not imported here.
        """
        self._t = trainer

    @property
    def model(self) -> torch.nn.Module:
        """The trainer's underlying module."""
        return self._t.model

    @property
    def device(self) -> torch.device:
        """The device the trainer's components live on."""
        return self._t.device_

    @property
    def train_data(self) -> Any:
        """The trainer's training DataLoader."""
        return self._t.train_data

    @property
    def inference_data(self) -> Any:
        """The trainer's validation DataLoader, with its evaluation collator."""
        return self._t.val_data

    @property
    def test_data(self) -> Any:
        """The trainer's test DataLoader (``None`` when not built)."""
        return self._t.test_data

    def forward(self, batch: Any) -> Any:
        """Run an evaluation forward with the stage's mode and grad context.

        Left grad-agnostic so the FLOPs profile can run it grad-enabled
        (FlopCounterMode needs ``grad_fn``); inference/trace stages wrap it in
        ``inference_mode`` themselves.
        """
        return self._t.forward_pass(batch)

    def test_forward(self, batch: Any) -> Any:
        """Run one test forward pass via the trainer's ``test_forward_pass``."""
        return self._t.test_forward_pass(batch)

    def compute_train_step(self, batch: Any) -> tuple[dict, torch.Tensor]:
        """Run one training step via the shared ``compute_train_step`` path."""
        return self._t.compute_train_step(batch)

    def prepare(self, device: torch.device) -> None:
        """Move the model and (if modular) loss onto the device."""
        self._t.model.to(device)
        if isinstance(self._t.loss, torch.nn.Module):
            self._t.loss.to(device)

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights into the underlying model."""
        self._t.load_weights(checkpoint_path)
