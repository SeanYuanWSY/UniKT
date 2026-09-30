"""Batch-level helpers: shape probing, valid-interaction counting, device moves."""

from collections.abc import Callable
from typing import Any

import torch

from ..device import synchronize
from ..target import BenchmarkTarget


def batch_size_of(batch: Any) -> int:
    """Number of rows (student sequences B) in a batch."""
    first = _first_tensor(batch)
    if first is None:
        raise ValueError("Batch contains no non-scalar tensor")
    return int(first.size(0))


def _first_tensor(batch: Any) -> torch.Tensor | None:
    """First non-scalar tensor found recursively in a batch."""
    if isinstance(batch, torch.Tensor):
        return batch if batch.dim() >= 1 else None
    if isinstance(batch, dict):
        batch = batch.values()
    elif not isinstance(batch, (tuple, list)):
        return None
    for value in batch:
        first = _first_tensor(value)
        if first is not None:
            return first
    return None


def _count_scored(
    forward: Callable[[], dict[str, torch.Tensor]], device: torch.device
) -> int:
    """``numel`` of the forward's aligned 1D ``y_label``, synchronized."""
    # no_grad (not inference_mode): counting forwards may be a model's first
    # execution, and models with lazily built seq-len constants (AKT family)
    # cache them here. Tensors created under inference_mode are rejected by
    # autograd in the later grad-enabled FLOPs/train stages; no_grad tensors
    # are not.
    with torch.no_grad():
        out = forward()
        n = int(out["y_label"].numel())
    synchronize(device)
    return n


def count_valid_interactions(target: BenchmarkTarget, sample_batch: Any) -> int:
    """Valid interactions per forward pass that participate in the loss.

    Throughput numerator: runs one evaluation forward via the target, takes ``numel`` of
    the aligned+masked 1D ``y_label`` — the interactions retained by
    ``_extract_valid_predictions`` after the adjacent-pair mask, i.e. the samples
    ``_compute_loss`` actually consumes.
    """
    return _count_scored(lambda: target.forward(sample_batch), target.device)


def count_test_predictions(target: BenchmarkTarget, sample_batch: Any) -> int:
    """Scored predictions per test forward pass.

    Same contract as :func:`count_valid_interactions` but through
    ``test_forward_pass``. For windowlate data this is far smaller than the
    dense-sequence count — each window scores only its final position — so the
    two paths must use their own scored counts.
    """
    return _count_scored(lambda: target.test_forward(sample_batch), target.device)


def to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move batch tensors to device (tuple/list/dict aware)."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch
