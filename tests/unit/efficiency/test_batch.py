"""Tests for batch helpers: shape probing, recursive device moves, scored counting.

``count_valid_interactions`` / ``count_test_predictions`` are exercised through
the duck-typed ``_StubTarget`` double (pattern shared with
``test_windowlate_stage.py``) so no trainer is built.
"""

from typing import Any, cast

import pytest
import torch

from utils.efficiency.measures.batch import (
    batch_size_of,
    count_test_predictions,
    count_valid_interactions,
    to_device,
)
from utils.efficiency.target import BenchmarkTarget


class _StubTarget:
    """Duck-typed BenchmarkTarget counting forward/test_forward calls."""

    def __init__(self, train_y: torch.Tensor, test_y: torch.Tensor) -> None:
        self.model = torch.nn.Linear(2, 2)
        self.device = torch.device("cpu")
        self.forward_calls = 0
        self.test_forward_calls = 0
        self._train_y = train_y
        self._test_y = test_y

    # ``batch`` is an opaque passthrough token; callers hand it back untouched.
    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        return {"y_label": self._train_y}

    def test_forward(self, batch: Any) -> dict[str, torch.Tensor]:
        self.test_forward_calls += 1
        return {"y_label": self._test_y}


# --- batch_size_of ---


class TestBatchSizeOf:
    def test_tuple_of_tensors_uses_first(self) -> None:
        batch = (torch.zeros(4, 3), torch.zeros(4))
        assert batch_size_of(batch) == 4

    def test_list_and_dict_batches(self) -> None:
        assert batch_size_of([torch.zeros(7, 2)]) == 7
        assert batch_size_of({"a": torch.zeros(9, 1)}) == 9

    @pytest.mark.parametrize("batch", [("meta", 42), None, torch.tensor(3)])
    def test_no_row_tensor_raises(self, batch: Any) -> None:
        with pytest.raises(ValueError, match="no non-scalar tensor"):
            batch_size_of(batch)

    def test_nested_augmentation_views(self) -> None:
        tensor = torch.zeros(64, 200)
        batch = ([tensor] * 3, [tensor] * 4, [tensor] * 3)
        assert batch_size_of(batch) == 64

    def test_tensor_and_nested_dict_skip_scalars(self) -> None:
        assert batch_size_of(torch.zeros(5, 2)) == 5
        assert (
            batch_size_of({"step": torch.tensor(3), "views": [[torch.zeros(7, 2)]]})
            == 7
        )


# --- to_device ---


class TestToDevice:
    def test_tuple_and_list_types_preserved(self) -> None:
        batch = (torch.zeros(2), torch.zeros(2))
        moved = to_device(batch, torch.device("cpu"))
        assert isinstance(moved, tuple)
        assert isinstance(to_device(list(batch), torch.device("cpu")), list)

    def test_dict_keys_and_values_preserved(self) -> None:
        batch = {"x": torch.zeros(2), "mask": torch.ones(2, dtype=torch.bool)}
        moved = to_device(batch, torch.device("cpu"))
        assert set(moved) == {"x", "mask"}
        assert moved["mask"].dtype == torch.bool

    def test_nested_structure_recurses(self) -> None:
        batch = {"pairs": [torch.zeros(2), (torch.zeros(2), "tag")]}
        moved = to_device(batch, torch.device("cpu"))
        assert isinstance(moved["pairs"][1], tuple)
        assert moved["pairs"][1][1] == "tag"

    def test_non_tensor_passes_through_unchanged(self) -> None:
        assert to_device("meta", torch.device("cpu")) == "meta"
        assert to_device(3, torch.device("cpu")) == 3


# --- counted forwards via the stub target ---


class TestCountScored:
    # The stubs are partial doubles (only forward/device are exercised), so
    # each call casts to the typed BenchmarkTarget boundary.
    def test_valid_interactions_counts_y_label_numel(self) -> None:
        target = _StubTarget(torch.zeros(6), torch.zeros(2))
        assert count_valid_interactions(cast(BenchmarkTarget, target), "batch") == 6
        assert target.forward_calls == 1

    def test_test_predictions_counts_y_label_numel(self) -> None:
        target = _StubTarget(torch.zeros(6), torch.zeros(3, 2))
        assert count_test_predictions(cast(BenchmarkTarget, target), "batch") == 6
        assert target.test_forward_calls == 1

    def test_train_and_test_denominators_are_independent(self) -> None:
        target = _StubTarget(torch.zeros(64), torch.zeros(8))
        assert count_valid_interactions(cast(BenchmarkTarget, target), "b") == 64
        assert count_test_predictions(cast(BenchmarkTarget, target), "b") == 8


class _LazyCacheTarget:
    """Duck-typed target around a module that caches a constant on first forward.

    The cached tensor feeds a grad-tracked multiplication, reproducing the
    AKT-family lazy-constant pattern (e.g. FlucKT's Kerple bias).
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.cached: torch.Tensor | None = None

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        x = batch
        if self.cached is None:
            self.cached = torch.arange(3, dtype=x.dtype, device=x.device)
        return {"y_label": ((x * self.weight) * self.cached).reshape(-1)}


class TestCountScoredCacheSafety:
    def test_counting_forward_keeps_lazy_caches_autograd_compatible(self) -> None:
        """The setup counting forward must not create inference-tensor caches.

        It is usually the session's first forward, so lazily built constants
        are cached here and reused by the grad-enabled FLOPs/train stages;
        inference tensors would be rejected there.
        """
        target = _LazyCacheTarget()
        batch = torch.randn(2, 3)
        assert count_valid_interactions(cast(BenchmarkTarget, target), batch) == 6
        assert target.cached is not None
        assert not target.cached.is_inference()

        out = target.forward(batch)
        out["y_label"].sum().backward()
        assert target.weight.grad is not None
