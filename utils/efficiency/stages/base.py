"""Efficiency stage contract: shared context + pluggable stage ABC.

A stage is one measurement dimension (params/FLOPs, inference latency, training
throughput, ...). Stages register via ``@register_efficiency_stage("name")`` and
are auto-discovered from this package — drop a file here, it shows up. The
session runs the enabled stages in ``priority`` order, each contributing a
serializable result to the report.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from rich.table import Table

from ..environment import EnvironmentInfo
from ..measures.formatting import (  # noqa: F401  — re-exported for stages
    format_duration,
    format_flops,
)
from ..target import BenchmarkTarget

# Title style for efficiency result table.
TITLE_STYLE = "bold cyan"


@dataclass
class StageContext:
    """Shared runtime context injected into every stage by the session.

    ``cfg`` is the composed EfficiencyConfig (general + per-stage sub-nodes),
    typed ``Any`` to avoid a config↔stages import cycle. Stages read it via
    :attr:`general` and :meth:`stage_cfg`.
    """

    target: BenchmarkTarget
    device: torch.device
    sample_batch: Any
    batch_size: int
    valid_tokens: float
    cfg: Any
    environment: EnvironmentInfo
    output_dir: Path | None = None
    # Provenance for ``valid_tokens`` when it is a full-split mean: total valid
    # interactions and the batch count the mean divides.
    valid_tokens_total: int = 0
    valid_tokens_batches: int = 0

    @property
    def general(self) -> Any:
        """Cross-stage config node (warmup, modes, routing, ...)."""
        return self.cfg.general

    def stage_cfg(self, name: str) -> Any:
        """This stage's own config sub-node (its registered name)."""
        return getattr(self.cfg, name)


class EfficiencyStage(ABC):
    """One pluggable efficiency measurement.

    Subclasses set ``name`` (registry key + report result key) and ``priority``
    (smaller runs earlier), then implement ``run`` and ``format_table``. Stages
    are stateless: all per-run state arrives via :class:`StageContext`.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 100

    @abstractmethod
    def run(self, ctx: StageContext) -> Any:
        """Execute the measurement and return a serializable result."""

    @classmethod
    @abstractmethod
    def format_table(cls, result: Any) -> Table | None:
        """Render the result as a Rich table, or ``None`` to print nothing."""

    @classmethod
    def make_kv_table(cls, title: str) -> Table:
        """Standard Key/Value result table (no header/frame) with uniform title."""
        table = Table(title=title, title_style=TITLE_STYLE, show_header=False, box=None)
        table.add_column("Key", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")
        return table

    @classmethod
    def add_latency_rows(cls, table: Table, result: Any) -> None:
        """Render the latency/peak rows shared by the forward-benchmark stages."""
        table.add_row("Latency mean", f"{result.latency_mean_ms:.3f} ms")
        table.add_row(
            "Latency p50 / p95 / p99",
            f"{result.latency_p50_ms:.3f} / {result.latency_p95_ms:.3f} / {result.latency_p99_ms:.3f} ms",
        )
        table.add_row("Latency cv", f"{result.latency_cv:.3f}")
        table.add_row("Repeat cv (stability)", f"{result.latency_repeat_cv:.3f}")
        if result.gpu_peak_allocated_mib is not None:
            table.add_row(
                "GPU peak (allocated)", f"{result.gpu_peak_allocated_mib:,.0f} MiB"
            )
        if result.gpu_peak_reserved_mib is not None:
            table.add_row(
                "GPU peak (reserved)", f"{result.gpu_peak_reserved_mib:,.0f} MiB"
            )


def format_valid_tokens(mean: float, total: int, batches: int) -> str:
    """Render a throughput numerator with its full-split provenance."""
    return f"{mean:,.1f} (split mean: {total:,} over {batches} batches)"
