"""Inference stage: latency distribution, throughput, peak memory."""

from dataclasses import dataclass
from typing import Any

import torch
from rich.table import Table

from utils.core import get_logger, register_efficiency_stage

from ..measures.batch import batch_size_of, count_valid_interactions
from ..measures.timing import (
    LatencyMetricsBase,
    benchmark_forward_loop,
)
from ..target import BenchmarkTarget
from .base import EfficiencyStage, StageContext

logger = get_logger(__name__)


@dataclass
class InferenceMetrics(LatencyMetricsBase):
    """Inference efficiency metrics."""

    iters: int = 0
    repeats: int = 0
    batch_size: int = 0
    data_split: str = "validation"
    count_basis: str = "measured_batch"
    valid_tokens_per_batch: int = 0
    throughput_interactions_per_sec: float = 0.0
    ns_per_interaction: float = 0.0


@dataclass
class InferenceStageConfig:
    """Inference stage knobs."""

    iters: int = 200
    repeats: int = 3


def benchmark_inference(
    target: BenchmarkTarget,
    sample_batch: Any,
    warmup_iters: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> InferenceMetrics:
    """Inference latency/throughput/memory benchmark.

    Reuses the prefetched ``sample_batch`` to avoid DataLoader IPC noise; a CUDA
    Event per iteration reads elapsed_time after ``end.synchronize()``, covering
    host launch through kernel completion. ``valid_tokens`` counts the scored
    interactions in this exact validation batch, using the trainer's normal
    evaluation pipeline with no sequence-length override.
    """
    target.model.eval()
    batch_size = batch_size_of(sample_batch)
    valid_tokens = count_valid_interactions(target, sample_batch)
    stats = benchmark_forward_loop(
        lambda: target.forward(sample_batch),
        warmup_iters,
        iters,
        repeats,
        device,
    )
    wall = stats.sustained_wall_s
    throughput = (valid_tokens * iters) / wall if wall > 0 else 0.0
    ns_per = (
        (wall * 1e9) / (valid_tokens * iters) if valid_tokens > 0 and iters > 0 else 0.0
    )

    logger.info(
        f"[Inference] latency_mean={stats.latency_mean_ms:.3f}ms "
        f"latency_p95={stats.latency_p95_ms:.3f}ms "
        f"latency_cv={stats.latency_cv:.3f} "
        f"repeat_cv={stats.latency_repeat_cv:.3f} | "
        f"throughput={throughput:,.0f} int/s"
        + (
            f" | gpu_peak={stats.gpu_peak_allocated_mib:.0f} MiB"
            if stats.gpu_peak_allocated_mib is not None
            else ""
        )
    )

    return InferenceMetrics(
        iters=iters,
        repeats=repeats,
        batch_size=batch_size,
        valid_tokens_per_batch=valid_tokens,
        throughput_interactions_per_sec=throughput,
        ns_per_interaction=ns_per,
        **LatencyMetricsBase.stats_kwargs(stats),
    )


@register_efficiency_stage("inference")
class InferenceStage(EfficiencyStage):
    """Inference efficiency: latency distribution, throughput, peak memory."""

    name = "inference"
    priority = 20
    config_cls = InferenceStageConfig

    def run(self, ctx: StageContext) -> InferenceMetrics:
        """Benchmark inference latency, throughput, and peak GPU memory."""
        cfg = ctx.stage_cfg(self.name)
        return benchmark_inference(
            ctx.target,
            ctx.inference_batch,
            ctx.general.warmup_iters,
            cfg.iters,
            cfg.repeats,
            ctx.device,
        )

    @classmethod
    def format_table(cls, result: InferenceMetrics) -> Table | None:
        """Render inference latency/throughput as a Rich table."""
        table = cls.make_kv_table("Inference")
        table.add_row("Iterations", f"{result.iters} x {result.repeats}")
        table.add_row("Data split", result.data_split)
        table.add_row("Batch size", f"{result.batch_size:,}")
        table.add_row("Count basis", result.count_basis)
        table.add_row("Valid tokens / batch", f"{result.valid_tokens_per_batch:,}")
        cls.add_latency_rows(table, result)
        table.add_row(
            "Throughput (sustained)",
            f"{result.throughput_interactions_per_sec:,.0f} interactions/s",
        )
        table.add_row("Per interaction", f"{result.ns_per_interaction:,.0f} ns")
        return table
