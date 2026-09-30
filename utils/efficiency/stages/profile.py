"""Profile stage: parameter counts, disk size, forward FLOPs."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from rich.table import Table

from utils.core import get_logger, register_efficiency_stage

from ..measures.batch import batch_size_of
from ..measures.flops import (
    count_flops as _count_flops,
)
from ..measures.flops import (
    estimate_disk_size_mb as _estimate_disk_size_mb,
)
from .base import EfficiencyStage, StageContext, format_flops

logger = get_logger(__name__)


@dataclass
class ModelProfile:
    """Static model profile."""

    params: int = 0
    trainable_params: int = 0
    model_size_mb: float = 0.0
    flops_forward: int | None = None
    op_breakdown: dict[str, int] = field(default_factory=dict)
    data_split: str | None = None
    batch_size: int | None = None


@dataclass
class ProfileStageConfig:
    """Profile stage knobs."""

    flops: bool = True


def profile_model(
    model: torch.nn.Module,
    forward_fn: Callable[[], Any],
    device: torch.device,
    count_flops: bool = True,
) -> ModelProfile:
    """Count parameter counts, disk size, and optionally forward FLOPs.

    Args:
        model: PyTorch model.
        forward_fn: Zero-argument callable that executes one full forward pass
            (typically ``trainer.forward_pass(batch)``). Provided by the caller to
            ensure correct model forward signature and trigger trainer-side state
            (e.g. GIKT's graph_data).
        device: Device the model is on.
        count_flops: Whether to estimate FLOPs (default True).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = _estimate_disk_size_mb(model)

    profile = ModelProfile(
        params=total,
        trainable_params=trainable,
        model_size_mb=round(size_mb, 3),
    )

    if count_flops:
        flops, breakdown = _count_flops(forward_fn, device)
        profile.flops_forward = flops
        profile.op_breakdown = breakdown

    if measured_flops := profile.flops_forward:
        gflops = measured_flops / 1e9
        flops_str = (
            f"{gflops:.2f} GFLOPs"
            if gflops >= 1
            else f"{measured_flops / 1e6:.2f} MFLOPs"
        )
        logger.info(
            f"[Profile] params={total:,} trainable={trainable:,} "
            f"size={size_mb:.2f}MB flops={flops_str}"
        )
    else:
        logger.info(
            f"[Profile] params={total:,} trainable={trainable:,} size={size_mb:.2f}MB"
        )

    return profile


@register_efficiency_stage("profile")
class ProfileStage(EfficiencyStage):
    """Static model profile: parameter counts, disk size, forward FLOPs."""

    name = "profile"
    priority = 10
    config_cls = ProfileStageConfig

    def run(self, ctx: StageContext) -> ModelProfile:
        """Measure parameter counts, disk size, and optional forward FLOPs."""
        count_flops = ctx.stage_cfg(self.name).flops
        if count_flops:
            ctx.target.model.eval()
        profile = profile_model(
            ctx.target.model,
            forward_fn=lambda: ctx.target.forward(ctx.inference_batch),
            device=ctx.device,
            count_flops=count_flops,
        )
        if count_flops:
            profile.data_split = "validation"
            profile.batch_size = batch_size_of(ctx.inference_batch)
        return profile

    @classmethod
    def format_table(cls, result: ModelProfile) -> Table | None:
        """Render the model profile as a Rich table."""
        table = cls.make_kv_table("Model Profile")
        table.add_row("Parameters", f"{result.params:,}")
        table.add_row("Trainable params", f"{result.trainable_params:,}")
        table.add_row("Model size", f"{result.model_size_mb:.2f} MB")
        if result.flops_forward is not None:
            table.add_row("Data split", str(result.data_split))
            table.add_row("Batch size", str(result.batch_size))
            table.add_row("FLOPs / forward", format_flops(result.flops_forward))
        if result.op_breakdown:
            top = list(result.op_breakdown.items())[:3]
            breakdown = ", ".join(f"{k}={format_flops(v)}" for k, v in top)
            table.add_row("Top ops", breakdown)
        return table
