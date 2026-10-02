"""Restore frozen KT models and windowlate samples for explanation research.

Read-only research module: loads an archived run (config + best checkpoint),
rebuilds the model via its registered trainer, and provides per-user
evidence/holdout splits straight from the windowlate parquet contract.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import model  # noqa: F401  — triggers trainer/model-config registry discovery
import numpy as np
import polars as pl
import torch

from utils.config import parse_run_archive
from utils.core import TRAINERS
from utils.data_process import get_data_source
from utils.training.checkpoint import CheckpointManager


@dataclass
class WindowSample:
    """One user's windowlate sample: evidence (early) + holdout (late)."""

    user_id: int
    sequence: np.ndarray  # dense skill ids, evidence+holdout concatenated
    response: np.ndarray  # 0/1 aligned with sequence
    question: np.ndarray  # dense question ids aligned with sequence
    holdout_idx: np.ndarray  # positions belonging to the late window

    @property
    def evidence_idx(self) -> np.ndarray:
        n = len(self.sequence)
        return np.array([i for i in range(n) if i not in set(self.holdout_idx.tolist())])


@dataclass
class RestoredModel:
    """A frozen KT model restored from an archived run directory."""

    model: torch.nn.Module
    model_name: str
    rc: Any
    data_src: Any
    device: torch.device
    run_dir: str
    skill_names: dict[int, str] = field(default_factory=dict)


@dataclass
class _RestoreEntry:
    """Minimal entry node so parse_run_archive accepts programmatic argv."""

    run_dir: str = ""


def restore(run_dir: str, device: str = "cuda") -> RestoredModel:
    """Restore model + data source from an archived run directory.

    Args:
        run_dir: Path like runs/normal/DKT_assistments09_..._fold0_bs128.
        device: "cuda" or "cpu".

    Returns:
        RestoredModel with model.eval() and weights loaded.
    """
    rc, _entry, resolved = parse_run_archive(
        [f"--explain.run_dir={run_dir}"],
        prog="llm_explain.restore",
        description="programmatic restore",
        entry_node="explain",
        entry_cls=_RestoreEntry,
    )
    data_src = get_data_source(rc)

    trainer_cls = TRAINERS.get(rc.experiment.model_name)
    trainer = trainer_cls.__new__(trainer_cls)  # build_components only, no training loop
    components = trainer.build_components(rc, data_src)

    model = components.model
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(resolved, "best_model.pth")
    CheckpointManager.load_weights(ckpt, model, dev)
    model.to(dev)
    model.eval()

    return RestoredModel(
        model=model,
        model_name=rc.experiment.model_name,
        rc=rc,
        data_src=data_src,
        device=dev,
        run_dir=str(resolved),
        skill_names=build_skill_names(data_src),
    )


def build_skill_names(data_src: Any) -> dict[int, str]:
    """Rebuild dense skill id -> human-readable skill name mapping.

    Replays the dataset's own load/clean/transform in memory (no save) to
    obtain the deterministic ``_id_mappings["skill"]`` (skill string -> dense
    id), then joins against the raw CSV's skill_id/skill_name columns.
    """
    data_src.load_src_data()
    data_src.clean_raw_data()
    data_src.transform_data()
    skill_map: dict[str, int] = data_src._id_mappings["skill"]  # noqa: SLF001

    pairs = data_src.raw_data.select(["skill_id", "skill_name"]).unique().collect()
    part_to_name: dict[str, str] = {}
    for sid, sname in zip(pairs["skill_id"].to_list(), pairs["skill_name"].to_list()):
        if sid is None or sname is None:
            continue
        id_parts = str(sid).split("_")
        name_parts = str(sname).split("_")
        if len(id_parts) == len(name_parts):
            for a, b in zip(id_parts, name_parts):
                part_to_name.setdefault(a, b)
        elif len(id_parts) == 1:
            part_to_name.setdefault(str(sid), str(sname))

    return {
        dense: (part_to_name.get(skill_str, "").strip() or f"KC {skill_str}")
        for skill_str, dense in skill_map.items()
    }


def load_user_samples(
    data_src: Any,
    fold: int = 0,
    holdout_ratio: float = 0.2,
    min_evidence: int = 10,
) -> list[WindowSample]:
    """Load per-user trajectories for a fold's validation users, split 80/20.

    Uses the split skill-sequence data (the exact sequences the models were
    trained/validated on, one row per expanded skill step) and the user->fold
    mapping, so selected students were never seen during training.

    Args:
        data_src: Data source with data_folder set.
        fold: Fold whose validation users to load (model-unseen at training).
        holdout_ratio: Tail fraction reserved as holdout (never enters prompts).
        min_evidence: Minimum evidence steps to keep a user.

    Returns:
        List of WindowSample sorted by user_id.
    """
    from utils.model_data import SkillModelData

    class _SeqOnly(SkillModelData):
        """SkillModelData without model-specific prepare_data (sequence loading only)."""

        def prepare_data(self, rc: Any) -> tuple:  # pragma: no cover - unused here
            raise NotImplementedError

    md = _SeqOnly(data_src)
    seq, resp, mask, uid, q = md.build_sequence_data()
    user_folds = md._build_user_folds(len(seq))  # noqa: SLF001
    val_idx = np.where(user_folds == fold)[0]

    samples: list[WindowSample] = []
    for i in val_idx:
        valid = np.where(mask[i] == 1)[0]
        n = len(valid)
        ev_len = int(round(n * (1 - holdout_ratio)))
        if ev_len < min_evidence or n - ev_len < 3:
            continue
        samples.append(
            WindowSample(
                user_id=int(uid[i][valid][0]),
                sequence=seq[i][valid].astype(np.int64),
                response=resp[i][valid].astype(np.int64),
                question=q[i][valid].astype(np.int64),
                holdout_idx=np.arange(ev_len, n),
            )
        )
    return sorted(samples, key=lambda s: s.user_id)
