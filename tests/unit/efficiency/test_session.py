"""Tests for ``_resolve_stages``: routing, validation, dedup, priority order.

Plus ``EfficiencySession.run`` stage-failure isolation: one stage raising
(e.g. CUDA OOM) is recorded in ``report.errors`` while later stages still run.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import call, patch

import pytest
import torch

from utils.config.run_config import RunConfig
from utils.core import EFFICIENCY_STAGES, register_efficiency_stage
from utils.efficiency.session import EfficiencySession, _resolve_stages
from utils.efficiency.stages.base import BenchmarkTarget, EfficiencyStage, StageContext
from utils.efficiency.stages.inference import InferenceStageConfig, benchmark_inference
from utils.efficiency.stages.profile import ProfileStageConfig
from utils.efficiency.stages.trace import TraceStageConfig
from utils.efficiency.stages.training import TrainStageConfig
from utils.efficiency.target import TrainerBenchmarkAdapter
from utils.training import BaseTrainer


class _FakeStage(EfficiencyStage):
    """Concrete stage double; priority is set per registered subclass."""

    def run(self, ctx: StageContext) -> Any:
        return None

    @classmethod
    def format_table(cls, result: Any) -> None:
        return None


@pytest.fixture
def two_fake_stages(registry_snapshot: None) -> None:
    """Register two same-priority stages under known registry keys."""
    for name in ("utest_stage_b", "utest_stage_a"):

        @register_efficiency_stage(name)
        class _Stage(_FakeStage):
            pass

        _Stage.name = name
        _Stage.priority = 50


class TestResolveStages:
    def test_empty_modes_selects_all_stages_by_priority(self) -> None:
        stages = _resolve_stages([])
        names = [name for name, _ in stages]
        from utils.core import get_supported_stages

        assert set(names) == set(get_supported_stages())
        priorities = [stage.priority for _, stage in stages]
        assert priorities == sorted(priorities)
        assert names == ["profile", "inference", "windowlate", "train", "trace"]

    def test_unknown_mode_exits_with_available_list(self) -> None:
        with pytest.raises(SystemExit, match="Unknown efficiency stage"):
            _resolve_stages(["profile", "utest_no_such_stage"])

    def test_selection_returns_fresh_stage_instances(self) -> None:
        stages = _resolve_stages(["train"])
        assert [name for name, _ in stages] == ["train"]
        again = _resolve_stages(["train"])
        assert again[0][1] is not stages[0][1]

    def test_dedup_preserves_first_seen_order(self, two_fake_stages: None) -> None:
        stages = _resolve_stages(["utest_stage_b", "utest_stage_a", "utest_stage_b"])
        assert [name for name, _ in stages] == ["utest_stage_b", "utest_stage_a"]

    def test_priority_sort_is_stable_within_equal_priority(
        self, two_fake_stages: None
    ) -> None:
        # Both fake stages share priority 50: first-seen order survives the sort.
        stages = _resolve_stages(["utest_stage_a", "utest_stage_b"])
        assert [name for name, _ in stages] == ["utest_stage_a", "utest_stage_b"]

    def test_registry_key_used_as_result_key(self, two_fake_stages: None) -> None:
        # Keys come from the registry name, not the stage's own ClassVar.
        stages = _resolve_stages(["utest_stage_a"])
        assert [name for name, _ in stages] == ["utest_stage_a"]
        assert EFFICIENCY_STAGES.get("utest_stage_a") is not None


class _FakeTarget:
    """Duck-typed BenchmarkTarget: CPU device, one synthetic batch."""

    device = torch.device("cpu")
    model = torch.nn.Linear(2, 2)

    def prepare(self, device: torch.device) -> None:
        pass

    @property
    def train_data(self) -> list[dict[str, Any]]:
        return [{"questions": torch.zeros(2, 3, dtype=torch.long)}]

    @property
    def inference_data(self) -> list[dict[str, Any]]:
        return [{"questions": torch.zeros(2, 3, dtype=torch.long)}]

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {"y_label": torch.zeros(6)}


def _make_session(tmp_path: Path, modes: str, target: Any = None) -> EfficiencySession:
    # eff_cfg must be a real dataclass: session serialization runs it through
    # ``config_to_dict`` → ``asdict``, which rejects SimpleNamespace.
    @dataclass
    class _GeneralCfg:
        # default_factory closure: a plain ``= modes`` default would shadow the
        # enclosing parameter inside the class body.
        modes: str = field(default_factory=lambda: modes)
        resource_sample_interval: float = 0.05
        warmup_iters: int = 1

    @dataclass
    class _EffCfg:
        general: _GeneralCfg = field(default_factory=_GeneralCfg)
        profile: ProfileStageConfig = field(default_factory=ProfileStageConfig)
        inference: InferenceStageConfig = field(
            default_factory=lambda: InferenceStageConfig(iters=2, repeats=1)
        )
        train: TrainStageConfig = field(
            default_factory=lambda: TrainStageConfig(iters=2, repeats=1)
        )
        trace: TraceStageConfig = field(
            default_factory=lambda: TraceStageConfig(iters=1, export=False)
        )

    rc = SimpleNamespace(
        experiment=SimpleNamespace(model_name="UTestModel"),
        data=SimpleNamespace(dataset="tinyds", max_seq_len=200),
        general=SimpleNamespace(seed=42),
        model=SimpleNamespace(batch_size=64),
    )
    # ``rc`` is a SimpleNamespace mirroring the RunConfig read-surface and the
    # stub target is a partial double: cast marks the typed session boundary.
    return EfficiencySession(
        target=cast(BenchmarkTarget, target or _FakeTarget()),
        rc=cast(RunConfig, rc),
        eff_cfg=_EffCfg(),
        data_src=cast(
            Any,
            SimpleNamespace(
                get_metadata=lambda: {
                    "max_question_seq_len": 3,
                    "max_skill_seq_len": 5,
                }
            ),
        ),
        output_dir=tmp_path,
    )


@pytest.fixture
def fail_then_ok_stages(registry_snapshot: None) -> None:
    """A stage that raises OOM (runs first) and a healthy stage (runs second)."""

    @register_efficiency_stage("utest_fail_stage")
    class _FailStage(_FakeStage):
        priority = 10

        def run(self, ctx: StageContext) -> None:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. (fake)")

    @register_efficiency_stage("utest_ok_stage")
    class _OkStage(_FakeStage):
        priority = 20

        def run(self, ctx: StageContext) -> dict[str, bool]:
            return {"ok": True}


class TestStageFailureIsolation:
    def test_failed_stage_recorded_and_later_stages_run(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        report = _make_session(tmp_path, "utest_fail_stage,utest_ok_stage").run()
        assert report.results == {"utest_ok_stage": {"ok": True}}
        assert "utest_fail_stage" in report.errors
        assert "OutOfMemoryError" in report.errors["utest_fail_stage"]
        assert report.modes == ["utest_fail_stage", "utest_ok_stage"]

    def test_errors_persist_to_report_json(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        _make_session(tmp_path, "utest_fail_stage,utest_ok_stage").run()
        payload = json.loads((tmp_path / "efficiency_report.json").read_text())
        assert "OutOfMemoryError" in payload["errors"]["utest_fail_stage"]
        assert payload["results"]["utest_ok_stage"] == {"ok": True}

    def test_all_stages_failed_raises_after_writing_report(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError, match="All efficiency stages failed"):
            _make_session(tmp_path, "utest_fail_stage").run()
        payload = json.loads((tmp_path / "efficiency_report.json").read_text())
        assert set(payload["errors"]) == {"utest_fail_stage"}
        assert payload["results"] == {}

    def test_no_failure_leaves_errors_empty(
        self, fail_then_ok_stages: None, tmp_path: Path
    ) -> None:
        report = _make_session(tmp_path, "utest_ok_stage").run()
        assert report.errors == {}
        assert report.results == {"utest_ok_stage": {"ok": True}}


@pytest.fixture
def ctx_capture_stage(registry_snapshot: None) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    @register_efficiency_stage("utest_capture_stage")
    class _CaptureStage(_FakeStage):
        priority = 5

        def run(self, ctx: StageContext) -> dict[str, bool]:
            metrics = benchmark_inference(
                ctx.target,
                ctx.inference_batch,
                warmup_iters=0,
                iters=1,
                repeats=1,
                device=ctx.device,
            )
            captured.update(
                valid_tokens=metrics.valid_tokens_per_batch,
                batch_size=metrics.batch_size,
                batch=ctx.inference_batch,
            )
            return {"ok": True}

    return captured


class TestMeasuredBatchCount:
    def test_counts_only_the_measured_validation_batch(
        self, ctx_capture_stage: dict[str, Any], tmp_path: Path
    ) -> None:
        class _TwoBatchTarget(_FakeTarget):
            calls = 0
            loader_accesses = 0

            @property
            def train_data(self) -> Any:
                raise AssertionError("inference accessed training data")

            @property
            def inference_data(self) -> list[dict[str, Any]]:
                self.loader_accesses += 1
                return [
                    {"n": 6, "questions": torch.zeros(2, 3)},
                    {"n": 3, "questions": torch.zeros(1, 3)},
                ]

            def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
                self.calls += 1
                return {"y_label": torch.zeros(batch["n"])}

        target = _TwoBatchTarget()
        report = _make_session(tmp_path, "utest_capture_stage", target).run()
        assert report.errors == {}
        assert report.batch_size == 64
        assert report.sequence_lengths == {
            "max_question_seq_len": 3,
            "max_skill_seq_len": 5,
        }
        assert ctx_capture_stage["valid_tokens"] == 6
        assert ctx_capture_stage["batch_size"] == 2
        assert target.calls == 3  # count, sustained timing, latency timing
        assert target.loader_accesses == 1

    def test_loader_error_is_recorded_as_stage_failure(
        self,
        ctx_capture_stage: dict[str, Any],
        fail_then_ok_stages: None,
        tmp_path: Path,
    ) -> None:
        class _BrokenInferenceTarget(_FakeTarget):
            @property
            def inference_data(self) -> Any:
                raise FileNotFoundError("validation data missing")

        report = _make_session(
            tmp_path, "utest_capture_stage,utest_ok_stage", _BrokenInferenceTarget()
        ).run()
        assert report.results == {"utest_ok_stage": {"ok": True}}
        assert (
            "FileNotFoundError: validation data missing"
            in report.errors["utest_capture_stage"]
        )


class _PhaseTrainer(BaseTrainer):
    """Trainer whose augmented train input cannot be used by its eval branch."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.device_ = torch.device("cpu")
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.loss = torch.nn.BCELoss()
        self.max_clip_grad_norm = None
        self.seen: list[tuple[str, int]] = []
        self.auxiliary_calls = 0
        self.cached: torch.Tensor | None = None
        train_x = torch.ones(4, 4)
        train_y = torch.zeros(4, 4)
        train_m = torch.ones(4, 4, dtype=torch.bool)
        train_m[:, 0] = False
        self.train_data = [
            (
                (train_x + 1, train_x - 1, train_x),
                (train_y, train_y, train_y, 1 - train_y),
                (train_m, train_m, train_m),
            )
        ]
        val_m = torch.ones(3, 4, dtype=torch.bool)
        val_m[:, 0] = False
        self.val_data = [(torch.ones(3, 4), torch.zeros(3, 4), val_m)]
        self.test_data = None

    def forward_pass(self, batch_data: Any) -> dict:
        result = {}
        if self.model.training:
            (x1, x2, x), (_, _, y, _), (_, _, mask) = batch_data
            result["cl_loss"] = (
                (self.model(x1.unsqueeze(-1)) - self.model(x2.unsqueeze(-1)))
                .square()
                .mean()
            )
            self.auxiliary_calls += 1
            self.seen.append(("train", x.size(0)))
        else:
            x, y, mask = batch_data
            # A nested training batch would fail here, as in the original CL4KT.
            self.seen.append(("validation", x.size(0)))
        if self.cached is None:
            self.cached = torch.arange(1, x.size(1) + 1, dtype=x.dtype)
        preds = (self.model(x.unsqueeze(-1)).squeeze(-1) * self.cached).sigmoid()
        result.update(y_hat=preds[mask], y_label=y[mask])
        return result

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        return self.loss(outputs["y_hat"], outputs["y_label"]) + outputs["cl_loss"]


class TestPhaseRouting:
    def test_all_paths_use_matching_batches_and_counts(self, tmp_path: Path) -> None:
        trainer = _PhaseTrainer()
        before = trainer.model.weight.detach().clone()
        report = _make_session(
            tmp_path, "profile,inference,train,trace", TrainerBenchmarkAdapter(trainer)
        ).run()
        assert report.errors == {}
        assert report.results["profile"].data_split == "validation"
        assert report.results["profile"].batch_size == 3
        inference = report.results["inference"]
        assert inference.data_split == "validation"
        assert inference.count_basis == "measured_batch"
        assert inference.batch_size == 3
        assert inference.valid_tokens_per_batch == 9
        train = report.results["train"]
        assert train.data_split == "train"
        assert train.batch_size == 4
        assert train.valid_tokens_per_batch == 12
        assert (
            train.throughput_interactions_per_sec * train.ns_per_interaction
            == pytest.approx(1e9)
        )
        trace = report.results["trace"]
        assert trace.forward.batch_size == 3
        assert trace.forward.data_split == "validation"
        assert trace.train.batch_size == 4
        assert trace.train.data_split == "train"
        assert set(trainer.seen) == {("validation", 3), ("train", 4)}
        assert trainer.auxiliary_calls == 5  # train warmup + timing, then trace
        assert not torch.equal(before, trainer.model.weight.detach())

    def test_train_only_does_not_require_validation(self, tmp_path: Path) -> None:
        trainer = _PhaseTrainer()
        trainer.val_data = None
        report = _make_session(
            tmp_path, "train", TrainerBenchmarkAdapter(trainer)
        ).run()
        assert report.errors == {}
        assert set(trainer.seen) == {("train", 4)}
        assert report.results["train"].valid_tokens_per_batch == 12

    def test_inference_only_does_not_require_training(self, tmp_path: Path) -> None:
        trainer = _PhaseTrainer()
        trainer.train_data = None
        report = _make_session(
            tmp_path, "inference", TrainerBenchmarkAdapter(trainer)
        ).run()
        assert report.errors == {}
        assert set(trainer.seen) == {("validation", 3)}

    def test_parameters_only_does_not_load_any_data(self, tmp_path: Path) -> None:
        trainer = _PhaseTrainer()
        trainer.train_data = trainer.val_data = None
        session = _make_session(tmp_path, "profile", TrainerBenchmarkAdapter(trainer))
        session.cfg.profile.flops = False
        report = session.run()
        assert report.errors == {}
        assert report.results["profile"].params == 2
        assert trainer.seen == []

    def test_empty_validation_is_an_error_with_no_training_fallback(
        self, tmp_path: Path
    ) -> None:
        trainer = _PhaseTrainer()
        trainer.val_data = []
        report = _make_session(
            tmp_path, "inference,train", TrainerBenchmarkAdapter(trainer)
        ).run()
        assert "StopIteration" in report.errors["inference"]
        assert set(report.results) == {"train"}
        assert set(trainer.seen) == {("train", 4)}

    def test_trace_only_initializes_autograd_compatible_caches(
        self, tmp_path: Path
    ) -> None:
        trainer = _PhaseTrainer()
        report = _make_session(
            tmp_path, "trace", TrainerBenchmarkAdapter(trainer)
        ).run()
        assert report.errors == {}
        assert trainer.cached is not None
        assert not trainer.cached.is_inference()
        assert trainer.auxiliary_calls == 2

    def test_inference_switches_model_mode_once(self, tmp_path: Path) -> None:
        trainer = _PhaseTrainer()
        with patch.object(trainer.model, "train", wraps=trainer.model.train) as mode:
            report = _make_session(
                tmp_path, "inference", TrainerBenchmarkAdapter(trainer)
            ).run()
        assert report.errors == {}
        assert mode.call_args_list == [call(False)]

    @pytest.mark.parametrize("warmup_iters,expected_forwards", [(0, 2), (1, 2), (2, 3)])
    def test_trace_warmup_primes_caches_without_an_extra_forward(
        self, tmp_path: Path, warmup_iters: int, expected_forwards: int
    ) -> None:
        trainer = _PhaseTrainer()
        session = _make_session(tmp_path, "trace", TrainerBenchmarkAdapter(trainer))
        session.cfg.general.warmup_iters = warmup_iters
        report = session.run()
        assert report.errors == {}
        assert trainer.cached is not None
        assert not trainer.cached.is_inference()
        assert trainer.seen.count(("validation", 3)) == expected_forwards
        assert trainer.auxiliary_calls == warmup_iters + 1
