"""HiTSKT registration and UniKT training adapter."""

from dataclasses import field
from typing import Literal

import torch

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("HiTSKT")
class HiTSKTConfig(ModelConfig):
    """HiTSKT configuration.

    Args:
        action_size: Padded actions per session, including its end token.
        session_size: History sessions, including its end session.
        session_gap_hours: Time gap that starts a new session.
        timestamp_unit: Unit of the framework dataset timestamps.
        d_model: Embedding and hidden dimension.
        d_inner: Feedforward hidden dimension.
        n_layers: Layers in each encoder and decoder.
        n_head: Attention heads.
        d_k: Per-head query/key dimension.
        d_v: Per-head value dimension.
        dropout: Encoder dropout; source decoder and attention dropout remain 0.1.
        epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
    """

    action_size: int = 64
    session_size: int = 16
    session_gap_hours: float = 10.0
    timestamp_unit: Literal["auto", "seconds", "milliseconds"] = "auto"
    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    d_inner: int = field(
        default=2048,
        metadata={"optuna": {"type": "categorical", "choices": [512, 1024, 2048]}},
    )
    n_layers: int = field(
        default=1,
        metadata={"optuna": {"type": "int", "low": 1, "high": 3}},
    )
    n_head: int = field(
        default=4,
        metadata={"optuna": {"type": "categorical", "choices": [2, 4, 8]}},
    )
    d_k: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )
    d_v: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )
    dropout: float = field(
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    epochs: int = 100
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    # categorical so the default 0.0 stays inside the space
    weight_decay: float = field(
        default=0.0,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]}
        },
    )


@register_trainer("HiTSKT")
class HiTSKTTrainer(BaseTrainer):
    """Train source-equivalent logits with framework next-item extraction."""

    def build_components(self, rc, data_src):
        from model.HiTSKT.HiTSKT_data import HiTSKTModelData
        from model.HiTSKT.HiTSKT_model import HiTSKT

        train_data, val_data, test_data, info = HiTSKTModelData(data_src).prepare_data(
            rc
        )
        m = rc.model
        model = HiTSKT(
            n_problem=info["n_problem"],
            n_skill=info["n_skill"],
            n_qno=info["n_qno"],
            action_size=m.action_size,
            session_size=m.session_size,
            d_model=m.d_model,
            d_inner=m.d_inner,
            n_layers=m.n_layers,
            n_head=m.n_head,
            d_k=m.d_k,
            d_v=m.d_v,
            dropout=m.dropout,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(reduction="sum"),
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def forward_pass(self, batch_data):
        packed, labels, mask = (self._move_tensor_to_device(x) for x in batch_data)
        prediction = self.model(packed)
        y_hat, y_label, _ = self._extract_valid_predictions(
            prediction, labels, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def _compute_eval_loss(self, outputs):
        return torch.nn.functional.binary_cross_entropy(
            outputs["y_hat"], outputs["y_label"]
        )
