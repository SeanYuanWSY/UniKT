"""CIKT 模型训练器模块。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("CIKT")
class CIKTConfig(ModelConfig):
    """CIKT 模型配置。

    Args:
        d_model: Hidden dimension (d_model).
        dropout: GCN dropout probability (matches the reference implementation).
        num_difficulty_levels: Number of question-difficulty bins (trivial head classes).
        loss_w_causal: Loss weight for the causal branch (lambda_1 in paper).
        loss_w_intervention: Loss weight for the response-invert intervention branch (lambda_3 in paper).
        loss_w_trivial: Loss weight for the trivial (difficulty) branch (lambda_4 in paper).
        loss_w_replace: Loss weight for the question-replace intervention branch (lambda_2 in paper).
        epochs: Number of training epochs.
        learning_rate: Learning rate for Adam.
        lr_decay: Exponential LR decay per epoch (None disables).
        weight_decay: Weight decay (L2).
        batch_size: Batch size.
    """

    d_model: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )
    dropout: float = field(
        default=0.5,
        metadata={"optuna": {"type": "float", "low": 0.2, "high": 0.5}},
    )
    num_difficulty_levels: int = 10
    loss_w_causal: float = 0.1
    loss_w_intervention: float = 0.2
    loss_w_trivial: float = 0.6
    loss_w_replace: float = 0.3
    epochs: int = 100
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=1e-5,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [8, 16, 32, 64]}},
    )


@register_trainer("CIKT")
class CIKTTrainer(BaseTrainer):
    """CIKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.CIKT.CIKT_data import CIKTModelData
        from model.CIKT.CIKT_model import CIKT

        model_data = CIKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            difficulty_table,
            collate_fn,
            eval_collate_fn,
        ) = model_data.prepare_data(rc)

        metadata = data_src.get_metadata()
        num_questions = metadata["num_questions"]
        num_concepts = metadata["num_skills"]
        m = rc.model
        logger.info(
            f"Initializing CIKT model (d_model={m.d_model}, "
            f"seq_len={metadata['max_question_seq_len']}, num_questions={num_questions}, "
            f"num_concepts={num_concepts})..."
        )

        model = CIKT(
            num_questions=num_questions,
            num_concepts=num_concepts,
            d_model=m.d_model,
            seq_len=metadata["max_question_seq_len"],
            dropout=m.dropout,
            num_difficulty_levels=m.num_difficulty_levels,
            difficulty_table=difficulty_table,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        self.w_causal = m.loss_w_causal
        self.w_intervention = m.loss_w_intervention
        self.w_trivial = m.loss_w_trivial
        self.w_replace = m.loss_w_replace
        self._ce_loss = torch.nn.CrossEntropyLoss()

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=collate_fn,
            val_collate_fn=eval_collate_fn,
            test_collate_fn=eval_collate_fn,
        )

    def forward_pass(self, batch_data):
        """训练 / 验证前向传播。

        batch_data: ``(Q, Y, mask, C, QR)``，模型输出 ``[B, L-1]``。
        """
        q, y, mask, c, qr = batch_data
        q = self._move_tensor_to_device(q)
        y = self._move_tensor_to_device(y)
        mask = self._move_tensor_to_device(mask)
        c = self._move_tensor_to_device(c)
        qr = self._move_tensor_to_device(qr)

        out = self.model(q, y, c, qr, mask)

        # Pad to [B, L]; the framework derives valid_mask as the overlap mask[:-1] & mask[1:]
        y_pred_full = self._pad_to_full_sequence(out["y_pred"])
        y_hat, y_label, valid_mask = self._extract_valid_predictions(
            y_pred_full, y, mask
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        outputs = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
        }
        # Auxiliary branch outputs are only consumed by the training loss.
        if self.model.training:
            a_true_full = self.model.difficulty_table[q][:, 1:]  # [B, L-1]
            outputs.update(
                {
                    "_aux_causal": out["y_causal"][valid_mask],
                    "_aux_intervention": out["y_intervention"][valid_mask],
                    "_aux_replace": out["y_replace"][valid_mask],
                    "_aux_trivial": out["y_trivial"][valid_mask],
                    "_aux_trivial_label": a_true_full[valid_mask],
                }
            )

        return outputs

    def _compute_loss(self, outputs):
        """多任务损失"""
        y_label = outputs["y_label"]
        bce = self.loss
        loss_causal = bce(outputs["_aux_causal"], y_label)
        loss_intervention = bce(outputs["_aux_intervention"], y_label)
        loss_replace = bce(outputs["_aux_replace"], y_label)
        loss_trivial = self._ce_loss(
            outputs["_aux_trivial"], outputs["_aux_trivial_label"]
        )
        return (
            self.w_causal * loss_causal
            + self.w_intervention * loss_intervention
            + self.w_trivial * loss_trivial
            + self.w_replace * loss_replace
        )
