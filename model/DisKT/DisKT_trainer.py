"""DisKT trainer."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["DisKTTrainer"]


@register_model_config("DisKT")
class DisKTConfig(ModelConfig):
    """DisKT model configuration.

    Args:
        embedding_size: Hidden dimension.
        num_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: Feed-forward inner dimension inside each block.
        final_fc_dim: Width of the first output MLP layer.
        final_fc_dim2: Width of the second output MLP layer.
        kq_same: If 1, key and query share the linear projection.
        separate_qr: Whether to embed question-response pairs separately.
        l2: Weight on the Rasch difficulty regulariser.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        max_grad_norm: Max gradient norm for clipping, 0 to disable.
        batch_size: Batch size.
    """

    # powers of two so d_model % num_attn_heads == 0 (assert in DisKT_model)
    embedding_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    num_blocks: int = field(
        default=2,
        metadata={"optuna": {"type": "int", "low": 1, "high": 4}},
    )
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    dropout: float = field(
        default=0.05,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.3}},
    )
    d_ff: int = 1024
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    kq_same: int = 1
    separate_qr: bool = False
    l2: float = 1e-5
    epochs: int = 200
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}},
    )
    # linear range: log sampling requires low > 0, default is 0.0
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 1e-2}},
    )
    max_grad_norm: float = 0.0
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128, 256]}},
    )


@register_trainer("DisKT")
class DisKTTrainer(BaseTrainer):
    """DisKT trainer."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DisKT.DisKT_data import DisKTModelData
        from model.DisKT.DisKT_model import DisKT

        model_data = DisKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_skills,
            num_questions,
        ) = model_data.prepare_data(rc)
        max_seq_len = data_src.get_metadata("max_question_seq_len")

        m = rc.model
        logger.info("Initializing DisKT model...")
        model = DisKT(
            num_skills=num_skills,
            num_questions=num_questions,
            seq_len=max_seq_len,
            embedding_size=m.embedding_size,
            num_blocks=m.num_blocks,
            dropout=m.dropout,
            kq_same=m.kq_same,
            d_ff=m.d_ff,
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            num_attn_heads=m.num_attn_heads,
            separate_qr=m.separate_qr,
            l2=m.l2,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        logger.info(
            f"DisKT Trainer: {num_skills} concepts (incl. padding), "
            f"{num_questions} questions"
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            max_clip_grad_norm=m.max_grad_norm or None,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: dict) -> dict:
        questions = self._move_tensor_to_device(batch_data["questions"])
        skills = self._move_tensor_to_device(batch_data["skills"])
        responses = self._move_tensor_to_device(batch_data["responses"])
        masks = self._move_tensor_to_device(batch_data["masks"])
        counter_masks = self._move_tensor_to_device(batch_data["counter_masks"])

        preds, reg_loss = self.model(questions, skills, responses, masks, counter_masks)

        y_hat, y_label, _ = self._extract_valid_predictions(
            preds, responses, masks, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "reg_loss": reg_loss,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        bce_loss = self.loss(outputs["y_hat"], outputs["y_label"])
        return bce_loss + outputs["reg_loss"]
