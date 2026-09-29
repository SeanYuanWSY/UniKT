"""RobustKT trainer."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("RobustKT")
class RobustKTConfig(ModelConfig):
    """RobustKT model configuration.

    Args:
        d_model: Hidden dimension of the model.
        n_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        d_ff: Feed-forward network dimension.
        final_fc_dim: Final fully connected layer dimension.
        kernel_size: Causal smoothing kernel size.
        dropout: Dropout probability.
        kq_same: Whether key and query use the same linear transformation.
        separate_qa: Whether to use separate QA embeddings.
        use_rasch: Whether to enable the Rasch problem-id difficulty model (True=yes).
        l2: Rasch difficulty regularization coefficient.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
        test_batch_size: Batch size for windowlate test evaluation.
    """

    # powers of two so d_model % num_attn_heads == 0 for every combination
    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256, 512]}},
    )
    n_blocks: int = field(
        default=4,
        metadata={"optuna": {"type": "int", "low": 2, "high": 6}},
    )
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    d_ff: int = 512
    final_fc_dim: int = 512
    kernel_size: int = 5
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    kq_same: int = 1
    separate_qa: int = 0
    use_rasch: bool = True
    l2: float = 1e-5
    epochs: int = 150
    learning_rate: float = field(
        default=1e-4,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    lr_decay: float | None = None
    # categorical so the default 0.0 stays inside the space
    weight_decay: float = field(
        default=0.0,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]}
        },
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )
    test_batch_size: int = 512


@register_trainer("RobustKT")
class RobustKTTrainer(BaseTrainer):
    """Trainer for RobustKT."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.RobustKT.RobustKT_data import RobustKTModelData

        model_data = RobustKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.RobustKT.RobustKT_model import RobustKT

        metadata = data_src.get_metadata()
        logger.info("Initializing RobustKT model...")
        num_questions = metadata.get("num_questions", 0) if rc.model.use_rasch else 0
        if num_questions > 0:
            logger.info(
                f"RobustKT: Using Problem ID (Rasch model) with {num_questions} questions"
            )
        else:
            logger.info("RobustKT: Using skill-only model (Rasch disabled)")

        m = rc.model
        model = RobustKT(
            num_skills=metadata["num_skills"],
            num_questions=num_questions,
            dropout=m.dropout,
            kq_same=m.kq_same,
            l2=m.l2,
            separate_qa=m.separate_qa,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            num_attn_heads=m.num_attn_heads,
            d_ff=m.d_ff,
            final_fc_dim=m.final_fc_dim,
            kernel_size=m.kernel_size,
            max_seq_len=metadata["max_skill_seq_len"],
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

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        y_hat_full, c_reg_loss = self.model(
            sequence,
            response,
            mask,
            question=question,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "c_reg_loss": c_reg_loss,
        }

    def test_forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        sequence, response, mask, late_group_id, true_labels, question, _ = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        valid_mask = late_group_id >= 0
        y_hat_full, _ = self.model(sequence, response, valid_mask, question=question)
        y_hat = torch.masked_select(y_hat_full, mask)
        y_label = torch.masked_select(true_labels, mask).float()
        group_ids = torch.masked_select(late_group_id, mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        bce_loss = self.loss(outputs["y_hat"], outputs["y_label"])
        return bce_loss + outputs.get(
            "c_reg_loss",
            torch.tensor(0.0, device=outputs["y_hat"].device),
        )
