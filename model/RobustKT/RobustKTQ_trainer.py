"""RobustKTQ trainer: question-level RobustKT for the skill-vs-question ablation."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("RobustKTQ")
class RobustKTQConfig(ModelConfig):
    """RobustKTQ configuration.

    Same architecture as RobustKT, but the question id is the concept embedding
    unit (``num_skills=num_questions``) and Rasch pid is disabled.
    Hyperparameters mirror RobustKT so the ablation varies only the modeling
    granularity.

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
        l2: Rasch regularization coefficient (inert when Rasch is off, but required by the model constructor).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
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


@register_trainer("RobustKTQ")
class RobustKTQTrainer(BaseTrainer):
    """Question-level RobustKT trainer (ablation variant).

    Reuses the RobustKT model with question-level concept embeddings and no
    Rasch pid. Test follows the standard K-fold split instead of windowlate.
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.RobustKT.RobustKT_model import RobustKT
        from model.RobustKT.RobustKTQ_data import RobustKTQModelData

        model_data = RobustKTQModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        num_questions = data_src.get_metadata("num_questions")
        logger.info(
            f"Initializing RobustKTQ (question-level) with {num_questions} questions"
        )

        m = rc.model
        model = RobustKT(
            num_skills=num_questions,
            num_questions=0,
            l2=m.l2,
            kernel_size=m.kernel_size,
            dropout=m.dropout,
            kq_same=m.kq_same,
            separate_qa=m.separate_qa,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            num_attn_heads=m.num_attn_heads,
            d_ff=m.d_ff,
            final_fc_dim=m.final_fc_dim,
            max_seq_len=data_src.get_metadata("max_question_seq_len"),
        )

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
            loss_fn=torch.nn.BCELoss(),
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Question-level forward pass.

        ``sequence`` is the question id and no pid is used (``question=None``).
        same_position=True: ``y_hat[:, t]`` predicts ``response[t]``.
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full, _ = self.model(sequence, response, mask, question=None)

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
        }
