"""Mamba4KTQ trainer: question-level Mamba4KT for the skill-vs-question ablation."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("Mamba4KTQ")
class Mamba4KTQConfig(ModelConfig):
    """Mamba4KTQ configuration.

    Same architecture as Mamba4KT, but the question id is the concept embedding
    unit (``num_c=num_questions``) and Rasch pid is disabled. Hyperparameters
    mirror Mamba4KT so the ablation varies only the modeling granularity.

    Args:
        d_model: Hidden dimension of the model.
        n_blocks: Number of Mamba blocks.
        d_state: SSM latent state dimension.
        d_conv: Conv1D kernel width in Mamba block.
        expand: Mamba internal expansion factor.
        dropout: Dropout probability.
        separate_qa: 1 uses a joint (question, response) interaction embedding table;
            0 (default) uses additive qa_embed(response) + q_embed(question).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
    """

    d_model: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    n_blocks: int = field(
        default=5,
        metadata={"optuna": {"type": "int", "low": 2, "high": 6}},
    )
    d_state: int = field(
        default=16,
        metadata={"optuna": {"type": "int", "low": 8, "high": 32}},
    )
    d_conv: int = field(
        default=4,
        metadata={"optuna": {"type": "categorical", "choices": [2, 4]}},
    )
    expand: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [2, 4]}},
    )
    dropout: float = field(
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    separate_qa: int = 0
    epochs: int = 150
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    # linear range: log sampling requires low > 0, default is 0.0
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 1e-3}},
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("Mamba4KTQ")
class Mamba4KTQTrainer(BaseTrainer):
    """Question-level Mamba4KT trainer (ablation variant).

    Reuses the Mamba4KT model with question-level concept embeddings and no
    Rasch pid. Test follows the standard K-fold split instead of windowlate.
    Output uses next-item alignment, matching Mamba4KT.
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.Mamba4KT.Mamba4KT_model import Mamba4KT
        from model.Mamba4KT.Mamba4KTQ_data import Mamba4KTQModelData

        model_data = Mamba4KTQModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        num_questions = data_src.get_metadata("num_questions")
        logger.info(
            f"Initializing Mamba4KTQ (question-level) with {num_questions} questions"
        )

        m = rc.model
        model = Mamba4KT(
            num_c=num_questions,
            n_pid=0,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            d_state=m.d_state,
            d_conv=m.d_conv,
            expand=m.expand,
            dropout=m.dropout,
            separate_qa=bool(m.separate_qa),
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """Question-level forward pass (next-item convention).

        ``sequence`` is the question id and no pid is used. ``out[t]`` predicts
        ``response[t+1]``, matching Mamba4KT's next-item alignment.
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full, _ = self.model(sequence, response, mask, None)

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=False
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
