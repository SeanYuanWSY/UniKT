"""UKTQ trainer: question-level UKT for the skill-vs-question ablation."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("UKTQ")
class UKTQConfig(ModelConfig):
    """UKTQ configuration.

    Same architecture as UKT, but the question id is the concept embedding
    unit (``num_c=num_questions``) and Rasch pid is disabled. Hyperparameters
    mirror UKT so the ablation varies only the modeling granularity.

    Args:
        d_model: Hidden dimension of the model.
        n_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: Feed-forward network dimension.
        final_fc_dim: First fully connected layer dimension.
        final_fc_dim2: Second fully connected layer dimension.
        kq_same: Whether key and query use the same linear transformation (1=yes, 0=no).
        separate_qa: Whether to use separate QA embeddings (1=yes, 0=no).
        use_CL: Enable contrastive learning (1=yes, 0=no).
        cl_weight: Weight for contrastive learning loss.
        no_uncertainty_aug: Disable uncertainty augmentation for contrastive learning.
        atten_type: Attention type: w2 (Wasserstein) or dp (dot product).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
    """

    # powers of two so d_model % num_attn_heads == 0 for every combination
    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    n_blocks: int = 4
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    d_ff: int = 512
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    kq_same: int = 1
    separate_qa: int = 0
    use_CL: int = 1
    cl_weight: float = 0.02
    no_uncertainty_aug: bool = False
    atten_type: str = "w2"
    epochs: int = 200
    learning_rate: float = field(
        default=1e-4,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    weight_decay: float = field(
        default=1e-5,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("UKTQ")
class UKTQTrainer(BaseTrainer):
    """Question-level UKT trainer (ablation variant).

    Reuses the UKT model with question-level concept embeddings and no Rasch
    pid. The granularity-independent contrastive response augmentation is
    preserved. Test follows the standard K-fold split instead of windowlate.
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.UKT.UKT_model import UKT
        from model.UKT.UKTQ_data import UKTQModelData

        model_data = UKTQModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        num_questions = data_src.get_metadata("num_questions")
        logger.info(
            f"Initializing UKTQ (question-level) with {num_questions} questions"
        )

        m = rc.model
        model = UKT(
            num_c=num_questions,
            n_pid=0,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            num_attn_heads=m.num_attn_heads,
            kq_same=m.kq_same,
            separate_qa=bool(m.separate_qa),
            use_CL=bool(m.use_CL),
            cl_weight=m.cl_weight,
            use_uncertainty_aug=not m.no_uncertainty_aug,
            atten_type=m.atten_type,
            seq_len=data_src.get_metadata("max_question_seq_len"),
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
        """Question-level forward pass.

        ``sequence`` is the question id; no pid is used; ``response_aug`` is
        preserved for contrastive learning. same_position=True: ``y_hat[:, t]``
        predicts ``response[t]``.
        """
        sequence, response, mask, response_aug = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        response_aug = self._move_tensor_to_device(response_aug)

        preds, cl_loss, _, _ = self.model(sequence, response, mask, None, response_aug)

        y_hat, y_label, _ = self._extract_valid_predictions(
            preds, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "cl_loss": cl_loss,
        }

    def _compute_loss(self, outputs):
        """BCE + contrastive learning loss (no Rasch regularization)."""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if self.model.use_CL and "cl_loss" in outputs:
            bce_loss = bce_loss + self.model.cl_weight * outputs["cl_loss"]

        return bce_loss
