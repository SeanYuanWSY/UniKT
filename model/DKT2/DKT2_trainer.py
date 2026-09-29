"""DKT2 模型训练器。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["DKT2Trainer"]


@register_model_config("DKT2")
class DKT2Config(ModelConfig):
    """DKT2 模型配置。

    Args:
        factor: mLSTM/sLSTM feed-forward up-projection factor.
        num_blocks: Number of xLSTM blocks.
        num_heads: Number of attention heads per block.
        slstm_at: Block indices using sLSTM (rest use mLSTM).
        slstm_backend: sLSTM backend: cuda or vanilla (auto-fallback to vanilla).
        conv1d_kernel_size: Conv1d kernel size in xLSTM blocks.
        qkv_proj_blocksize: Block size of the mLSTM qkv projection.
        embedding_size: Embedding / hidden dimension.
        dropout: Dropout probability.
        length: Prediction horizon (next-item when 1).
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        max_grad_norm: Max gradient norm for clipping, 0 to disable.
        batch_size: Batch size.
    """

    factor: float = 1.3
    num_blocks: int = field(
        default=1,
        metadata={"optuna": {"type": "int", "low": 1, "high": 3}},
    )
    # powers of two so embedding_size (64) % num_heads == 0 for every choice
    num_heads: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    slstm_at: list[int] = field(default_factory=lambda: [0])
    slstm_backend: str = "cuda"
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    embedding_size: int = 64
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    length: int = 1
    epochs: int = 300
    learning_rate: float = field(
        default=1e-3,
        metadata={
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
    )
    # categorical so the default 0.0 stays inside the space
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4]}},
    )
    max_grad_norm: float = 2.0
    batch_size: int = field(
        default=512,
        metadata={"optuna": {"type": "categorical", "choices": [256, 512, 1024]}},
    )


@register_trainer("DKT2")
class DKT2Trainer(BaseTrainer):
    """DKT2 训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DKT2.DKT2_data import DKT2ModelData
        from model.DKT2.DKT2_model import DKT2

        model_data = DKT2ModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_skills,
            num_questions,
        ) = model_data.prepare_data(rc)
        max_seq_len = data_src.get_metadata("max_question_seq_len")

        m = rc.model
        logger.info("Initializing DKT2 model...")
        model = DKT2(
            num_skills=num_skills,
            num_questions=num_questions,
            batch_size=m.batch_size,
            seq_len=max_seq_len,
            factor=m.factor,
            num_blocks=m.num_blocks,
            num_heads=m.num_heads,
            slstm_at=m.slstm_at,
            conv1d_kernel_size=m.conv1d_kernel_size,
            qkv_proj_blocksize=m.qkv_proj_blocksize,
            embedding_size=m.embedding_size,
            dropout=m.dropout,
            slstm_backend=m.slstm_backend,
            length=m.length,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        logger.info(
            f"DKT2 Trainer: {num_skills} concepts (incl. padding), "
            f"{num_questions} questions, sLSTM backend = {model.slstm_backend}"
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

    def forward_pass(self, batch_data: dict) -> dict[str, torch.Tensor]:
        """next-item 前向传播。

        模型输出已过 sigmoid 的概率，长度 S-1，output[t] 预测 response[t+1]。
        用 _pad_to_full_sequence 补一列后走基类的 next-item 对齐。
        """
        questions = self._move_tensor_to_device(batch_data["questions"])
        responses = self._move_tensor_to_device(batch_data["responses"])
        masks = self._move_tensor_to_device(batch_data["masks"])
        skills = self._move_tensor_to_device(batch_data["skills"])

        output, _ = self.model(questions, skills, responses, masks)  # [B, S-1] probs

        y_hat_full = self._pad_to_full_sequence(output)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, responses, masks
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
        }
