"""SAINT trainer."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("SAINT")
class SAINTConfig(ModelConfig):
    """SAINT model configuration.

    Args:
        emb_size: Embedding dimension.
        num_attn_heads: Number of attention heads (encoder and decoder share it).
        n_blocks: Number of encoder and decoder blocks (one value for both).
        dropout: Dropout probability.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
    """

    emb_size: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [64, 256]}},
    )
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8]}},
    )
    n_blocks: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.05, 0.1, 0.2, 0.3, 0.5]}
        },
    )
    epochs: int = 200
    learning_rate: float = field(
        default=1e-3,
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


@register_trainer("SAINT")
class SAINTTrainer(BaseTrainer):
    """SAINT 模型训练器

    Args:
        rc: RunConfig 实例
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.SAINT.SAINT_data import SAINTModelData

        model_data = SAINTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.SAINT.SAINT_model import SAINT

        logger.info("Initializing SAINT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = SAINT(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            seq_len=metadata["max_skill_seq_len"],
            emb_size=m.emb_size,
            num_attn_heads=m.num_attn_heads,
            dropout=m.dropout,
            n_blocks=m.n_blocks,
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
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SAINT 前向传播

        Args:
            batch_data: 包含 (sequence, response, mask, question) 的元组，
                其中 sequence 为技能序列、question 为题目序列

        Returns:
            包含 y_hat, y_label, y_predict 的字典；模型输出形状 [B, S]，
            p[:, t] 由历史 0..t-1 预测位置 t（same-position 约定）
        """
        sequence, response, mask, question = batch_data
        question = self._move_tensor_to_device(question)
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(question, sequence, response)  # [B, S]

        # same-position output: p[:, t] predicts response[t] from history 0..t-1
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

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估

        数据格式：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]
        - mask: [0, ..., 0, 1]
        - late_group_id: [g1, ..., gN]
        - true_labels: [历史标签, 真实标签]
        - question: [题目历史, 目标题目]

        SAINT 预测语义：
        - p[:, t] 使用历史 0..t-1 预测位置 t
        - 所有位置均有有效预测
        """
        sequence, response, mask, late_group_id, true_labels, question, _ = batch_data

        question = self._move_tensor_to_device(question)
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full = self.model(question, sequence, response)  # [B, S]

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
