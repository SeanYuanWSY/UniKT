"""ATDKT 模型训练器模块"""

from dataclasses import field
from typing import Literal

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("ATDKT")
class ATDKTConfig(ModelConfig):
    """ATDKT model configuration.

    Args:
        embedding_dim: Embedding size (also the LSTM hidden size).
        num_qt_layers: Number of layers of the QT relation network.
        num_attn_heads: Number of attention heads of the QT transformer.
        dropout: Dropout probability.
        use_qt: Whether to enable the question tagging auxiliary task.
        use_ik: Whether to enable the individualized prior knowledge
            auxiliary task.
        qt_encoder: QT relation network type, "transformer" or "lstm".
        qt_with_interaction: Whether the QT encoder input also includes the
            interaction embedding (False matches the paper's default variant
            where the input is question + concept embeddings only).
        ik_start: Start position of the IK loss (early positions have
            unstable history correctness estimates).
        qt_weight: Weight of the QT auxiliary loss.
        ik_weight: Weight of the IK auxiliary loss.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    # constrained so embedding_dim % num_attn_heads == 0 for every combination
    embedding_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [64, 256]}},
    )
    num_qt_layers: int = field(
        default=1,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    num_attn_heads: int = field(
        default=4,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8]}},
    )
    dropout: float = field(
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    use_qt: bool = True
    use_ik: bool = True
    qt_encoder: Literal["transformer", "lstm"] = "transformer"
    qt_with_interaction: bool = False
    ik_start: int = field(
        default=50,
        metadata={
            "optuna": {
                "type": "categorical",
                "choices": [0, 10, 30, 50, 70, 100, 120, 150],
            }
        },
    )
    qt_weight: float = field(
        default=0.5,
        metadata={
            "optuna": {
                "type": "categorical",
                "choices": [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
            }
        },
    )
    ik_weight: float = field(
        default=0.5,
        metadata={
            "optuna": {
                "type": "categorical",
                "choices": [0.01, 0.1, 0.3, 0.5, 0.7, 1.0],
            }
        },
    )
    epochs: int = 200
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "categorical", "choices": [1e-3, 1e-4, 1e-5]}},
    )
    weight_decay: float = 0.0
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("ATDKT")
class ATDKTTrainer(BaseTrainer):
    """ATDKT 模型训练器

    负责初始化 ATDKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig
        data_src: 数据源实例
        exp_manager: 实验管理器
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.ATDKT.ATDKT_data import ATDKTModelData
        from model.ATDKT.ATDKT_model import ATDKT

        train_dataset, val_dataset, test_dataset = ATDKTModelData(
            data_src
        ).prepare_data(rc)

        metadata = data_src.get_metadata()
        m = rc.model
        if (
            m.use_qt
            and m.qt_encoder == "transformer"
            and m.embedding_dim % m.num_attn_heads != 0
        ):
            raise ValueError(
                f"embedding_dim ({m.embedding_dim}) must be divisible by "
                f"num_attn_heads ({m.num_attn_heads})"
            )
        if m.use_ik and m.ik_start >= metadata["max_skill_seq_len"]:
            raise ValueError(
                f"ik_start ({m.ik_start}) must be smaller than max_seq_len "
                f"({metadata['max_skill_seq_len']}), otherwise the IK auxiliary task "
                "covers no positions"
            )
        logger.info("Initializing ATDKT model...")
        model = ATDKT(
            num_q=metadata["num_questions"],
            num_c=metadata["num_skills"],
            emb_size=m.embedding_dim,
            seq_len=metadata["max_skill_seq_len"],
            dropout=m.dropout,
            use_qt=m.use_qt,
            use_ik=m.use_ik,
            qt_encoder=m.qt_encoder,
            qt_num_layers=m.num_qt_layers,
            num_attn_heads=m.num_attn_heads,
            qt_with_interaction=m.qt_with_interaction,
            ik_start=m.ik_start,
        )
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=None,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def _build_history_corr(
        self, response: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """截止当前（含当前交互）的全局历史正确率，IK 任务目标"""
        valid = mask.float()
        right = torch.cumsum(response.float() * valid, dim=1)
        total = torch.cumsum(valid, dim=1)
        return right / total.clamp(min=1.0)

    def forward_pass(self, batch_data) -> dict[str, torch.Tensor]:
        """ATDKT 前向传播

        预测语义（same-position）：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 训练态额外计算 QT/IK 辅助损失
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        question = self._move_tensor_to_device(question)

        # IK target is only consumed in training mode; skip the cumsums on val
        history_corr = (
            self._build_history_corr(response, mask)
            if self.model.training and self.run_config.model.use_ik
            else None
        )
        out = self.model(sequence, response, mask, question, history_corr)

        result: dict[str, torch.Tensor] = {}
        if out["qt_loss"] is not None:
            result["qt_loss"] = out["qt_loss"]
        if out["ik_loss"] is not None:
            result["ik_loss"] = out["ik_loss"]

        y_hat, y_label, _ = self._extract_valid_predictions(
            out["preds"], response, mask, same_position=True
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result.update(
            {
                "y_hat": y_hat,
                "y_label": y_label,
                "y_predict": y_predict,
                "y_score": y_hat,
                "y_prob": y_hat,
            }
        )
        return result

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估"""
        sequence, response, mask, late_group_id, true_labels, question, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        y_hat_full = self.model(sequence, response, mask, question)["preds"]

        y_hat_aligned = y_hat_full[:, 1:]
        true_labels_aligned = true_labels[:, 1:]
        mask_aligned = mask[:, 1:]
        group_id_aligned = late_group_id[:, 1:]

        y_hat = torch.masked_select(y_hat_aligned, mask_aligned)
        y_label = torch.masked_select(true_labels_aligned, mask_aligned).float()
        group_ids = torch.masked_select(group_id_aligned, mask_aligned)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """BCE + QT/IK 辅助项"""
        m = self.run_config.model
        loss = self.loss(outputs["y_hat"], outputs["y_label"])
        if "qt_loss" in outputs:
            loss = loss + m.qt_weight * outputs["qt_loss"]
        if "ik_loss" in outputs:
            loss = loss + m.ik_weight * outputs["ik_loss"]
        return loss
