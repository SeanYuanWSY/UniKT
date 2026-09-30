"""MTKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("MTKT")
class MTKTConfig(ModelConfig):
    """MTKT model configuration.

    Args:
        d_model: Hidden dimension of the model.
        n_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: CIC hidden dimension.
        final_fc_dim: Output FC layer dimension 1.
        final_fc_dim2: Output FC layer dimension 2.
        kq_same: Whether key and query use the same linear transformation (1=yes, 0=no).
        separate_qa: Whether to use separate QA embeddings (1=yes, 0=no).
        l2: L2 regularization coefficient for Rasch model.
        k1: CIC convolution kernel size 1.
        k2: CIC convolution kernel size 2.
        num_rgap: Number of review gap buckets.
        num_sgap: Number of sequential gap buckets.
        num_pcount: Number of practice count buckets.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
    """

    # powers of two so d_model % num_attn_heads == 0 for every combination
    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    n_blocks: int = field(
        default=2,
        metadata={"optuna": {"type": "int", "low": 1, "high": 4}},
    )
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    d_ff: int = 256
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    kq_same: int = 1
    separate_qa: int = 0
    l2: float = 1e-5
    k1: int = 1
    k2: int = 3
    num_rgap: int = 100
    num_sgap: int = 100
    num_pcount: int = 15
    epochs: int = 150
    learning_rate: float = field(
        default=1e-4,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    lr_decay: float | None = None
    # linear range: log sampling requires low > 0, default is 0.0
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 1e-3}},
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("MTKT")
class MTKTTrainer(BaseTrainer):
    """MTKT 模型训练器

    负责初始化 MTKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.MTKT.MTKT_data import MTKTModelData
        from model.MTKT.MTKT_model import MTKT

        model_data = MTKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        logger.info("Initializing MTKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(f"MTKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.info("MTKT: Problem ID not available, using skill-only model")

        m = rc.model
        model = MTKT(
            num_skills=metadata["num_skills"],
            n_pid=n_pid,
            num_rgap=m.num_rgap,
            num_sgap=m.num_sgap,
            num_pcount=m.num_pcount,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            kq_same=m.kq_same,
            separate_qa=bool(m.separate_qa),
            l2=m.l2,
            k1=m.k1,
            k2=m.k2,
            num_attn_heads=m.num_attn_heads,
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            seq_len=max(
                metadata["max_skill_seq_len"],
                metadata.get("max_windowlate_seq_len", 0),
            ),
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=m.lr_decay,
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

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """构建 Rasch pid 数据, 0 保留给 padding"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """MTKT 前向传播

        MTKT 预测语义:
        - preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]
        - 同位置输出，trainer 用 same_position=True 由内置函数归一化为 next-item

        Args:
            batch_data: (sequence, response, mask, question, rgap, sgap, pcount)

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, question, rgap, sgap, pcount = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)
        rgap = self._move_tensor_to_device(rgap)
        sgap = self._move_tensor_to_device(sgap)
        pcount = self._move_tensor_to_device(pcount)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else question

        y_hat_full, c_reg_loss = self.model(
            sequence,
            response,
            pid_data,
            rgap,
            sgap,
            pcount,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response,
            mask,
            same_position=True,
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
        if use_pid:
            result["c_reg_loss"] = c_reg_loss
        return result

    def test_forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """测试前向传播, 支持 windowlateauc_mean 评估

        Windowlate 数据为 9-元组:
            (sequence, response, mask, late_group_id, true_labels, question,
             rgap, sgap, pcount)
        """
        (
            sequence,
            response,
            mask,
            late_group_id,
            true_labels,
            question,
            rgap,
            sgap,
            pcount,
        ) = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)
        rgap = self._move_tensor_to_device(rgap)
        sgap = self._move_tensor_to_device(sgap)
        pcount = self._move_tensor_to_device(pcount)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else question

        y_hat_full, _ = self.model(
            sequence,
            response,
            pid_data,
            rgap,
            sgap,
            pcount,
        )

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
        """计算损失, 包含 BCE 损失和 Rasch 正则化损失"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]

        return bce_loss
