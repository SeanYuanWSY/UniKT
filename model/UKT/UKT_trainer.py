"""UKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("UKT")
class UKTConfig(ModelConfig):
    """UKT model configuration.

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
        use_rasch: Whether to enable the Rasch problem-id difficulty model (True=yes).
        use_CL: Enable contrastive learning (1=yes, 0=no).
        cl_weight: Weight for contrastive learning loss.
        l2: L2 regularization coefficient for Rasch difficulty.
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
    use_rasch: bool = True
    use_CL: int = 1
    cl_weight: float = 0.02
    l2: float = 1e-5
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


@register_trainer("UKT")
class UKTTrainer(BaseTrainer):
    """UKT 模型训练器

    负责初始化UKT模型、优化器和训练数据，并实现前向传播逻辑。
    支持对比学习损失的组合训练。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.UKT.UKT_data import UKTModelData

        model_data = UKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.UKT.UKT_model import UKT

        logger.info("Initializing UKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0) if rc.model.use_rasch else 0
        if n_pid > 0:
            logger.info(f"UKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.info("UKT: Using skill-only model (Rasch disabled)")

        m = rc.model
        model = UKT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
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
            l2=m.l2,
            atten_type=m.atten_type,
            seq_len=metadata["max_skill_seq_len"],
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build UKT Rasch pid data with 0 reserved for padding."""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data):
        """UKT 前向传播

        UKT 使用因果注意力，preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]。
        """
        if len(batch_data) == 5:
            sequence, response, mask, question, response_aug = batch_data
        else:
            sequence, response, mask, question = batch_data
            response_aug = None

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)
        if response_aug is not None:
            response_aug = self._move_tensor_to_device(response_aug)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        preds, cl_loss, _, c_reg_loss = self.model(
            sequence, response, mask, pid_data, response_aug
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            preds, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "cl_loss": cl_loss,
        }
        if use_pid:
            result["c_reg_loss"] = c_reg_loss
        return result

    def test_forward_pass(self, batch_data):
        sequence, response, mask, late_group_id, true_labels, question, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else None

        preds, _, _, _ = self.model(sequence, response, mask, pid_data)

        y_hat = torch.masked_select(preds, mask)
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

    def _compute_loss(self, outputs):
        """计算损失：BCE + 对比学习损失"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if self.model.use_CL and "cl_loss" in outputs:
            bce_loss = bce_loss + self.model.cl_weight * outputs["cl_loss"]

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            bce_loss = bce_loss + outputs["c_reg_loss"]

        return bce_loss
