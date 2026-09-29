"""KeenKT 模型训练器模块"""

from dataclasses import field
from typing import Literal

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("KeenKT")
class KeenKTConfig(ModelConfig):
    """KeenKT model configuration.

    Args:
        d_model: Dimension of the model.
        d_ff: Dimension of feed-forward network.
        n_blocks: Number of transformer blocks.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        final_fc_dim: First fully connected layer dimension in output.
        final_fc_dim2: Second fully connected layer dimension in output.
        se_ratio: Reduction ratio of the squeeze-excitation gate.
        emb_type: Embedding mode, "stoc_qid" uses both mean/cov streams,
            "qid" keeps only the mean stream.
        use_rasch: Whether to enable the Rasch problem-id difficulty model.
        use_cl: Whether to enable the NIG contrastive loss.
        cl_weight: Weight of the contrastive loss.
        use_uncertainty_aug: Whether to build the polarity-flip augmented
            view for contrastive learning.
        use_diffusion: Whether to enable the denoising auxiliary loss.
        diffusion_weight: Weight of the denoising loss.
        noise_level: Std of the noise injected for denoising.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization) for optimizer.
    """

    # powers of two so d_model % n_heads == 0 for every combination
    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    d_ff: int = field(
        default=512,
        metadata={"optuna": {"type": "categorical", "choices": [256, 512]}},
    )
    n_blocks: int = field(
        default=4,
        metadata={"optuna": {"type": "int", "low": 1, "high": 4}},
    )
    n_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [2, 4, 8, 16]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    final_fc_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    final_fc_dim2: int = 256
    se_ratio: int = field(
        default=16,
        metadata={"optuna": {"type": "categorical", "choices": [8, 16]}},
    )
    emb_type: Literal["stoc_qid", "qid"] = "stoc_qid"
    use_rasch: bool = True
    use_cl: bool = True
    cl_weight: float = field(
        default=0.0004,
        metadata={
            "optuna": {"type": "float", "low": 2.5e-5, "high": 0.01, "log": True}
        },
    )
    use_uncertainty_aug: bool = True
    use_diffusion: bool = True
    diffusion_weight: float = field(
        default=0.0016,
        metadata={"optuna": {"type": "float", "low": 2e-4, "high": 4e-3, "log": True}},
    )
    noise_level: float = field(
        default=0.3,
        metadata={"optuna": {"type": "float", "low": 0.1, "high": 0.5}},
    )
    epochs: int = 200
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128]}},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    # categorical so the default 1e-5 stays inside the space
    weight_decay: float = field(
        default=1e-5,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]}
        },
    )


@register_trainer("KeenKT")
class KeenKTTrainer(BaseTrainer):
    """KeenKT 模型训练器

    负责初始化 KeenKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig
        data_src: 数据源实例
        exp_manager: 实验管理器
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.KeenKT.KeenKT_data import KeenKTModelData

        model_data = KeenKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.KeenKT.KeenKT_model import KeenKT

        logger.info("Initializing KeenKT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        n_pid = metadata["num_questions"] if m.use_rasch else 0
        if n_pid > 0:
            logger.info(
                f"KeenKT: Using Problem ID (Rasch model) with {n_pid} questions"
            )
        else:
            logger.info("KeenKT: Using skill-only model (Rasch disabled)")

        model = KeenKT(
            num_skills=metadata["num_skills"],
            n_pid=n_pid,
            d_model=m.d_model,
            d_ff=m.d_ff,
            n_blocks=m.n_blocks,
            n_heads=m.n_heads,
            dropout=m.dropout,
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            se_ratio=m.se_ratio,
            # data arrays are always built at the preprocessed width
            seq_len=metadata["max_skill_seq_len"],
            emb_type=m.emb_type,
            use_cl=m.use_cl,
            use_diffusion=m.use_diffusion,
            noise_level=m.noise_level,
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

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data) -> dict[str, torch.Tensor]:
        """KeenKT 前向传播

        预测语义（same-position）：
        - y_hat[:, t] 基于 q[0:t+1] 和 response[0:t] 预测 response[t]
        - 训练态额外计算对比/去噪辅助损失

        batch 为 4 元组（sequence, response, mask, question）或训练数据的
        5 元组（多一个 target_aug）；按元组长度分支，保证 eval 模式消费
        训练 batch（如 efficiency benchmark）也能正常前向。
        """
        if len(batch_data) == 5:
            sequence, response, mask, question, target_aug = batch_data
            target_aug = self._move_tensor_to_device(target_aug)
        else:
            sequence, response, mask, question = batch_data
            target_aug = None
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None
        out = self.model(sequence, response, pid_data, mask=mask, target_aug=target_aug)

        result: dict[str, torch.Tensor] = {}
        if self.model.training:
            result["cl_loss"] = out["cl_loss"]
            result["diffusion_loss"] = out["diffusion_loss"]

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
        """测试前向传播，支持 windowlateauc_mean 评估

        数据格式说明：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]  # 目标位置 response=0 用于避免数据泄露
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id
        - true_labels: [历史标签, 真实标签]  # 用于评估
        - question: [题目历史, 目标题目]  # 用于Rasch pid
        """
        sequence, response, mask, late_group_id, true_labels, question, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        valid_mask = late_group_id >= 0
        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else None

        y_hat_full = self.model(sequence, response, pid_data)["preds"]

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
        """BCE + 对比/去噪辅助项"""
        m = self.run_config.model
        loss = self.loss(outputs["y_hat"], outputs["y_label"])
        if "cl_loss" in outputs:
            loss = loss + m.cl_weight * outputs["cl_loss"]
        if "diffusion_loss" in outputs:
            loss = loss + m.diffusion_weight * outputs["diffusion_loss"]
        return loss
