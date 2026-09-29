"""SimpleKT 模型训练器模块"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("SimpleKT")
class SimpleKTConfig(ModelConfig):
    """SimpleKT 模型配置

    Args:
        d_model: Dimension of the model.
        n_blocks: Number of transformer blocks.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: Dimension of feed-forward network.
        kq_same: Whether to share key and query weights (1 for yes, 0 for no).
        separate_qa: Whether to use separate interaction embedding (1 for yes, 0 for no).
        use_rasch: Whether to enable the Rasch problem-id difficulty model (True=yes).
        final_fc_dim: First fully connected layer dimension in output.
        final_fc_dim2: Second fully connected layer dimension in output.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    # powers of two so d_model % n_heads == 0 for every combination
    d_model: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    n_blocks: int = field(
        default=2,
        metadata={"optuna": {"type": "int", "low": 1, "high": 4}},
    )
    n_heads: int = field(
        default=4,
        metadata={"optuna": {"type": "categorical", "choices": [2, 4, 8, 16]}},
    )
    dropout: float = field(
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    d_ff: int = 256
    kq_same: int = 1
    separate_qa: int = 0
    use_rasch: bool = True
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    epochs: int = 100
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
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("SimpleKT")
class SimpleKTTrainer(BaseTrainer):
    """SimpleKT 模型训练器

    负责初始化 SimpleKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.SimpleKT.SimpleKT_data import SimpleKTModelData

        model_data = SimpleKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.SimpleKT.SimpleKT_model import SimpleKT

        logger.info("Initializing SimpleKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata["num_questions"] if rc.model.use_rasch else 0
        if n_pid > 0:
            logger.info(
                f"SimpleKT: Using Problem ID (Rasch model) with {n_pid} questions"
            )
        else:
            logger.info("SimpleKT: Using skill-only model (Rasch disabled)")

        m = rc.model
        model = SimpleKT(
            num_skills=metadata["num_skills"],
            n_pid=n_pid,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            n_heads=m.n_heads,
            seq_len=metadata["max_skill_seq_len"],
            kq_same=m.kq_same,
            separate_qa=bool(m.separate_qa),
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
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

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SimpleKT 前向传播

        SimpleKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 第一个位置：y_hat[:, 0] 基于 sequence[0:1] 和空历史预测 response[0]
        - y_hat[:, t] 直接对应 response[t]，需要对齐

        Args:
            batch_data: 包含 (sequence, response, mask, question) 的元组（question 必选，用于Rasch pid）

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        y_hat_full = self.model(sequence, response, mask, pid_data)

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

        数据格式说明：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]  # 目标位置 response=0 用于避免数据泄露
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id
        - true_labels: [历史标签, 真实标签]  # 用于评估
        - question: [题目历史, 目标题目]  # 用于Rasch pid

        SimpleKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 模型内部已经使用移位后的 response 作为 target，所以 y_hat[:, t] 直接对应 response[t]
        - 测试时 response[:, -1] = 0（占位），模型会忽略（使用移位后的目标）
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

        y_hat_full = self.model(sequence, response, mask, pid_data)

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
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)
