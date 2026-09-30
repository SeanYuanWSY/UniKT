"""Mamba4KT 模型数据处理模块

数据格式与 AKT 一致（基于技能序列 + 题目 ID，用于 Rasch 模型）：
    训练/验证模式: (sequence, response, mask, question)
    窗口测试模式:  (sequence, response, mask, late_group_id, true_labels, question)
"""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class Mamba4KTDataset(Dataset):
    """Mamba4KT 数据集。"""

    def __init__(self, sequences, responses, masks, questions):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        question = torch.tensor(self.questions[idx], dtype=torch.long)
        return sequence, response, mask, question


class Mamba4KTModelData(SkillModelData):
    """Mamba4KT 模型数据加载器，复用技能序列构建逻辑。"""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        user_sequence, user_response, user_mask, _, user_question = (
            self.build_sequence_data()
        )

        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            train_data, val_data, _ = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
            train_question, val_question, _ = self.split_kfold_data(
                user_question, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        window_test_data = self.create_windowlate_iterable_dataset()

        train_dataset = Mamba4KTDataset(
            train_data[0], train_data[1], train_data[2], train_question[0]
        )
        val_dataset = Mamba4KTDataset(
            val_data[0], val_data[1], val_data[2], val_question[0]
        )
        test_dataset = window_test_data.create_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"Mamba4KT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
