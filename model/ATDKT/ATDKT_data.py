"""ATDKT 数据处理模块"""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class ATDKTDataset(Dataset):
    """ATDKT 数据集

    KC 级序列外加对齐的题目序列（QT 辅助任务需要）。

    Args:
        sequences: 概念ID序列
        responses: 响应序列
        masks: 掩码序列
        questions: 题目ID序列

    Returns:
        (sequence, response, mask, question) 元组
    """

    def __init__(self, sequences, responses, masks, questions):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions

    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        question = torch.tensor(self.questions[idx], dtype=torch.long)
        return sequence, response, mask, question


class ATDKTModelData(SkillModelData):
    """ATDKT 模型数据加载器

    继承自 SkillModelData，使用技能序列构建方式，
    额外携带对齐的题目序列。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练和验证数据

        Args:
            rc: RunConfig

        Returns:
            训练数据集、验证数据集和窗口测试 DataLoader
        """
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
                user_sequence,
                user_response,
                user_mask,
                user_question,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        stream_dataset = self.create_windowlate_iterable_dataset()

        train_dataset = ATDKTDataset(
            train_data[0], train_data[1], train_data[2], train_data[3]
        )
        val_dataset = ATDKTDataset(val_data[0], val_data[1], val_data[2], val_data[3])
        test_dataset = stream_dataset.create_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"ATDKT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
