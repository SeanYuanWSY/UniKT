"""KeenKT 数据处理模块"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


def _build_augmented_target(response: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """构建对比学习第二视图的确定性极性翻转响应序列。

    依据序列首/末有效位的响应取值，条件性翻转位置 0 与位置 1
    （上游循环在 i=0 处原地翻转判定条件后再读取，故仅翻转一位）。
    pad 位恒为 0，不参与翻转。
    """
    n = int(mask.sum())
    target_aug = (response * mask).astype(np.int64)
    if n < 2:
        return target_aug
    last, first, shft0 = response[n - 1], response[0], response[1]
    if first == last:
        target_aug[0] = 1 - target_aug[0]
    if shft0 == last:
        target_aug[1] = 1 - target_aug[1]
    return target_aug


class KeenKTTrainDataset(Dataset):
    """KeenKT 训练数据集

    Args:
        sequences: 技能ID序列
        responses: 响应序列
        masks: 掩码序列
        questions: 题目ID序列
        use_uncertainty_aug: 是否生成极性翻转增强视图

    Returns:
        (sequence, response, mask, question, target_aug) 元组
    """

    def __init__(
        self,
        sequences,
        responses,
        masks,
        questions,
        use_uncertainty_aug: bool,
    ):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions
        self.use_uncertainty_aug = use_uncertainty_aug
        # deterministic per sample, so precompute once
        if use_uncertainty_aug:
            self.target_augs = np.stack(
                [_build_augmented_target(r, m) for r, m in zip(responses, masks)]
            )
        else:
            self.target_augs = None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        question = torch.tensor(self.questions[idx], dtype=torch.long)

        if self.target_augs is not None:
            target_aug = torch.from_numpy(self.target_augs[idx])
        else:
            target_aug = response * mask

        return sequence, response, mask, question, target_aug


class KeenKTEvalDataset(Dataset):
    """KeenKT 验证数据集

    Returns:
        (sequence, response, mask, question) 元组
    """

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


class KeenKTModelData(SkillModelData):
    """KeenKT 模型数据加载器"""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练、验证和测试数据

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

        train_dataset = KeenKTTrainDataset(
            train_data[0],
            train_data[1],
            train_data[2],
            train_data[3],
            use_uncertainty_aug=rc.model.use_uncertainty_aug,
        )
        val_dataset = KeenKTEvalDataset(
            val_data[0], val_data[1], val_data[2], val_data[3]
        )
        test_dataset = DataLoader(
            stream_dataset,
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"KeenKT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
