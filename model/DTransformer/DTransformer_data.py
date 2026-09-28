"""DTransformer 模型数据处理模块"""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class DTransformerDataset(Dataset):
    """DTransformer 数据集

    训练/验证模式:
        (sequence, response, mask, question)
    窗口测试模式:
        (sequence, response, mask, late_group_id, true_labels, question)
    """

    def __init__(
        self,
        sequences,
        responses,
        masks,
        questions,
        late_group_ids=None,
        true_labels=None,
    ):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions
        self.late_group_ids = late_group_ids
        self.true_labels = true_labels

        self._is_window_mode = late_group_ids is not None and true_labels is not None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        question = torch.tensor(self.questions[idx], dtype=torch.long)

        if self._is_window_mode:
            late_group_id = torch.tensor(self.late_group_ids[idx], dtype=torch.long)
            true_labels = torch.tensor(self.true_labels[idx], dtype=torch.long)
            return sequence, response, mask, late_group_id, true_labels, question

        return sequence, response, mask, question


class DTransformerModelData(SkillModelData):
    """DTransformer 模型数据加载器"""

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

        train_dataset = DTransformerDataset(
            train_data[0], train_data[1], train_data[2], train_question[0]
        )
        val_dataset = DTransformerDataset(
            val_data[0], val_data[1], val_data[2], val_question[0]
        )
        test_dataset = DataLoader(
            window_test_data,
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"DTransformer data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
