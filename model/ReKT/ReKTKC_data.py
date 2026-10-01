from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class ReKTKCDataset(Dataset):
    def __init__(self, questions, skills, responses, masks):
        self.questions = questions
        self.skills = skills
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.questions[index], dtype=torch.long),
            torch.tensor(self.skills[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.questions)


class ReKTKCModelData(SkillModelData):
    """ReKT 的 skill-level 数据变体。

    与 ReKTModelData 的差异：继承 SkillModelData，skill 序列直接取自
    数据源的 split_skill_sequence，而非问题→组合技能映射。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        # build_sequence_data
        (
            user_skill_sequence,
            user_response,
            user_mask,
            _,
            user_question_sequence,
        ) = self.build_sequence_data()

        if fold_idx is not None:
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            train_data, val_data, _ = self.split_kfold_data(
                user_question_sequence,
                user_skill_sequence,
                user_response,
                user_mask,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        train_dataset = ReKTKCDataset(
            train_data[0], train_data[1], train_data[2], train_data[3]
        )
        val_dataset = ReKTKCDataset(val_data[0], val_data[1], val_data[2], val_data[3])

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        num_skills = self.data_src.get_metadata("num_skills")
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            # Model reads skill count under the num_combined_skills key.
            {"num_combined_skills": num_skills},
        )
