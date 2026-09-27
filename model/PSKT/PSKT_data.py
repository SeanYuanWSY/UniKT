"""PSKT model data preparation."""

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class PSKTDataset(Dataset):
    """PSKT dataset.

    Args:
        sequences: Question id sequence [N, S] (1-based, 0 = padding).
        responses: Response sequence [N, S] (0/1 valid, 2 = padding).
        masks: Valid position mask [N, S].
        skills: Multi-concept skill sequence [N, S, max_concepts] (1-based, -1 = empty).
        timestamps: Timestamp sequence [N, S].
    """

    def __init__(self, sequences, responses, masks, skills, timestamps):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.skills = skills
        self.timestamps = timestamps

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.skills[idx], dtype=torch.long),
            torch.tensor(self.timestamps[idx], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class PSKTModelData(QuestionModelData):
    REQUIRED_FEATURE_COLUMNS = ("timestamp",)

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.max_concepts = None

    @override
    def prepare_data(self, rc):
        """Prepare train, validation, and test datasets.

        Args:
            rc: RunConfig instance.

        Returns:
            (train_dataset, val_dataset, test_dataset) tuple.
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        user_sequence, user_response, user_mask, user_timestamp = self._load_sequences()

        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        self.max_concepts = int(q_matrix.sum(axis=1).max())
        user_skills = self._build_skill_sequences(user_sequence, q_matrix)

        pad = ~user_mask.astype(bool)
        user_sequence = np.where(pad, 0, user_sequence + 1)
        user_skills[pad] = -1

        logger.info(
            f"PSKT data prepared: max_concepts={self.max_concepts}, "
            f"num_questions={q_matrix.shape[0]}, num_skills={q_matrix.shape[1]}"
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
            train_data, val_data, test_data = self.split_kfold_data(
                user_sequence,
                user_response,
                user_mask,
                user_skills,
                user_timestamp,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        train_dataset = PSKTDataset(*train_data)
        val_dataset = PSKTDataset(*val_data)
        test_dataset = PSKTDataset(*test_data)

        logger.info(
            f"PSKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    def _load_sequences(self):
        """Return (questions 0-based, responses, masks, timestamps) as [num_users, max_seq_len] int arrays."""
        data = self._load_split_data()
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_users = data["sequence_id"].n_unique()

        user_sequence = np.zeros((num_users, max_seq_len), dtype=np.int64)
        user_response = np.full((num_users, max_seq_len), 2, dtype=np.int64)
        user_mask = np.zeros((num_users, max_seq_len), dtype=np.int64)
        user_timestamp = np.zeros((num_users, max_seq_len), dtype=np.int64)

        user_idx = data["sequence_id"].to_numpy()
        seq_pos = data["seq_pos"].to_numpy()
        user_sequence[user_idx, seq_pos] = data["question"].to_numpy()
        user_response[user_idx, seq_pos] = data["label"].to_numpy()
        user_mask[user_idx, seq_pos] = 1
        user_timestamp[user_idx, seq_pos] = data["timestamp"].to_numpy()

        return user_sequence, user_response, user_mask, user_timestamp

    def _build_skill_sequences(self, user_sequence, q_matrix):
        """Return multi-concept skill sequence [N, S, max_concepts] (1-based, -1 = empty)."""
        num_questions = q_matrix.shape[0]

        question_skills = np.full(
            (num_questions, self.max_concepts), -1, dtype=np.int64
        )
        for q_id in range(num_questions):
            related = np.where(q_matrix[q_id] > 0)[0]
            for k in range(min(len(related), self.max_concepts)):
                question_skills[q_id, k] = related[k] + 1

        return question_skills[user_sequence]
