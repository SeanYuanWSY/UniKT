from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class HawkesKTDataset(Dataset):
    def __init__(self, skill_seqs, problem_seqs, time_seqs, labels, masks):
        self.skill_seqs = skill_seqs
        self.problem_seqs = problem_seqs
        self.time_seqs = time_seqs
        self.labels = labels
        self.masks = masks

    def __len__(self) -> int:
        return len(self.skill_seqs)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.skill_seqs[idx], dtype=torch.long),
            torch.tensor(self.problem_seqs[idx], dtype=torch.long),
            torch.tensor(self.time_seqs[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )


class HawkesKTModelData(QuestionModelData):
    """HawkesKT model data loader.

    Builds all sequences (skill, problem, time, label) from the question
    split data. Skills are obtained by joining with question_data (first skill per question).
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        max_seq_len = self.data_src.get_metadata("max_seq_len")

        # 1. Load question split data (one row per question, has timestamp)
        q_data = self.load_split_data(
            required=("timestamp",), optional=("ms_first_response",)
        )
        num_users = q_data["sequence_id"].n_unique()

        # 2. Get first skill per question from question_data
        question_data = self.data_src.get_relation("question_skill")
        question_skill_map = question_data.group_by("question").agg(
            pl.col("skill").first().alias("skill")
        )
        merged = q_data.join(question_skill_map, on="question", how="left")
        merged_pd = merged.to_pandas()

        # 3. Build arrays from the same merged data
        skill_seqs = np.zeros((num_users, max_seq_len), dtype=np.int64)
        problem_seqs = np.zeros((num_users, max_seq_len), dtype=np.int64)
        time_seqs = np.zeros((num_users, max_seq_len), dtype=np.int64)
        label_seqs = np.zeros((num_users, max_seq_len), dtype=np.int64)
        mask_seqs = np.zeros((num_users, max_seq_len), dtype=np.int64)

        user_indices = merged_pd["sequence_id"].values
        seq_positions = merged_pd["seq_pos"].values

        skill_seqs[user_indices, seq_positions] = merged_pd["skill"].values
        problem_seqs[user_indices, seq_positions] = merged_pd["question"].values
        label_seqs[user_indices, seq_positions] = merged_pd["label"].values
        mask_seqs[user_indices, seq_positions] = 1

        # Build time_seqs per dataset (target unit: seconds)
        dataset = self.data_src.dataset
        if dataset == "assistments09" and "ms_first_response" in merged_pd.columns:
            # Match original: cumulative dwell time from ms_first_response (seconds)
            # timestamps[i] = timestamps[i-1] + dwell[i-1]/1000 + 1
            dwell = merged_pd["ms_first_response"].fillna(0).values.astype(np.float64)
            dwell /= 1000.0
            cumshift = np.zeros(num_users, dtype=np.float64)
            order = np.lexsort((seq_positions, user_indices))
            for idx in order:
                uid = user_indices[idx]
                pos = seq_positions[idx]
                time_seqs[uid, pos] = int(cumshift[uid])
                cumshift[uid] += dwell[idx] + 1.0
        else:
            # Default: convert timestamp milliseconds → seconds
            time_seqs[user_indices, seq_positions] = (
                merged_pd["timestamp"].values.astype(np.float64) / 1000.0
            ).astype(np.int64)

        # 4. K-fold split
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} out of range [0, {kfold_n_splits})"
                )
            logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")
            train_data, val_data, test_data = self.split_kfold_data(
                skill_seqs,
                problem_seqs,
                time_seqs,
                label_seqs,
                mask_seqs,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        train_dataset = HawkesKTDataset(*train_data)
        val_dataset = HawkesKTDataset(*val_data)
        test_dataset = HawkesKTDataset(*test_data)

        logger.debug(
            f"HawkesKT data ready: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
