import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class ClusterKTDataset(Dataset):
    """Dataset for ClusterKT model.

    Returns (skill_group_seq, question_seq, response, mask, lagtime).
    """

    def __init__(self, sequences, questions, responses, masks, timestamps, n_et):
        self.sequences = sequences
        self.questions = questions
        self.responses = responses
        self.masks = masks
        self.timestamps = timestamps
        self.n_et = n_et

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        question = torch.tensor(self.questions[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.long)

        # Compute lagtime from timestamps
        timestamps = np.array(self.timestamps[idx], dtype=np.float64)
        lagtime = self._compute_lagtime(timestamps, mask)
        lagtime = torch.tensor(lagtime, dtype=torch.long)

        return sequence, question, response, mask, lagtime

    def _compute_lagtime(self, timestamps, mask):
        """Compute elapsed time intervals from absolute timestamps."""
        max_len = len(timestamps)
        lagtime = np.full(max_len, self.n_et + 1, dtype=np.int64)

        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        valid = np.where(mask_np > 0)[0]
        if len(valid) <= 1:
            return lagtime

        for i in range(1, len(valid)):
            idx = valid[i]
            prev_idx = valid[i - 1]
            diff = int(timestamps[idx] - timestamps[prev_idx])
            lagtime[idx] = max(0, min(diff, self.n_et))

        lagtime[valid[0]] = self.n_et + 1
        return lagtime


class ClusterKTModelData(QuestionModelData):
    """ClusterKT data loader.

    Extends QuestionModelData with:
    - Skill group mapping: bundles multi-skill questions into single skill_group IDs
    - Timestamp extraction for lagtime computation
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    def _build_skill_group_mapping(self):
        """Map each question's skill combination to a unique skill_group ID.

        Questions sharing the same set of skills map to the same skill_group.
        """
        question_data = self.data_src.get_relation("question_skill")
        question_skills = question_data.group_by("question").agg(
            pl.col("skill").sort().alias("skill_list")
        )

        question_skills = question_skills.with_columns(
            pl.col("skill_list")
            .cast(pl.List(pl.Utf8))
            .list.join(",")
            .rank("dense")
            .sub(1)
            .alias("skill_group_id")
        )

        num_skill_groups = question_skills["skill_group_id"].n_unique()
        question_to_sg = dict(
            zip(
                question_skills["question"].to_list(),
                question_skills["skill_group_id"].to_list(),
            )
        )
        logger.info(
            f"Built skill group mapping: {num_skill_groups} groups from "
            f"{len(question_to_sg)} questions"
        )
        return question_to_sg, num_skill_groups

    def load_sequence_data(self):
        """Load question-level sequences with skill_group mapping and timestamps.

        Returns:
            (user_sequence, user_question, user_response, user_mask, user_timestamp)
            - user_sequence: skill_group IDs
            - user_question: original question IDs (for Rasch)
            - user_response: correctness labels
            - user_mask: validity mask
            - user_timestamp: absolute timestamps
        """
        data = self.load_split_data(required=("timestamp",))

        data_pd = data.to_pandas()

        # Build skill group mapping
        q_to_sg, num_skill_groups = self._build_skill_group_mapping()
        self.num_skill_groups = num_skill_groups

        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_users = data_pd["sequence_id"].nunique()

        logger.info(
            f"Building ClusterKT sequences: {num_users} users, "
            f"max_seq_len={max_seq_len}, num_skill_groups={num_skill_groups}"
        )

        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_question = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)
        user_timestamp = np.zeros((num_users, max_seq_len), dtype=np.float64)

        user_indices = data_pd["sequence_id"].values
        seq_positions = data_pd["seq_pos"].values

        # Map question → skill_group
        sg_values = data_pd["question"].map(q_to_sg).values
        user_sequence[user_indices, seq_positions] = sg_values
        user_question[user_indices, seq_positions] = data_pd["question"].values
        user_response[user_indices, seq_positions] = data_pd["label"].values
        user_mask[user_indices, seq_positions] = 1
        user_timestamp[user_indices, seq_positions] = data_pd["timestamp"].values

        return user_sequence, user_question, user_response, user_mask, user_timestamp

    @override
    def prepare_data(self, rc):
        """Prepare ClusterKT datasets.

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        n_et = rc.model.n_et

        # Build sequences
        (
            user_sequence,
            user_question,
            user_response,
            user_mask,
            user_timestamp,
        ) = self.load_sequence_data()

        # K-fold split
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
                user_question,
                user_response,
                user_mask,
                user_timestamp,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        # Build datasets
        train_dataset = ClusterKTDataset(
            train_data[0],
            train_data[1],
            train_data[2],
            train_data[3],
            train_data[4],
            n_et=n_et,
        )
        val_dataset = ClusterKTDataset(
            val_data[0], val_data[1], val_data[2], val_data[3], val_data[4], n_et=n_et
        )
        test_dataset = ClusterKTDataset(
            test_data[0],
            test_data[1],
            test_data[2],
            test_data[3],
            test_data[4],
            n_et=n_et,
        )

        return train_dataset, val_dataset, test_dataset
