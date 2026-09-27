from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)

__all__ = ["GRKTModelData", "GRKTDataset"]


class GRKTDataset(Dataset):
    def __init__(self, questions, knows, responses, times, masks):
        self.questions = questions
        self.knows = knows
        self.responses = responses
        self.times = times
        self.masks = masks

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.knows[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.times[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )


class GRKTModelData(QuestionModelData):
    """GRKT model data preparation.

    - question sequences [num_users, max_seq_len]
    - multi-skill sequences [num_users, max_seq_len, K]  (K = max skills per question)
    - response sequences [num_users, max_seq_len]
    - time sequences [num_users, max_seq_len]  (absolute seconds from millisecond
      timestamps / 1000 when available, else 1-indexed seq_pos + 1 as fallback)
    - mask sequences [num_users, max_seq_len]
    Also computes rel_map and pre_map (skill relationship and prerequisite
    graphs) from train-fold co-occurrence statistics, matching
    split_kfold_data's train definition.
    """

    OPTIONAL_FEATURE_COLUMNS = ("timestamp",)

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        # Load split question sequence data
        q_data = self._load_split_data()
        num_users = q_data["sequence_id"].n_unique()
        merged_pd = q_data.to_pandas()

        # Build multi-skill per question from question_skill relation
        question_skill_df = self.data_src.get_relation("question_skill")
        qs_grouped = question_skill_df.group_by("question").agg(
            pl.col("skill").sort().alias("skills")
        )
        K = qs_grouped.select(pl.col("skills").list.len().max()).item()
        logger.info(f"Max skills per question (K): {K}")

        # Create question_skills array: [num_questions, K]
        question_skills = np.full((num_questions, K), num_skills, dtype=np.int64)
        for row in qs_grouped.iter_rows(named=True):
            qid = row["question"]
            skills = row["skills"]
            question_skills[qid, : len(skills)] = skills

        # Build sequence arrays
        user_sequence = np.zeros((num_users, max_seq_len), dtype=np.int64)
        user_knows = np.full((num_users, max_seq_len, K), num_skills, dtype=np.int64)
        user_response = np.zeros((num_users, max_seq_len), dtype=np.int64)
        user_time = np.zeros((num_users, max_seq_len), dtype=np.int64)
        user_mask = np.zeros((num_users, max_seq_len), dtype=np.int64)

        user_indices = merged_pd["sequence_id"].values
        seq_positions = merged_pd["seq_pos"].values
        questions = merged_pd["question"].values
        labels = merged_pd["label"].values

        user_sequence[user_indices, seq_positions] = questions
        user_knows[user_indices, seq_positions] = question_skills[questions]
        user_response[user_indices, seq_positions] = labels
        user_mask[user_indices, seq_positions] = 1

        # Use actual timestamps converted to seconds for temporal dynamics.
        # Fall back to seq_pos + 1 when timestamps are unavailable.
        if "timestamp" in merged_pd.columns:
            user_time[user_indices, seq_positions] = (
                merged_pd["timestamp"].values.astype(np.float64) / 1000.0
            ).astype(np.int64)
        else:
            user_time[user_indices, seq_positions] = seq_positions + 1

        # K-fold split
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} out of range [0, {kfold_n_splits})"
                )
            logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")
            train_data, val_data, test_data = self.split_kfold_data(
                user_sequence,
                user_knows,
                user_response,
                user_time,
                user_mask,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        train_dataset = GRKTDataset(*train_data)
        val_dataset = GRKTDataset(*val_data)
        test_dataset = GRKTDataset(*test_data)

        user_folds = self._build_user_folds(num_users)
        train_user_indices = np.where((user_folds != fold_idx) & (user_folds != -1))[0]
        rel_map, pre_map = self._build_skill_graph(
            user_sequence[train_user_indices],
            user_knows[train_user_indices],
            user_response[train_user_indices],
            user_mask[train_user_indices],
            num_skills,
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            num_questions,
            num_skills,
            K,
            rel_map,
            pre_map,
        )

    def _build_skill_graph(self, questions, knows, responses, masks, num_skills):
        """Build skill relationship and prerequisite graphs from the given data.

        Computes four co-occurrence matrices (tt, tf, ft, ff) based on the
        correctness patterns of skill pairs within each user sequence. Then
        derives:
        - rel_map: skill relevance probability (agreement rate)
        - pre_map: prerequisite probability (incorrect→correct pattern)

        Uses only the first skill per question, matching the original GRKT
        preprocessing.
        """
        tt = np.zeros((num_skills, num_skills))
        tf = np.zeros((num_skills, num_skills))
        ft = np.zeros((num_skills, num_skills))
        ff = np.zeros((num_skills, num_skills))

        for i in tqdm(range(len(questions)), desc="Building skill graph"):
            mask = masks[i]
            seq_len = int(mask.sum())
            t_vec = np.zeros(num_skills)
            f_vec = np.zeros(num_skills)
            for t in range(seq_len):
                know = knows[i, t, 0]  # First skill (as in original)
                corr = bool(responses[i, t])
                if corr:
                    tt[:, know] += t_vec
                    tt[know, :] += t_vec
                    ft[:, know] += f_vec
                    tf[know, :] += f_vec
                else:
                    ff[:, know] += f_vec
                    ff[know, :] += f_vec
                    tf[:, know] += t_vec
                    ft[know, :] += t_vec
                if corr:
                    t_vec[know] += 1
                else:
                    f_vec[know] += 1

        cold_thresh = 5
        rel_filt = (tt + ff + tf + ft) >= 4 * cold_thresh
        pre_filt = (tf + ft) >= 2 * cold_thresh

        rel_map = (tt + ff) / np.clip(tt + ff + tf + ft, a_min=1, a_max=None)
        pre_map = ft / np.clip(tf + ft, a_min=1, a_max=None)

        for i in range(len(rel_map)):
            rel_map[i, i] = 0
            pre_map[i, i] = 0

        rel_map = rel_map * rel_filt
        pre_map = pre_map * pre_filt

        return rel_map.astype(np.float32), pre_map.astype(np.float32)
