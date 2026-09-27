"""TCKT 模型数据处理模块。

- 题目序列 ``e``、答案序列 ``a``、掩码 ``mask``
- 响应时间桶序列 ``at``（由 ``ms_first_response`` 按秒离散化）
- 间隔时间桶序列 ``it``（由相邻 ``timestamp`` 之差按分钟离散化）
- 概念序列 ``c``（每题的主知识点）
- Q-matrix（题目-概念关联）
- 每题难度（贝叶斯估计，论文式 2，仅用训练折数据计算）

时间离散化遵循论文 4.4 节：“response time 按秒、interval time 按分钟”，并分别截断
到 ``max_rt_seconds`` / ``max_it_minutes`` 以滤除噪声长尾。
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class TCKTDataset(Dataset):
    """TCKT 数据集。

    每个样本返回 ``(e, at, a, it, c, mask)``：
        e:    题目序列 [S]
        at:   响应时间桶序列 [S]
        a:    答案序列 [S]（同时作为输入特征与预测目标）
        it:   间隔时间桶序列 [S]
        c:    概念（主知识点）序列 [S]
        mask: 有效位置掩码 [S]
    """

    def __init__(self, e, at, a, it, c, mask):
        self.e = e
        self.at = at
        self.a = a
        self.it = it
        self.c = c
        self.mask = mask

    def __len__(self) -> int:
        return len(self.e)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.e[idx], dtype=torch.long),
            torch.tensor(self.at[idx], dtype=torch.long),
            torch.tensor(self.a[idx], dtype=torch.long),
            torch.tensor(self.it[idx], dtype=torch.long),
            torch.tensor(self.c[idx], dtype=torch.long),
            torch.tensor(self.mask[idx], dtype=torch.bool),
        )


class TCKTModelData(QuestionModelData):
    """TCKT 模型数据加载器。"""

    REQUIRED_FEATURE_COLUMNS = ("ms_first_response", "timestamp")

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的元信息。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            为含 ``q_matrix``、``primary_skill``、``difficulty``、``n_at``、
            ``n_it``、``num_questions``、``num_skills``、``max_seq_len`` 的字典。
        """
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        user_sequence, user_response, user_mask, _ = self.load_sequence_data()

        at_seq, it_seq, n_at, n_it = self._build_time_sequences(
            max_seq_len, rc.model.max_rt_seconds, rc.model.max_it_minutes
        )

        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        primary_skill = self._build_primary_skill(q_matrix)  # [num_questions]
        c_seq = primary_skill[user_sequence]

        logger.info(
            f"TCKT data: num_questions={num_questions}, num_skills={num_skills}, "
            f"n_at={n_at}, n_it={n_it}, max_seq_len={max_seq_len}"
        )

        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        if fold_idx is None:
            raise ValueError("K-fold cross-validation is not enabled (fold < 0).")

        train_slices, val_slices, test_slices = self.split_kfold_data(
            user_sequence,
            user_response,
            user_mask,
            at_seq,
            it_seq,
            c_seq,
            fold_idx=fold_idx,
        )
        (tr_e, tr_a, tr_mask, tr_at, tr_it, tr_c) = train_slices
        (va_e, va_a, va_mask, va_at, va_it, va_c) = val_slices
        (te_e, te_a, te_mask, te_at, te_it, te_c) = test_slices

        difficulty = self._compute_bayes_difficulty(tr_e, tr_a, tr_mask, num_questions)

        train_dataset = TCKTDataset(tr_e, tr_at, tr_a, tr_it, tr_c, tr_mask)
        val_dataset = TCKTDataset(va_e, va_at, va_a, va_it, va_c, va_mask)
        test_dataset = TCKTDataset(te_e, te_at, te_a, te_it, te_c, te_mask)

        logger.info(
            f"TCKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        info = {
            "q_matrix": q_matrix,
            "primary_skill": primary_skill,
            "difficulty": difficulty,
            "n_at": n_at,
            "n_it": n_it,
            "num_questions": num_questions,
            "num_skills": num_skills,
            "max_seq_len": max_seq_len,
        }
        return train_dataset, val_dataset, test_dataset, info

    def _build_time_sequences(
        self, max_seq_len: int, max_rt_seconds: int, max_it_minutes: int
    ):
        """由 ``ms_first_response`` 与 ``timestamp`` 构建响应/间隔时间桶序列。

        Returns:
            at_seq: 响应时间桶 [N, S] ∈ [0, max_rt_seconds]
            it_seq: 间隔时间桶 [N, S] ∈ [0, max_it_minutes]
            n_at:   max_rt_seconds + 1
            n_it:   max_it_minutes + 1
        """
        data = self._load_split_data()
        num_users = data["sequence_id"].n_unique()

        sub = data.select(
            ["sequence_id", "seq_pos", "ms_first_response", "timestamp"]
        ).to_pandas()
        user_idx = sub["sequence_id"].to_numpy()
        seq_pos = sub["seq_pos"].to_numpy()
        ms = sub["ms_first_response"].to_numpy(dtype=np.float64)
        ts = sub["timestamp"].to_numpy(dtype=np.float64)

        # Pivot into [N, S] grids; missing/padded positions become 0.
        ms_grid = np.zeros((num_users, max_seq_len), dtype=np.float64)
        ts_grid = np.zeros((num_users, max_seq_len), dtype=np.float64)
        ms_grid[user_idx, seq_pos] = np.nan_to_num(ms, nan=0.0, posinf=0.0, neginf=0.0)
        ts_grid[user_idx, seq_pos] = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

        # Response time in seconds, truncated to [0, max_rt_seconds].
        rt_sec = np.floor(ms_grid / 1000.0)
        rt_sec = np.clip(rt_sec, 0, max_rt_seconds).astype(np.int32)

        # Interval time in minutes: adjacent timestamp diff per row, first position 0, truncated to [0, max_it_minutes].
        diff_ms = np.zeros_like(ts_grid)
        diff_ms[:, 1:] = ts_grid[:, 1:] - ts_grid[:, :-1]
        it_min = np.floor(diff_ms / 60000.0)
        it_min = np.clip(it_min, 0, max_it_minutes).astype(np.int32)

        n_at = int(max_rt_seconds) + 1
        n_it = int(max_it_minutes) + 1
        return rt_sec, it_min, n_at, n_it

    @staticmethod
    def _build_primary_skill(q_matrix: np.ndarray) -> np.ndarray:
        """每题取第一个关联知识点作为主知识点；无关联则为 0。"""
        num_questions = q_matrix.shape[0]
        primary = np.zeros(num_questions, dtype=np.int32)
        for q in range(num_questions):
            skills = np.where(q_matrix[q] > 0)[0]
            if skills.size > 0:
                primary[q] = skills[0] + 1  # +1: reserve 0 for “no skill” (padding_idx)
        return primary

    @staticmethod
    def _compute_bayes_difficulty(
        e: np.ndarray,
        a: np.ndarray,
        mask: np.ndarray,
        num_questions: int,
    ) -> np.ndarray:
        """基于贝叶斯估计的每题难度（式 2）。

            dif_q = NE_q/(NE_q+m) · D_q + m/(NE_q+m) · TD

        其中 D_q 为该题正确率，NE_q 为该题作答次数，m 为所有题的平均作答次数，
        TD 为整体正确率。仅用训练折交互计算；未见过的题难度取 TD。
        """
        valid = mask.astype(bool)
        flat_e = e[valid]
        flat_a = a[valid].astype(np.float64)

        count_inter = np.bincount(flat_e, minlength=num_questions).astype(np.float64)
        count_correct = np.bincount(
            flat_e, weights=flat_a, minlength=num_questions
        ).astype(np.float64)

        total_inter = float(count_inter.sum())
        total_correct = float(count_correct.sum())
        TD = total_correct / total_inter if total_inter > 0 else 0.5
        m = total_inter / num_questions if num_questions > 0 else 0.0

        with np.errstate(invalid="ignore", divide="ignore"):
            normal_diff = np.where(
                count_inter > 0, count_correct / np.maximum(count_inter, 1.0), 0.0
            )
            denom = count_inter + m
            bayes = np.where(
                denom > 0,
                count_inter / np.maximum(denom, 1e-12) * normal_diff
                + m / np.maximum(denom, 1e-12) * TD,
                TD,
            )
        return bayes.astype(np.float32)
