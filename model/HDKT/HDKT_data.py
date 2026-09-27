"""HDKT 模型数据处理模块。

- 题目序列 ``e``、答案序列 ``a``、掩码 ``mask``、用户 id ``uid``（学生嵌入
  索引）、知识点序列 ``skill``（每题一个关联 skill，由 Q-matrix 派生）
- 答题用时 ``at``：``ms_first_response`` 毫秒→秒；间隔时间 ``it``：相邻
  ``timestamp`` 差→分钟、截断 ``max_it_minutes``。两者均为词表式离散化：
  全数据唯一值各一桶、id 从 1 起，0 为零向量 padding
- Q-matrix（模型内现算 kc 投影，多概念题为多 hot）
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class HDKTDataset(Dataset):
    """HDKT 数据集。

    每个样本返回 ``(e, at, a, it, mask, uid, skill)``：
        e:     题目序列 [S]
        at:    答题用时词表 id 序列 [S]（0 = padding 零向量）
        a:     答案序列 [S]（同时作为输入特征与预测目标）
        it:    间隔时间词表 id 序列 [S]（0 = padding 零向量）
        mask:  有效位置掩码 [S]
        uid:   用户（序列行）id 标量，学生嵌入索引
        skill: 知识点序列 [S]（每题一个关联 skill，padding 为 0）
    """

    def __init__(self, e, at, a, it, mask, uid, skill):
        self.e = e
        self.at = at
        self.a = a
        self.it = it
        self.mask = mask
        self.uid = uid
        self.skill = skill

    def __len__(self) -> int:
        return len(self.e)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.e[idx], dtype=torch.long),
            torch.tensor(self.at[idx], dtype=torch.long),
            torch.tensor(self.a[idx], dtype=torch.long),
            torch.tensor(self.it[idx], dtype=torch.long),
            torch.tensor(self.mask[idx], dtype=torch.bool),
            torch.tensor(self.uid[idx], dtype=torch.long),
            torch.tensor(self.skill[idx], dtype=torch.long),
        )


class HDKTModelData(QuestionModelData):
    """HDKT 模型数据加载器。"""

    REQUIRED_FEATURE_COLUMNS = ("ms_first_response", "timestamp")

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的元信息。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            为含 ``q_matrix``、``n_at``、``n_it``、``num_questions``、
            ``num_skills``、``num_users``、``max_seq_len`` 的字典。
        """
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")
        num_users = self.data_src.get_metadata("num_users")

        user_sequence, user_response, user_mask, user_id_sequence = (
            self.load_sequence_data()
        )

        # Negative labels would wrap the knowledge-embedding index around the
        # vocabulary, so fail fast on dirty data.
        if not np.isin(user_response, [0, 1]).all():
            raise ValueError("HDKT requires binary labels (0/1) in the data.")

        at_seq, it_seq, n_at, n_it = self._build_time_sequences(
            max_seq_len, rc.model.max_it_minutes
        )

        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))

        # The denoising front end indexes one skill per step: derive the first
        # associated skill per question from the raw binary Q-matrix. Questions
        # without any relation fall back to skill 0.
        skill_of_question = np.argmax(q_matrix, axis=1)
        if not (q_matrix.sum(axis=1) > 0).all():
            n_orphan = int((q_matrix.sum(axis=1) == 0).sum())
            logger.warning(
                f"HDKT: {n_orphan}/{num_questions} questions have no skill "
                f"relation; their skill id falls back to 0."
            )
        skill_seq = skill_of_question[user_sequence]
        skill_seq = np.where(user_mask, skill_seq, 0)  # padding -> 0

        uid = user_id_sequence[:, 0]

        logger.info(
            f"HDKT data: num_questions={num_questions}, num_skills={num_skills}, "
            f"num_users={num_users}, n_at={n_at}, n_it={n_it}, "
            f"max_seq_len={max_seq_len}"
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
            uid,
            skill_seq,
            fold_idx=fold_idx,
        )
        (tr_e, tr_a, tr_mask, tr_at, tr_it, tr_uid, tr_skill) = train_slices
        (va_e, va_a, va_mask, va_at, va_it, va_uid, va_skill) = val_slices
        (te_e, te_a, te_mask, te_at, te_it, te_uid, te_skill) = test_slices

        train_dataset = HDKTDataset(tr_e, tr_at, tr_a, tr_it, tr_mask, tr_uid, tr_skill)
        val_dataset = HDKTDataset(va_e, va_at, va_a, va_it, va_mask, va_uid, va_skill)
        test_dataset = HDKTDataset(te_e, te_at, te_a, te_it, te_mask, te_uid, te_skill)

        logger.info(
            f"HDKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        info = {
            "q_matrix": q_matrix,
            "n_at": n_at,
            "n_it": n_it,
            "num_questions": num_questions,
            "num_skills": num_skills,
            "num_users": num_users,
            "max_seq_len": max_seq_len,
        }
        return train_dataset, val_dataset, test_dataset, info

    def _build_time_sequences(
        self, max_seq_len: int, max_it_minutes: int
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """构建答题用时 / 间隔时间的词表 id 序列。

        词表对全数据统计唯一值（不分 fold——时间值不含标签信息）；有效位置
        （含值 0）映射到 ≥1，padding 位置恒为 0。

        Returns:
            at_seq / it_seq: 词表 id [N, S]
            n_at / n_it: 词表大小 = 唯一值数 + 1
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

        # Filled values may legitimately be 0, so validity must come from fill marks.
        valid = np.zeros((num_users, max_seq_len), dtype=bool)
        valid[user_idx, seq_pos] = True

        # Answer time in seconds; negative dirty values clipped to 0. No upper
        # cap — every distinct second keeps its own bucket (vocabulary style).
        at_sec = np.clip(np.floor(ms_grid / 1000.0), 0, None).astype(np.int64)

        # Interval time in minutes: adjacent diff per row, first position 0.
        diff_ms = np.zeros_like(ts_grid)
        diff_ms[:, 1:] = ts_grid[:, 1:] - ts_grid[:, :-1]
        it_min = np.clip(
            np.floor(np.clip(diff_ms, 0, None) / 60000.0), 0, max_it_minutes
        )
        it_min = it_min.astype(np.int64)

        at_vocab = np.unique(at_sec[valid])
        it_vocab = np.unique(it_min[valid])

        at_seq = np.zeros_like(at_sec, dtype=np.int32)
        it_seq = np.zeros_like(it_min, dtype=np.int32)
        at_seq[valid] = np.searchsorted(at_vocab, at_sec[valid]) + 1
        it_seq[valid] = np.searchsorted(it_vocab, it_min[valid]) + 1

        return at_seq, it_seq, int(at_vocab.size) + 1, int(it_vocab.size) + 1
