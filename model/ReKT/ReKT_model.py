from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReKT(nn.Module):
    """ReKT: Repetition-aware Knowledge Tracing.

    通过时间感知的遗忘机制跟踪问题状态、技能状态和全局状态。
    模型在每个位置 t 预测 response[t]，使用历史 0..t-1 的状态。
    返回原始 logits，配合 BCEWithLogitsLoss 使用。
    """

    def __init__(
        self,
        data_metadata: dict[str, Any],
        max_seq_len: int,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Explicit max_seq_len: the same model class serves both granularities
        # (ReKT question-level, ReKTKC skill-level), so the caller picks the
        # matching metadata padded-length key.
        num_questions = data_metadata["num_questions"]
        num_skills = data_metadata["num_combined_skills"]

        d = hidden_dim
        p = dropout

        self.num_questions = num_questions
        self.num_skills = num_skills
        self.max_seq_len = max_seq_len

        self.question_embed = nn.Embedding(num_questions, d)
        self.skill_embed = nn.Embedding(num_skills, d)
        self.answer_embed = nn.Embedding(2, d)
        self.time_embed = nn.Embedding(max_seq_len, d)

        self.question_diff = nn.Embedding(num_questions, 1)
        self.skill_change = nn.Embedding(num_skills, d)

        self.global_state = nn.Parameter(torch.randn(1, d))
        self.pro_state_init = nn.Parameter(torch.randn(max_seq_len, d))
        self.skill_state_init = nn.Parameter(torch.randn(max_seq_len, d))

        self.obtain_pro_forget = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.obtain_pro_state = nn.Sequential(nn.Linear(2 * d, d))
        self.obtain_skill_forget = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.obtain_skill_state = nn.Sequential(nn.Linear(2 * d, d))
        self.obtain_all_forget = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.obtain_all_state = nn.Sequential(nn.Linear(2 * d, d))

        self.out = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(d, 1),
        )

        self.dropout = nn.Dropout(p=p)

    def forward(
        self,
        question_seq: torch.Tensor,
        skill_seq: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, S = question_seq.shape
        device = question_seq.device

        q_embed = (
            self.question_embed(question_seq)
            + self.skill_embed(skill_seq)
            + self.question_diff(question_seq) * self.skill_change(skill_seq)
        )
        interaction_embed = q_embed + self.answer_embed(response)

        time_w = self.time_embed.weight
        global_time_emb = time_w[1]

        last_pro_time = torch.zeros(
            B, self.num_questions, device=device, dtype=torch.long
        )
        last_skill_time = torch.zeros(
            B, self.num_skills, device=device, dtype=torch.long
        )
        pro_state = self.pro_state_init.unsqueeze(0).expand(B, -1, -1).clone()
        skill_state = self.skill_state_init.unsqueeze(0).expand(B, -1, -1).clone()
        all_state = self.global_state.expand(B, -1).clone()

        batch_idx = torch.arange(B, device=device)
        max_tg = time_w.shape[0] - 1

        dp = self.dropout
        pro_fg = self.obtain_pro_forget
        pro_st = self.obtain_pro_state
        sk_fg = self.obtain_skill_forget
        sk_st = self.obtain_skill_state
        all_fg = self.obtain_all_forget
        all_st = self.obtain_all_state
        out = self.out

        res = []

        for t in range(S):
            pro_id = question_seq[:, t]
            sk_id = skill_seq[:, t]

            pt = last_pro_time[batch_idx, pro_id]
            st_ = last_skill_time[batch_idx, sk_id]

            ps = pro_state[batch_idx, pt]
            ss = skill_state[batch_idx, st_]

            pt_emb = F.embedding((t - pt).clamp(max=max_tg), time_w)
            st_emb = F.embedding((t - st_).clamp(max=max_tg), time_w)

            ps = ps * pro_fg(dp(torch.cat([ps, pt_emb], -1)))
            ss = ss * sk_fg(dp(torch.cat([ss, st_emb], -1)))
            a_s = all_state * all_fg(
                dp(torch.cat([all_state, global_time_emb.expand(B, -1)], -1))
            )

            res.append(out(dp(torch.cat([a_s, ps, ss, q_embed[:, t]], -1))).squeeze(-1))

            x = interaction_embed[:, t]
            all_state = a_s + torch.tanh(all_st(dp(torch.cat([a_s, x], -1))))
            pro_state[:, t] = ps + torch.tanh(pro_st(dp(torch.cat([ps, x], -1))))
            last_pro_time[batch_idx, pro_id] = t
            skill_state[:, t] = ss + torch.tanh(sk_st(dp(torch.cat([ss, x], -1))))
            last_skill_time[batch_idx, sk_id] = t

        return torch.stack(res, dim=1)
