"""Mamba4KT (Mamba-based Knowledge Tracing) 模型实现

原始论文: Cao & Zhang, "Mamba4KT: An Efficient and Effective Mamba-based
Knowledge Tracing Model", 2024 (arXiv:2405.16542)

模型由三部分组成：
1. Rasch model embed-based Embeddings（Eq. 4）：题目/交互嵌入引入题目难度 μ
2. Mamba Block（Eq. 5）：N 个堆叠的选择性状态空间块（S6 + Conv1D + 门控），带残差与 LayerNorm
3. FFN Block（Eq. 9）与预测头（Eq. 10）

预测遵循 next-item 对齐：输出 out[t] 利用 0..t 时刻的历史预测 response[t+1]，
并通过左移一位的题目嵌入引入待预测题目 q_{t+1} 的信息，避免标签泄漏。
"""

import torch
from mamba_ssm import Mamba
from torch import nn


class MambaBlock(nn.Module):
    """单个 Mamba 块（论文 Eq. 5）。

    内部使用 mamba_ssm.Mamba 实现 S6 + Conv1D + 门控分支（ẑ = SiLU(Linear(x̂'))），
    外层补充论文 Eq. 5 最后一行的残差连接与 LayerNorm：
        ŷ = LayerNorm(ŷ' + x̂'),  ŷ' = out_proj(S6(x̂) ⊗ ẑ)

    注：mamba_ssm.Mamba 内部不含残差与归一化，故在此显式添加。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 论文 Eq. 5：LayerNorm(Mamba(x) + x)
        y = self.mamba(x)
        return self.norm(self.dropout(y) + x)


class FFNBlock(nn.Module):
    """前馈网络块（论文 Eq. 9）。

    FFN(H) = GELU(H·W^(1) + b^(1))·W^(2) + b^(2)
    其中 W^(1) ∈ R^{D×4D}，W^(2) ∈ R^{4D×D}。
    外层补充残差连接与 LayerNorm（标准 Transformer FFN 结构）。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(x))))
        return self.norm(self.dropout(h) + x)


class Mamba4KT(nn.Module):
    """Mamba4KT 模型。

    Args:
        num_c: 概念（技能）数量
        n_pid: 题目（Problem ID）数量，>0 时启用 Rasch 模型嵌入
        d_model: 隐藏维度
        n_blocks: Mamba 块数量（论文 N=5）
        d_state: SSM 隐状态维度
        d_conv: 因果卷积核宽度
        expand: Mamba 内部扩展系数（Conv1D 输出通道数 = expand * d_model）
        dropout: Dropout 概率
        l2: Rasch 难度参数 μ 的 L2 正则化系数（论文 Eq. 11 中的 λ）
        separate_qa: True 时交互嵌入查 (concept, response) 联合索引表；
            False 时为 qa_embed(response) + q_embed(sequence) 相加
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        d_model: int = 128,
        n_blocks: int = 5,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        l2: float = 1e-5,
        separate_qa: bool = False,
    ):
        super().__init__()
        self.model_name = "mamba4kt"
        self.num_c = num_c
        self.n_pid = n_pid
        self.l2 = l2
        self.separate_qa = separate_qa
        embed_l = d_model

        # Rasch model embed-based Embeddings（论文 Eq. 4）
        # 基础嵌入
        self.q_embed = nn.Embedding(num_c, embed_l)  # c_{c_t}
        if separate_qa:
            self.qa_embed = nn.Embedding(2 * num_c + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        if self.n_pid > 0:
            # μ_{q_t}：题目难度标量
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            # d_{c_t}：题目对所属 concept 的偏移向量
            self.q_embed_diff = nn.Embedding(num_c, embed_l)
            # f(c_t, r_t)：concept-result 配对的变差向量
            if separate_qa:
                self.qa_embed_diff = nn.Embedding(2 * num_c + 1, embed_l)
            else:
                self.qa_embed_diff = nn.Embedding(2, embed_l)
            self.reset()

        # Mamba Block
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # FFN Block（论文 Eq. 9）
        self.ffn = FFNBlock(d_model=d_model, dropout=dropout)

        # 预测头（论文 Eq. 10）
        # 输入 = concat(FFN 输出, 题目嵌入)，经单层线性后 sigmoid
        self.out = nn.Linear(2 * embed_l, 1)

    def reset(self):
        """初始化 Rasch 难度参数 μ 为 0。"""
        if self.n_pid > 0:
            nn.init.constant_(self.difficult_param.weight, 0.0)

    def embed(self, sequence, response, pid_data):
        """计算题目嵌入 Q_t 与交互嵌入 R_t（论文 Eq. 4）。

        Args:
            sequence: 技能 ID 序列 [B, S]
            response: 响应序列 [B, S]
            pid_data: 题目 ID 序列（+1 偏移，0 为填充）[B, S]

        Returns:
            q_embed_data: 题目嵌入 [B, S, D]
            qa_embed_data: 交互嵌入 [B, S, D]
            c_reg_loss: Rasch 正则化损失（标量）
        """
        c_reg_loss = torch.tensor(0.0, device=sequence.device)

        q_embed_data = self.q_embed(sequence)  # c_{c_t}
        if self.separate_qa:
            # concept-response 联合索引：c + num_c * r
            qa_data = sequence + self.num_c * response
            qa_embed_data = self.qa_embed(qa_data)  # e_{r_t}（含 concept 信息）
        else:
            qa_embed_data = self.qa_embed(response) + q_embed_data

        if self.n_pid > 0 and pid_data is not None:
            q_diff = self.q_embed_diff(sequence)  # d_{c_t}
            qa_diff = self.qa_embed_diff(
                qa_data if self.separate_qa else response
            )  # f(c_t, r_t)
            mu = self.difficult_param(pid_data)  # μ_{q_t}

            # Q_t = c_{c_t} + μ_{q_t} · d_{c_t}
            q_embed_data = q_embed_data + mu * q_diff
            # R_t = e_{r_t} + μ_{q_t} · f(c_t, r_t)
            qa_embed_data = qa_embed_data + mu * qa_diff

            # 论文 Eq. 11：λ||μ||²
            c_reg_loss = (mu**2).sum() * self.l2

        return q_embed_data, qa_embed_data, c_reg_loss

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor = None,
        pid_data: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            sequence: 技能 ID 序列 [B, S]
            response: 响应序列 [B, S]
            mask: 有效位置掩码 [B, S]（未使用，保留接口）
            pid_data: 题目 ID 序列 [B, S]

        Returns:
            preds: next-item 约定下的预测概率 [B, S]，preds[t] 预测 response[t+1]
            c_reg_loss: Rasch 正则化损失
        """
        q_embed_data, qa_embed_data, c_reg_loss = self.embed(
            sequence, response, pid_data
        )

        # Mamba 处理交互序列
        h = qa_embed_data
        for block in self.blocks:
            h = block(h)

        # FFN
        f = self.ffn(h)

        # 引入待预测题目 q_{t+1}：将题目嵌入左移一位（最后一位置零占位）
        q_next = torch.roll(q_embed_data, shifts=-1, dims=1)
        q_next[:, -1] = 0.0

        # 论文 Eq. 10：p_t = Sigmoid(f' · W + b)，f' = concat(FFN 输出, q_{t+1} 嵌入)
        out = self.out(torch.cat([f, q_next], dim=-1)).squeeze(-1)
        preds = torch.sigmoid(out)

        return preds, c_reg_loss


__all__ = ["Mamba4KT"]
