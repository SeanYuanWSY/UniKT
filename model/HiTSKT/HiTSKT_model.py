"""HiTSKT's hierarchical action, session, and response-shifted attention model."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq: torch.Tensor) -> torch.Tensor:
    length = seq.size(1)
    return torch.ones((1, length, length), device=seq.device, dtype=torch.bool).tril()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e32)
        attention = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attention, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout, encoder_type):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.encoder_type = encoder_type
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k**0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, time_affect=False):
        action = self.encoder_type == "Action"
        batch_size = q.size(0)
        length = q.size(2) if action else q.size(1)
        session_size = q.size(1) if action else None
        residual = q
        shape = (
            (batch_size, session_size, length, self.n_head, self.d_k)
            if action
            else (batch_size, length, self.n_head, self.d_k)
        )
        value_shape = (*shape[:-1], self.d_v)
        axis = 2 if action else 1
        q = self.w_qs(q).view(shape).transpose(axis, axis + 1)
        k = self.w_ks(k).view(shape).transpose(axis, axis + 1)
        v = self.w_vs(v).view(value_shape).transpose(axis, axis + 1)
        if mask is not None:
            mask = mask.unsqueeze(axis)
        if time_affect:
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
            expanded = mask.squeeze(1).to(dtype=torch.int64).expand(-1, length, length)
            cumulative = torch.cumsum(expanded, dim=-1)
            distance = (cumulative - cumulative.transpose(1, 2)).abs().float()
            scores = scores * (1 / (distance[:, None] + 1)).detach()
            scores = scores.masked_fill(mask == 0, -1e32)
            attention = self.dropout(F.softmax(scores, dim=-1))
            q = torch.matmul(attention, v)
        else:
            q = self.attention(q, k, v, mask)
        q = q.transpose(axis, axis + 1).contiguous()
        q = (
            q.view(batch_size, session_size, length, -1)
            if action
            else q.view(batch_size, length, -1)
        )
        q = self.dropout(self.fc(q))
        return self.layer_norm(q + residual)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.layer_norm(self.dropout(self.w_2(F.relu(self.w_1(x)))) + x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout, encoder_type):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout, encoder_type
        )
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, x, mask=None, time_affect=False):
        return self.pos_ffn(self.slf_attn(x, x, x, mask, time_affect))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, n_position, encoder_type):
        super().__init__()
        positions = np.arange(n_position)[:, None]
        dimensions = np.arange(d_model)[None, :]
        table = positions / np.power(10000, 2 * (dimensions // 2) / d_model)
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        self.register_buffer("pos_table", torch.FloatTensor(table).unsqueeze(0))
        self.encoder_type = encoder_type

    def forward(self, x):
        if self.encoder_type == "Action":
            return x + self.pos_table[:, None, : x.size(2)]
        return x + self.pos_table[:, : x.size(1)]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout):
        super().__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout, "Dec")
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, x, encoded, mask):
        return self.pos_ffn(self.enc_attn(x, encoded, encoded, mask))


class ActionEncoder(nn.Module):
    def __init__(
        self,
        n_problem,
        n_skill,
        n_correct,
        n_qno,
        d_model,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_inner,
        dropout,
        n_position,
    ):
        super().__init__()
        self.problem_emb = nn.Embedding(n_problem, d_model)
        self.skill_emb = nn.Embedding(n_skill, d_model)
        self.correctness_emb = nn.Embedding(n_correct, d_model)
        self.qno_emb = nn.Embedding(n_qno, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position, "Action")
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList(
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, "Action")
            for _ in range(n_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, attention_mask):
        embedded = self.correctness_emb(x[:, :, 2]) + self.problem_emb(x[:, :, 0])
        embedded = embedded + self.skill_emb(x[:, :, 1]) + self.qno_emb(x[:, :, 3])
        x = self.layer_norm(self.dropout(self.position_enc(embedded)))
        for layer in self.layer_stack:
            x = layer(x, attention_mask)
        return x


class SessionEncoder(nn.Module):
    def __init__(
        self, d_model, n_layers, n_head, d_k, d_v, d_inner, dropout, n_position
    ):
        super().__init__()
        self.position_enc = PositionalEncoding(d_model, n_position, "Session")
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList(
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, "Session")
            for _ in range(n_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        x = self.layer_norm(self.dropout(self.position_enc(x)))
        for layer in self.layer_stack:
            x = layer(x, mask, time_affect=True)
        return x


class CorrectPadEncoder(nn.Module):
    def __init__(
        self, d_model, n_layers, n_head, d_k, d_v, d_inner, dropout, n_position
    ):
        super().__init__()
        self.position_enc = PositionalEncoding(d_model, n_position, "PC")
        self.cor_pad_emb = nn.Embedding(6, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList(
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, "PC")
            for _ in range(n_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, correct_pad, mask):
        x = self.layer_norm(
            self.dropout(self.position_enc(x + self.cor_pad_emb(correct_pad)))
        )
        for layer in self.layer_stack:
            x = layer(x, mask)
        return x


class HierarchicalDecoder(nn.Module):
    def __init__(
        self,
        n_problem,
        n_skill,
        n_qno,
        d_model,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_inner,
        n_position,
    ):
        super().__init__()
        self.problem_emb = nn.Embedding(n_problem, d_model)
        self.skill_emb = nn.Embedding(n_skill, d_model)
        self.qno_emb = nn.Embedding(n_qno, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position, "Dec")
        self.dropout = nn.Dropout(0.1)
        self.layer_stack = nn.ModuleList(
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, 0.1)
            for _ in range(n_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, problem, skill, qno, encoded, mask):
        x = self.problem_emb(problem) + self.skill_emb(skill) + self.qno_emb(qno)
        x = self.layer_norm(self.dropout(self.position_enc(x)))
        for layer in self.layer_stack:
            x = layer(x, encoded, mask)
        return x


class HiTSKT(nn.Module):
    """Source-equivalent HiTSKT network on packed [B, S+1, 5, A] inputs."""

    def __init__(
        self,
        n_problem,
        n_skill,
        n_qno,
        action_size=64,
        session_size=16,
        d_model=256,
        d_inner=2048,
        n_layers=1,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
    ):
        super().__init__()
        self.session_size = session_size
        self.eos_skill = n_skill - 2
        self.fc_layer1 = nn.Linear(d_model, 1)
        self.actionEncoder = ActionEncoder(
            n_problem,
            n_skill,
            6,
            n_qno,
            d_model,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_inner,
            dropout,
            action_size,
        )
        self.sessionEncoder = SessionEncoder(
            d_model, n_layers, n_head, d_k, d_v, d_inner, dropout, session_size
        )
        self.PCEncoder = CorrectPadEncoder(
            d_model, n_layers, n_head, d_k, d_v, d_inner, dropout, action_size
        )
        self.my_decoder = HierarchicalDecoder(
            n_problem,
            n_skill,
            n_qno,
            d_model,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_inner,
            action_size,
        )

    def forward(self, input_array):
        session_size = self.session_size
        history = input_array[:, :session_size]
        target = input_array[:, session_size]
        action_mask = get_pad_mask(history[:, :, 2], 2)
        action_out = self.actionEncoder(history, action_mask)
        session_input = action_out[:, :, -1, :]
        session_mask = get_subsequent_mask(history[:, :, 2, 0]) & get_pad_mask(
            history[:, :, 2, 0], 2
        )
        session_out = self.sessionEncoder(session_input, session_mask)
        correct_pad = target[:, 4]
        encoded = self.PCEncoder(
            session_out[:, -1].unsqueeze(1),
            correct_pad,
            get_subsequent_mask(correct_pad),
        )
        target_mask = (
            get_subsequent_mask(target[:, 0])
            & get_pad_mask(target[:, 1], 0)
            & get_pad_mask(target[:, 1], self.eos_skill)
        )
        decoded = self.my_decoder(
            target[:, 0], target[:, 1], target[:, 3], encoded, target_mask
        )
        return torch.sigmoid(self.fc_layer1(decoded).squeeze(-1))
