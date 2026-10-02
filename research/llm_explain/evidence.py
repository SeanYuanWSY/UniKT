"""Evidence pack construction: auditable per-student evidence for LLM prompts.

Design (v2): rows addressable by KC dense id, values rounded to 2 decimals,
row order fixed by KC id, evidence segment only (holdout never enters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from restore import RestoredModel, WindowSample
from signals import evidence_predictions, readiness_scan


@dataclass
class KCStats:
    """Per-KC evidence statistics (evidence segment only)."""

    kc: int
    name: str
    attempts: int
    acc: float
    pred_mean: float
    readiness: float
    recent5: float  # correctness of the last up-to-5 attempts on this KC


@dataclass
class EvidencePack:
    """Addressable evidence for one student."""

    user_id: int
    kcs: list[KCStats]
    anomalies: list[dict[str, Any]]  # top-3 |p_t - y_t| steps in evidence
    global_stats: dict[str, float]
    holdout_target_kcs: list[int]  # KCs learned in evidence AND retested in holdout
    model_name: str
    extra: dict[str, Any] = field(default_factory=dict)

    def kc_row(self, kc: int) -> KCStats | None:
        for row in self.kcs:
            if row.kc == kc:
                return row
        return None

    def to_markdown(self) -> str:
        lines = [
            f"学生 ID: U{self.user_id}（冻结模型: {self.model_name}）",
            "",
            "E1 已学知识点统计（evidence 段，行 = KC ID）：",
            "",
            "| KC | 名称 | 尝试次数 | 正确率 | 模型预测均值 | 模型就绪度 | 最近5次正确率 |",
            "|----|------|---------|--------|-------------|-----------|--------------|",
        ]
        for r in sorted(self.kcs, key=lambda x: x.kc):
            lines.append(
                f"| {r.kc} | {r.name} | {r.attempts} | {r.acc:.2f} | {r.pred_mean:.2f} | {r.readiness:.2f} | {r.recent5:.2f} |"
            )
        lines.append("")
        lines.append("E2 模型意外事件（|模型预测 − 实际| 最大的步骤）：")
        for a in self.anomalies:
            lines.append(
                f"- 步骤{a['step']}: KC {a['kc']}({a['name']}) 模型预测 {a['pred']:.2f}，实际{'对' if a['label'] else '错'}"
            )
        g = self.global_stats
        lines.append("")
        lines.append(
            f"E3 全局：共 {g['steps']:.0f} 步，总体正确率 {g['acc']:.2f}，"
            f"模型平均预测 {g['pred_mean']:.2f}，模型对该生的 AUC {g['auc']:.2f}"
        )
        lines.append("")
        lines.append(
            f"待预测知识点（将出现在后续练习中）: {', '.join(str(k) for k in self.holdout_target_kcs)}"
        )
        return "\n".join(lines)


def _recent5(labels: np.ndarray) -> float:
    return float(labels[-5:].mean()) if len(labels) else 0.0


def build_pack(rm: RestoredModel, ws: WindowSample, top_anomalies: int = 3) -> EvidencePack:
    """Build the evidence pack for one student from the frozen model's signals."""
    ev = ws.evidence_idx
    seq, resp = ws.sequence[ev], ws.response[ev]

    preds = evidence_predictions(rm, ws)  # [L-1] predicting steps 1..L-1
    pred_at = np.concatenate([[np.nan], preds])  # aligned to evidence steps

    # Model AUC on the evidence segment (steps with a prediction)
    ys, ps = resp[1:], preds
    if len(ys) > 0 and 0 < ys.mean() < 1:
        order = np.argsort(ps)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(ps) + 1)
        n1, n0 = ys.sum(), (1 - ys).sum()
        pos = ranks[ys == 1].sum()
        auc = float((pos - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
    else:
        auc = float("nan")

    kcs = sorted(set(seq.tolist()))
    readiness = readiness_scan(rm, ws, kcs)

    rows: list[KCStats] = []
    for c in kcs:
        m = seq == c
        rows.append(
            KCStats(
                kc=int(c),
                name=rm.skill_names.get(int(c), f"KC {c}"),
                attempts=int(m.sum()),
                acc=float(resp[m].mean()),
                pred_mean=float(np.nanmean(pred_at[m])),
                readiness=float(readiness[int(c)]),
                recent5=_recent5(resp[m]),
            )
        )

    abs_err = np.abs(preds - ys)
    order = np.argsort(-abs_err)[:top_anomalies]
    anomalies = [
        {
            "step": int(ev[1:][i]),
            "kc": int(seq[1:][i]),
            "name": rm.skill_names.get(int(seq[1:][i]), f"KC {seq[1:][i]}"),
            "pred": float(preds[i]),
            "label": int(ys[i]),
        }
        for i in order
    ]

    hold_kcs = sorted(
        {int(ws.sequence[t]) for t in ws.holdout_idx} & set(kcs)
    )

    return EvidencePack(
        user_id=ws.user_id,
        kcs=rows,
        anomalies=anomalies,
        global_stats={
            "steps": float(len(ev)),
            "acc": float(resp.mean()),
            "pred_mean": float(preds.mean()),
            "auc": round(auc, 2),
        },
        holdout_target_kcs=hold_kcs,
        model_name=rm.model_name,
    )
