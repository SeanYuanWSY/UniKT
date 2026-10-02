"""Internal-signal extraction from frozen KT models.

Two signal families, both training-free:
- Generic: per-step predictions p_t on the evidence segment.
- Readiness scan: feed each candidate KC as a hypothetical next item and
  read the model's predicted correctness -> readiness(c). Unified across
  DKT/DKVMN/SAKT/AKT via a small forward adapter.
- DKVMN-only: memory-slot readout mastery proxy (declared as a self-made
  probe reusing the frozen weights).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from restore import RestoredModel, WindowSample


def _forward_probs(
    rm: RestoredModel, sequence: torch.Tensor, response: torch.Tensor
) -> np.ndarray:
    """Model-agnostic forward returning per-step predicted probabilities [B, S]."""
    name = rm.model_name
    if name in ("DKT", "DKVMN"):
        mask = torch.ones_like(sequence, dtype=torch.bool)
        out = rm.model(sequence, response, mask)
    elif name == "SAKT":
        out = rm.model(sequence, response)
    elif name == "AKT":
        mask = torch.ones_like(sequence, dtype=torch.bool)
        out = rm.model(sequence, response, mask, None)
        if isinstance(out, tuple):
            out = out[0]
    else:
        raise ValueError(f"Unsupported model for signal extraction: {name}")
    if isinstance(out, tuple):
        out = out[0]
    return out.detach().float().cpu().numpy()


def evidence_predictions(rm: RestoredModel, ws: WindowSample) -> np.ndarray:
    """Per-step predictions on the evidence segment (teacher-forced, p_t uses 0..t-1)."""
    ev = ws.evidence_idx
    seq = torch.tensor(ws.sequence[ev], dtype=torch.long, device=rm.device).unsqueeze(0)
    resp = torch.tensor(ws.response[ev], dtype=torch.long, device=rm.device).unsqueeze(0)
    probs = _forward_probs(rm, seq, resp)[0]  # [L], probs[t] predicts step t
    # Step 0 has no history; drop it. probs[1:] predicts evidence steps 1..L-1.
    return probs[1:]


def readiness_scan(
    rm: RestoredModel,
    ws: WindowSample,
    candidate_kcs: list[int],
    batch_size: int = 32,
) -> dict[int, float]:
    """Readiness(c): model's predicted correctness for a hypothetical next item of KC c.

    Appends c to the full evidence segment and reads the last-position
    prediction. The appended response value never influences that position's
    prediction (all four models predict position t from positions < t).
    """
    ev = ws.evidence_idx
    base_seq = ws.sequence[ev].tolist()
    base_resp = ws.response[ev].tolist()

    results: dict[int, float] = {}
    for i in range(0, len(candidate_kcs), batch_size):
        chunk = candidate_kcs[i : i + batch_size]
        seq = torch.tensor(
            [base_seq + [c] for c in chunk], dtype=torch.long, device=rm.device
        )
        resp = torch.tensor(
            [base_resp + [0] for _ in chunk], dtype=torch.long, device=rm.device
        )
        probs = _forward_probs(rm, seq, resp)[:, -1]
        for c, p in zip(chunk, probs):
            results[int(c)] = float(p)
    return results


def dkvmn_mastery(rm: RestoredModel, candidate_kcs: list[int]) -> dict[int, float]:
    """DKVMN-only mastery probe: replay the memory write with the student's evidence
    segment, then read out each KC via the model's own read/predict head.

    This reuses frozen weights but is a self-made probe (drops the
    question-dependence of the real head) -- reported as such.
    """
    if rm.model_name != "DKVMN":
        raise ValueError("dkvmn_mastery is DKVMN-only")
    raise NotImplementedError("wired after smoke test; see run_explain TODO")


def holdout_actuals(
    rm: RestoredModel, ws: WindowSample
) -> dict[int, dict[str, float]]:
    """Ground truth per KC on the holdout window: {kc: {attempts, acc, pred_mean}}.

    pred_mean uses the model run over evidence+holdout-so-far (positions
    before each holdout step) -- i.e. the model's own online view.
    """
    n = len(ws.sequence)
    seq = torch.tensor(ws.sequence, dtype=torch.long, device=rm.device).unsqueeze(0)
    resp = torch.tensor(ws.response, dtype=torch.long, device=rm.device).unsqueeze(0)
    probs = _forward_probs(rm, seq, resp)[0]

    per_kc: dict[int, dict[str, list[float]]] = {}
    for t in ws.holdout_idx:
        kc = int(ws.sequence[t])
        d = per_kc.setdefault(kc, {"labels": [], "preds": []})
        d["labels"].append(float(ws.response[t]))
        d["preds"].append(float(probs[t]))
    out: dict[int, dict[str, float]] = {}
    for kc, d in per_kc.items():
        out[kc] = {
            "attempts": float(len(d["labels"])),
            "acc": float(np.mean(d["labels"])),
            "pred_mean": float(np.mean(d["preds"])),
        }
    return out
