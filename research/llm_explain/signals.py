"""Internal-signal extraction from frozen KT models.

Two signal families, both training-free:
- Generic: per-step predictions p_t on the evidence segment.
- Readiness scan: feed each candidate KC as a hypothetical next item and
  read the model's predicted correctness -> readiness(c).

Model-specific forward contracts (verified against each model's source):
- DKT/DKVMN: forward(sequence, response, mask) -> [B, S]; out[t] predicts
  response[t] from strictly earlier steps; out[:, 0] is a dummy.
- SAKT: forward(sequence, response) -> [B, S-1]; out[:, j] predicts
  response[:, j+1]. We left-pad a dummy column to restore the [B, S]
  "out[t] <-> response[t]" alignment.
- AKT: forward(sequence, response, mask=None, pid_data=None) -> tuple;
  when the run uses Rasch embeddings (n_pid > 0) training passed
  pid_data = question + 1, so we do the same; readiness uses pid 0 for the
  virtual next item (padding row: zero modulation, documented).
"""

from __future__ import annotations

import numpy as np
import torch

from restore import RestoredModel, WindowSample


def _forward_probs(
    rm: RestoredModel,
    sequence: torch.Tensor,
    response: torch.Tensor,
    question: torch.Tensor | None = None,
) -> np.ndarray:
    """Model-agnostic forward returning per-step predicted probabilities [B, S].

    Column convention after this adapter: out[:, t] predicts response[:, t]
    using strictly earlier steps; column 0 is a dummy.
    """
    name = rm.model_name
    with torch.no_grad():
        if name in ("DKT", "DKVMN"):
            mask = torch.ones_like(sequence, dtype=torch.bool)
            out = rm.model(sequence, response, mask)
        elif name == "SAKT":
            out = rm.model(sequence, response)  # [B, S-1]
            dummy = torch.zeros(out.shape[0], 1, dtype=out.dtype, device=out.device)
            out = torch.cat([dummy, out], dim=1)  # -> [B, S], aligned
        elif name == "AKT":
            mask = torch.ones_like(sequence, dtype=torch.bool)
            pid = None
            if question is not None and getattr(rm.model, "n_pid", 0):
                pid = question + 1  # training-time convention (0 reserved for padding)
            out = rm.model(sequence, response, mask, pid)
            if isinstance(out, tuple):
                out = out[0]
        else:
            raise ValueError(f"Unsupported model for signal extraction: {name}")
    if isinstance(out, tuple):
        out = out[0]
    return out.detach().float().cpu().numpy()


def _pid_for_virtual(rm: RestoredModel) -> torch.Tensor | None:
    """AKT Rasch pid for a hypothetical next item: 0 (padding row, no modulation)."""
    if rm.model_name == "AKT" and getattr(rm.model, "n_pid", 0):
        return torch.zeros(1, dtype=torch.long, device=rm.device)
    return None


def evidence_predictions(rm: RestoredModel, ws: WindowSample) -> np.ndarray:
    """Per-step predictions on the evidence segment (teacher-forced, p_t uses 0..t-1)."""
    ev = ws.evidence_idx
    seq = torch.tensor(ws.sequence[ev], dtype=torch.long, device=rm.device).unsqueeze(0)
    resp = torch.tensor(ws.response[ev], dtype=torch.long, device=rm.device).unsqueeze(0)
    q = torch.tensor(ws.question[ev], dtype=torch.long, device=rm.device).unsqueeze(0)
    probs = _forward_probs(rm, seq, resp, q)[0]  # [L]; probs[t] predicts step t
    return probs[1:]  # step 0 has no history: drop the dummy


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
    base_q = ws.question[ev].tolist()
    q_tensor = torch.tensor(base_q, dtype=torch.long, device=rm.device).unsqueeze(0)
    virtual_pid = _pid_for_virtual(rm)

    results: dict[int, float] = {}
    for i in range(0, len(candidate_kcs), batch_size):
        chunk = candidate_kcs[i : i + batch_size]
        b = len(chunk)
        seq = torch.tensor(
            [base_seq + [c] for c in chunk], dtype=torch.long, device=rm.device
        )
        resp = torch.tensor(
            [base_resp + [0] for _ in chunk], dtype=torch.long, device=rm.device
        )
        q = q_tensor.expand(b, -1)
        if virtual_pid is not None:
            # history keeps real pids; the virtual last item gets pid 0
            q = torch.cat([q, virtual_pid.expand(b, 1)], dim=1)
        probs = _forward_probs(rm, seq, resp, q)[:, -1]
        for c, p in zip(chunk, probs):
            results[int(c)] = float(p)
    return results


def dkvmn_mastery(rm: RestoredModel, candidate_kcs: list[int]) -> dict[int, float]:
    """DKVMN-only mastery probe (self-made, reuses frozen weights)."""
    if rm.model_name != "DKVMN":
        raise ValueError("dkvmn_mastery is DKVMN-only")
    raise NotImplementedError("deferred; not part of frozen holdout protocol")


def holdout_actuals(
    rm: RestoredModel, ws: WindowSample
) -> dict[int, dict[str, float]]:
    """Ground truth per KC on the holdout window: {kc: {attempts, acc, pred_mean}}.

    pred_mean uses the model run over the full sequence (online view: each
    holdout step is predicted from everything before it). Never enters prompts.
    """
    seq = torch.tensor(ws.sequence, dtype=torch.long, device=rm.device).unsqueeze(0)
    resp = torch.tensor(ws.response, dtype=torch.long, device=rm.device).unsqueeze(0)
    q = torch.tensor(ws.question, dtype=torch.long, device=rm.device).unsqueeze(0)
    probs = _forward_probs(rm, seq, resp, q)[0]

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
