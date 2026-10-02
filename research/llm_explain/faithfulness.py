"""Grounding verification and perturbation probes (design v2).

Cell-level citation checks (kc+field+value must match exactly after
normalisation), KC-membership in the student's own E1 rows, holdout
prediction sanity -- all scored on the FIRST attempt. Plus constructors for
perturbation probes (permuted readiness, swapped packs, contradictions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evidence import EvidencePack

FIELD_ACCESS = {
    "尝试次数": "attempts",
    "正确率": "acc",
    "模型预测均值": "pred_mean",
    "模型就绪度": "readiness",
    "最近5次正确率": "recent5",
}


@dataclass
class Verification:
    """Per-report verification outcome."""

    n_claims: int
    violations: list[dict[str, Any]]
    holdout_coverage: float  # fraction of target KCs predicted
    weakest_valid: bool

    @property
    def violation_rate(self) -> float:
        return len(self.violations) / self.n_claims if self.n_claims else 0.0


def _fmt(value) -> str:
    if value is None:
        return "—"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return "—"
    if fv != fv:  # NaN
        return "—"
    return f"{round(fv, 2):.2f}"


def verify_report(parsed: dict[str, Any], pack: EvidencePack) -> Verification:
    """Cell-level grounding check of one parsed LLM report."""
    violations: list[dict[str, Any]] = []
    claims = parsed.get("claims", [])
    if not isinstance(claims, list) or not claims:
        violations.append({"type": "no_claims", "detail": "claims 缺失或为空"})

    for ci, claim in enumerate(claims or []):
        if not isinstance(claim, dict):
            violations.append({"type": "malformed_claim", "claim_index": ci})
            continue
        cits = claim.get("citations", [])
        if not cits:
            violations.append(
                {"type": "uncited_claim", "claim_index": ci, "statement": claim.get("statement", "")[:80]}
            )
            continue
        for cit in cits:
            kc, fld, val = cit.get("kc"), cit.get("field"), cit.get("value")
            row = pack.kc_row(int(kc)) if kc is not None else None
            if row is None:
                violations.append({"type": "kc_not_in_e1", "claim_index": ci, "kc": kc})
                continue
            attr = FIELD_ACCESS.get(fld)
            if attr is None:
                violations.append({"type": "unknown_field", "claim_index": ci, "field": fld})
                continue
            actual = getattr(row, attr)
            if attr == "attempts":
                ok = int(val) == int(actual)
            else:
                ok = _fmt(val) == _fmt(actual)
            if not ok:
                violations.append(
                    {
                        "type": "value_mismatch",
                        "claim_index": ci,
                        "kc": kc,
                        "field": fld,
                        "claimed": val,
                        "actual": _fmt(actual) if attr != "attempts" else int(actual),
                    }
                )

    preds = parsed.get("holdout_predictions", [])
    pred_kcs = {p.get("kc") for p in preds if isinstance(p, dict)}
    target = set(pack.holdout_target_kcs)
    coverage = (len(pred_kcs & target) / len(target)) if target else 1.0

    weakest = parsed.get("weakest_kcs", [])
    weakest_valid = isinstance(weakest, list) and all(
        (w in {r.kc for r in pack.kcs}) for w in weakest
    )

    return Verification(
        n_claims=len(claims),
        violations=violations,
        holdout_coverage=coverage,
        weakest_valid=weakest_valid,
    )


# ---------- Perturbation probes ----------


def permute_readiness(pack: EvidencePack, rng: Any) -> EvidencePack:
    """Shuffle the readiness column across KCs (format/length unchanged)."""
    values = [r.readiness for r in pack.kcs]
    rng.shuffle(values)
    for r, v in zip(pack.kcs, values):
        r.readiness = v
    return pack


def inject_contradiction(pack: EvidencePack) -> EvidencePack:
    """Make E3 global accuracy contradict E1 rows (mean of rows != global)."""
    pack.global_stats["acc"] = 0.99
    return pack


def blank_fields(pack: EvidencePack, fields: list[str]) -> EvidencePack:
    """Blank out fields to probe fabrication on missing data."""
    for r in pack.kcs:
        for f in fields:
            setattr(r, f, float("nan"))
    return pack
