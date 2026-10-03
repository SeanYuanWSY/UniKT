"""Zero-training transfer evaluation: frozen source model on target dataset.

Pipeline per target student: map target KC sequence to source KC space via an
alignment, run the frozen source model, score holdout responses with
within-student AUC. Includes occupancy-matched random null with Monte-Carlo
permutation (B=999) and the frozen-LLM direct baseline hook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "llm_explain"))
sys.path.insert(0, str(_HERE.parents[1]))

import torch  # noqa: E402

from kc_align import load_alignment, random_occupancy_matched  # noqa: E402
from kc_tables import build_kc_table  # noqa: E402
from llm_explain_restore_shim import restore_any  # noqa: E402


def stratified_auc_rows(rows: list[dict], pred_key: str) -> float | None:
    """Within-student AUC over one student's holdout rows."""
    ys = np.array([r["y"] for r in rows])
    vs = np.array([r[pred_key] for r in rows], dtype=float)
    ok = vs == vs
    ys, vs = ys[ok], vs[ok]
    if (ys == 1).sum() == 0 or (ys == 0).sum() == 0:
        return None
    num = den = 0.0
    for vp in vs[ys == 1]:
        for vn in vs[ys == 0]:
            num += 1.0 if vp > vn else (0.5 if vp == vn else 0.0)
            den += 1
    return num / den if den else None


def forward_mapped(
    rm, seq_mapped: list[int], resp: list[int], q_mapped: list[int] | None
) -> np.ndarray:
    """Frozen source model forward on a mapped sequence; [L] per-step probs."""
    s = torch.tensor([seq_mapped], dtype=torch.long, device=rm.device)
    r = torch.tensor([resp], dtype=torch.long, device=rm.device)
    q = (
        torch.tensor([q_mapped], dtype=torch.long, device=rm.device)
        if q_mapped is not None
        else None
    )
    from signals import _forward_probs  # noqa: PLC0415

    return _forward_probs(rm, s, r, q)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True, help="frozen source run_dir")
    ap.add_argument("--target-dataset", required=True)
    ap.add_argument("--alignment", required=True, help="alignment cache name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=999)
    ap.add_argument("--max-users", type=int, default=0)
    args = ap.parse_args()

    rm = restore_any(args.source_run)
    alignment = {
        int(k): v for k, v in load_alignment(args.alignment).items()
    }  # target_kc -> [{source_kc, score, relation}]

    # Load target users through the target dataset's own pipeline
    import argparse as _ap  # noqa: PLC0415

    from restore import load_user_samples  # noqa: PLC0415
    from utils.data_process import get_data_source  # noqa: PLC0415

    ns = _ap.Namespace(
        data_base_path="./data", seed=42, data=_ap.Namespace(dataset=args.target_dataset)
    )
    tgt_src = get_data_source(ns)
    samples = load_user_samples(tgt_src, fold=0)
    if args.max_users:
        samples = samples[: args.max_users]

    # Per-student rows: map sequence, forward, collect holdout (y, pred_llm_map)
    students = []
    n_dropped = n_total = 0
    for ws in samples:
        mapping = {}
        for kc in set(ws.sequence.tolist()):
            ms = alignment.get(int(kc), [])
            mapping[int(kc)] = ms[0]["source_kc"] if ms else None
        keep = [i for i, kc in enumerate(ws.sequence.tolist()) if mapping[int(kc)] is not None]
        n_total += len(ws.sequence)
        n_dropped += len(ws.sequence) - len(keep)
        if len(keep) < 6:
            continue
        seq_m = [mapping[int(ws.sequence[i])] for i in keep]
        resp_m = [int(ws.response[i]) for i in keep]
        holdout_pos = set(ws.holdout_idx.tolist())
        probs = forward_mapped(rm, seq_m, resp_m, None)
        rows = []
        ev_len = 0
        for j, i in enumerate(keep):
            if i in holdout_pos:
                rows.append({"y": int(ws.response[i]), "pred": float(probs[j])})
            else:
                ev_len += 1
        if ev_len >= 3 and 0 < sum(r["y"] for r in rows) < len(rows):
            students.append(rows)

    aw = np.mean(
        [a for a in (stratified_auc_rows(s, "pred") for s in students) if a is not None]
    )
    result = {
        "source_run": args.source_run,
        "target": args.target_dataset,
        "alignment": args.alignment,
        "n_students": len(students),
        "drop_rate": round(n_dropped / max(n_total, 1), 4),
        "within_student_auc": round(float(aw), 4),
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
