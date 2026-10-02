"""One-student full-pipeline smoke test (writes /tmp/smoke_out.txt)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from llm_client import call_llm  # noqa: E402
from evidence import build_pack  # noqa: E402
from faithfulness import verify_report  # noqa: E402
from prompts import build_messages  # noqa: E402
from restore import load_user_samples, restore  # noqa: E402
from signals import holdout_actuals  # noqa: E402

lines = []


def log(s: str = "") -> None:
    print(s, flush=True)


rm = restore("runs/normal/DKT_assistments09_20261003-024700_fold0_bs128")
samples = load_user_samples(rm.data_src, fold=0)
big = [s for s in samples if len({int(c) for c in s.sequence[s.evidence_idx]}) >= 10]
ws = big[len(big) // 2]
pack = build_pack(rm, ws)
call = call_llm("glm", build_messages(pack.to_markdown()), purpose="smoke", user_id=ws.user_id, condition="smoke")
log(f"parse_ok: {call.parse_ok} | attempts: {len(call.attempts)}")
if call.parse_ok:
    ver = verify_report(call.parsed, pack)
    log(f"claims: {ver.n_claims} | violations: {len(ver.violations)} | coverage: {ver.holdout_coverage} | weakest_valid: {ver.weakest_valid}")
    for v in ver.violations[:5]:
        log(f"  viol: {v}")
    log(f"summary: {call.parsed['summary'][:200]}")
    log(f"weakest: {call.parsed['weakest_kcs']}")
    actuals = holdout_actuals(rm, ws)
    preds = {int(p["kc"]): p["pred_acc"] for p in call.parsed["holdout_predictions"]}
    for k in pack.holdout_target_kcs[:8]:
        log(f"  KC {k}: llm={preds.get(k)}, actual_acc={actuals[k]['acc']:.2f}, model_pred={actuals[k]['pred_mean']:.2f}")
else:
    log(f"last attempt head: {call.attempts[-1] if call.attempts else 'none'}")
