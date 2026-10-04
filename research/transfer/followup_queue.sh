#!/bin/bash
# Review-mandated follow-ups (X1/X4/X6), queued after the coverage ladder.
P=/root/.pixi/bin/pixi
cd /root/unikt-fork
T=research/transfer/transfer_eval.py
O=research/transfer/results

while ! grep -q "LADDER-ALL-DONE" /root/ladder.log 2>/dev/null; do sleep 120; done
echo "FOLLOWUP-START $(date)"

DKT=runs/normal/DKT_assistments09_20261003-024700_fold0_bs128
AKT=runs/normal/AKT_assistments09_20261003-025410_fold0_bs64

# X1: occupancy nulls for edit and embed (B=999)
$P run python $T --source-run $DKT --target-dataset assistments17 --alignment assistments09__to__assistments17__edit__glm__c25k3__rep0 --out $O/x1_dkt_edit_null.json --b 999 --max-users 264
$P run python $T --source-run $AKT --target-dataset assistments17 --alignment assistments09__to__assistments17__edit__glm__c25k3__rep0 --out $O/x1_akt_edit_null.json --b 999 --max-users 264
$P run python $T --source-run $DKT --target-dataset assistments17 --alignment assistments09__to__assistments17__embed__bge_m3__rep0 --out $O/x1_dkt_embed_null.json --b 999 --max-users 264
$P run python $T --source-run $AKT --target-dataset assistments17 --alignment assistments09__to__assistments17__embed__bge_m3__rep0 --out $O/x1_akt_embed_null.json --b 999 --max-users 264
echo "X1-DONE $(date)"

# X4: mechanism decomposition replicated on the k3-high alignment (prior_only
# test currently hardcodes the flash cache; run via env override below)
K3AL=assistments09__to__assistments17__llm__k3-high__c25k3__rep0 python3 - <<'PYEOF'
import os, sys, json
sys.path.insert(0, "/root/unikt-fork/research/transfer")
sys.path.insert(0, "/root/unikt-fork/research/llm_explain")
# reuse prior_only_test with a parameterised cache name
src = open("/root/unikt-fork/research/transfer/prior_only_test.py").read()
src = src.replace('assistments09__to__assistments17__llm__glm__rep0', os.environ["K3AL"])
exec(compile(src, "prior_only_k3high", "exec"))
PYEOF
echo "X4-DONE $(date)"
echo "FOLLOWUP-ALL-DONE $(date)"
