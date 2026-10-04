#!/bin/bash
# Transfer-matrix cells (stage 3). Fixed vs v3: grep patterns anchored on
# `.json` (the old `rep0$` never matched `rep0.json`), edit cache name carries
# the provider tag (`edit__glm__`). Run from the repo root inside pixi env.
#
# Usage: bash research/transfer/run_transfer_cells.sh [MAX_USERS]
set -u
P=${PIXI:-/root/.pixi/bin/pixi}
T=research/transfer/transfer_eval.py
O=research/transfer/results
K=research/transfer/align_cache
MU=${1:-300}

DKT9=runs/normal/DKT_assistments09_20261003-024700_fold0_bs128
AKT9=runs/normal/AKT_assistments09_20261003-025410_fold0_bs64

run_cell() {  # src_run src_tag target alignment_name out_suffix extra...
  local RUN=$1 STAG=$2 TGT=$3 ALN=$4 SUF=$5; shift 5
  local OUT=$O/mx_${STAG}_to_${TGT}_${SUF}.json
  if [ -f "$OUT" ]; then echo "skip (exists): $OUT"; return; fi
  $P run python $T --source-run "$RUN" --target-dataset "$TGT" \
      --alignment "$ALN" --out "$OUT" --max-users "$MU" "$@"
}

for PAIR in "$DKT9 DKT assistments17" "$AKT9 AKT assistments17" \
            "$DKT9 DKT junyi2015" "$AKT9 AKT junyi2015" \
            "$DKT9 DKT ednet_kt1" "$AKT9 AKT ednet_kt1"; do
  set -- $PAIR
  RUN=$1; STAG=$2; TGT=$3
  K3=$(ls $K 2>/dev/null | grep "^assistments09__to__${TGT}__llm__k3-high__c25k3__rep0\.json$" | head -1 | sed 's/\.json$//')
  K3B=$(ls $K 2>/dev/null | grep "^assistments09__to__${TGT}__llm__k3-high__c25k3__rep1\.json$" | head -1 | sed 's/\.json$//')
  G5=$(ls $K 2>/dev/null | grep "^assistments09__to__${TGT}__llm__glm-5.3__c25k3__rep0\.json$" | head -1 | sed 's/\.json$//')
  ED=$(ls $K 2>/dev/null | grep "^assistments09__to__${TGT}__edit__glm__c25k3__rep0\.json$" | head -1 | sed 's/\.json$//')
  EM=$(ls $K 2>/dev/null | grep "^assistments09__to__${TGT}__embed__bge_m3__rep0\.json$" | head -1 | sed 's/\.json$//')
  echo "== $STAG -> $TGT =="
  [ -n "$K3" ]  && run_cell "$RUN" "$STAG" "$TGT" "$K3"  k3high --b 399
  [ -n "$K3B" ] && run_cell "$RUN" "$STAG" "$TGT" "$K3B" k3high_rep1
  [ -n "$G5" ]  && run_cell "$RUN" "$STAG" "$TGT" "$G5"  glm53
  [ -n "$ED" ]  && run_cell "$RUN" "$STAG" "$TGT" "$ED"  edit
  [ -n "$EM" ]  && run_cell "$RUN" "$STAG" "$TGT" "$EM"  embed
done
echo "TRANSFER-CELLS-DONE $(date)"
