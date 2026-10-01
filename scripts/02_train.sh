#!/usr/bin/env bash
# 결과 모델(EventXI) 학습 — 두 규약, 1단계(가산) → 2단계(집합 블록).
# 출력: outputs/player_encoder_set_<tag>_fold{1,2}.pt
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
BASE="FOLDS=season FULLCH=1 FAMMIX=1 DTDAYS=1 LANE=none H=16 SEEDS=5 \
ONBALL_FILE=onball_gk_resid_merged_defresp_ref.parquet"

for TS in 2024,2025 2025,2026; do
  if [ "$TS" = "2024,2025" ]; then PFX="ssn-"; else PFX="ssn$(echo "$TS" | tr -d ',')-"; fi
  STAGE1="${PFX}set0-cross0-H16-L1-none-full-days-fam5-s5"
  echo "== $TS 1단계 (가산 점수 머리)"
  env $BASE TESTSEASONS="$TS" SET=0 SAVE=1 TUNETAG=-fam5 $PY -m experiments.player_encoder_set
  echo "== $TS 2단계 (선수 집합 블록, 1단계 동결)"
  env $BASE TESTSEASONS="$TS" SET=1 SETSTAGE=2 SAVE=1 TUNETAG=-fam5 WARMTAG="$STAGE1" \
      $PY -m experiments.player_encoder_set
done
