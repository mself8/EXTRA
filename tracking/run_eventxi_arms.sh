#!/usr/bin/env bash
# EventXI 세 팔 — scripts/02_train.sh 의 통합 규약(보류 2024·2025) 그대로, 1단계(가산) → 2단계(선수 집합 블록)
#   A: 이벤트만 · T: + 트래킹 페이즈 행(6) · P: + 트래킹 합친 행(1, 대조군)
# 저장소 루트에서 돌고, 산출물은 outputs/ 에 쓴다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=logs; mkdir -p "$LOG"
TRKF=outputs/trk_channels.parquet
BASE="FOLDS=season FULLCH=1 FAMMIX=1 DTDAYS=1 LANE=none H=16 SEEDS=${SEEDS:-5} ONBALL_FILE=onball_gk_resid_merged_defresp_ref.parquet TESTSEASONS=2024,2025"

arm () {  # $1 이름  $2 GPU  $3 추가 환경
  local T="-fam5-ssac$1"; local S1="ssn-set0-cross0-H16-L1-none-full-days${T}-s${SEEDS:-5}"
  env $BASE $3 CUDA_VISIBLE_DEVICES=$2 SET=0 SAVE=1 TUNETAG=$T $PY -W ignore -m experiments.player_encoder_set > "$LOG/$1_stage1.log" 2>&1
  env $BASE $3 CUDA_VISIBLE_DEVICES=$2 SET=1 SETSTAGE=2 SAVE=1 TUNETAG=$T WARMTAG=$S1 $PY -W ignore -m experiments.player_encoder_set > "$LOG/$1_stage2.log" 2>&1
}

arm A ${GPU_A:-0} "" &
arm T ${GPU_B:-2} "TRK=1 TRKMODE=phase TRKFILE=$TRKF" &
arm P ${GPU_C:-3} "TRK=1 TRKMODE=all TRKFILE=$TRKF" &
wait
echo "done $(date '+%F %T')" > "$LOG/ALL_DONE"
