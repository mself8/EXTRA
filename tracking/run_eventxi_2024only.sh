#!/usr/bin/env bash
# EventXI 2024-only 학습 — run_eventxi_arms.sh 와 같은 규약(2025 보류, 5시드, 1단계 → 2단계), 학습 대상 시즌만 2024 로 제한
#   setting A (tag ssac24*):  TRAINFROM=2024 — 학습·검증 경기는 2024, 선수 이력은 2021년부터 그대로(2021–23 트래킹은 0 + trk_has=0)
#   setting B (tag ssac24h*), used in the abstract: TRAINFROM=2024 HISTFROM=2024 — 이력 토큰도 2024년 이후 경기만 (0으로 채운 트래킹 이력 제거)
#   팔: A 이벤트만 · T + 트래킹 페이즈 행(6)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=logs; mkdir -p "$LOG"
TRKF=outputs/trk_channels.parquet
BASE="FOLDS=season FULLCH=1 FAMMIX=1 DTDAYS=1 LANE=none H=16 SEEDS=5 ONBALL_FILE=onball_gk_resid_merged_defresp_ref.parquet TESTSEASONS=2025 TRAINFROM=2024"

arm () {  # $1 태그  $2 GPU  $3 추가 환경
  local T="-fam5-$1"; local S1="ssn2025-set0-cross0-H16-L1-none-full-days${T}-s5"
  env $BASE $3 CUDA_VISIBLE_DEVICES=$2 SET=0 SAVE=1 TUNETAG=$T $PY -W ignore -m experiments.player_encoder_set > "$LOG/$1_stage1.log" 2>&1
  env $BASE $3 CUDA_VISIBLE_DEVICES=$2 SET=1 SETSTAGE=2 SAVE=1 TUNETAG=$T WARMTAG=$S1 $PY -W ignore -m experiments.player_encoder_set > "$LOG/$1_stage2.log" 2>&1
}

TRK="TRK=1 TRKMODE=phase TRKFILE=$TRKF"
arm ssac24A ${GPU_A:-1} "" &
arm ssac24T ${GPU_B:-3} "$TRK" &
arm ssac24hA ${GPU_A:-1} "HISTFROM=2024" &
arm ssac24hT ${GPU_B:-3} "HISTFROM=2024 $TRK" &
wait
echo "done $(date '+%F %T')" > "$LOG/ALL_DONE_2024only"
