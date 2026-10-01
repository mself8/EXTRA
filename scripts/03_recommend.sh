#!/usr/bin/env bash
# 학습된 모델 → 선수 점수 → 제약 추천 → 격차 효과·위약·24시드 셔플 귀무.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export FAMMIX=1
RIDGE_P1=outputs/gap_soft_fullonball_gk_resid_merged_defresp_ref_pc20_all_npxg_nozvzn_fixsd_ssn.pkl
RIDGE_P2=outputs/gap_soft_fullonball_gk_resid_merged_defresp_ref_pc20_all_npxg_nozvzn_fixsd_ssn20252026.pkl

score () {  # $1 태그  $2 릿지 기준 pkl  $3 출력 이름
  BLEND=0 TAG="$1" PROD="$2" \
    OUT="outputs/gap_soft_fullonball_gk_resid_merged_$3_pc20_all_npxg_nozvzn_fixsd.pkl" \
    $PY -m experiments.set_score
}
recommend () {  # $1 점수 pkl 이름  $2 시즌들
  local C="outputs/gap_soft_fullonball_gk_resid_merged_$1_pc20_all_npxg_nozvzn_fixsd.pkl"
  for s in $2; do
    PROD="$C" HOLDOUT=$s EVALSEASON=$s LAM=0.01 SHAPEPRIOR=0.1 FCAP=4 FKAP=1 \
      SEEDS=${NULLSEEDS:-0} TOPK=6 NW=${NW:-6} \
      SAVENULL="outputs/eval_$1_add_$s.pkl" SAVEREC="outputs/rec_$1_$s.pkl" \
      $PY -m experiments.role_balance_parallel
  done
}
score ssn-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-s5           "$RIDGE_P1" fam
score ssn20252026-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-s5   "$RIDGE_P2" fam26
recommend fam   "2024 2025"
recommend fam26 "2025 2026"
$PY -m experiments.pool_eval outputs/eval_fam_add_2024.pkl   outputs/eval_fam_add_2025.pkl
$PY -m experiments.pool_eval outputs/eval_fam26_add_2025.pkl outputs/eval_fam26_add_2026.pkl
echo "== 24시드 셔플 귀무 (최종 구성에만; 오래 걸립니다)"
NULLSEEDS=24 recommend fam "2024 2025"
$PY -m experiments.pool_eval outputs/eval_fam_add_2024.pkl outputs/eval_fam_add_2025.pkl
