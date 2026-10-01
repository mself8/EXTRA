#!/usr/bin/env bash
# 기준선 점수·정수계획·HIGFormer, 교체 정렬, 추천기 절제.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export FAMMIX=1
echo "== 학습 없는 기준선 점수 (VAEP 합·출전분·맥락)"
$PY -m experiments.baseline_scores
echo "== TeamBuilder 식 정수계획"
for s in 2024 2025; do
  PROD=outputs/gap_soft_fullonball_gk_resid_merged_fam_pc20_all_npxg_nozvzn_fixsd.pkl \
    EVALSEASON=$s OUT=outputs/eval_familp_$s.pkl SAVEREC=outputs/rec_familp_$s.pkl \
    $PY -m experiments.ilp_baseline
done
echo "== HIGFormer 재현"
$PY -m experiments.higformer_t1
echo "== 교체 정렬 (표 2 의 Substitution alignment)"
NBOOT=300 NPERM=200 $PY -m experiments.sub_alignment fam ridge mins vaepsum prev
PROTO=26 SEASONS=2025,2026 NBOOT=300 NPERM=200 \
  $PY -m experiments.sub_alignment fam26 ridge26 mins26 vaepsum26 prev
echo "== 추천기 절제 (무제약 · 정수계획)"
for s in 2024 2025; do
  PROD=outputs/gap_soft_fullonball_gk_resid_merged_fam_pc20_all_npxg_nozvzn_fixsd.pkl \
    HOLDOUT=$s EVALSEASON=$s LAM=0 SHAPEPRIOR=0 FCAP=4 FKAP=1 SEEDS=0 TOPK=6 NW=6 \
    SAVENULL=outputs/eval_famuncon_$s.pkl SAVEREC=outputs/rec_famuncon_$s.pkl \
    $PY -m experiments.role_balance_parallel
done
$PY -m experiments.pool_eval outputs/eval_famuncon_2024.pkl outputs/eval_famuncon_2025.pkl
