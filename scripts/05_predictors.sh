#!/usr/bin/env bash
# 선발 예측(EventXI-Select·DraftRec·선수별 분류기)과 포메이션 예측(EventXI-Form).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export FAMMIX=1
TAG=ssn20252026-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-s5
FULL=outputs/gap_soft_fullonball_gk_resid_merged_famfull_pc20_all_npxg_nozvzn_fixsd.pkl
echo "== 전 시즌 점수 (선발 프레임 학습에 필요)"
BLEND=0 TAG=$TAG \
  PROD=outputs/gap_soft_fullonball_gk_resid_merged_defresp_ref_pc20_all_npxg_nozvzn_fixsd.pkl \
  OUT="$FULL" $PY -m experiments.set_score
echo "== 선수별 분류기·규칙 기준선 + 포메이션 (이력 20경기)"
TAG=$TAG SCPKL="$FULL" FORMK=20 DUMP_E=outputs/t3_frame_fam.parquet \
  $PY -m experiments.multitask_heads
echo "== 드래프트 정책: EventXI-Select(FEATS=all) 와 DraftRec(FEATS=base)"
for F in all base; do
  FRAME=outputs/t3_frame_fam.parquet SEEDS=3 FEATS=$F $PY -m experiments.draftrec_t3
done
