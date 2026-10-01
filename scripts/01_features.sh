#!/usr/bin/env bash
# 원본 이벤트 → SPADL·VAEP → 선수 식별 브릿지 → 선수-경기 피처.
# 원본 이벤트 데이터(raw-data-2026/)가 있어야 합니다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
echo "== 1. SPADL 변환 + VAEP 액션 가치"
$PY vaep/run_vaep.py
echo "== 2. 선수 식별: K리그 공식 등록번호 브릿지"
$PY -m experiments.kleague_scrape          # 공식 명부·프로필 수집
$PY -m experiments.kleague_bridge          # 등록번호 ↔ 제공자 pid 매칭
$PY -m experiments.apply_kl_bridge         # 파이프라인 파일에 적용
$PY -m experiments.apply_person_bridge     # 2026 명단 단절 연결
echo "== 3. 온볼 피처와 보정"
$PY -m experiments.onball_vaep_event       # 유형×결과×밴드×면
$PY -m experiments.build_refined           # 세분 패스·슛·수비 채널
$PY -m experiments.gk_features             # 골키퍼 열
$PY -m experiments.gk_resid                # 팀 수비 부담에 잔차화
$PY -m experiments.build_defresp           # 수비 책임 배분
$PY -m experiments.build_bio_features      # 나이·신장·국적
echo "== 4. 라벨과 보조 테이블"
$PY -m experiments.build_xg                # 무페널티 xG (평가 시즌 제외 학습)
$PY -m experiments.build_subs              # 교체 시각
$PY -m experiments.build_redcards          # 퇴장
echo "완료: outputs/onball_gk_resid_merged_defresp_ref.parquet"
