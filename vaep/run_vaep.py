"""
K-League VAEP 전체 파이프라인 실행 스크립트

이 스크립트를 실행하면 다음을 순서대로 수행한다:
  1. raw-data/에서 전 시즌 경기 목록 로드
  2. Bepro 이벤트 → SPADL 액션 변환 (~15분)
  3. 선수/팀 메타데이터 저장 (teams.csv, players.csv, games.csv)
  4. Leave-One-Season-Out OOF VAEP 학습 (~10분)
  5. 결과 저장 (output/vaep_oof.parquet, vaep_oof_metrics.json)

실행 방법:
  cd <저장소 루트>
  pip install -r requirements.txt
  python run_vaep.py

전제 조건:
  - raw-data-2026/ 폴더에 Bepro raw JSON 데이터가 있어야 한다.
  - 폴더 구조: raw-data/{KLEAGUE1|KLEAGUE2}/{season}/match/{game_id}/
"""

from pathlib import Path

from core import (
    build_loader,
    load_all_games,
    convert_games_to_spadl,
    load_players_and_teams,
    run_oof_vaep,
)

# ---------------------------------------------------------------------------
# 경로 설정
#
# HERE: 이 스크립트가 있는 폴더 (metric/VAEP/)
# RAW_DATA: K-League raw JSON 데이터 폴더 (raw-data-2026/)
# OUTPUT: 결과물 저장 폴더 (vaep/output/)
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
# 2026 어댑터가 만든 트리(2021~25 는 심볼릭 링크, 2026 은 실물)
RAW_DATA = HERE.parent / "raw-data-2026"
OUTPUT = HERE / "output"

print("=== K-League VAEP OOF 학습 ===")
print(f"Raw data: {RAW_DATA}")
print(f"Output:   {OUTPUT}")

# ---------------------------------------------------------------------------
# 1단계: 경기 메타데이터 로드
#
# BeproLoader를 통해 KLEAGUE1, KLEAGUE2의 전 시즌 경기 목록을 가져온다.
# 반환 DataFrame에는 game_id, home/away_team_id, season, competition_name 포함.
# ---------------------------------------------------------------------------
loader = build_loader(RAW_DATA)
games = load_all_games(loader, competition_names=["KLEAGUE1", "KLEAGUE2"])
print(f"\n전체 경기 수: {len(games):,}")
print(games.groupby(["season"]).size().to_string())

# ---------------------------------------------------------------------------
# 2단계: SPADL 변환
#
# 각 경기의 Bepro 이벤트(이동, 패스, 슈팅 등)를 SPADL 표준 포맷으로 변환한다.
# 변환에 실패한 경기는 games DataFrame에서 자동 제거된다.
# 약 2,300경기 처리 시 15~20분 소요.
# ---------------------------------------------------------------------------
print("\n=== SPADL 변환 ===")
games, actions_dict = convert_games_to_spadl(loader, games, verbose=True)
print(f"변환 완료: {len(actions_dict):,}경기")

import pandas as pd
pd.concat(
    [df.assign(game_id=gid) for gid, df in actions_dict.items()],
    ignore_index=True,
).to_parquet(OUTPUT / "spadl_all.parquet", index=False)
print(f"SPADL 캐시 저장 완료: {OUTPUT / 'spadl_all.parquet'}")

# ---------------------------------------------------------------------------
# 3단계: 선수/팀 정보 저장
#
# 노트북 분석에서 player_id → 선수 이름 매핑에 사용한다.
# output/ 폴더에 CSV로 저장해 두면 매번 재계산할 필요 없다.
# ---------------------------------------------------------------------------
print("\n=== 선수/팀 정보 로드 ===")
teams_df, players_df = load_players_and_teams(loader, games)
teams_df.to_csv(OUTPUT / "teams.csv", index=False)
players_df.to_csv(OUTPUT / "players.csv", index=False)
games.to_csv(OUTPUT / "games.csv", index=False)
print(f"팀: {len(teams_df):,}, 선수: {len(players_df):,}")

# ---------------------------------------------------------------------------
# 4단계: OOF VAEP 학습 및 저장
#
# Leave-One-Season-Out 방식으로 5개 fold를 순환 학습한다.
# 각 fold에서 득점/실점 XGBoost 모델 2개를 학습하고,
# 해당 시즌의 모든 액션에 대한 VAEP 값을 예측한다.
#
# 출력 파일:
#   output/vaep_oof.parquet — 전 시즌 unbiased VAEP 값 (약 350만 행)
#   output/vaep_oof_metrics.json — fold별 AUC 성능 기록
# ---------------------------------------------------------------------------
print("\n=== OOF VAEP 학습 ===")
run_oof_vaep(games, actions_dict, OUTPUT)
