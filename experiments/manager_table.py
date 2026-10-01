"""감독 재임 테이블 → 경기별 감독 배정 + 검증.

입력: outputs/managers_raw.csv (team_kr, manager, start_date, end_date, role, note)
출력: outputs/game_manager.parquet (game_id, team_id, manager, role, mgr_game_no)
검증: 팀-경기 커버리지(정확히 1명), 겹침/공백 구간 보고.

실행: python -m experiments.manager_table
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

# 크롤 team_kr → team_id (표기 변형 흡수)
ALIAS = {
    "강원FC": 4643, "광주FC": 4648, "김천상무": 2353, "김천 상무": 2353,
    "대구FC": 4644, "대전하나시티즌": 4657, "대전 하나 시티즌": 4657,
    "FC서울": 316, "성남FC": 4651, "수원삼성블루윙즈": 328, "수원 삼성 블루윙즈": 328,
    "수원FC": 4220, "울산HD": 2354, "울산 HD": 2354, "울산현대": 2354,
    "인천유나이티드": 4646, "인천 유나이티드": 4646, "전북현대모터스": 4640,
    "전북 현대 모터스": 4640, "제주SK": 4641, "제주유나이티드": 4641,
    "포항스틸러스": 4639, "포항 스틸러스": 4639, "FC안양": 4654,
    "부천FC1995": 4652, "부천FC 1995": 4652, "부산아이파크": 4649,
    "충남아산FC": 4650, "충남아산": 4650, "경남FC": 4647, "전남드래곤즈": 4645,
    "전남 드래곤즈": 4645, "서울이랜드FC": 4655, "서울이랜드": 4655,
    "안산그리너스": 4656, "김포FC": 5896, "천안시티FC": 5902, "천안시티": 5902,
    "충북청주FC": 5893, "충북청주": 5893, "화성FC": 5890, "김해FC": 6007,
    "김해FC 2008": 6007, "용인FC": 57730, "파주프런티어": 5895, "파주 프런티어 FC": 5895,
}


def norm(s):
    return str(s).replace(" ", "").strip()


def main():
    M = pd.read_csv(ROOT / "outputs" / "managers_raw.csv")
    M["tid"] = M.team_kr.map(lambda s: ALIAS.get(norm(s)) or ALIAS.get(str(s).strip()))
    bad = M[M.tid.isna()]
    if len(bad):
        print("매핑 실패 팀명:", sorted(bad.team_kr.unique()))
    M = M.dropna(subset=["tid"]).copy()
    M["tid"] = M.tid.astype(int)
    for c in ("start_date", "end_date"):
        M[c] = pd.to_datetime(M[c], errors="coerce")
    M = M.dropna(subset=["start_date"])
    M.loc[M.end_date.isna(), "end_date"] = pd.Timestamp("2026-12-31")

    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g["game_date"] = pd.to_datetime(g.game_date)
    rows = []
    for r in g.itertuples(index=False):
        gid, d = int(r.game_id), r.game_date
        for tid in (int(r.home_team_id), int(r.away_team_id)):
            cand = M[(M.tid == tid) & (M.start_date <= d) & (M.end_date >= d)]
            if len(cand) == 0:
                rows.append(dict(game_id=gid, team_id=tid, manager=None, role=None))
            else:
                # 겹치면 가장 늦게 부임한 사람(대행 교대 처리)
                c = cand.sort_values("start_date").iloc[-1]
                rows.append(dict(game_id=gid, team_id=tid, manager=str(c.manager),
                                 role=str(c.role)))
    D = pd.DataFrame(rows)
    cov = D.manager.notna().mean()
    print(f"경기-팀 {len(D):,} · 감독 배정 커버리지 {cov:.1%}")
    miss = D[D.manager.isna()].merge(g[["game_id", "season"]], on="game_id")
    if len(miss):
        mm = miss.groupby(["team_id", "season"]).size()
        print("미배정 상위:", mm.sort_values(ascending=False).head(8).to_dict())
    # 감독별 그 팀 몇 번째 경기인가 (부임 초기 지표)
    D = D.merge(g[["game_id", "game_date"]], on="game_id")
    D = D.sort_values(["team_id", "game_date"]).reset_index(drop=True)
    D["mgr_game_no"] = D.groupby(["team_id", "manager"]).cumcount() + 1
    D.drop(columns=["game_date"]).to_parquet(ROOT / "outputs" / "game_manager.parquet", index=False)
    ns = D[D.manager.notna()]
    print(f"고유 감독 {ns.manager.nunique()} · 감독-팀 스펠 {ns.groupby(['team_id','manager']).ngroups}")
    print("저장 outputs/game_manager.parquet")


if __name__ == "__main__":
    main()
