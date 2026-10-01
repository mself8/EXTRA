"""Bepro team_stats.json → 팀-경기 스타일 통계 (포메이션 예측 입력). 선수 합이 아니라 팀 단위 기록.
행 = (game_id, team_id) · 열 = ts_<stat> (패스 길이·방향·구역, 크로스, 경합, 태클, 클리어, 슛 위치 …)
출력: outputs/teamstats.parquet
실행: python -m experiments.build_teamstats
"""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); from config import VAEP_OUTPUT_DIR  # noqa
RAW = ROOT / "raw-data-2026"


def main():
    G = set(pd.read_csv(VAEP_OUTPUT_DIR / "games.csv").game_id.astype(int)); rows = []
    for f in glob.glob(str(RAW / "*/*/match/*/team_stats.json")):
        gid = int(f.split("/")[-2])
        if gid not in G: continue
        try: ts = json.load(open(f))["result"]
        except Exception: continue
        for t in ts:
            r = {"game_id": gid, "team_id": int(t["team_id"])}
            for k, v in t["stats"].items():
                if isinstance(v, (int, float)): r["ts_" + k] = float(v)
            rows.append(r)
    D = pd.DataFrame(rows).fillna(0.0)
    D.to_parquet(ROOT / "outputs/teamstats.parquet", index=False)
    print(f"팀-경기 {len(D):,} · 경기 {D.game_id.nunique():,} · 열 {sum(c.startswith('ts_') for c in D.columns)}", flush=True)


if __name__ == "__main__":
    main()
