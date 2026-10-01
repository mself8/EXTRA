"""교체 이벤트 — `event_data.json` 의 Substitution (Off/On) 을 정확한 시각으로.

기존 `lineup_event_stints.py` 는 `minutes_played` 로 교체 시각을 역산했다
(나간 선발 수 == 들어온 교체 수 87.3%, 시각 오차 ≤1분 96.1%). 원본에 교체 이벤트가
그대로 있으므로 역산할 필요가 없다.

출력: outputs/subs.parquet (game_id, team_id, player_id, dir, period, t)
      dir: 'Off' | 'On' · t: 경기 시작부터의 분 (period 2 는 45 를 더한 표기 그대로)
실행: python -m experiments.build_subs
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import RAW_DATA_DIR  # noqa: E402

OUT = ROOT / "outputs" / "subs.parquet"


def main():
    rows = []
    for comp in ("KLEAGUE1", "KLEAGUE2"):
        for sd in sorted((RAW_DATA_DIR / comp).glob("*")):
            if not sd.is_dir():
                continue
            for f in sorted((sd / "match").glob("*/event_data.json")):
                try:
                    d = json.load(open(f))
                except Exception:
                    continue
                r = d.get("result", d)
                if isinstance(r, dict):
                    r = list(r.values())[0]
                gid = int(f.parent.name)
                for e in r:
                    for t in (e.get("event_types") or []):
                        if t.get("event_type") != "Substitution":
                            continue
                        rows.append(dict(
                            game_id=gid,
                            team_id=int(e["team_id"]) if e.get("team_id") is not None else -1,
                            player_id=int(e["player_id"]) if e.get("player_id") is not None else -1,
                            dir=t.get("sub_event_type"),
                            period=str(e.get("event_period") or "FIRST_HALF"),
                            t=float(e.get("event_time") or 0) / 60000.0))   # ms → 분
    D = pd.DataFrame(rows)
    # event_time 은 피리어드 내 초. 2피리어드는 45분을 더해 경기 시각으로.
    # event_time 은 이미 **경기 시각**이다 (FIRST_HALF 1.6~50.5 · SECOND_HALF 44.2~105.3).
    # 피리어드 오프셋을 더하면 이중 계산이 된다.
    D["tmin"] = D.t
    D.to_parquet(OUT, index=False)
    print(f"교체 이벤트 {len(D):,} · 경기 {D.game_id.nunique():,}")
    print(f"방향 분포 {D.dir.value_counts().to_dict()}")
    per = D[D.dir == "Off"].groupby(["game_id", "team_id"]).size()
    print(f"팀-경기당 교체 {per.mean():.2f}회 (중앙 {per.median():.0f})")
    print(f"교체 시각 분포: p25 {D.tmin.quantile(.25):.0f}분 · 중앙 {D.tmin.median():.0f} · "
          f"p75 {D.tmin.quantile(.75):.0f}")


if __name__ == "__main__":
    main()
