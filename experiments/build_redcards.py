"""퇴장 이벤트 — event_data.json 의 카드 sub_event_type 에서 퇴장(Red / Second Yellow)을 시각과 함께.

출력: outputs/red_cards.parquet (game_id, team_id, player_id, kind, period, t)   t: 경기 시각(분)
실행: python -m experiments.build_redcards
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import RAW_DATA_DIR  # noqa: E402

OUT = ROOT / "outputs" / "red_cards.parquet"


def main():
    rows, kinds = [], Counter()
    for comp in ("KLEAGUE1", "KLEAGUE2"):
        for sd in sorted((RAW_DATA_DIR / comp).glob("*")):
            if not sd.is_dir(): continue
            for f in sorted((sd / "match").glob("*/event_data.json")):
                try: d = json.load(open(f))
                except Exception: continue
                r = d.get("result", d)
                if isinstance(r, dict): r = list(r.values())[0]
                gid = int(f.parent.name)
                for e in r:
                    for t in (e.get("event_types") or []):
                        st = str(t.get("sub_event_type") or "")
                        if "card" not in st.lower(): continue
                        kinds[st] += 1
                        if "red" in st.lower() or "second yellow" in st.lower():
                            rows.append(dict(game_id=gid, team_id=int(e["team_id"]) if e.get("team_id") is not None else -1,
                                             player_id=int(e["player_id"]) if e.get("player_id") is not None else -1,
                                             kind=st, period=str(e.get("event_period") or ""),
                                             t=float(e.get("event_time") or 0) / 60000.0))
    print("카드 종류:", dict(kinds))
    D = pd.DataFrame(rows).drop_duplicates(["game_id", "player_id"])
    D.to_parquet(OUT, index=False)
    print(f"퇴장 {len(D):,}건 · 경기 {D.game_id.nunique():,} · 종류 {D.kind.value_counts().to_dict()}")
    print(f"시각 분포: 최소 {D.t.min():.1f} · 중앙 {D.t.median():.1f} · 최대 {D.t.max():.1f}")
    print(f"팀 미상 {int((D.team_id < 0).sum())} · 선수 미상 {int((D.player_id < 0).sum())}")


if __name__ == "__main__":
    main()
