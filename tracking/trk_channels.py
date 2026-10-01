"""깊이 구간 페이즈 표 → EventXI 입력 형식의 트래킹 열 (선수-경기 한 행).

입력  outputs/player_phase_band.parquet (phase_tables.py)
출력  outputs/trk_channels.parquet: game_id, player_id(EventXI ID), trk{mode}_{면}_{라벨}_{구간}
  mode  phase = 페이즈 6 · third = 점유×공 위치 3분할 6 · all = 페이즈 합침 1
  면    v = 평균 속도(m/s) · o = 고강도(>5.5 m/s) 비율 · d = 압박 참여(공 10 m 안, 3 m/s 이상 접근) 비율 · n = 그 구간에 머문 분
  구간  0(자기 골문) → 5(상대 골문), EventXI 밴드와 같은 방향
  트래킹이 있는 선수-경기에서 비는 (라벨, 구간) 칸은 0 이다. 트래킹이 없는 선수-경기는 행이 없다(EventXI 쪽에서 0 + 표시).
실행  python tracking/trk_channels.py
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"
FACES = {"v": "v", "o": "hi", "d": "engage", "n": "minutes"}
NB = 6


def main():
    B = pd.read_parquet(OUT / "player_phase_band.parquet")
    B = B[B.ex_pid.notna()].copy(); B["player_id"] = B.ex_pid.astype(int)
    parts = []
    for face, col in FACES.items():
        w = B.pivot_table(index=["game_id", "player_id"], columns=["tax", "label", "band"], values=col, aggfunc="mean")
        w.columns = [f"trk{m}_{face}_{l}_{int(b)}" for m, l, b in w.columns]; parts.append(w)
    W = pd.concat(parts, axis=1)
    full = [f"trk{m}_{face}_{l}_{b}" for m in ("phase", "third", "all") for l in sorted(B[B.tax == m].label.unique()) for face in FACES for b in range(NB)]
    W = W.reindex(columns=full).fillna(0.0).astype("float32").reset_index()
    W.to_parquet(OUT / "trk_channels.parquet", index=False)
    print(f"선수-경기 {len(W):,} · 경기 {W.game_id.nunique()} · 열 {len(full)} (phase {sum(c.startswith('trkphase_') for c in full)} · third {sum(c.startswith('trkthird_') for c in full)} · all {sum(c.startswith('trkall_') for c in full)})")


if __name__ == "__main__":
    main()
