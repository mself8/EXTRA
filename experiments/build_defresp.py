"""수비 책임 피처 — 상대의 위협 온볼 액션을 책임 구역 수비수에게 배분 (versa+ 유형).

방법
  ① 캘리브레이션: 공시 격자 칸 (단 k, 레인 l) 별로, 그 칸에 공시된 선수들의
     실제 액션 평균 위치(자기 공격 방향 좌표)를 리그 전체에서 추정 → 칸 중심점.
  ② 상대 팀의 위협 액션(offensive_value > 0)을 수비 좌표계로 반사
     (x_def = L − x_att, y_def = W − y_att) 하고, 그 경기 우리 선발들의
     칸 중심점 중 **최근접 선수**에게 위협값을 귀속.
  ③ 선수-경기 집계: dr_v{deep,mid,high}(허용 위협 합·깊이 3분할) + dr_n{...} 6열.

주의: 공시 위치는 실제 수비 위치의 정적 대리(잡음) · 팀 혼입 위험(약팀은 어디서나
허용)은 릿지 팀차 회귀와 관문이 판정한다.

출력: outputs/onball_gk_resid_merged_defresp.parquet (프로덕션 + 6열)
실행: python -m experiments.build_defresp
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR  # noqa: E402
from gap_lane import declared_grid  # noqa: E402

L, W = 105.0, 68.0
OUT = ROOT / "outputs" / "onball_gk_resid_merged_defresp.parquet"


def main():
    V = pd.read_parquet(ROOT / "outputs" / "vaep_oof_pre10.parquet",
                        columns=["game_id", "action_id", "team_id", "player_id",
                                 "offensive_value"])
    S = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet",
                        columns=["game_id", "action_id", "start_x", "start_y"])
    for D in (V, S):
        D["game_id"] = D.game_id.astype(np.int64); D["action_id"] = D.action_id.astype(np.int64)
    V = V.merge(S, on=["game_id", "action_id"], how="inner").dropna(
        subset=["player_id", "team_id", "start_x", "start_y"])
    gm = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    hmap = gm.set_index("game_id").home_team_id
    V["home"] = V.game_id.map(hmap)
    away = V.team_id.astype(float) != V.home.astype(float)
    V["xa"] = np.where(away, L - V.start_x, V.start_x)   # 행위자 공격 방향 좌표
    V["ya"] = np.where(away, W - V.start_y, V.start_y)

    CELL, PER, _ = declared_grid()
    # ① 칸 중심점 캘리브레이션 — 선수-경기 평균 위치를 칸별로 평균
    pg = V.groupby(["game_id", "player_id"]).agg(x=("xa", "mean"), y=("ya", "mean"),
                                                 tid=("team_id", "first")).reset_index()
    cx = {}
    for r in pg.itertuples(index=False):
        kl = PER.get((int(r.game_id), int(r.tid), int(r.player_id)))
        if kl is None: continue
        cx.setdefault(kl, []).append((r.x, r.y))
    CEN = {kl: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))
           for kl, v in cx.items() if len(v) >= 200}
    print("칸 중심점:", {k: (round(a, 1), round(b, 1)) for k, (a, b) in sorted(CEN.items())})

    # ② 상대 위협 → 최근접 책임 수비수
    opp = {}
    for r in gm.itertuples(index=False):
        opp[(int(r.game_id), int(r.home_team_id))] = int(r.away_team_id)
        opp[(int(r.game_id), int(r.away_team_id))] = int(r.home_team_id)
    T = V[V.offensive_value > 0].copy()
    T["def_tid"] = [opp.get((int(g), int(t)), -1) for g, t in zip(T.game_id, T.team_id)]
    T["xd"] = L - T.xa; T["yd"] = W - T.ya          # 수비 좌표계로 반사

    # 경기·수비팀별 선발 책임자 좌표 테이블
    ROSTER = {}
    for (gid, tid, pid), kl in PER.items():
        if kl in CEN: ROSTER.setdefault((gid, tid), []).append((pid, *CEN[kl]))
    rows = []
    cur = None; arr = None
    T = T.sort_values(["game_id", "def_tid"])
    for r in T.itertuples(index=False):
        key = (int(r.game_id), int(r.def_tid))
        if key != cur:
            cur = key
            lst = ROSTER.get(key)
            arr = (np.array([p for p, _, _ in lst]),
                   np.array([[x, y] for _, x, y in lst])) if lst else None
        if arr is None: continue
        d = ((arr[1] - [r.xd, r.yd]) ** 2).sum(1)
        rows.append((key[0], int(arr[0][int(np.argmin(d))]), float(r.offensive_value), r.xd))
    D = pd.DataFrame(rows, columns=["game_id", "player_id", "v", "xd"])
    D["band"] = np.where(D.xd < L / 3, 0, np.where(D.xd < 2 * L / 3, 1, 2))
    print(f"위협 배분 {len(D):,}건 · 커버 경기 {D.game_id.nunique():,}")

    Q = D.pivot_table(index=["game_id", "player_id"], columns="band", values="v",
                      aggfunc=["sum", "count"], fill_value=0.0)
    Q.columns = [f"dr_{a}{b}" for a, b in
                 [("v", i) for i in range(3)] + [("n", i) for i in range(3)]][:len(Q.columns)]
    Q = Q.reset_index()
    for c in [f"dr_v{i}" for i in range(3)] + [f"dr_n{i}" for i in range(3)]:
        if c not in Q.columns: Q[c] = 0.0

    M = pd.read_parquet(ROOT / "outputs" / "onball_gk_resid_merged.parquet")
    M2 = M.merge(Q, on=["game_id", "player_id"], how="left")
    dc = [c for c in M2.columns if c.startswith("dr_")]
    M2[dc] = M2[dc].fillna(0.0)
    cov = (M2[dc].abs().sum(1) > 0).mean()
    M2.to_parquet(OUT, index=False)
    print(f"저장 {OUT.name} {M2.shape} · 선수-경기 커버 {cov:.1%}")

    # 신뢰도 사전 관문: 과거감쇠 dr_v(깊은 1/3) → 다음 경기
    g2 = gm.copy(); g2["game_date"] = pd.to_datetime(g2.game_date)
    ORD = {int(r.game_id): i for i, r in
           g2.sort_values("game_date").reset_index(drop=True).iterrows()}
    X = M2[M2[dc].abs().sum(1) > 0][["game_id", "player_id", "dr_v0"]].copy()
    X["t"] = X.game_id.map(ORD); X = X.sort_values(["player_id", "t"])
    A, B = [], []
    for pid, gg in X.groupby("player_id"):
        a = w = 0.0
        for r in gg.itertuples(index=False):
            if w >= 5: A.append(a / w); B.append(r.dr_v0)
            a = a * np.exp(-np.log(2) / 40) + float(r.dr_v0); w = w * np.exp(-np.log(2) / 40) + 1
    A, B = np.array(A), np.array(B)
    print(f"신뢰도: 과거감쇠 깊은구역 허용위협 → 다음경기 r={np.corrcoef(A, B)[0,1]:+.4f} (n={len(A):,})")


if __name__ == "__main__":
    main()
