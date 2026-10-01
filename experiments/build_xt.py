"""xT (expected threat) — 표준 마르코프 격자(12×8)를 우리 SPADL 로 적합해 액션별 xT 이득과 구간 합을 만든다.

정의 (Singh 2018): 격자 셀 z 의 xT(z) = P(shot|z)·P(goal|shot,z) + P(move|z)·Σ_z' T(z→z')·xT(z')  (반복 수렴)
액션 가치: 성공한 이동(패스·드리블 계열)은 xT(end) − xT(start), 실패는 −xT(start)·0 (0으로 둠), 슛은 P(goal|z) − xT(start) 대신 xG 로 대체하지 않고 0 (슛은 npxG 가 따로 있으므로 이동 위협만 본다 = "빌드업 위협").
출력: outputs/bin_xt.parquet (game_id, team_id, bin, xt)   구간별 팀 xT 이득 합
실행: python -m experiments.build_xt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR  # noqa: E402
from segment_panel import EDGES     # noqa: E402

NX, NY = 12, 8; MOVE = {0, 1, 2, 3, 4, 5, 6, 7, 8, 21}   # 패스·크로스·스로인·프리킥·코너·테이크온·드리블 계열 (SPADL id)
SHOT = {11, 12, 13}


def cell(x, y):
    return np.clip((x / 105.0 * NX).astype(int), 0, NX - 1) * NY + np.clip((y / 68.0 * NY).astype(int), 0, NY - 1)


def main():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); home = {int(r.game_id): int(r.home_team_id) for r in g.itertuples(index=False)}
    sp = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet", columns=["game_id", "action_id", "period_id", "time_seconds", "team_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"])
    sp = sp.dropna(subset=["team_id", "start_x", "start_y"]).copy(); sp["team_id"] = sp.team_id.astype(int)
    away = sp.team_id.to_numpy() != np.array([home.get(int(gg), -1) for gg in sp.game_id])
    sx = np.where(away, 105 - sp.start_x, sp.start_x); sy = np.where(away, 68 - sp.start_y, sp.start_y)
    ex = np.where(away, 105 - sp.end_x.fillna(sp.start_x), sp.end_x.fillna(sp.start_x)); ey = np.where(away, 68 - sp.end_y.fillna(sp.start_y), sp.end_y.fillna(sp.start_y))
    cs, ce = cell(sx, sy), cell(ex, ey); ty = sp.type_id.to_numpy(int); ok = sp.result_id.to_numpy(int) == 1
    is_move, is_shot = np.isin(ty, list(MOVE)), np.isin(ty, list(SHOT))
    Z = NX * NY; n_start = np.bincount(cs, minlength=Z).astype(float)
    n_shot = np.bincount(cs[is_shot], minlength=Z).astype(float); n_goal = np.bincount(cs[is_shot & ok], minlength=Z).astype(float)
    n_move = np.bincount(cs[is_move], minlength=Z).astype(float)
    T = np.zeros((Z, Z)); np.add.at(T, (cs[is_move & ok], ce[is_move & ok]), 1.0); T = T / np.maximum(T.sum(1, keepdims=True), 1)
    p_shot = n_shot / np.maximum(n_start, 1); p_goal = n_goal / np.maximum(n_shot, 1); p_move = n_move / np.maximum(n_start, 1)
    xt = np.zeros(Z)
    for _ in range(50): xt = p_shot * p_goal + p_move * (T @ xt)
    print(f"xT 격자: 최소 {xt.min():.3f} 최대 {xt.max():.3f} · 셀별 슛확률 최대 {p_shot.max():.2f}")
    val = np.where(is_move & ok, xt[ce] - xt[cs], 0.0)
    minute = np.where(sp.period_id.to_numpy() >= 2, 45.0, 0.0) + sp.time_seconds.to_numpy() / 60.0
    sp["bin"] = np.digitize(minute, EDGES[1:-1]); sp["xt"] = val
    out = sp.groupby(["game_id", "team_id", "bin"]).xt.sum().reset_index(); out.to_parquet(ROOT / "outputs/bin_xt.parquet", index=False)
    np.save(ROOT / "outputs/xt_grid_12x8.npy", xt.reshape(NX, NY))
    print(f"구간-팀 {len(out):,} · xT/구간 평균 {out.xt.mean():.3f} SD {out.xt.std():.3f} → bin_xt.parquet")


if __name__ == "__main__":
    main()
