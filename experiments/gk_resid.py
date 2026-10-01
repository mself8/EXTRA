"""GK 피처를 팀 수비 부담에 잔차화 — 상대 강도 대리변수 성분을 제거한다.

문제
  gk_xg_faced ↔ 그 경기 팀이 내준 xG 상관 **+0.963**. 골키퍼 피처는 사실상
  "이 팀이 얼마나 얻어맞았나" 다. 자유도를 주면 릿지가 그걸 팀 강도로 쓴다
  (GK 점수 음수 100%, AUC 0.601 → 0.453).

해법 두 가지
  (a) 잔차화  각 gk_* 를 [내준 xG · 마주한 슛 · 만든 xG] 에 회귀한 **잔차**로 대체
  (b) 비율화  볼륨 열을 마주한 슛 수로 나눠 **단위당** 값으로

누수 아님: 스냅샷은 **과거 경기**의 피처만 지수감쇠 평균하므로, 과거 경기의 값을
그 경기 자신의 부담으로 보정하는 것은 현재 예측에 정보를 주지 않는다.

실행: python -m experiments.gk_resid
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

VOL = ["gk_faced", "gk_xg_faced", "gk_ga", "gk_faced_op", "gk_xg_op", "gk_ga_op",
       "gk_save_n", "gk_claim_n", "gk_punch_n", "gk_pickup_n", "gk_dist_n",
       "gk_dist_ok", "gk_sweep_n", "gk_save_v", "gk_claim_v", "gk_punch_v",
       "gk_pickup_v", "gk_dist_v"]


def main():
    G = pd.read_parquet(ROOT / "outputs" / "gk_feat.parquet")
    GC = [c for c in G.columns if c.startswith("gk_")]
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    opp = {}
    for r in g.itertuples(index=False):
        opp[(int(r.game_id), int(r.home_team_id))] = int(r.away_team_id)
        opp[(int(r.game_id), int(r.away_team_id))] = int(r.home_team_id)
    xg = pd.read_parquet(ROOT / "outputs" / "team_xg.parquet").set_index(["game_id", "team_id"]).xg
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    tm = pl.groupby(["game_id", "player_id"]).team_id.first()
    G["team_id"] = [tm.get((int(a), int(b)), np.nan) for a, b in zip(G.game_id, G.player_id)]
    G = G.dropna(subset=["team_id"]).copy(); G["team_id"] = G.team_id.astype(int)
    G["xga"] = [xg.get((int(a), opp.get((int(a), int(b)), -1)), np.nan) for a, b in zip(G.game_id, G.team_id)]
    G["xgf"] = [xg.get((int(a), int(b)), np.nan) for a, b in zip(G.game_id, G.team_id)]
    G = G.dropna(subset=["xga", "xgf"]).reset_index(drop=True)
    print(f"GK 경기 {len(G):,}")

    # (a) 잔차화
    Z = np.column_stack([np.ones(len(G)), G.xga, G.gk_faced, G.xgf])
    A = G.copy()
    for c in GC:
        if c in ("gk_is",): continue
        b = np.linalg.lstsq(Z, G[c].to_numpy(float), rcond=None)[0]
        A[c] = G[c].to_numpy(float) - Z @ b
    # (b) 비율화
    B = G.copy()
    den = G.gk_faced.to_numpy(float).clip(min=1.0)
    for c in GC:
        if c in ("gk_is", "gk_save_rate", "gk_dist_pct", "gk_gsax", "gk_gsax_all"): continue
        if c in VOL: B[c] = G[c].to_numpy(float) / den
    B["gk_gsax"] = G.gk_gsax / den; B["gk_gsax_all"] = G.gk_gsax_all / den
    B["gk_load"] = G.gk_faced / 10.0

    print(f"\n{'열':16s}{'원본 r(내준xG)':>15s}{'잔차 r':>10s}{'비율 r':>10s}")
    for c in ["gk_xg_faced", "gk_faced", "gk_ga", "gk_gsax", "gk_claim_n", "gk_dist_n", "gk_punch_n"]:
        print(f"{c:16s}{G[c].corr(G.xga):+15.3f}{A[c].corr(G.xga):+10.3f}{B[c].corr(G.xga):+10.3f}")

    P = pd.read_parquet(ROOT / "outputs" / "onball_vaep_event_pre10.parquet")
    for tag, D in (("resid", A), ("rate", B)):
        cols = ["game_id", "player_id"] + [c for c in D.columns if c.startswith("gk_")]
        M = P.merge(D[cols], on=["game_id", "player_id"], how="left")
        gcs = [c for c in M.columns if c.startswith("gk_")]
        M[gcs] = M[gcs].fillna(0.0)
        out = ROOT / "outputs" / f"onball_gk_{tag}.parquet"
        M.to_parquet(out, index=False)
        print(f"저장 {out.name}  {M.shape}")


if __name__ == "__main__":
    main()
