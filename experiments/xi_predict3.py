"""XI 예측기 v3 — 감독의 선발 선택을 직접 학습한다.

왜
  우리 라인업 점수는 **팀 xG 차** 를 맞추도록 학습된 것이라 선택을 학습한 적이 없다.
  그래서 "최근 평균 출전분 상위 11명" 이라는 한 줄 규칙(8.380/11)에 진다(8.322).

무엇을 바꾸나
  ① 목표를 선택으로  — 선발 여부를 직접 학습 (로지스틱 → 부스팅)
  ② 정보를 넓힌다    — opp_panel_v2 의 32 피처 + **우리 VAEP 점수** +
                     **칸 내 경쟁 피처**(같은 (단,레인) 안에서의 순위·후보 수)
  ③ 선택을 제약 하에 — GK+상위10 이 아니라 공시 (단,레인) 격자 제약으로 뽑는다
  ④ 평가를 통일      — gap_soft 와 **같은 2,552 팀-경기**, 같은 출전분 베이스라인

실행: python -m experiments.xi_predict3
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR  # noqa: E402
from gap_soft import pick_soft  # noqa: E402
from minutes_control import hist_minutes  # noqa: E402

import os as _os
PKL = _os.environ.get("PKL",  # 기본 = 현행 프로덕션 케이스 테이블
    "outputs/gap_soft_fullonball_gk_resid_merged_mix25d_pc20_all_npxg_nozvzn_fixsd.pkl")
BETA = 0.04
BASE_F = ["st_rate", "st_w", "st_last", "st3", "min_w", "app_rate", "gap", "rating", "is_gk",
          "npast", "last_mins", "load3", "load5", "miss_run", "nostart_run", "ret",
          "rest", "own_rank", "opp_rank", "home", "prog", "yc_cum", "rc_cum",
          "foul_cum", "yc_risk", "last_red", "lv_n", "lv_rank"] + [f"lv{i}" for i in range(5)]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    D = pd.read_parquet(ROOT / "outputs" / "opp_panel_v2.parquet")
    C = pd.read_pickle(ROOT / PKL)
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    HM = hist_minutes(gpos)
    LP = C.LPk.iloc[0]

    # 우리 점수 · 칸(단,레인) · 요구 인원 을 케이스에서 뽑아 패널에 붙인다
    SC, CELL, NEED, POOL = {}, {}, {}, {}
    for r in C.itertuples(index=False):
        k = (int(r.gid), int(r.tid))
        SC[k] = r.sc; POOL[k] = set(r.pool)
        cell = {}
        for p in r.pool:
            q = r.qmap.get(p)
            lv = int(np.argmax(q)) if q is not None else 2
            cell[p] = (lv, int(r.lane.get(p, 1)))
        CELL[k] = cell; NEED[k] = r.G
    _pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    _pos = _pl.groupby("player_id").starting_position_name.agg(
        lambda z: z.mode().iloc[0] if len(z.mode()) else "?")
    ISGK = {int(k): (v == "GK") for k, v in _pos.items()}
    keys = set(SC)
    D["key"] = list(zip(D.game_id.astype(int), D.team_id.astype(int)))
    D = D[D.key.isin(keys)].reset_index(drop=True)
    print(f"평가 대상 팀-경기 {D.key.nunique():,} · 선수-경기 {len(D):,}")

    sc, hm, lvl, lane, need, ncand = [], [], [], [], [], []
    for r in D.itertuples(index=False):
        k = r.key; p = int(r.player_id)
        sc.append(SC[k].get(p, np.nan))
        v = HM.get((int(r.game_id), p)); hm.append(np.nan if v is None else v)
        c = CELL[k].get(p, (2, 1)); lvl.append(c[0]); lane.append(c[1])
        need.append(float(NEED[k][c[0], c[1]]))
        ncand.append(float(sum(1 for q in POOL[k] if CELL[k].get(q, (2, 1)) == c)))
    D["is_gk"] = [1.0 if ISGK.get(int(p), False) else 0.0 for p in D.player_id]
    D["our_sc"] = sc; D["hist_min"] = hm; D["cell_lv"] = lvl; D["cell_ln"] = lane
    D["cell_need"] = need; D["cell_n"] = ncand
    D["cell_ratio"] = D.cell_need / D.cell_n.clip(lower=1)
    # 칸 안 순위 (팀-경기 안에서)
    for c, nm in [("our_sc", "r_sc"), ("hist_min", "r_min"), ("st_rate", "r_st")]:
        D[nm] = D.groupby(["key", "cell_lv", "cell_ln"])[c].rank(ascending=False, pct=True)
        D[nm + "_t"] = D.groupby("key")[c].rank(ascending=False, pct=True)
    F3 = BASE_F + ["our_sc", "hist_min", "cell_need", "cell_n", "cell_ratio",
                   "r_sc", "r_min", "r_st", "r_sc_t", "r_min_t", "r_st_t"]
    D[F3] = D[F3].astype(float)

    # 전진 연쇄: 시즌 s 테스트, 이전 시즌 학습
    seasons = sorted(D.season.unique())
    D["prob"] = np.nan; D["prob_lr"] = np.nan
    for s in seasons[1:]:
        tr = D[D.season < s]; te = D.season == s
        if len(tr) < 5000 or te.sum() == 0: continue
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                           max_depth=6, l2_regularization=1.0,
                                           random_state=0)
        m.fit(tr[F3].to_numpy(float), tr.y.to_numpy())
        D.loc[te, "prob"] = m.predict_proba(D.loc[te, F3].to_numpy(float))[:, 1]
        lr = LogisticRegression(max_iter=3000)
        X = np.nan_to_num(tr[BASE_F].to_numpy(float)); lr.fit(X, tr.y.to_numpy())
        D.loc[te, "prob_lr"] = lr.predict_proba(np.nan_to_num(D.loc[te, BASE_F].to_numpy(float)))[:, 1]
    E = D.dropna(subset=["prob"]).copy()
    print(f"표본외 평가 {E.key.nunique():,} 팀-경기 · 시즌 {sorted(E.season.unique())}")

    res = {}
    for nm, col in [("v3 부스팅 (전 피처)", "prob"), ("v2 로지스틱 (32피처)", "prob_lr"),
                    ("우리 VAEP 점수", "our_sc"), ("과거 평균 출전분", "hist_min"),
                    ("선발률 st_rate", "st_rate")]:
        hits, gk_all, gk_multi = [], [], []
        for k, sub in E.groupby("key"):
            truth = set(sub[sub.y == 1].player_id.astype(int))
            if len(truth) != 11: continue
            s = {int(p): (0.0 if not np.isfinite(v) else float(v))
                 for p, v in zip(sub.player_id, sub[col])}
            gk_pool = [int(p) for p in sub.player_id if ISGK.get(int(p), False)]
            qmap = {int(p): np.eye(5)[CELL[k].get(int(p), (2, 1))[0]] for p in sub.player_id}
            lane = {int(p): CELL[k].get(int(p), (2, 1))[1] for p in sub.player_id}
            rec, _, _ = pick_soft(s, qmap, lane, NEED[k], gk_pool, set(s), LP, BETA)
            hits.append(len(truth & set(rec)))
            tg = [p for p in gk_pool if p in truth]
            pg = [p for p in rec if p in set(gk_pool)]
            if tg and pg:
                ok = float(pg[0] == tg[0]); gk_all.append(ok)
                if len(gk_pool) >= 2: gk_multi.append(ok)
        res[nm] = np.array(hits, float)
        gk_all = np.array(gk_all, float); gk_multi = np.array(gk_multi, float)
        print(f"  {nm:22s} {res[nm].mean():.3f}/11   n={len(hits):,}   "
              f"GK 정확도 전체 {100*gk_all.mean():.1f}% · 후보 2명이상 "
              f"{100*gk_multi.mean() if len(gk_multi) else float('nan'):.1f}% (n={len(gk_multi):,})")
    # DraftRec 을 같은 행에 붙인다
    try:
        DR = pd.read_parquet(ROOT / "outputs" / "dr_hits_by_case.parquet")
        drm = {(int(a_), int(b_)): float(v) for (a_, b_), v in DR.dr_hit.items()}
        ks = [k for k, sub in E.groupby("key") if len(set(sub[sub.y == 1].player_id)) == 11]
        dv = np.array([drm.get(k, np.nan) for k in ks], float)
        m = np.isfinite(dv)
        print(f"  {'DraftRec (디코딩)':22s} {dv[m].mean():.3f}/11   n={int(m.sum()):,}")
        res["DraftRec"] = dv
    except Exception as e:
        print("  DraftRec 로드 실패:", e)
    a = res["v3 부스팅 (전 피처)"]; b = res["과거 평균 출전분"]
    n = min(len(a), len(b)); d = a[:n] - b[:n]
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
    print(f"\nv3 − 출전분: {d.mean():+.3f}  95% CI [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    for nm2, key2 in [("v2", "v2 로지스틱 (32피처)"), ("DraftRec", "DraftRec"),
                      ("VAEP 점수", "우리 VAEP 점수")]:
        if key2 not in res: continue
        cc = res[key2]; n2 = min(len(a), len(cc))
        dd = a[:n2] - cc[:n2]; ok = np.isfinite(dd); dd = dd[ok]
        bs2 = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
        print(f"v3 − {nm2:10s} {dd.mean():+.3f}  95% CI "
              f"[{np.percentile(bs2,2.5):+.3f},{np.percentile(bs2,97.5):+.3f}]  n={len(dd):,}")


if __name__ == "__main__":
    main()
