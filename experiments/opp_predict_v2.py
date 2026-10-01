"""XI 예측기 v2 — 피처를 넓힌다. 구조가 아니라 정보가 천장인지 확인.

v1 은 "과거에 얼마나 나왔나" 만 썼다 (선발률·출전분·이탈간격·평점).
SetTransformer 실험에서 **구조를 바꿔도 +0.02명** 뿐이었으므로, 남은 여지가
정보 쪽인지 본다. 감독이 실제로 쓰는 정보 중 데이터에 있는 것을 전부 넣는다.

추가한 것
  부하    직전 경기 출전분 · 최근 3경기 누적분 · 최근 5경기 누적분
  부상 대리 연속 결장 횟수 · 명단에서 빠진 연속 횟수 · 복귀 직후 여부
          (부상 기록은 없다. "명단에 없다가 돌아옴" 으로 부분 대리한다)
  일정    팀 휴식일(team_state.rest) · 직전 경기로부터의 경기 간격
  맥락    홈원정 · 시즌 진행률(game_day) · 팀 순위 · 상대 순위
  경쟁    같은 단(공시 5단) 후보 수 · 그 단에서의 선발률 순위

실행: python -m experiments.opp_predict_v2
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
sys.path.insert(0, str(ROOT / "experiments"))
import os as _o
from config import VAEP_OUTPUT_DIR  # noqa: E402
from opp_predict import FEATS as F1, pick_xi as _pick  # noqa: E402

W = 10
TEST = {2025}
OUT = ROOT / "outputs" / "opp_panel_v2.parquet"


def build():
    p = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    g["t"] = np.arange(len(g))
    ts = pd.read_parquet(ROOT / "outputs/team_state.parquet").set_index(["game_id", "team_id"])
    CD = pd.read_parquet(ROOT / "outputs/cards.parquet")
    CARD = {}
    for r in CD.itertuples(index=False):
        CARD[(int(r.game_id), int(r.player_id))] = (int(r.yellow), int(r.red), int(r.foul))
    LV = pd.read_parquet(ROOT / "outputs/player_level.parquet")
    lv = {int(r.player_id): int(r.level) for r in LV.itertuples(index=False)}
    gk = set(int(r.player_id) for r in LV[LV.is_gk == 1].itertuples(index=False))
    opp, home = {}, {}
    for r in g.itertuples():
        opp[(r.game_id, r.home_team_id)] = r.away_team_id
        opp[(r.game_id, r.away_team_id)] = r.home_team_id
        home[(r.game_id, r.home_team_id)] = 1.0
        home[(r.game_id, r.away_team_id)] = 0.0
    p = p.join(g.set_index("game_id")[["t", "season", "game_day"]], on="game_id")
    p = p.dropna(subset=["t"]).sort_values("t")
    p["is_starter"] = p.is_starter.astype(bool)
    p["minutes_played"] = p.minutes_played.fillna(0.0)
    nday = g.groupby("season").game_day.max().to_dict()

    rows = []
    for tid, gr in p.groupby("team_id"):
        hist = []
        for (gid, t, ss, gd), gg in gr.groupby(["game_id", "t", "season", "game_day"], sort=True):
            rec = {int(r.player_id): (bool(r.is_starter), float(r.minutes_played),
                                      float(r.rating) if np.isfinite(r.rating) else np.nan)
                   for r in gg.itertuples(index=False)}
            past = [h for h in hist if h[1] == ss][-W:]
            hist_gids = [h[3] for h in hist if h[1] == ss]
            if len(past) >= 3:
                truth = {q for q, v in rec.items() if v[0]}
                cand = sorted({q for _, _, r, _g in past for q in r})
                n = len(past)
                oid = opp.get((int(gid), int(tid)))
                try:
                    st_own = ts.loc[(int(gid), int(tid))]
                    rest = float(st_own.rest); orank = float(st_own.rank_pct)
                except KeyError:
                    rest, orank = np.nan, np.nan
                try:
                    orank_o = float(ts.loc[(int(gid), int(oid)), "rank_pct"])
                except (KeyError, TypeError):
                    orank_o = np.nan
                base = {}
                for q in cand:
                    seen = [(k, r[q]) for k, (_, _, r, _g) in enumerate(past) if q in r]
                    ks = np.array([k for k, _ in seen], float)
                    stv = np.array([v[0] for _, v in seen], float)
                    mnv = np.array([v[1] for _, v in seen], float)
                    rtv = np.array([v[2] for _, v in seen], float)
                    w = np.exp((ks - (n - 1)) / 3.0)
                    last_k = ks.max()
                    # 결장·복귀 패턴 (부상 대리)
                    inlist = np.zeros(n, bool); inlist[ks.astype(int)] = True
                    miss_run = 0
                    for k in range(n - 1, -1, -1):
                        if inlist[k]:
                            break
                        miss_run += 1
                    ret = 1.0 if (miss_run == 0 and n >= 2 and not inlist[n - 2]) else 0.0
                    nostart_run = 0
                    for k, fl in reversed(list(enumerate(inlist))):
                        s_ = dict(seen).get(k)
                        if fl and s_ is not None and s_[0]:
                            break
                        nostart_run += 1
                    # 카드: 시즌 누적 경고·직전 경기 퇴장 (출장정지 위험)
                    ylist = [CARD.get((gid_h, q), (0, 0, 0)) for gid_h in hist_gids]
                    cy = sum(z[0] for z in ylist); cr = sum(z[1] for z in ylist)
                    cf = sum(z[2] for z in ylist)
                    last_red = ylist[-1][1] if ylist else 0
                    base[q] = dict(
                        game_id=int(gid), team_id=int(tid), player_id=q, t=int(t),
                        season=int(ss), y=int(q in truth),
                        st_rate=stv.mean(), st_w=(stv * w).sum() / w.sum(),
                        st_last=float(stv[-1]) if last_k == n - 1 else 0.0,
                        st3=stv[ks >= n - 3].mean() if (ks >= n - 3).any() else 0.0,
                        min_w=(mnv * w).sum() / w.sum() / 90.0,
                        app_rate=len(seen) / n, gap=(n - 1 - last_k) / n,
                        rating=np.nanmean(rtv) if np.isfinite(rtv).any() else np.nan,
                        is_gk=int(q in gk), npast=n,
                        # ── v2 추가 ──
                        last_mins=float(mnv[-1]) / 90.0 if last_k == n - 1 else 0.0,
                        load3=mnv[ks >= n - 3].sum() / 270.0,
                        load5=mnv[ks >= n - 5].sum() / 450.0,
                        miss_run=miss_run / n, nostart_run=nostart_run / n, ret=ret,
                        rest=rest, own_rank=orank, opp_rank=orank_o,
                        home=home.get((int(gid), int(tid)), 0.5),
                        prog=float(gd) / max(nday.get(int(ss), 38), 1),
                        lvl=lv.get(q, 2),
                        yc_cum=cy, rc_cum=cr, foul_cum=cf / max(n, 1),
                        yc_risk=float(cy % 5) / 4.0, last_red=float(last_red))
                # 같은 단 경쟁
                for q, d in base.items():
                    same = [r2 for r2 in base.values() if r2["lvl"] == d["lvl"]]
                    d["lv_n"] = len(same)
                    d["lv_rank"] = float(sum(1 for r2 in same if r2["st_rate"] > d["st_rate"])) / max(len(same), 1)
                rows += list(base.values())
            hist.append((int(t), int(ss), rec, int(gid)))
    D = pd.DataFrame(rows)
    for c in ["rating", "rest", "own_rank", "opp_rank"]:
        D[c] = D[c].fillna(D[c].median())
    for i in range(5):
        D[f"lv{i}"] = (D.lvl == i).astype(float)
    D.to_parquet(OUT, index=False)
    return D


F2 = list(F1) + ["yc_cum", "rc_cum", "foul_cum", "yc_risk", "last_red", "last_mins", "load3", "load5", "miss_run", "nostart_run", "ret",
                 "rest", "own_rank", "opp_rank", "home", "prog", "lv_n", "lv_rank"] + \
     [f"lv{i}" for i in range(5)]


def evaluate(D, feats, label):
    tr = D[~D.season.isin(TEST | {2026})]
    te = D[D.season.isin(TEST)].copy()
    m = LogisticRegression(max_iter=3000).fit(tr[feats].to_numpy(float), tr.y.to_numpy())
    te["prob"] = m.predict_proba(te[feats].to_numpy(float))[:, 1]
    h = []
    for _, sub in te.groupby(["game_id", "team_id"]):
        truth = set(sub[sub.y == 1].player_id.astype(int))
        if len(truth) != 11:
            continue
        h.append(len(set(_pick(sub, "prob")) & truth))
    return np.array(h, float), m


if __name__ == "__main__":
    import os
    D = pd.read_parquet(OUT) if os.path.exists(OUT) else build()
    print(f"패널 {len(D):,} · 피처 v1 {len(F1)} → v2 {len(F2)}")
    h1, _ = evaluate(D, F1, "v1")
    h2, m2 = evaluate(D, F2, "v2")
    d = h2 - h1
    print(f"\n{'모델':<28}{'적중/11':>9}{'비율':>8}")
    print(f"{'v1 (기존 9피처)':<28}{h1.mean():>9.3f}{h1.mean()/11*100:>7.1f}%")
    print(f"{'v2 (전체 27피처)':<28}{h2.mean():>9.3f}{h2.mean()/11*100:>7.1f}%")
    print(f"\n증분 {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f} "
          f"(t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}, n={len(d)})")
    co = sorted(zip(F2, m2.coef_[0]), key=lambda x: -abs(x[1]))[:12]
    print(f"\n계수 상위 12: " + " · ".join(f"{k}={v:+.2f}" for k, v in co))
