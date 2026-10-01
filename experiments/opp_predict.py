"""상대 라인업 예측 — 감독처럼 "상대가 누구로 나올지" 를 먼저 읽는다.

왜
  지금 추천기는 상대의 **실제 선발**과 그 경기에서 **실제로 뛴 좌표**를 입력으로
  쓴다(milp_lineup.py: opp_pids, _played_pos(game_id)). 경기 전에는 둘 다 알 수
  없다. 즉 현재 상대-조건화는 배치 가능한 결정이 아니라 사후 정보다.
  감독의 실제 순서는 (1) 상대 XI·포메이션을 예측하고 (2) 거기에 맞춰 우리를
  정한다. 이 파일은 (1)이다.

무엇
  팀-경기마다 후보 풀을 **직전 W경기 스쿼드**로 만들고, 각 후보의 선발 확률을
  **직전 경기들만** 쓰는 피처로 예측한다. 사후 정보(그날 명단·결과) 없음.
  GK 1 + 아웃필드 10 으로 XI 를 구성한다.

기준선
  P) 직전 XI 복사 (persistence) — 이미 7.89/11 이다. 이걸 못 이기면 의미 없다.
  S) 최근 W경기 선발률 상위 11
  R) 무작위 11

실행: python -m experiments.opp_predict --test 2025,2026
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

W = 10          # 후보 풀·피처 윈도 (팀 경기 수)
GKPOS = {"GK"}


def load():
    p = pd.read_csv("vaep/output/players.csv")
    g = pd.read_csv("vaep/output/games.csv")
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    g["t"] = np.arange(len(g))
    p = p.join(g.set_index("game_id")[["game_date", "t", "season", "competition_name"]],
               on="game_id")
    p = p.dropna(subset=["t"]).sort_values("t")
    p["is_starter"] = p.is_starter.astype(bool)
    p["minutes_played"] = p.minutes_played.fillna(0.0)
    return p, g


def build_panel(p):
    """팀별로 시간순 진행하며 (경기, 후보) 행을 만든다. 모든 피처는 직전 경기까지."""
    rows = []
    for tid, gr in p.groupby("team_id"):
        gr = gr.sort_values("t")
        hist = []                       # [(t, season, {pid: (started, minutes, rating, pos)})]
        for (gid, t, season), gg in gr.groupby(["game_id", "t", "season"], sort=True):
            rec = {int(r.player_id): (bool(r.is_starter), float(r.minutes_played),
                                      float(r.rating) if np.isfinite(r.rating) else np.nan,
                                      str(r.starting_position_name))
                   for r in gg.itertuples(index=False)}
            past = [h for h in hist if h[1] == season][-W:]
            if len(past) >= 3:
                truth = {pid for pid, v in rec.items() if v[0]}
                cand = sorted({pid for _, _, r in past for pid in r})
                n = len(past)
                for pid in cand:
                    seen = [(k, r[pid]) for k, (_, _, r) in enumerate(past) if pid in r]
                    ks = np.array([k for k, _ in seen], float)
                    stv = np.array([v[0] for _, v in seen], float)
                    mnv = np.array([v[1] for _, v in seen], float)
                    rtv = np.array([v[2] for _, v in seen], float)
                    poss = [v[3] for _, v in seen]
                    w = np.exp((ks - (n - 1)) / 3.0)          # 최근 가중(반감 ~2경기)
                    last_k = ks.max()
                    rows.append(dict(
                        game_id=int(gid), team_id=int(tid), player_id=int(pid), t=int(t),
                        season=int(season), y=int(pid in truth),
                        st_rate=stv.mean(), st_w=(stv * w).sum() / w.sum(),
                        st_last=float(stv[-1]) if last_k == n - 1 else 0.0,
                        st3=stv[ks >= n - 3].mean() if (ks >= n - 3).any() else 0.0,
                        min_w=(mnv * w).sum() / w.sum() / 90.0,
                        app_rate=len(seen) / n,
                        gap=(n - 1 - last_k) / n,             # 최근 몇 경기 명단 밖
                        rating=np.nanmean(rtv) if np.isfinite(rtv).any() else np.nan,
                        is_gk=int(sum(q in GKPOS for q in poss) > len(poss) / 2),
                        npast=n))
            hist.append((int(t), int(season), rec))
    D = pd.DataFrame(rows)
    D["rating"] = D.rating.fillna(D.rating.mean())
    return D


FEATS = ["st_rate", "st_w", "st_last", "st3", "min_w", "app_rate", "gap", "rating", "is_gk"]


def pick_xi(sub, col):
    """GK 1 + 아웃필드 10. 열 col 을 점수로."""
    s = sub.sort_values(col, ascending=False)
    gk = s[s.is_gk == 1]
    out = s[s.is_gk == 0]
    xi = set()
    if len(gk):
        xi.add(int(gk.iloc[0].player_id))
    xi |= set(out.head(11 - len(xi)).player_id.astype(int))
    return xi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="2025,2026")
    ap.add_argument("--cache", default="outputs/opp_panel.parquet")
    args = ap.parse_args()
    test_seasons = {int(s) for s in args.test.split(",")}

    import os
    if os.path.exists(args.cache):
        D = pd.read_parquet(args.cache)
    else:
        p, g = load()
        D = build_panel(p)
        D.to_parquet(args.cache, index=False)
    print(f"패널 {len(D):,}행 · 팀-경기 {D.groupby(['game_id','team_id']).ngroups:,} "
          f"· 후보/경기 {len(D)/D.groupby(['game_id','team_id']).ngroups:.1f}")

    tr = D[~D.season.isin(test_seasons)]
    te = D[D.season.isin(test_seasons)].copy()
    print(f"학습 {len(tr):,} ({sorted(tr.season.unique())}) · 평가 {len(te):,} "
          f"({sorted(te.season.unique())})")
    m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit(tr[FEATS].to_numpy(float), tr.y.to_numpy())
    te["prob"] = m.predict_proba(te[FEATS].to_numpy(float))[:, 1]
    print("계수:", ", ".join(f"{f}={c:+.2f}" for f, c in zip(FEATS, m.coef_[0])))

    rng = np.random.default_rng(0)
    hits = {k: [] for k in ["model", "persist", "strate", "random"]}
    for (gid, tid), sub in te.groupby(["game_id", "team_id"]):
        truth = set(sub[sub.y == 1].player_id.astype(int))
        if len(truth) != 11:
            continue
        sub = sub.copy()
        sub["_rand"] = rng.random(len(sub))
        hits["model"].append(len(pick_xi(sub, "prob") & truth))
        hits["persist"].append(len(pick_xi(sub, "st_last") & truth))
        hits["strate"].append(len(pick_xi(sub, "st_rate") & truth))
        hits["random"].append(len(pick_xi(sub, "_rand") & truth))
    print(f"\n평가 팀-경기 {len(hits['model']):,}")
    print(f"{'방법':<10}{'적중/11':>9}{'표준오차':>9}{'≥9명':>8}")
    for k in ["model", "persist", "strate", "random"]:
        h = np.array(hits[k], float)
        print(f"{k:<10}{h.mean():>9.3f}{h.std(ddof=1)/np.sqrt(len(h)):>9.3f}"
              f"{(h >= 9).mean()*100:>7.1f}%")
    d = np.array(hits["model"], float) - np.array(hits["persist"], float)
    print(f"\nmodel − persist: {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f} "
          f"(t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f})")


if __name__ == "__main__":
    main()
