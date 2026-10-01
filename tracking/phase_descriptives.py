"""초록용 기술 통계: (1) 상대가 페이즈 비중을 얼마나 바꾸나 (2) 선수 페이즈 프로필의 반쪽 신뢰도 (3) 페이즈마다 선수 순위가 달라지나.

(1) 팀-경기 페이즈 비중 ~ 팀-시즌 고정효과 [+ 상대-시즌 고정효과]: 상대를 넣었을 때 늘어나는 설명력(수정 R²)
(2) 트래킹 10경기 이상 선수(필드), 경기를 홀수·짝수 번째로 나눠 반쪽 평균 → 선수 간 상관 → 스피어만-브라운 보정
(3) 같은 지표의 페이즈 간 선수 순위 상관 (스피어만)
실행  python tracking/phase_descriptives.py
"""
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"
EX = Path(__file__).resolve().parent.parent / "vaep/output"
PH = ["build_up", "settled_attack", "transition_attack", "pressing", "settled_defense", "transition_defense"]


def adj_r2(y, X):
    X = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    n, k = X.shape; return 1 - (r @ r / (n - k)) / (y.var(ddof=1))


def main():
    g = pd.read_csv(EX / "games.csv", usecols=["game_id", "season", "home_team_id", "away_team_id", "game_date"])
    tf = pd.read_parquet(OUT / "team_phase_match.parquet"); tf = tf[tf.tax == "phase"]
    sh = tf.pivot_table(index=["game_id", "team"], columns="label", values="share").fillna(0).reset_index().merge(g, on="game_id")
    sh["opp"] = np.where(sh.team == sh.home_team_id, sh.away_team_id, sh.home_team_id)
    sh["ts"] = sh.team.astype(str) + "_" + sh.season.astype(str); sh["os"] = sh.opp.astype(str) + "_" + sh.season.astype(str)
    sh["home"] = (sh.team == sh.home_team_id).astype(float)
    T = pd.get_dummies(sh.ts, drop_first=True).to_numpy(float); O = pd.get_dummies(sh.os, drop_first=True).to_numpy(float)
    print(f"(1) 팀-경기 {len(sh):,} · 팀-시즌 {sh.ts.nunique()} — 페이즈 비중의 수정 R²: 팀+홈 → 팀+홈+상대")
    for p in PH + ["poss"]:
        y = (sh[["build_up", "settled_attack", "transition_attack"]].sum(1) if p == "poss" else sh[p]).to_numpy(float)
        a = adj_r2(y, np.column_stack([T, sh.home])); b = adj_r2(y, np.column_stack([T, O, sh.home]))
        print(f"   {p:18s} 평균 {y.mean():.3f} SD {y.std():.3f} · 팀 {a:.3f} → +상대 {b:.3f} (상대 몫 +{b - a:.3f})")

    pp = pd.read_parquet(OUT / "player_phase_match.parquet"); pp = pp[(pp.tax == "phase") & (pp.pos != "GK") & pp.ex_pid.notna()].merge(g[["game_id", "game_date"]], on="game_id")
    pp = pp[pp.minutes >= 1.0]
    cnt = pp.groupby("ex_pid").game_id.nunique(); keep = cnt[cnt >= 10].index; q = pp[pp.ex_pid.isin(keep)].copy()
    q["k"] = q.groupby(["ex_pid", "label"]).game_date.rank(method="first"); q["half"] = (q.k % 2).astype(int)
    print(f"\n(2) 반쪽 신뢰도 (트래킹 10경기 이상 필드 선수 {len(keep)}명, 스피어만-브라운 보정)")
    rows = []
    for m in ["ax", "ayw", "v", "hi", "engage", "dball"]:
        h = q.groupby(["ex_pid", "label", "half"])[m].mean().unstack("half").dropna()
        for p in PH:
            if p not in h.index.get_level_values("label"): continue
            x = h.xs(p, level="label"); r = x[0].corr(x[1]); rows.append((m, p, 2 * r / (1 + r)))
    R = pd.DataFrame(rows, columns=["metric", "phase", "rel"]).pivot(index="metric", columns="phase", values="rel")[PH]
    print(R.round(2).to_string())
    print("\n(3) 같은 지표의 페이즈 간 선수 순위 상관 (스피어만, 선수 평균)")
    M = q.groupby(["ex_pid", "label"])[["engage", "hi", "ax"]].mean().unstack("label")
    for m in ["engage", "hi", "ax"]:
        C = M[m][PH].corr(method="spearman")
        print(f"   {m}: 빌드업↔압박 {C.loc['build_up', 'pressing']:.2f} · 압박↔기본수비 {C.loc['pressing', 'settled_defense']:.2f} · 기본공격↔압박 {C.loc['settled_attack', 'pressing']:.2f} · 공격전환↔수비전환 {C.loc['transition_attack', 'transition_defense']:.2f}")

    # (4) abstract: same comparison within position (subtract the mean of each player's modal position)
    pos = q.groupby("ex_pid").pos.agg(lambda s: s.mode().iloc[0])
    Mw = M["hi"][PH].sub(M["hi"][PH].groupby(pos).transform("mean"))
    print(f"\n(4) 고강도 비율, 같은 포지션 안 순위 상관: 압박↔기본공격 {Mw.pressing.corr(Mw.settled_attack, method='spearman'):.2f} · 압박↔공격전환 {Mw.pressing.corr(Mw.transition_attack, method='spearman'):.2f}")


if __name__ == "__main__":
    main()
