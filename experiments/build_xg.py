r"""xG 모델 — 슛 단위와 팀 단위를 한 곳에서 만든다.

왜 통합하나
  team_xg.parquet 은 빌더 없이 존재했고, 나중에 만든 shot_xg.parquet 과 모델이
  달라 두 파일의 합이 일치하지 않았다. 하나의 모델에서 슛 단위를 내고 그것을
  합쳐 팀 단위를 만든다.

주의 — 방향 정규화
  SPADL 좌표는 홈 공격 +x 로 고정이다. 원정 슛은 반전하지 않으면 골대까지의
  거리가 반대로 계산된다(과거 이 버그로 페널티 xG 가 0.06 으로 나왔다).

유형 코드: 11 오픈플레이 슛 · 12 페널티 · 13 프리킥
출력  outputs/shot_xg.parquet  (슛 단위, 분 단위 시각 포함)
      outputs/team_xg.parquet  (game_id, team_id, xg)

실행: python -m experiments.build_xg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

L, W = 105.0, 68.0
OUT_S = ROOT / "outputs" / "shot_xg.parquet"
OUT_T = ROOT / "outputs" / "team_xg.parquet"
OUT_N = ROOT / "outputs" / "team_npxg.parquet"      # 무페널티 xG — 논문의 결과 라벨


def main():
    sp = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet",
                         columns=["game_id", "period_id", "time_seconds", "team_id",
                                  "player_id", "start_x", "start_y", "type_id",
                                  "bodypart_id", "result_id"])
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")[["game_id", "home_team_id", "season"]]
    sp["game_id"] = sp.game_id.astype(int)
    sp = sp.merge(g, on="game_id", how="left")
    S = sp[sp.type_id.isin([11, 12, 13])].copy()
    away = S.team_id.astype(float) != S.home_team_id.astype(float)
    S["x"] = np.where(away, L - S.start_x, S.start_x)
    S["y"] = np.where(away, W - S.start_y, S.start_y)
    dx = L - S.x
    dy = np.abs(S.y - W / 2)
    S["dist"] = np.hypot(dx, dy)
    a = np.arctan2(7.32 * dx, dx ** 2 + dy ** 2 - (7.32 / 2) ** 2)
    S["ang"] = np.where(a < 0, a + np.pi, a)
    S["goal"] = (S.result_id == 1).astype(int)
    S["pen"] = (S.type_id == 12).astype(int)
    S["fk"] = (S.type_id == 13).astype(int)
    S["head"] = (S.bodypart_id == 0).astype(int)
    Z = S[["dist", "ang", "pen", "fk", "head"]].to_numpy(float)
    Z = np.hstack([Z, Z[:, [0]] ** 2, np.log1p(Z[:, [0]])])
    y = S.goal.to_numpy()
    # 라벨도 전방 전용: 시즌 s 의 xG 는 s 이전 시즌으로만 적합한다.
    # 가장 이른 시즌은 앞선 시즌이 없으므로 그 시즌 안에서만 경기 단위 폴드로 낸다.
    oof = np.zeros(len(S)); sea = S.season.to_numpy(); gid = S.game_id.to_numpy()
    for s_ in sorted(pd.unique(sea)):
        te = sea == s_; tr = sea < s_
        if tr.sum() >= 2000:
            m = LogisticRegression(max_iter=2000, C=1.0).fit(Z[tr], y[tr])
            oof[te] = m.predict_proba(Z[te])[:, 1]
            print(f"  시즌 {s_}: 학습 {tr.sum():,} (이전 시즌) · 슛 {te.sum():,}", flush=True)
        else:
            idx = np.flatnonzero(te)
            for a_, b_ in GroupKFold(n_splits=4).split(Z[idx], y[idx], gid[idx]):
                m = LogisticRegression(max_iter=2000, C=1.0).fit(Z[idx][a_], y[idx][a_])
                oof[idx[b_]] = m.predict_proba(Z[idx][b_])[:, 1]
            print(f"  시즌 {s_}: 앞선 시즌 없음 → 시즌 내부 4폴드 · 슛 {te.sum():,}", flush=True)
    S["xg"] = oof
    print(f"슛 {len(S):,} | 골 {y.sum():,} ({y.mean():.3f}) | AUC {roc_auc_score(y, oof):.3f}")
    for lab, m in (("페널티", S.pen == 1), ("프리킥", S.fk == 1),
                   ("오픈플레이", (S.pen == 0) & (S.fk == 0))):
        print(f"  {lab:<6} xG 평균 {S.xg[m].mean():.3f} / 실제 {S.goal[m].mean():.3f}"
              f" (n={int(m.sum()):,})")
    S["minute"] = np.where(S.period_id == 1, S.time_seconds / 60,
                           45 + S.time_seconds / 60)
    S[["game_id", "team_id", "period_id", "minute", "xg", "goal",
       "pen", "fk"]].to_parquet(OUT_S, index=False)
    T = (S.groupby(["game_id", "team_id"], as_index=False).xg.sum())
    T["team_id"] = T.team_id.astype(int)
    T.to_parquet(OUT_T, index=False)
    # 무페널티 판: 페널티만 빼고 같은 모델로 합친다. 종전에는 빌더 없이 파일만 있었다.
    N = (S[S.pen == 0].groupby(["game_id", "team_id"], as_index=False).xg.sum()
         .rename(columns={"xg": "npxg"}))
    N["team_id"] = N.team_id.astype(int)
    allk = T[["game_id", "team_id"]]
    N = allk.merge(N, on=["game_id", "team_id"], how="left").fillna({"npxg": 0.0})
    N.to_parquet(OUT_N, index=False)
    print(f"\n저장: {OUT_S} ({len(S):,}행)")
    print(f"저장: {OUT_T} ({len(T):,}행, 경기 {T.game_id.nunique():,})")
    print(f"저장: {OUT_N} ({len(N):,}행, 경기 {N.game_id.nunique():,})")
    print(f"팀-경기 xG 평균 {T.xg.mean():.3f} SD {T.xg.std():.3f} · 무페널티 {N.npxg.mean():.3f} SD {N.npxg.std():.3f}")


if __name__ == "__main__":
    main()
