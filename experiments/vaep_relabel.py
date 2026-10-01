"""VAEP 레이블 창 재검토 — 수비가치가 왜 4배 눌리는가, 고치면 어떻게 되는가.

현행 `run_oof_vaep` 은 `scores_by_seconds`(10**초**)를 쓴다. socceraction 표준은
`scores_by_actions`(10**액션**)다. 10초 창은 두 레이블을 비대칭으로 자른다 —

    레이블        득점 양성률  실점 양성률   비
    10 액션        1.206%     0.429%    0.355
    10 초         0.586%     0.095%    0.163   ← 실점만 4.5배 잘린다

실점하려면 먼저 공을 뺏겨야 해서 시간이 더 걸리는데, 짧은 시간창이 그걸 잘라낸다.
결과가 |수비가치| = |공격가치| × 0.263 이다.

여기서는 SPADL 캐시로부터 **액션 기반 레이블**로 VAEP 를 다시 계산하고
(득점·실점 헤드 각각, 시즌 LOSO OOF, 같은 XGB 설정) 창 길이별로 비교한다.

실행: python -m experiments.vaep_relabel [10 20 ...]
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vaep")); sys.path.insert(0, str(ROOT / "vaep" / "lib"))
sys.path.insert(0, str(ROOT / "gnn"))
import core as C  # noqa: E402
from config import VAEP_OUTPUT_DIR  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SHOT, SUCCESS, OWNGOAL = {11, 12, 13}, 1, 3
FEAT_CACHE = ROOT / "outputs" / "vaep_features.parquet"


def build_features(sp, games):
    """경기별 피처 → 하나의 표 (캐시)."""
    if FEAT_CACHE.exists():
        F = pd.read_parquet(FEAT_CACHE)
        print(f"피처 캐시 {len(F):,}행 × {F.shape[1]}열")
        return F
    home = dict(zip(games.game_id.astype(int), games.home_team_id.astype(int)))
    t0, parts = time.time(), []
    for i, (gid, a) in enumerate(sp.groupby("game_id", sort=False)):
        try:
            X = C.compute_features(a, home[int(gid)])
        except Exception as e:
            print(f"  ✗ {gid}: {e}"); continue
        X = X.copy(); X["game_id"] = int(gid); X["action_id"] = a.action_id.to_numpy()
        parts.append(X)
        if (i + 1) % 500 == 0:
            print(f"  {i+1} 경기 {time.time()-t0:.0f}초", flush=True)
    F = pd.concat(parts, ignore_index=True)
    F.to_parquet(FEAT_CACHE)
    print(f"피처 {len(F):,}행 × {F.shape[1]}열  ({time.time()-t0:.0f}초)")
    return F


def labels_actions(d, n):
    """socceraction 표준: 이후 n 액션 안에 그 팀이 득점/실점했는가."""
    goal = (d.type_id.isin(SHOT) & (d.result_id == SUCCESS)).to_numpy()
    og = (d.result_id == OWNGOAL).to_numpy(); tm = d.team_id.to_numpy()
    sc, cd = goal.copy(), og.copy()
    for i in range(1, n):
        g = np.r_[goal[i:], np.zeros(i, bool)]; o = np.r_[og[i:], np.zeros(i, bool)]
        t = np.r_[tm[i:], np.full(i, -1)]
        sc |= (g & (t == tm)) | (o & (t != tm))
        cd |= (g & (t != tm)) | (o & (t == tm))
    return sc, cd


def main():
    ns = [int(x) for x in sys.argv[1:]] or [10, 20]
    sp = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet")
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    sp = sp.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)
    F = build_features(sp, games)
    F = F.set_index(["game_id", "action_id"])
    key = pd.MultiIndex.from_arrays([sp.game_id.to_numpy(), sp.action_id.to_numpy()])
    F = F.reindex(key)
    XCOLS = [c for c in F.columns]
    print(f"정렬 후 피처 {F.shape}  결측행 {int(F.isna().all(1).sum()):,}")

    season = dict(zip(games.game_id.astype(int), games.season.astype(int)))
    sp["season"] = sp.game_id.map(season)
    ok = sp.season.notna().to_numpy() & (~F.isna().all(1).to_numpy())
    print(f"사용 가능한 액션 {int(ok.sum()):,} / {len(sp):,}")

    for n in ns:
        print(f"\n{'='*60}\n=== 레이블: {n} 액션 ===", flush=True)
        SC, CD = [], []
        for gid, d in sp.groupby("game_id", sort=False):
            a, c = labels_actions(d, n); SC.append(a); CD.append(c)
        ys = np.concatenate(SC); yc = np.concatenate(CD)
        print(f"양성률  득점 {100*ys.mean():.3f}%  실점 {100*yc.mean():.3f}%  비 {yc.mean()/ys.mean():.3f}")
        pS = np.full(len(sp), np.nan); pC = np.full(len(sp), np.nan)
        Xall = F[XCOLS]
        metrics = []
        for s in sorted(set(sp.season.dropna())):
            te = (sp.season == s).to_numpy() & ok
            tr = (sp.season != s).to_numpy() & ok
            if te.sum() == 0 or tr.sum() == 0: continue
            t0 = time.time()
            ms = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                   eval_metric="logloss", enable_categorical=True,
                                   random_state=42, n_jobs=-1, tree_method="hist")
            mc = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                   eval_metric="logloss", enable_categorical=True,
                                   random_state=42, n_jobs=-1, tree_method="hist")
            ms.fit(Xall[tr], ys[tr]); mc.fit(Xall[tr], yc[tr])
            pS[te] = ms.predict_proba(Xall[te])[:, 1]
            pC[te] = mc.predict_proba(Xall[te])[:, 1]
            a1 = roc_auc_score(ys[te], pS[te]); a2 = roc_auc_score(yc[te], pC[te])
            metrics.append(dict(season=int(s), auc_scores=a1, auc_concedes=a2))
            print(f"  [{int(s)}] 학습 {int(tr.sum()):,} 테스트 {int(te.sum()):,}  "
                  f"AUC 득점 {a1:.4f} 실점 {a2:.4f}  ({time.time()-t0:.0f}초)", flush=True)
        D = sp[["game_id", "action_id", "period_id", "time_seconds", "team_id",
                "player_id", "type_id", "result_id"]].copy()
        D["p_scores"] = pS; D["p_concedes"] = pC
        D = D[np.isfinite(pS)].reset_index(drop=True)
        V = C._compute_vaep_values(D)
        out = ROOT / "outputs" / f"vaep_oof_a{n}.parquet"
        V.to_parquet(out)
        r = V.defensive_value.abs().mean() / V.offensive_value.abs().mean()
        print(f"  p_scores 평균 {V.p_scores.mean():.5f}  p_concedes 평균 {V.p_concedes.mean():.5f}")
        print(f"  |수비|/|공격| = {r:.3f}   (현행 10초판 0.263)")
        print(f"  저장 {out}")
        json.dump(metrics, open(ROOT / "outputs" / f"vaep_oof_a{n}_metrics.json", "w"))


if __name__ == "__main__":
    main()
