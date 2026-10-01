"""결과 누수 제거 — 수비에 넘겨주는 기준값을 **결과 이전**으로.

문제
  `fs.result` 가 XFNS 에 있어 **현재 액션의 결과**가 상태가치에 들어간다. 그래서

      슛의 p_scores   골 0.9961 · 노골 0.0263

  즉 막힌 슛은 이미 "실패한 슛"으로 값이 매겨지고, 바로 다음 액션인 골키퍼
  세이브가 물려받을 가치가 남지 않는다(세이브 수비가치 +0.026). 0.30 xG 짜리
  선방도 0.026 을 받는다. 수비 전반이 같은 구조다.

해법
  결과를 통째로 빼면 공격 평가가 망가진다(골은 1.0 이어야 한다). 대신 **결과 이전**
  헤드를 따로 학습해서 수비 기준값만 바꾼다.

      V_off(a) = p_scores(S_a) − prev_off           (그대로 — 슈터는 결과로 평가)
      V_def(a) = prev_def_PRE − p_concedes(S_a)     (수비수는 **막지 않았다면**의 위협 대비)

  prev_def_PRE 는 result_*_a0 열을 가린 채 학습한 헤드의 예측이다.

실행: python -m experiments.vaep_pre [N]      (기본 10 액션)
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vaep")); sys.path.insert(0, str(ROOT / "vaep" / "lib"))
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
import core as C  # noqa: E402
from config import VAEP_OUTPUT_DIR  # noqa: E402
from vaep_relabel import labels_actions, FEAT_CACHE  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MASK = ["result_a0", "result_fail_a0", "result_success_a0", "result_offside_a0",
        "result_owngoal_a0", "result_yellow_card_a0", "result_red_card_a0"]


def xgbm():
    return xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             eval_metric="logloss", enable_categorical=True,
                             random_state=42, n_jobs=-1, tree_method="hist")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sp = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet")
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    sp = sp.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)
    F = pd.read_parquet(FEAT_CACHE).set_index(["game_id", "action_id"])
    F = F.reindex(pd.MultiIndex.from_arrays([sp.game_id.to_numpy(), sp.action_id.to_numpy()]))
    XPRE = [c for c in F.columns if c not in MASK]
    print(f"피처 {F.shape[1]}열 → 결과이전 헤드 {len(XPRE)}열 (가린 열 {len(MASK)})")

    season = dict(zip(games.game_id.astype(int), games.season.astype(int)))
    sp["season"] = sp.game_id.map(season)
    ok = sp.season.notna().to_numpy()
    SC, CD = [], []
    for gid, d in sp.groupby("game_id", sort=False):
        a, c = labels_actions(d, n); SC.append(a); CD.append(c)
    ys, yc = np.concatenate(SC), np.concatenate(CD)
    print(f"레이블 {n}액션 — 득점 {100*ys.mean():.3f}% 실점 {100*yc.mean():.3f}%")

    pSp = np.full(len(sp), np.nan); pCp = np.full(len(sp), np.nan)
    Xp = F[XPRE]
    for s in sorted(set(sp.season.dropna())):
        te = (sp.season == s).to_numpy() & ok; tr = (sp.season != s).to_numpy() & ok
        if te.sum() == 0 or tr.sum() == 0: continue
        t0 = time.time()
        ms, mc = xgbm(), xgbm()
        ms.fit(Xp[tr], ys[tr]); mc.fit(Xp[tr], yc[tr])
        pSp[te] = ms.predict_proba(Xp[te])[:, 1]; pCp[te] = mc.predict_proba(Xp[te])[:, 1]
        print(f"  [{int(s)}] AUC 득점 {roc_auc_score(ys[te], pSp[te]):.4f} "
              f"실점 {roc_auc_score(yc[te], pCp[te]):.4f}  ({time.time()-t0:.0f}초)", flush=True)

    # 결과 포함 헤드는 vaep_oof_a{n} 에서 가져온다
    A = pd.read_parquet(ROOT / "outputs" / f"vaep_oof_a{n}.parquet")
    A = A.set_index(["game_id", "action_id"])
    key = pd.MultiIndex.from_arrays([sp.game_id.to_numpy(), sp.action_id.to_numpy()])
    A = A.reindex(key)
    D = sp[["game_id", "action_id", "period_id", "time_seconds", "team_id",
            "player_id", "type_id", "result_id"]].copy()
    D["p_scores"] = A.p_scores.to_numpy(); D["p_concedes"] = A.p_concedes.to_numpy()
    D["p_scores_pre"] = pSp; D["p_concedes_pre"] = pCp
    D = D[np.isfinite(D.p_scores) & np.isfinite(D.p_scores_pre)].reset_index(drop=True)
    print(f"슛 p_scores(결과포함) vs (결과이전):")
    sh = D[D.type_id.isin([11, 12, 13])]
    print(sh.groupby("result_id").agg(n=("p_scores", "size"), 포함=("p_scores", "mean"),
                                      이전=("p_scores_pre", "mean")).round(4).to_string())

    # V_def 를 결과이전 기준으로 다시 계산
    V = C._compute_vaep_values(D.rename(columns={"p_scores": "_ps", "p_concedes": "_pc"})
                              .assign(p_scores=lambda d: d._ps, p_concedes=lambda d: d._pc))
    # prev_def_PRE 로 다시: 같은 로직을 결과이전 확률로 한 번 더 돌려 prev 만 취한다
    D2 = D.copy(); D2["p_scores"] = D.p_scores_pre; D2["p_concedes"] = D.p_concedes_pre
    V2 = C._compute_vaep_values(D2)
    V2 = V2.sort_values(["game_id", "action_id"]).reset_index(drop=True)
    V = V.sort_values(["game_id", "action_id"]).reset_index(drop=True)
    # V2.defensive_value = prev_def_pre - p_concedes_pre  →  prev_def_pre 복원
    prev_def_pre = V2.defensive_value.to_numpy() + V2.p_concedes.to_numpy()
    V["defensive_value_pre"] = prev_def_pre - V.p_concedes.to_numpy()
    V["vaep_pre"] = V.offensive_value + V.defensive_value_pre
    out = ROOT / "outputs" / f"vaep_oof_pre{n}.parquet"
    V.to_parquet(out)
    r0 = V.defensive_value.abs().mean() / V.offensive_value.abs().mean()
    r1 = V.defensive_value_pre.abs().mean() / V.offensive_value.abs().mean()
    print(f"\n|수비|/|공격|   결과포함 {r0:.3f}  →  결과이전 {r1:.3f}")
    for t_, nm in [(14, "keeper_save"), (16, "keeper_punch"), (15, "keeper_claim"),
                   (10, "interception"), (9, "tackle"), (18, "clearance")]:
        m = V.type_id == t_
        print(f"  {nm:13s} n={int(m.sum()):6,}  수비가치 {V.loc[m,'defensive_value'].mean():+.4f}"
              f"  →  {V.loc[m,'defensive_value_pre'].mean():+.4f}")
    print(f"\n저장 {out}")


if __name__ == "__main__":
    main()
