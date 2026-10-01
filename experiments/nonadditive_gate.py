"""비가산 관문 — f(우리, 상대)가 선형 가산 f(우리−상대)를 이기는가.

동기: 가산 모델에선 argmax 에서 상대 항이 소거돼 조건부 추천이 불가능.
비가산 f 는 소거가 깨지지만, 그 투자가 정당하려면 먼저 **깨끗한 경기 단위
데이터에서 예측이라도** 이겨야 한다 (구간 데이터는 상태 누수로 부적격 판정).

설계: 입력 = [우리 XI 합 PCA20, 상대 XI 합 PCA20] (차분 아님 — 교차항 학습 가능)
      모델 = MLP(64,32)·HGB vs 기준 = 현행 선형(차분 PCA20)
      판정 = 전방 연쇄 보류 폴드의 경기 npxG 차 R², 경기 블록 부트스트랩.

실행: python -m experiments.nonadditive_gate
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp.parquet")
os.environ.setdefault("DROPFAM", "zv,zn")
from sklearn.linear_model import Ridge              # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from sklearn.neural_network import MLPRegressor     # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from lineup_pipeline import r2w, AL                 # noqa: E402
from lineup_event_panel import _snapshots, MINPL    # noqa: E402
from config import VAEP_OUTPUT_DIR                  # noqa: E402

NPC = 20


def build_sides():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    by, T, dim, FC = _snapshots(gpos)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(  # noqa: E712
        lambda s: [int(x) for x in s])
    st = st[st.str.len() == 11]
    npxg = pd.read_parquet(ROOT / "outputs/team_npxg.parquet").set_index(
        ["game_id", "team_id"]).npxg

    def fx(ids, tt):
        v = np.zeros(dim, np.float32); c = 0
        for p in ids:
            a = T.get(p)
            if a is None: continue
            j = np.searchsorted(a, tt, "right") - 1
            if j < 0: continue
            v += by[p][j][1]; c += 1
        return v, c

    rows, FA, FB = [], [], []
    for r in g.itertuples(index=False):
        gid = int(r.game_id)
        ha, aw = int(r.home_team_id), int(r.away_team_id)
        if (gid, ha) not in st.index or (gid, aw) not in st.index: continue
        tt = gpos[gid]
        va, ca = fx(st.loc[(gid, ha)], tt); vb, cb = fx(st.loc[(gid, aw)], tt)
        if min(ca, cb) < MINPL: continue
        for me, opp, vm, vo in ((ha, aw, va, vb), (aw, ha, vb, va)):
            yv = float(npxg.get((gid, me), np.nan)) - float(npxg.get((gid, opp), np.nan))
            if not np.isfinite(yv): continue
            rows.append(dict(game_id=gid, t=tt, season=int(r.season),
                             home=1.0 if me == ha else 0.0, y=yv))
            FA.append(vm); FB.append(vo)
    R = pd.DataFrame(rows)
    A = np.vstack(FA).astype(np.float64); Bm = np.vstack(FB).astype(np.float64)
    keep = (A - Bm).std(0) > 1e-9
    A, Bm = A[:, keep], Bm[:, keep]
    SEA = pd.get_dummies(R.season.astype(str)).to_numpy(float)
    BASE = np.hstack([R[["home"]].to_numpy(float), SEA])
    if os.environ.get("TEAMFE", "0") == "1":                                   # 팀 고정효과: 우리 팀·상대 팀 더미를 기저에 넣어 점수가 팀 전력을 흡수하지 않게 한다
        import pandas as _pd
        hm = g.set_index("game_id"); me = np.where(R.home.to_numpy() == 1, hm.home_team_id.reindex(R.game_id), hm.away_team_id.reindex(R.game_id)).astype(int)
        op = np.where(R.home.to_numpy() == 1, hm.away_team_id.reindex(R.game_id), hm.home_team_id.reindex(R.game_id)).astype(int)
        tid = sorted(set(me) | set(op)); ix = {t: j for j, t in enumerate(tid)}
        TFE = np.zeros((len(R), len(tid)), float)
        for i, (a_, b_) in enumerate(zip(me, op)): TFE[i, ix[a_]] += 1.0; TFE[i, ix[b_]] -= 1.0
        BASE = np.hstack([BASE, TFE]); print(f"TEAMFE: 팀 더미 {len(tid)} 열 추가 (우리 +1 / 상대 −1)", flush=True)
    BASE = (BASE - BASE.mean(0)) / np.where(BASE.std(0) == 0, 1, BASE.std(0))
    gord = R.groupby("game_id").t.first().sort_values().index.to_numpy()
    IDX = [np.flatnonzero(R.game_id.isin(set(x)).to_numpy()) for x in np.array_split(gord, 5)]
    return R, A, Bm, BASE, IDX


def main():
    R, A, Bm, BASE, IDX = build_sides()
    y = R.y.to_numpy()
    D = A - Bm
    D = (D - D.mean(0)) / D.std(0).clip(1e-9)
    mu, sd = A.mean(0), A.std(0).clip(1e-9)
    As = (A - mu) / sd; Bs = (Bm - mu) / sd          # 같은 척도 (대칭성)
    print(f"사례 {len(R):,}")

    preds = {k: np.full(len(R), np.nan) for k in ("linear", "mlp", "hgb")}
    for f in range(1, 5):
        tr = np.concatenate(IDX[:f]); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
        pc = PCA(n_components=NPC, random_state=0).fit(D[i1])
        Z = pc.transform(D); Z = (Z - Z[i1].mean(0)) / Z[i1].std(0).clip(1e-9)
        X = np.hstack([BASE, Z])
        ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(X[i1], y[i1]).predict(X[i2]),
                                       np.ones(len(i2))))
        preds["linear"][IDX[f]] = Ridge(alpha=ba).fit(X[tr], y[tr]).predict(X[IDX[f]])
        pc2 = PCA(n_components=NPC, random_state=0).fit(np.vstack([As[i1], Bs[i1]]))
        Za = pc2.transform(As); Zb = pc2.transform(Bs)
        zm, zs = Za[i1].mean(0), Za[i1].std(0).clip(1e-9)
        Za = (Za - zm) / zs; Zb = (Zb - zm) / zs
        X2 = np.hstack([BASE, Za, Zb])
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                           validation_fraction=0.15, max_iter=600, random_state=0)
        mlp.fit(X2[tr], y[tr]); preds["mlp"][IDX[f]] = mlp.predict(X2[IDX[f]])
        hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                            early_stopping=True, validation_fraction=0.15,
                                            random_state=0)
        hgb.fit(X2[tr], y[tr]); preds["hgb"][IDX[f]] = hgb.predict(X2[IDX[f]])

    m = np.isfinite(preds["linear"])
    ix0 = np.flatnonzero(m)
    gids = R.game_id.to_numpy()
    G = {}
    for i in ix0: G.setdefault(int(gids[i]), []).append(i)
    keys = list(G); rng = np.random.default_rng(0)
    def r2(pv, ix):
        return 1 - ((y[ix] - pv[ix]) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    print(f"기준(차분 선형) R² {r2(preds['linear'], ix0):+.4f}")
    for k in ("mlp", "hgb"):
        ds = []
        for _ in range(2000):
            pick = rng.choice(len(keys), len(keys), replace=True)
            ix = np.concatenate([G[keys[j]] for j in pick])
            ds.append(r2(preds[k], ix) - r2(preds["linear"], ix))
        ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"{k.upper():4s} (양측 비가산) R² {r2(preds[k], ix0):+.4f}  "
              f"Δ={np.mean(ds):+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ>0)={np.mean(ds > 0):.3f}")


if __name__ == "__main__":
    main()
