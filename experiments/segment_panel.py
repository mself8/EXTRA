"""10분 구간 패널 — "쪼개기 + 실제 뛴 선수" 제안의 직접 검정.

행 = (경기, 팀측, 10분 구간) ≈ 45,000. 입력 = 그 구간 실제 온필드 선수들의
과거-only 스냅샷 합(교체 반영, subs.parquet), 라벨 = 구간 npxG 차 per-90,
가중 = 구간 길이. 심판 = 보류 경기의 **경기 전체** npxG 차 — 구간 예측을
노출 가중 합산해 현행(선발 XI·경기 단위) 모델과 짝지어 비교.

실행: python -m experiments.segment_panel
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp.parquet")
os.environ.setdefault("DROPFAM", "zv,zn")
from sklearn.linear_model import Ridge              # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from lineup_pipeline import r2w, AL                 # noqa: E402
import lineup_event_panel as LEP                    # noqa: E402
from config import VAEP_OUTPUT_DIR                  # noqa: E402

NPC, NBIN = 20, 9
EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200.0]


def pid_maps():
    PM = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_merge.json")).items()}
    PMb = {int(k): int(v) for k, v in
           json.load(open(ROOT / ("outputs/pid_merge_old.json" if (ROOT / "outputs/pid_merge_old.json").exists() else "outputs/backup_pre_person_bridge/pid_merge.json"))).items()}
    B1 = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_bridge_2026.json")).items()}
    _f26 = ROOT / "outputs/pid_merge_2026.json"
    B2 = {int(k): int(v) for k, v in json.load(open(_f26)).items()} if _f26.exists() else {k: v for k, v in PM.items() if PMb.get(k) != v}
    comp = {}
    for k, v in B1.items():
        mk = PMb.get(k)
        if mk is not None and mk not in B1 and mk not in comp: comp[mk] = v
    F1 = {**comp, **B1}
    def old_map(p): return PMb.get(int(p), int(p))
    def new_map(p):
        x = F1.get(int(p), int(p)); return B2.get(x, x)
    return old_map, new_map


def main():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    g26 = set(g[g.season == 2026].game_id.astype(int))
    old_map, new_map = pid_maps()
    by, T, dim, FC = LEP._snapshots(gpos)

    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == 1].groupby(["game_id", "team_id"]).player_id.apply(
        lambda s: [int(x) for x in s])
    S = pd.read_parquet(ROOT / "outputs/subs.parquet")
    S["pid"] = [new_map(p) if gid in g26 else old_map(p)
                for p, gid in zip(S.player_id, S.game_id)]
    subs = defaultdict(list)
    for r in S.itertuples(index=False):
        subs[(int(r.game_id), int(r.team_id))].append((float(r.t), str(r.dir), int(r.pid)))
    sx = pd.read_parquet(ROOT / "outputs/shot_xg.parquet")
    sx = sx[sx.pen == 0]
    sx["bin"] = np.digitize(sx.minute, EDGES[1:-1])
    binxg = sx.groupby(["game_id", "team_id", "bin"]).xg.sum()
    gend = sx.groupby("game_id").minute.max().clip(lower=90.0)

    def snap(p, tt):
        a = T.get(int(p))
        if a is None: return None
        j = np.searchsorted(a, tt, "right") - 1
        return by[int(p)][j][1] if j >= 0 else None

    rows, X = [], []
    for r in g.itertuples(index=False):
        gid = int(r.game_id); tt = gpos[gid]
        ha, aw = int(r.home_team_id), int(r.away_team_id)
        if (gid, ha) not in st.index or (gid, aw) not in st.index: continue
        SN, ONF = {}, {}
        okg = True
        for tid in (ha, aw):
            cur = set(st.loc[(gid, tid)])
            ev = sorted(subs.get((gid, tid), []))
            tl, cnt = [], 0
            for b in range(NBIN):
                lo = EDGES[b]
                while ev and ev[0][0] <= lo + 1e-9:
                    t_, d_, p_ = ev.pop(0)
                    if d_ == "Off": cur.discard(p_)
                    else: cur.add(p_)
                tl.append(frozenset(cur))
            ONF[tid] = tl
            for p in set().union(*tl):
                if p not in SN:
                    SN[p] = snap(p, tt)
        MODE = os.environ.get("SEGMODE", "actual")
        def tsum(tid, b):
            src = set(st.loc[(gid, tid)]) if MODE == "starters" else ONF[tid][b]
            v = np.zeros(dim, np.float32); c = 0
            for p in src:
                if p not in SN: SN[p] = snap(p, tt)
                s = SN.get(p)
                if s is not None: v += s; c += 1
            return v, c
        ge = float(gend.get(gid, 95.0))
        for tid, oid in ((ha, aw), (aw, ha)):
            for b in range(NBIN):
                lo = EDGES[b]; hi = min(EDGES[b + 1], ge)
                ln = hi - lo
                if ln <= 1: continue
                va, ca = tsum(tid, b); vb, cb = tsum(oid, b)
                if min(ca, cb) < 8: continue
                yv = (float(binxg.get((gid, tid, b), 0.0))
                      - float(binxg.get((gid, oid, b), 0.0))) * 90.0 / ln
                rows.append(dict(game_id=gid, me=tid, t=tt, b=b, w=ln / 90.0, y=yv,
                                 home=1.0 if tid == ha else 0.0, season=int(r.season)))
                X.append(va - vb)
    R = pd.DataFrame(rows)
    M = np.vstack(X).astype(np.float64)
    keep = M.std(0) > 1e-9; M = M[:, keep]
    Msd = M.std(0)
    M = (M - M.mean(0)) / Msd
    SEA = pd.get_dummies(R.season.astype(str)).to_numpy(float)
    B = np.hstack([R[["home"]].to_numpy(float), SEA])
    B = (B - B.mean(0)) / np.where(B.std(0) == 0, 1, B.std(0))
    print(f"구간 행 {len(R):,} · 경기 {R.game_id.nunique():,}")

    gord = R.groupby("game_id").t.first().sort_values().index.to_numpy()
    IDX = [np.flatnonzero(R.game_id.isin(set(x)).to_numpy()) for x in np.array_split(gord, 5)]
    y = R.y.to_numpy(); wt = R.w.to_numpy()
    pred = np.full(len(R), np.nan); WSAVE = {}
    for f in range(1, 5):
        tr = np.concatenate(IDX[:f]); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
        pc = PCA(n_components=NPC, random_state=0).fit(M[i1])
        Z = pc.transform(M); zsd_ = Z[i1].std(0).clip(1e-9)
        Z = (Z - Z[i1].mean(0)) / zsd_
        Xd = np.hstack([B, Z])
        ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(
            Xd[i1], y[i1], sample_weight=wt[i1]).predict(Xd[i2]), wt[i2]))
        mm = Ridge(alpha=ba).fit(Xd[tr], y[tr], sample_weight=wt[tr])
        pred[IDX[f]] = mm.predict(Xd[IDX[f]])
        if os.environ.get("SAVEW") == "1":
            WSAVE[f] = ((pc.components_.T / Z[np.concatenate(IDX[:f])[:int(.8*len(np.concatenate(IDX[:f])))]].std(0).clip(1e-9) if False else pc.components_.T / zsd_) @ mm.coef_[B.shape[1]:]) / Msd

    # 경기 수준 합산 예측: Σ ŷ_bin · w
    R["pred"] = pred
    E = R.dropna(subset=["pred"])
    gp = E.groupby(["game_id", "me"]).apply(
        lambda d: pd.Series(dict(yhat=float((d.pred * d.w).sum()),
                                 yg=float((d.y * d.w).sum()))))
    # 기준: 현행 경기 단위 모델 (선발 XI)
    R0, F0, B0, IDX0, _ = LEP.build_event("npxg")
    y0 = R0.y.to_numpy(); p0 = np.full(len(R0), np.nan)
    M0 = F0["vaepon"]
    for f in range(1, 5):
        tr = np.concatenate(IDX0[:f]); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
        pc = PCA(n_components=NPC, random_state=0).fit(M0[i1])
        Z = pc.transform(M0); Z = (Z - Z[i1].mean(0)) / Z[i1].std(0).clip(1e-9)
        Xd = np.hstack([B0, Z])
        ba = max(AL, key=lambda a: r2w(y0[i2], Ridge(alpha=a).fit(Xd[i1], y0[i1]).predict(Xd[i2]),
                                       np.ones(len(i2))))
        mm = Ridge(alpha=ba).fit(Xd[tr], y0[tr])
        p0[IDX0[f]] = mm.predict(Xd[IDX0[f]])
    K0 = {(int(a), int(b)): (p0[i], y0[i]) for i, (a, b) in
          enumerate(zip(R0.game_id, R0.me)) if np.isfinite(p0[i])}
    common = [k for k in gp.index if (int(k[0]), int(k[1])) in K0]
    ys = np.array([K0[(int(a), int(b))][1] for a, b in common])
    pa = np.array([K0[(int(a), int(b))][0] for a, b in common])
    pb = np.array([gp.loc[k, "yhat"] for k in common])
    gids = np.array([int(a) for a, _ in common])
    def r2(pv, ix):
        return 1 - ((ys[ix] - pv[ix]) ** 2).sum() / ((ys[ix] - ys[ix].mean()) ** 2).sum()
    G = {}
    for i, gg_ in enumerate(gids): G.setdefault(gg_, []).append(i)
    keys = list(G); rng = np.random.default_rng(0); ds = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ix = np.concatenate([G[keys[j]] for j in pick])
        ds.append(r2(pb, ix) - r2(pa, ix))
    ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
    allix = np.arange(len(common))
    print(f"경기 수준 판정 (공통 {len(common):,} 팀-경기, 경기 전체 y)")
    print(f"  현행(선발 XI·경기 단위)  R² {r2(pa, allix):+.4f}")
    print(f"  구간(실제출전·10분×9)   R² {r2(pb, allix):+.4f}")
    print(f"  ΔR²(구간−현행) = {np.mean(ds):+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ>0)={np.mean(ds>0):.3f}")
    if os.environ.get("SAVEW") == "1":
        fold_of = {}
        for f in range(1, 5):
            for gid_ in set(R.game_id.iloc[IDX[f]]): fold_of[int(gid_)] = f
        np.savez(ROOT / "outputs" / "segment_W.npz",
                 keep=keep, FC=np.array(FC), fold_gids=np.array(list(fold_of)),
                 fold_ids=np.array([fold_of[k] for k in fold_of]),
                 **{f"W{f}": WSAVE[f] for f in WSAVE})
        print("저장 outputs/segment_W.npz")


if __name__ == "__main__":
    main()
