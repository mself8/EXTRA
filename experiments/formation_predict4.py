"""포메이션 예측 v4 — 전 정보 블록 투입 + 블록별 절제 (병합 라벨).

블록
  S 모양 이력   직전 1~3 모양 원핫 + 최빈 + 습관강도            (v3 와 동일)
  R 성적       최근 3·5경기 승점, 5경기 득실차
  M 감독       감독의 과거 모양 최빈(팀 무관, 과거-only) 원핫 + 신임(≤10) + log(재임경기)
  O 상대       상대의 모양 최빈 원핫 + 상대 pts5 + 상대 신임
  P 선수단     결장 역할 7 + 오늘 명단의 역할별 공격 스냅샷 질량 6 (과거-only)

평가: 병합 라벨(동치류) · ≤2023 학습+2024 · 2025~26 정확일치.
절제: 지속성 → S → S+R → S+R+M → S+R+M+P → 전부(+O).

실행: python -m experiments.formation_predict4
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ["ONBALL_FILE"] = "onball_gk_resid_merged_defresp.parquet"
os.environ.setdefault("DROPFAM", "zv,zn")
from config import VAEP_OUTPUT_DIR, RAW_DATA_DIR  # noqa: E402
from gap_constrained import GK_MAX_LVL  # noqa: E402
from gap_lane import declared_grid  # noqa: E402
import json  # noqa: E402

TOL = 0.02


def merged_labels():
    """(gid,tid) -> 병합 모양 클래스 (formation_canon 의 union-find 재현)."""
    YS = {}
    for f in RAW_DATA_DIR.glob("*/*/match/*/lineup.json"):
        try: d = json.load(open(f))["result"]
        except Exception: continue
        gid = int(f.parent.name)
        buck = defaultdict(list)
        for e in d:
            q = e.get("position") or {}
            if e.get("is_starting_lineup") and "y" in q:
                buck[int(e.get("team_id", -1))].append(float(q["y"]))
        for tid, ys in buck.items():
            ys = sorted(y for y in ys if y > GK_MAX_LVL)
            if len(ys) == 10: YS[(gid, tid)] = np.array(ys)
    CELL, _, _ = declared_grid()
    OLD = {k: tuple(int(x) for x in np.asarray(G).sum(1) if x) for k, G in CELL.items()}
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date")
    seq = defaultdict(list)
    for r in g.itertuples(index=False):
        for tid in (int(r.home_team_id), int(r.away_team_id)):
            k = (int(r.game_id), tid)
            if k in YS and k in OLD: seq[tid].append(k)
    pairD = defaultdict(list)
    for tid, ks in seq.items():
        for k1, k2 in zip(ks, ks[1:]):
            if OLD[k1] != OLD[k2]:
                pairD[frozenset((OLD[k1], OLD[k2]))].append(float(np.abs(YS[k1] - YS[k2]).mean()))
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    # BACKLINE=1: 뒷선 인원이 다른 쌍은 병합하지 않는다 — 공시 y 는 템플릿이라
    # 격자 한 칸 차이가 백4/백3 를 가르는데, 그 구분은 표기 잡음으로 볼 수 없다.
    KEEP_BACK = os.environ.get("BACKLINE", "0") == "1"
    MODE = os.environ.get("MERGE", "1")
    if MODE == "0":                                              # 병합 없음 — 공시 라벨 그대로
        return {k: s for k, s in OLD.items()}
    if MODE == "3part":
        # 관례 표기 (수비-미드필드-공격): 첫 밴드=뒷선, 마지막 밴드=최전방, 사이는 미드필드로 합친다.
        # 공시 y 는 템플릿이라 중간 밴드의 쪼개짐이 표기 잡음의 자리이고, 뒷선·최전방은 보존된다.
        def three(t): return t if len(t) < 3 else (t[0], sum(t[1:-1]), t[-1])
        return {k: three(s) for k, s in OLD.items()}
    for pr, ds in pairD.items():
        if len(ds) >= 10 and np.median(ds) <= TOL:
            a, b = list(pr)
            if KEEP_BACK and a[0] != b[0]: continue
            parent[find(a)] = find(b)
    return {k: find(s) for k, s in OLD.items()}


def main():
    NEW = merged_labels()
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    SEA = dict(zip(g.game_id, g.season)); CMP = dict(zip(g.game_id, g.competition_name))
    GM = pd.read_parquet(ROOT / "outputs" / "game_manager.parquet")
    MGR = {(int(r.game_id), int(r.team_id)): (str(r.manager), int(r.mgr_game_no))
           for r in GM.itertuples(index=False) if r.manager}
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    stl = pl[pl.is_starter == 1].groupby(["game_id", "team_id"]).player_id.apply(
        lambda s: [int(x) for x in s])
    sq = pl.groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])
    _, PER, _ = declared_grid()
    role = defaultdict(Counter)
    for (gid, tid, pid), (k, l) in PER.items(): role[int(pid)][(int(k), int(l != 1))] += 1
    ROLE = {p: c.most_common(1)[0][0] for p, c in role.items()}

    from lineup_event_panel import _snapshots
    by, T, dim, FC = _snapshots(gpos)
    itv = np.array([i for i, c in enumerate(FC) if c.startswith("tv_")])
    def tvtot(p, tt):
        a = T.get(int(p))
        if a is None: return 0.0
        j = np.searchsorted(a, tt, "right") - 1
        return float(by[int(p)][j][1][itv].sum()) if j >= 0 else 0.0

    res = defaultdict(list); shp = defaultdict(list); xh = defaultdict(list)
    mshp = defaultdict(list)   # 감독별 과거 모양
    rows = []
    for r in g.itertuples(index=False):
        gid = int(r.game_id)
        pair = ((int(r.home_team_id), r.home_score, r.away_score, int(r.away_team_id)),
                (int(r.away_team_id), r.away_score, r.home_score, int(r.home_team_id)))
        # 상대 이력은 같은 시점 스냅샷이어야 하므로 먼저 읽고 나중에 갱신
        snap_info = {}
        for tid, gf, ga, oid in pair:
            snap_info[tid] = dict(h5=list(shp[tid][-5:]), r5=list(res[tid][-5:]),
                                  prev=list(shp[tid][-5:]))
        for tid, gf, ga, oid in pair:
            s = NEW.get((gid, tid)); pts = 3 if gf > ga else (1 if gf == ga else 0)
            xi = stl.get((gid, tid))
            si, so = snap_info[tid], snap_info[oid]
            if s is not None and xi is not None and len(si["h5"]) >= 3 and len(si["r5"]) >= 3:
                c = Counter()
                for xs in xh[tid][-5:]:
                    for p in xs: c[p] += 1
                reg = [p for p, nn in c.items() if nn >= 3]
                today = set(sq.get((gid, tid), []))
                av = np.zeros(7)
                for p in reg:
                    if p not in today:
                        k, w = ROLE.get(p, (2, 0)); av[k] += 1; av[5] += w; av[6] += 1
                rq = np.zeros(6); tt = gpos[gid]
                for p in today:
                    k, w = ROLE.get(int(p), (2, 0))
                    v = tvtot(p, tt); rq[k] += v; rq[5] += v * w
                mg = MGR.get((gid, tid)); og = MGR.get((gid, oid))
                cnt = Counter(si["h5"])
                rows.append(dict(
                    tid=tid, sea=SEA[gid], y=s,
                    prev=si["prev"], pers=cnt.most_common(1)[0][0],
                    hab=cnt.most_common(1)[0][1] / len(si["h5"]),
                    pts3=sum(si["r5"][-3:]), pts5=sum(si["r5"]),
                    newm=float(mg[1] <= 10) if mg else 0.0,
                    logten=np.log1p(mg[1]) if mg else 0.0,
                    mghab=(Counter(mshp[mg[0]][-15:]).most_common(1)[0][0]
                           if mg and mshp.get(mg[0]) else None),
                    ohab=(Counter(so["h5"]).most_common(1)[0][0] if so["h5"] else None),
                    opts5=sum(so["r5"]) if so["r5"] else 7.0,
                    onew=float(og[1] <= 10) if og else 0.0,
                    home=float(tid == pair[0][0]), k1=float(CMP[gid] == "KLEAGUE1"),
                    av=av, rq=rq))
            if s is not None:
                shp[tid].append(s)
                if xi is not None: xh[tid].append(xi)
                mg = MGR.get((gid, tid))
                if mg: mshp[mg[0]].append(s)
            res[tid].append(pts)
    D = pd.DataFrame(rows).reset_index(drop=True)
    lib = [k for k, v in Counter(D.y).items() if v >= 20]
    CL = {s: i for i, s in enumerate(lib)}; OTH = len(lib)
    D["yc"] = [CL.get(s, OTH) for s in D.y]
    def oh_series(vals):
        M = np.zeros((len(D), len(lib) + 1))
        for i, s in enumerate(vals): M[i, CL.get(s, OTH) if s is not None else OTH] = 1
        return M
    S_blk = np.hstack([oh_series([p[-1] if len(p) > 0 else None for p in D.prev]),
                       oh_series([p[-2] if len(p) > 1 else None for p in D.prev]),
                       oh_series([p[-3] if len(p) > 2 else None for p in D.prev]),
                       oh_series(D.pers), D[["hab", "home", "k1"]].to_numpy(float)])
    R_blk = D[["pts3", "pts5"]].to_numpy(float)
    M_blk = np.hstack([oh_series(D.mghab), D[["newm", "logten"]].to_numpy(float)])
    O_blk = np.hstack([oh_series(D.ohab), D[["opts5", "onew"]].to_numpy(float)])
    P_blk = np.hstack([np.vstack(D.av.to_numpy()), np.vstack(D.rq.to_numpy())])
    TS = [int(x) for x in os.environ.get("TESTSEASONS", "2025,2026").split(",")]   # 통합 규약이면 2024,2025
    tr = (D.sea < min(TS)).to_numpy(); te = D.sea.isin(TS).to_numpy()
    y = D.yc.to_numpy()
    pers_acc = (D.pers[te] == D.y[te]).mean()
    ok_p = (D.pers[te] == D.y[te]).to_numpy(float)
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0); n2 = te.sum()
    print(f"표본 {len(D):,} · 평가 {n2:,} · 클래스 {len(lib)}+other · 지속성 {pers_acc:.3f}")
    np.savez(ROOT / "outputs" / "formation_design.npz",
             S=S_blk, R=R_blk, M=M_blk, O=O_blk, P=P_blk, y=y,
             sea=D.sea.to_numpy(), tid=D.tid.to_numpy(),
             pers=np.array([CL.get(s, OTH) for s in D.pers]),
             prev_seq=np.array([[CL.get(p[-i], OTH) if len(p) >= i else OTH
                                 for i in range(1, 6)] for p in D.prev]))
    combos = [("S 모양만", [S_blk]), ("S+R 성적", [S_blk, R_blk]),
              ("S+R+M 감독", [S_blk, R_blk, M_blk]),
              ("S+R+M+P 선수단", [S_blk, R_blk, M_blk, P_blk]),
              ("전부(+O 상대)", [S_blk, R_blk, M_blk, P_blk, O_blk])]
    for name, blks in combos:
        X = np.hstack(blks)
        mu, sd = X[tr].mean(0), X[tr].std(0).clip(1e-9); Xs = (X - mu) / sd
        m = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
        pred = m.predict(Xs[te]); acc = (pred == y[te]).mean()
        okm = (pred == y[te]).astype(float); ds = []
        for _ in range(1000):
            ix = rng.integers(0, n2, n2); ds.append(okm[ix].mean() - ok_p[ix].mean())
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"  {name:16s} {acc:.3f}  Δvs지속 {acc-pers_acc:+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
