"""제약 추천 vs 무제약 추천 — 격차↔결과 관계가 살아남는가.

무제약(점수 상위 10 + GK 1)은 센터백 1명 같은 실현 불가 XI 를 만든다.
여기서는 **공시 5단 인원**을 제약으로 걸어 같은 모양 안에서만 인원을 바꾼다.

  모양 A (shape=coach)  그 경기 감독의 실제 공시 5단 인원  — 모양 고정, 인원만 최적화
  모양 B (shape=modal)  그 팀의 **과거** 최빈 5단 인원      — 누수 없음

각 선수는 단 하나의 단에 배정되므로(동적 단 argmax) 제약 하 최적해는
**단별 상위 n_i 명**이다 — ILP 불필요, 정확히 최적.

실행: python -m experiments.gap_constrained
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR, RAW_DATA_DIR  # noqa: E402

LEVELS = np.array([0.247, 0.416, 0.581, 0.747, 0.916])
BAND_TOL = 0.02
GK_MAX_LVL = 0.15


def lvl_of(y):
    return int(np.argmin(np.abs(LEVELS - y)))


def declared_counts():
    """(gid,tid) -> (5,) 아웃필드 10명의 공시 단 인원 + (gid,tid,pid)->단."""
    cnt, per = {}, {}
    for f in RAW_DATA_DIR.glob("*/*/match/*/lineup.json"):
        try:
            d = json.load(open(f))["result"]
        except Exception:
            continue
        gid = int(f.parent.name)
        buck = defaultdict(list)
        for e in d:
            q = e.get("position") or {}
            if not (e.get("is_starting_lineup") and "x" in q and "y" in q):
                continue
            tid = int(e.get("team_id", -1)); yv = float(q["y"])
            buck[tid].append((int(e["player_id"]), yv))
        for tid, ps in buck.items():
            out = [(p, y) for p, y in ps if y > GK_MAX_LVL]
            if len(out) != 10:
                continue
            v = [0] * 5
            for p, y in out:
                k = lvl_of(y); v[k] += 1; per[(gid, tid, p)] = k
            cnt[(gid, tid)] = np.array(v)
    return cnt, per


_DYN = None


def dyn_level(gid, pid, static):
    """그 경기의 동적 단 argmax, 없으면 정적 단."""
    global _DYN
    if _DYN is None:
        f = ROOT / "outputs" / "player_level_dyn.parquet"
        t = pd.read_parquet(f)
        qs = t[[f"q{i}" for i in range(5)]].to_numpy(float)
        _DYN = {(int(a), int(b)): int(np.argmax(qs[k]))
                for k, (a, b) in enumerate(zip(t.game_id.to_numpy(), t.player_id.to_numpy()))}
    v = _DYN.get((int(gid), int(pid)))
    return v if v is not None else static.get(int(pid))


def pick_constrained(sc, lv, counts, gks, ps=None):
    """단별 상위 n_i — 제약 하 정확한 최적해. 부족하면 잔여를 점수순으로 채운다."""
    rec = [max(gks, key=lambda p: sc[p])] if gks else []
    used, short = set(rec), 0
    ok = (lambda p: True) if ps is None else (lambda p: p in ps)
    for i in range(5):
        pool = sorted((p for p in sc if ok(p) and p not in used and lv.get(p) == i),
                      key=lambda p: -sc[p])
        take = pool[:int(counts[i])]
        short += int(counts[i]) - len(take)
        used.update(take); rec += take
    if short:                                   # 단 인원 부족 → 잔여 최고점으로 보충
        rest = sorted((p for p in sc if ok(p) and p not in used), key=lambda p: -sc[p])[:short]
        rec += rest
    return rec[:11], short


def fe_ols(df, xcol, extra=()):
    """팀-시즌 고정효과 하 y ~ x — 그룹 내 차분 + 부트스트랩 CI."""
    d = df.dropna(subset=[xcol, "y"]).copy()
    d["grp"] = d.tid.astype(str) + "_" + d.season.astype(str)
    cols = [xcol] + list(extra)
    for c in cols + ["y"]:
        d[c] = d[c] - d.groupby("grp")[c].transform("mean")
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
    b = np.linalg.lstsq(X, d.y.to_numpy(), rcond=None)[0]
    rng = np.random.default_rng(0); bs = []
    gs = d.grp.unique()
    # 그룹별 행 위치를 **한 번만** 만든다 (복제마다 만들면 O(그룹×행)로 폭발)
    _gv = d.grp.to_numpy()
    loc = {g: np.flatnonzero(_gv == g) for g in gs}
    for _ in range(600):                        # 팀-시즌 클러스터 부트스트랩
        pick = rng.choice(gs, len(gs), replace=True)
        ix = np.concatenate([loc[g] for g in pick])
        try:
            bs.append(np.linalg.lstsq(X[ix], d.y.to_numpy()[ix], rcond=None)[0][1])
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return b[1], lo, hi, len(d)


def main():
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from lineup_pipeline import r2w, AL
    from lineup_event_panel import _snapshots, build_event

    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    by, T_, dim, FC = _snapshots(gpos)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    NAME, POS = {}, {}
    for pid, gg in pl.groupby("player_id"):
        nk = gg["nickname"].dropna()
        NAME[int(pid)] = str(nk.mode().iloc[0]) if len(nk) and nk.mode().iloc[0] else str(gg.player_name.mode().iloc[0])
        m = gg["starting_position_name"].mode(); POS[int(pid)] = str(m.iloc[0]) if len(m) else "?"
    stat = pd.read_parquet(ROOT / "outputs" / "player_level.parquet")
    STAT = {int(r.player_id): int(r.level) for r in stat.itertuples(index=False)}

    st = pl[pl.is_starter == 1].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])
    st = st[st.str.len() == 11]
    squad = pl.groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])
    xg = pd.read_parquet(ROOT / "outputs/team_xg.parquet").set_index(["game_id", "team_id"]).xg

    R, F, BASE, IDX, cols = build_event("xg")
    M, B = F["vaepon"], BASE; y = R.y.to_numpy(); WT = R.nlvl.to_numpy(float)
    kx = np.array([i for i, c in enumerate(FC) if c in set(cols)])
    Wf, fold_of = {}, {}
    for f in range(1, 5):
        tr = np.concatenate(IDX[:f]); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
        pc = PCA(n_components=5, random_state=0).fit(M[i1])
        Z = pc.transform(M); zmu, zsd = Z[i1].mean(0), Z[i1].std(0).clip(1e-9)
        X = np.hstack([B, (Z - zmu) / zsd])
        ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(X[i1], y[i1], sample_weight=WT[i1]).predict(X[i2]), WT[i2]))
        mm = Ridge(alpha=ba).fit(X[tr], y[tr], sample_weight=WT[tr])
        Wf[f] = (pc.components_.T / zsd) @ mm.coef_[B.shape[1]:]
        for i in IDX[f]: fold_of[int(R.game_id.iloc[i])] = f

    def snap(p, tt_):
        a = T_.get(int(p))
        if a is None: return None
        j = np.searchsorted(a, tt_, "right") - 1
        return by[int(p)][j][1][kx] if j >= 0 else None

    print("공시 단 인원 집계 중…", flush=True)
    CNT, PER = declared_counts()
    print(f"  공시 5단 인원이 읽히는 팀-경기 {len(CNT):,}")
    # 팀의 **과거** 최빈 모양 (누수 없음)
    rows = [(k[0], k[1], tuple(v)) for k, v in CNT.items()]
    S = pd.DataFrame(rows, columns=["gid", "tid", "shape"])
    S["ord"] = S.gid.map(gpos); S = S.sort_values("ord")
    past = {}
    hist = defaultdict(list)
    for r in S.itertuples(index=False):
        past[(r.gid, r.tid)] = Counter(hist[r.tid]).most_common(1)[0][0] if hist[r.tid] else None
        hist[r.tid].append(r.shape)

    prev_xi = {}                                  # 위약용: 직전 경기 XI
    lastxi = {}
    for r in S.itertuples(index=False):
        if (r.gid, r.tid) in st.index:
            prev_xi[(r.gid, r.tid)] = lastxi.get(r.tid)
            lastxi[r.tid] = list(st.loc[(r.gid, r.tid)])

    cases, nshort_a, nshort_b, noshape = [], 0, 0, 0
    for r in g.itertuples():
        gid = int(r.game_id)
        if gid not in fold_of or r.season != 2025: continue
        w = Wf[fold_of[gid]]; tt_ = gpos[gid]
        for tid, oid in ((int(r.home_team_id), int(r.away_team_id)),
                         (int(r.away_team_id), int(r.home_team_id))):
            if (gid, tid) not in st.index: continue
            pool = [int(p) for p in squad.loc[(gid, tid)]]
            pv = prev_xi.get((gid, tid)) or []
            sc = {}
            for p in pv:                          # 위약 표본 확보용 (선택 대상 아님)
                v = snap(p, tt_)
                if v is not None: sc.setdefault(int(p), float(w @ v))
            for p in pool:
                v = snap(p, tt_)
                if v is not None: sc[p] = float(w @ v)
            coach = list(st.loc[(gid, tid)])
            if len(sc) < 12 or not set(coach) <= set(sc): continue
            yv = xg.get((gid, tid), np.nan) - xg.get((gid, oid), np.nan)
            if not np.isfinite(yv): continue
            ps = set(pool)
            gks = [p for p in sc if p in ps and POS.get(p) == "GK"]
            of = [p for p in sc if p in ps and POS.get(p) != "GK"]
            # 무제약 (기존)
            rec_u = ([max(gks, key=lambda p: sc[p])] if gks else [])
            rec_u += sorted(of, key=lambda p: -sc[p])[:11 - len(rec_u)]
            # 단 배정
            lv = {}
            for p in sc:
                if p in gks: continue
                k = PER.get((gid, tid, p))            # 그날 공시가 있으면 그대로
                if k is None: k = dyn_level(gid, p, STAT)
                if k is not None: lv[p] = int(k)
            cA = CNT.get((gid, tid)); cB = past.get((gid, tid))
            if cA is None: noshape += 1; continue
            rec_a, sa = pick_constrained(sc, lv, cA, gks, ps); nshort_a += sa > 0
            rec_b, sb = (rec_a, 0) if cB is None else pick_constrained(sc, lv, np.array(cB), gks, ps)
            nshort_b += sb > 0
            gp = lambda R_: sum(sc[p] for p in R_) - sum(sc[p] for p in coach)
            cb = lambda R_: sum(1 for p in R_ if lv.get(p) == 0)
            cases.append(dict(gid=gid, tid=tid, oid=oid, season=int(r.season),
                              date=str(r.game_date.date()), y=yv,
                              gap_u=gp(rec_u), gap_a=gp(rec_a), gap_b=gp(rec_b),
                              hit_u=len(set(rec_u) & set(coach)), hit_a=len(set(rec_a) & set(coach)),
                              hit_b=len(set(rec_b) & set(coach)),
                              def_u=cb(rec_u), def_a=cb(rec_a), def_c=cb(coach),
                              coach_sc=sum(sc[p] for p in coach),
                              prev=prev_xi.get((gid, tid)), sc=sc, lv=lv,
                              cA=tuple(int(t) for t in cA), gks=tuple(gks), pool=tuple(pool),
                              coach_xi=coach, rec_u=rec_u, rec_a=rec_a))
    C = pd.DataFrame(cases)
    print(f"\n사례 {len(C):,}  (모양 없음 {noshape}, 단인원 부족 A {nshort_a} · B {nshort_b})")

    # 위약: 직전 경기 XI 를 '추천' 으로 둔 가짜 격차
    pg = []
    for r in C.itertuples(index=False):
        if r.prev and set(r.prev) <= set(r.sc):
            pg.append(sum(r.sc[p] for p in r.prev) - r.coach_sc)
        else: pg.append(np.nan)
    C["gap_placebo"] = pg

    print("\n=== 구성 현실성 ===")
    print(f"  수비단(단0) 인원   감독 {C.def_c.mean():.2f}   무제약 {C.def_u.mean():.2f}   "
          f"제약A {C.def_a.mean():.2f}")
    print(f"  감독 XI 적중       무제약 {C.hit_u.mean():.3f}/11   제약A {C.hit_a.mean():.3f}/11   "
          f"제약B(과거모양) {C.hit_b.mean():.3f}/11")
    print(f"  주장 격차 평균     무제약 {C.gap_u.mean():+.4f}   제약A {C.gap_a.mean():+.4f}   "
          f"제약B {C.gap_b.mean():+.4f}")

    print("\n=== 격차 → 실제 xG 차 (팀-시즌 고정효과) ===")
    for nm, col in [("무제약", "gap_u"), ("제약A 감독모양", "gap_a"), ("제약B 과거모양", "gap_b"),
                    ("위약(직전 XI)", "gap_placebo")]:
        b, lo, hi, n = fe_ols(C, col)
        star = "*" if (lo > 0) == (hi > 0) else " "
        print(f"  {nm:14s} {b:+.4f} [{lo:+.4f},{hi:+.4f}] n={n:4d} {star}")
    print("  ── 감독 XI 점수 통제 추가 ──")
    for nm, col in [("무제약", "gap_u"), ("제약A", "gap_a"), ("제약B", "gap_b")]:
        b, lo, hi, n = fe_ols(C, col, extra=["coach_sc"])
        star = "*" if (lo > 0) == (hi > 0) else " "
        print(f"  {nm:14s} {b:+.4f} [{lo:+.4f},{hi:+.4f}] n={n:4d} {star}")

    print("\n=== 점수 셔플 위약 (제약A 절차 그대로, 6시드) ===")
    sh = []
    for seed in range(6):
        rng = np.random.default_rng(seed); gg2 = []
        for r in C.itertuples(index=False):
            keys = list(r.sc); vals = [r.sc[k] for k in keys]
            rng.shuffle(vals); s2 = dict(zip(keys, vals))
            ps2 = set(r.pool)
            gk2 = [max(r.gks, key=lambda p: s2[p])] if r.gks else []
            rr, _ = pick_constrained(s2, r.lv, np.array(r.cA), list(r.gks), ps2)
            gg2.append(sum(s2[p] for p in rr) - sum(s2[p] for p in r.coach_xi))
        C["gap_sh"] = gg2
        b, lo, hi, n = fe_ols(C, "gap_sh")
        sd = C.gap_sh.std(); sh.append(b * sd)
        print(f"  시드{seed}  {b*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}]  격차평균 {C.gap_sh.mean():+.4f}")
    print(f"  셔플 평균 {np.mean(sh):+.4f} ± {np.std(sh):.4f}")

    print("\n=== 제약A 격차 오분위 ===")
    q = pd.qcut(C.gap_a, 5, labels=False)
    m = C.groupby(q).agg(g=("gap_a", "mean"), y=("y", "mean"), n=("y", "size"))
    print(m.round(3).to_string())
    C.drop(columns=["gap_sh"], errors="ignore").to_pickle(ROOT / "outputs" / "gap_constrained_full.pkl")
    C.drop(columns=["sc", "lv", "gks", "pool", "cA"]).to_pickle(ROOT / "outputs" / "gap_constrained.pkl")
    print(f"\n저장 outputs/gap_constrained.pkl")


if __name__ == "__main__":
    main()
