"""5단 × 3레인 격자 제약 — 깊이만이 아니라 좌우까지 세는 추천.

단 인원만 세면 "왼쪽 선수 둘, 오른쪽 없음" 같은 XI 를 못 막는다(좌우 배정 오차
감독 4.20m vs 제약추천 5.42m). 여기서는 감독의 실제 XI 가 만든 (단, 레인) 칸
인원을 그대로 제약으로 걸어 배정 자체가 최적화의 출력이 되게 한다.

레인: 좌 x<0.35 (21.4%) · 중 0.35~0.65 (57.1%) · 우 x>0.65 (21.4%)
단   : 공시 5단 (LEVELS)

누수: 선수의 (단,레인)은 **과거 전용**으로 매긴다 — 단은 player_level_dyn(그날
실제와 80.8% 일치 = 과거 전용 확인), 레인은 과거 선발의 반감기 가중 최빈.

실행: python -m experiments.gap_lane
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR, RAW_DATA_DIR  # noqa: E402
from gap_constrained import LEVELS, GK_MAX_LVL, fe_ols, dyn_level, lvl_of  # noqa: E402

LANE_LO, LANE_HI = 0.35, 0.65
HALFLIFE = 10.0            # 과거 선발 반감기 (레인 추정)
LEFTPOS = {"LB", "LWB", "LW", "LM", "LF"}
RIGHTPOS = {"RB", "RWB", "RW", "RM", "RF"}


def _cidmap():
    """원시 명단 pid → 정규 pid. 2021~25 공간(pid_merge.json)과 2026 공간(pid_merge_2026.json)은 숫자가 겹치므로 경기 시즌으로 고른다."""
    import json
    f = ROOT / "outputs" / "pid_merge.json"; f26 = ROOT / "outputs" / "pid_merge_2026.json"
    if not f.exists(): return {}, {}, set()
    M = {int(k): int(v) for k, v in json.load(open(f)).items()}
    M26 = {int(k): int(v) for k, v in json.load(open(f26)).items()} if f26.exists() else {}
    try:
        g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv", usecols=["game_id", "season"]); g26 = set(g[g.season >= 2026].game_id.astype(int))
        pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv", usecols=["game_id", "player_id"])
        cur = set(pl[~pl.game_id.isin(g26)].player_id.astype(int))
    except Exception:
        return {}, {}, set()
    return (M if not ({k for k, v in M.items() if k != v} & cur) else {}), M26, g26


_CIDMAP, _CIDMAP26, _G26 = _cidmap()
def _cid(p, gid=None):
    p = int(p)
    if gid is not None and int(gid) in _G26: return _CIDMAP26.get(p, p)
    return _CIDMAP.get(p, p)


def lane_of(x):
    return 0 if x < LANE_LO else (2 if x > LANE_HI else 1)


def declared_grid():
    """(gid,tid) -> (5,3) 칸 인원 · (gid,tid,pid) -> (단,레인) · 선발 이력 목록."""
    cells, per, hist = {}, {}, []
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
            buck[int(e.get("team_id", -1))].append((_cid(e["player_id"], gid), float(q["y"]), float(q["x"])))
        for tid, ps in buck.items():
            out = [(p, y, x) for p, y, x in ps if y > GK_MAX_LVL]
            if len(out) != 10:
                continue
            G = np.zeros((5, 3), int)
            for p, y, x in out:
                k, l = lvl_of(y), lane_of(x)
                G[k, l] += 1; per[(gid, tid, p)] = (k, l)
                hist.append((gid, p, l))
            cells[(gid, tid)] = G
    return cells, per, pd.DataFrame(hist, columns=["gid", "pid", "lane"])


def past_lanes(hist, gpos):
    """(gid,pid) -> 과거 선발의 반감기 가중 최빈 레인 (그 경기 제외)."""
    hist = hist.copy(); hist["ord"] = hist.gid.map(gpos)
    hist = hist.dropna(subset=["ord"]).sort_values("ord")
    acc, out = defaultdict(lambda: np.zeros(3)), {}
    dec = 0.5 ** (1.0 / HALFLIFE)
    for r in hist.itertuples(index=False):
        v = acc[r.pid]
        out[(int(r.gid), int(r.pid))] = int(np.argmax(v)) if v.sum() > 0 else None
        v *= dec; v[int(r.lane)] += 1.0
    return out


def pick_grid(sc, cell, G, gks, ps):
    """칸별 상위 n 명 — 각 선수가 정확히 한 칸에 속하므로 제약 하 정확한 최적해."""
    rec = [max(gks, key=lambda p: sc[p])] if gks else []
    used, short = set(rec), 0
    for k in range(5):
        for l in range(3):
            n = int(G[k, l])
            if n == 0:
                continue
            pool = sorted((p for p in sc if p in ps and p not in used and cell.get(p) == (k, l)),
                          key=lambda p: -sc[p])[:n]
            short += n - len(pool); used.update(pool); rec += pool
    if short:
        rest = sorted((p for p in sc if p in ps and p not in used), key=lambda p: -sc[p])[:short]
        rec += rest
    return rec[:11], short


def main():
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from lineup_pipeline import r2w, AL
    from lineup_event_panel import _snapshots, build_event
    from gap_constrained import pick_constrained

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

    print("공시 격자 집계 중…", flush=True)
    CELL, PER, HIST = declared_grid()
    PLANE = past_lanes(HIST, gpos)
    print(f"  격자가 읽히는 팀-경기 {len(CELL):,}   과거 레인 {len(PLANE):,}")

    def lane_fallback(p):
        q = POS.get(int(p), "?")
        return 0 if q in LEFTPOS else (2 if q in RIGHTPOS else 1)

    prev_xi, lastxi = {}, {}
    for gid in sorted(set(k[0] for k in CELL), key=lambda z: gpos.get(z, 1 << 30)):
        for (gg, tt) in [k for k in CELL if k[0] == gid]:
            if (gg, tt) in st.index:
                prev_xi[(gg, tt)] = lastxi.get(tt); lastxi[tt] = list(st.loc[(gg, tt)])

    cases, sh_g, sh_a = [], 0, 0
    for r in g.itertuples():
        gid = int(r.game_id)
        if gid not in fold_of or r.season != 2025: continue
        w = Wf[fold_of[gid]]; tt_ = gpos[gid]
        for tid, oid in ((int(r.home_team_id), int(r.away_team_id)),
                         (int(r.away_team_id), int(r.home_team_id))):
            if (gid, tid) not in st.index or (gid, tid) not in CELL: continue
            pool = [int(p) for p in squad.loc[(gid, tid)]]
            pv = prev_xi.get((gid, tid)) or []
            sc = {}
            for p in list(pool) + list(pv):
                v = snap(p, tt_)
                if v is not None: sc.setdefault(int(p), float(w @ v))
            coach = list(st.loc[(gid, tid)])
            if len(sc) < 12 or not set(coach) <= set(sc): continue
            yv = xg.get((gid, tid), np.nan) - xg.get((gid, oid), np.nan)
            if not np.isfinite(yv): continue
            ps = set(pool)
            gks = [p for p in sc if p in ps and POS.get(p) == "GK"]
            of = [p for p in sc if p in ps and POS.get(p) != "GK"]
            rec_u = ([max(gks, key=lambda p: sc[p])] if gks else [])
            rec_u += sorted(of, key=lambda p: -sc[p])[:11 - len(rec_u)]
            # 과거 전용 (단, 레인)
            cell, lvp = {}, {}
            for p in sc:
                if p in gks: continue
                k = dyn_level(gid, p, STAT)
                if k is None: k = STAT.get(int(p), 2)
                l = PLANE.get((gid, int(p)))
                if l is None: l = lane_fallback(p)
                cell[p] = (int(k), int(l)); lvp[p] = int(k)
            G = CELL[(gid, tid)]
            rec_g, s1 = pick_grid(sc, cell, G, gks, ps); sh_g += s1 > 0
            rec_h, s2 = pick_constrained(sc, lvp, G.sum(1), gks, ps); sh_a += s2 > 0
            gp = lambda R_: sum(sc[p] for p in R_) - sum(sc[p] for p in coach)
            cases.append(dict(gid=gid, tid=tid, oid=oid, season=int(r.season),
                              date=str(r.game_date.date()), y=yv,
                              gap_u=gp(rec_u), gap_g=gp(rec_g), gap_h=gp(rec_h),
                              hit_u=len(set(rec_u) & set(coach)), hit_g=len(set(rec_g) & set(coach)),
                              hit_h=len(set(rec_h) & set(coach)),
                              coach_sc=sum(sc[p] for p in coach), prev=prev_xi.get((gid, tid)),
                              sc=sc, cell=cell, lv=lvp, G=G, gks=tuple(gks), pool=tuple(pool),
                              coach_xi=coach, rec_u=rec_u, rec_g=rec_g, rec_h=rec_h))
    C = pd.DataFrame(cases)
    print(f"\n사례 {len(C):,}   칸 인원 부족 격자 {sh_g} · 단만 {sh_a}")
    pg = []
    for r in C.itertuples(index=False):
        pg.append(sum(r.sc[p] for p in r.prev) - r.coach_sc
                  if r.prev and set(r.prev) <= set(r.sc) else np.nan)
    C["gap_placebo"] = pg

    print("\n=== 적중 · 격차 ===")
    for nm, h, gc in [("무제약", "hit_u", "gap_u"), ("단만 (과거전용)", "hit_h", "gap_h"),
                      ("단×레인 격자", "hit_g", "gap_g")]:
        print(f"  {nm:16s} 적중 {C[h].mean():.3f}/11   격차 {C[gc].mean():+.4f} (SD {C[gc].std():.4f})"
              f"   11/11 {int((C[h]==11).sum()):3d}건")

    print("\n=== 격차 → 실제 xG 차 (팀-시즌 FE, 격차 1SD 당) ===")
    for nm, col in [("무제약", "gap_u"), ("단만 (과거전용)", "gap_h"), ("단×레인 격자", "gap_g"),
                    ("위약 직전 XI", "gap_placebo")]:
        sd = C[col].std(); b, lo, hi, n = fe_ols(C, col)
        star = "*" if (lo > 0) == (hi > 0) else " "
        print(f"  {nm:16s} {b*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}] n={n:4d} {star}")
    print("  ── 감독 XI 점수 통제 ──")
    for nm, col in [("단만", "gap_h"), ("단×레인", "gap_g")]:
        sd = C[col].std(); b, lo, hi, n = fe_ols(C, col, extra=["coach_sc"])
        star = "*" if (lo > 0) == (hi > 0) else " "
        print(f"  {nm:16s} {b*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}] n={n:4d} {star}")

    print("\n=== 점수 셔플 위약 (격자 절차 그대로, 6시드) ===")
    outs = []
    for seed in range(6):
        rng = np.random.default_rng(seed); gg2 = []
        for r in C.itertuples(index=False):
            ks = list(r.sc); vs = [r.sc[k] for k in ks]; rng.shuffle(vs)
            s2 = dict(zip(ks, vs)); ps2 = set(r.pool)
            rr, _ = pick_grid(s2, r.cell, r.G, list(r.gks), ps2)
            gg2.append(sum(s2[p] for p in rr) - sum(s2[p] for p in r.coach_xi))
        C["gap_sh"] = gg2
        sd = C.gap_sh.std(); b, lo, hi, n = fe_ols(C, "gap_sh")
        outs.append(b * sd)
        print(f"  시드{seed}  {b*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}]")
    mu, sg = float(np.mean(outs)), float(np.std(outs))
    sd = C.gap_g.std(); b, lo, hi, n = fe_ols(C, "gap_g")
    print(f"  셔플 평균 {mu:+.4f} ± {sg:.4f}   →  격자는 셔플 대비 z = {(b*sd-mu)/max(sg,1e-9):+.2f}")
    C.drop(columns=["gap_sh"], errors="ignore").to_pickle(ROOT / "outputs" / "gap_lane_full.pkl")
    print("\n저장 outputs/gap_lane_full.pkl")


if __name__ == "__main__":
    main()
