"""대칭 포메이션 제약 — 좌/우를 따로 세지 말고 **와이드**로 묶는다.

공시 포메이션은 5,080 중 99.4% 가 완전 좌우대칭이고(비대칭 33건), 와이드 선수는
실제로 13.5% 가 반대쪽에서 뛴다. 그런데 엄격한 (단,좌/중/우) 격자는 그 교차를
금지해 65% 사례에서 칸 인원이 모자랐다.

대칭 제약: 단 k 마다 **중앙 n_c 명 · 와이드 n_w 명**만 요구하고, 뽑힌 와이드를
좌/우로 나누는 것은 선호도 기준 사후 배정. 각 선수는 (단, 중앙/와이드) 한 칸에만
속하므로 여전히 칸별 상위 n 이 정확한 최적해다.

실행: python -m experiments.gap_sym
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR
from gap_constrained import fe_ols, dyn_level  # noqa: E402
TAU = 0.10
from gap_lane import declared_grid, past_lanes, pick_grid, LEFTPOS, RIGHTPOS  # noqa: E402


def pick_sym(sc, lvl, wide, G, gks, ps):
    """단별 (중앙 n_c · 와이드 n_w). wide[p]=True 면 와이드 부류."""
    rec = [max(gks, key=lambda p: sc[p])] if gks else []
    used, short, chosen = set(rec), 0, defaultdict(list)
    for k in range(5):
        for is_w, n in ((False, int(G[k, 1])), (True, int(G[k, 0] + G[k, 2]))):
            if n == 0:
                continue
            pool = sorted((p for p in sc if p in ps and p not in used
                           and lvl.get(p) == k and bool(wide.get(p)) == is_w),
                          key=lambda p: -sc[p])[:n]
            short += n - len(pool); used.update(pool); rec += pool
            if is_w: chosen[k] += pool
    if short:
        rest = sorted((p for p in sc if p in ps and p not in used), key=lambda p: -sc[p])[:short]
        rec += rest
    return rec[:11], short, chosen


_Q = None


def dyn_q(gid, pid):
    """그 경기의 단 확률 (5,) — 과거 전용. 없으면 None."""
    global _Q
    if _Q is None:
        t = pd.read_parquet(ROOT / "outputs" / "player_level_dyn.parquet")
        qs = t[[f"q{i}" for i in range(5)]].to_numpy(float)
        _Q = {(int(a), int(b)): qs[k] for k, (a, b) in
              enumerate(zip(t.game_id.to_numpy(), t.player_id.to_numpy()))}
    return _Q.get((int(gid), int(pid)))


def pick_flex(sc, elig, wide, G, gks, ps, BIG=1e6):
    """대칭 + **단 허용도** — 선수가 여러 단에 설 수 있으므로 배정 문제로 정확히 푼다.

    elig[p] = 그 선수가 설 수 있는 단들의 집합. 슬롯 10개(단×중앙/와이드)에
    선수를 1:1 배정해 총점 최대 — linear_sum_assignment 로 정확해.
    """
    from scipy.optimize import linear_sum_assignment
    rec = [max(gks, key=lambda p: sc[p])] if gks else []
    slots = []
    for k in range(5):
        slots += [(k, 0)] * int(G[k, 1]) + [(k, 1)] * int(G[k, 0] + G[k, 2])
    cand = [p for p in sc if p in ps and p not in set(rec) and p in elig]
    if len(cand) < len(slots):
        return rec + sorted((p for p in sc if p in ps and p not in set(rec)),
                            key=lambda p: -sc[p])[:len(slots)], len(slots) - len(cand), {}
    cost = np.full((len(cand), len(slots)), BIG, float)
    for i, p in enumerate(cand):
        for j, (k, isw) in enumerate(slots):
            if k in elig[p] and int(bool(wide.get(p))) == isw:
                cost[i, j] = -sc[p]
    ri, ci = linear_sum_assignment(cost)
    bad = int((cost[ri, ci] >= BIG / 2).sum())
    chosen = defaultdict(list)
    for i, j in zip(ri, ci):
        k, isw = slots[j]; rec.append(cand[i])
        if isw: chosen[k].append(cand[i])
    return rec[:11], bad, chosen


def assign_sides(chosen, lane, G):
    """뽑힌 와이드를 좌/우로 — 선호 쪽과 어긋남 최소."""
    out = {}
    for k, ps in chosen.items():
        nl, nr = int(G[k, 0]), int(G[k, 2])
        if nl + nr != len(ps):                       # 부족분 보충 등으로 어긋나면 균등
            nl = len(ps) // 2; nr = len(ps) - nl
        # 왼쪽 선호(과거 레인 0)가 강한 순으로 왼쪽 배정
        order = sorted(ps, key=lambda p: (lane.get(p, 1) != 0, lane.get(p, 1) == 2))
        for i, p in enumerate(order):
            out[p] = 0 if i < nl else 2
    return out


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
    POS, NAME = {}, {}
    for pid, gg in pl.groupby("player_id"):
        m = gg["starting_position_name"].mode(); POS[int(pid)] = str(m.iloc[0]) if len(m) else "?"
        nk = gg["nickname"].dropna()
        NAME[int(pid)] = str(nk.mode().iloc[0]) if len(nk) and nk.mode().iloc[0] else str(gg.player_name.mode().iloc[0])
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

    print("공시 격자…", flush=True)
    CELL, PER, HIST = declared_grid()
    PLANE = past_lanes(HIST, gpos)

    def lane_fb(p):
        q = POS.get(int(p), "?")
        return 0 if q in LEFTPOS else (2 if q in RIGHTPOS else 1)

    prev_xi, lastxi = {}, {}
    for gid in sorted(set(k[0] for k in CELL), key=lambda z: gpos.get(z, 1 << 30)):
        for (gg, tt) in [k for k in CELL if k[0] == gid]:
            if (gg, tt) in st.index:
                prev_xi[(gg, tt)] = lastxi.get(tt); lastxi[tt] = list(st.loc[(gg, tt)])

    cases, shg, shs, shf = [], 0, 0, 0
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
            cell, lvl, lane, wide = {}, {}, {}, {}
            for p in sc:
                if p in gks: continue
                k = dyn_level(gid, p, STAT)
                if k is None: k = STAT.get(int(p), 2)
                l = PLANE.get((gid, int(p)))
                if l is None: l = lane_fb(p)
                cell[p] = (int(k), int(l)); lvl[p] = int(k)
                lane[p] = int(l); wide[p] = int(l) != 1
            G = CELL[(gid, tid)]
            rec_g, s1 = pick_grid(sc, cell, G, gks, ps); shg += s1 > 0
            rec_s, s2, chosen = pick_sym(sc, lvl, wide, G, gks, ps); shs += s2 > 0
            elig = {}
            for p in sc:
                if p in gks: continue
                q = dyn_q(gid, p)
                a = lvl[p]
                E = {a, max(0, a - 1), min(4, a + 1)}
                if q is not None: E |= {k for k in range(5) if q[k] >= TAU}
                elig[p] = E
            rec_f, s3, chos_f = pick_flex(sc, elig, wide, G, gks, ps); shf += s3 > 0
            side = assign_sides(chosen, lane, G)
            gp = lambda R_: sum(sc[p] for p in R_) - sum(sc[p] for p in coach)
            cases.append(dict(gid=gid, tid=tid, oid=oid, season=int(r.season),
                              date=str(r.game_date.date()), y=yv,
                              gap_u=gp(rec_u), gap_g=gp(rec_g), gap_s=gp(rec_s), gap_f=gp(rec_f),
                              hit_u=len(set(rec_u) & set(coach)), hit_g=len(set(rec_g) & set(coach)),
                              hit_s=len(set(rec_s) & set(coach)), hit_f=len(set(rec_f) & set(coach)),
                              short_g=s1, short_s=s2, coach_sc=sum(sc[p] for p in coach),
                              prev=prev_xi.get((gid, tid)), sc=sc, cell=cell, lvl=lvl,
                              lane=lane, wide=wide, side=side, G=G, gks=tuple(gks),
                              pool=tuple(pool), coach_xi=coach, rec_u=rec_u,
                              rec_g=rec_g, rec_s=rec_s, rec_f=rec_f, elig=elig))
    C = pd.DataFrame(cases)
    print(f"\n사례 {len(C):,}   칸 부족: 엄격격자 {shg} ({100*shg/len(C):.0f}%) · "
          f"대칭 {shs} ({100*shs/len(C):.0f}%) · 대칭+허용도 {shf} ({100*shf/len(C):.0f}%)")
    C["gap_placebo"] = [sum(r.sc[p] for p in r.prev) - r.coach_sc
                        if r.prev and set(r.prev) <= set(r.sc) else np.nan
                        for r in C.itertuples(index=False)]

    # 모양 위반: 추천 XI 의 실제 (단, 중앙/와이드) 인원 vs 요구
    def viol(xi, gks_, lvl_, wide_, G_):
        got = np.zeros((5, 2), int)
        for p in xi:
            if p in gks_: continue
            got[lvl_.get(p, 2), 1 if wide_.get(p) else 0] += 1
        need = np.stack([G_[:, 1], G_[:, 0] + G_[:, 2]], 1)
        return int(np.abs(got - need).sum()) // 2
    for nm, k in [("엄격격자", "rec_g"), ("대칭", "rec_s"), ("대칭+허용도", "rec_f"), ("무제약", "rec_u")]:
        v = np.array([viol(getattr(r, k), set(r.gks), r.lvl, r.wide, r.G) for r in C.itertuples(index=False)])
        print(f"  모양 위반 {nm:8s} 평균 {v.mean():.2f}명/10   완전충족 {100*(v==0).mean():5.1f}%")

    # 어느 축이 부족을 만드나
    nl, nw, nb = 0, 0, 0
    for r in C.itertuples(index=False):
        ps2 = set(r.pool); need_l = np.zeros(5, int); need_w = np.zeros((5, 2), int)
        for k in range(5):
            need_l[k] = int(r.G[k].sum()); need_w[k, 0] = int(r.G[k, 1]); need_w[k, 1] = int(r.G[k, 0] + r.G[k, 2])
        have_l = Counter(r.lvl[p] for p in ps2 if p in r.lvl)
        have_w = Counter((r.lvl[p], 1 if r.wide[p] else 0) for p in ps2 if p in r.lvl)
        dl = sum(max(0, need_l[k] - have_l.get(k, 0)) for k in range(5))
        dw = sum(max(0, need_w[k, j] - have_w.get((k, j), 0)) for k in range(5) for j in range(2))
        nl += dl > 0; nw += dw > 0; nb += (dw > dl)
    print(f"\n  부족 원인:  단(깊이)만으로도 부족 {nl} ({100*nl/len(C):.0f}%)   "
          f"단×와이드에서 부족 {nw} ({100*nw/len(C):.0f}%)   레인이 추가로 만든 부족 {nb}")

    print("\n=== 적중 · 격차 ===")
    for nm, h, gc in [("무제약", "hit_u", "gap_u"), ("엄격 격자", "hit_g", "gap_g"),
                      ("대칭 제약", "hit_s", "gap_s"), ("대칭+허용도", "hit_f", "gap_f")]:
        print(f"  {nm:10s} 적중 {C[h].mean():.3f}/11   격차 {C[gc].mean():+.4f}   "
              f"11/11 {int((C[h]==11).sum()):3d}건")

    print("\n=== 격차 → 실제 xG 차 (팀-시즌 FE, 1SD 당) ===")
    for nm, col in [("무제약", "gap_u"), ("엄격 격자", "gap_g"), ("대칭 제약", "gap_s"),
                    ("대칭+허용도", "gap_f"), ("위약 직전 XI", "gap_placebo")]:
        sd = C[col].std(); b, lo, hi, n = fe_ols(C, col)
        print(f"  {nm:14s} {b*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}] n={n:4d} "
              f"{'*' if (lo>0)==(hi>0) else ' '}")
    print("  ── 감독 XI 점수 통제 ──")
    for nm, col in [("엄격 격자", "gap_g"), ("대칭 제약", "gap_s"), ("대칭+허용도", "gap_f")]:
        sd = C[col].std(); b, lo, hi, n = fe_ols(C, col, extra=["coach_sc"])
        print(f"  {nm:14s} {b*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}] n={n:4d} "
              f"{'*' if (lo>0)==(hi>0) else ' '}")

    print("\n=== 점수 셔플 위약 (대칭 절차, 6시드) ===")
    outs = []
    for seed in range(6):
        rng = np.random.default_rng(seed); gg2 = []
        for r in C.itertuples(index=False):
            ks = list(r.sc); vs = [r.sc[k] for k in ks]; rng.shuffle(vs)
            s2 = dict(zip(ks, vs))
            rr, _, _ = pick_flex(s2, r.elig, r.wide, r.G, list(r.gks), set(r.pool))
            gg2.append(sum(s2[p] for p in rr) - sum(s2[p] for p in r.coach_xi))
        C["gap_sh"] = gg2
        sd = C.gap_sh.std(); b, lo, hi, n = fe_ols(C, "gap_sh"); outs.append(b * sd)
        print(f"  시드{seed}  {b*sd:+.4f}")
    mu, sg = float(np.mean(outs)), float(np.std(outs))
    sd = C.gap_f.std(); b, lo, hi, n = fe_ols(C, "gap_f")
    print(f"  셔플 {mu:+.4f} ± {sg:.4f}   →  대칭+허용도는 셔플 대비 z = {(b*sd-mu)/max(sg,1e-9):+.2f}")
    C.drop(columns=["gap_sh"], errors="ignore").to_pickle(ROOT / "outputs" / "gap_sym_full.pkl")
    print("\n저장 outputs/gap_sym_full.pkl")


if __name__ == "__main__":
    main()
