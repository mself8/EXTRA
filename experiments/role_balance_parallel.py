"""칸 역할 균형 벌점 — 병렬 선택기 + 셔플 귀무 (24시드) + 포메이션 분포.

속도: multiprocessing(fork, 워커 10) 로 사례를 나누고, 국소 교환 탐색은 헝가리안 점수 상위 TOPK 포메이션에만 적용.
귀무 규약은 nulltest 와 동일(선택·채점 모두 셔플 점수), 팀-시즌 FE(fe_ols), 층 4개.
실행: LAM=0.01 SEEDS=24 TOPK=6 python -m experiments.role_balance_parallel
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from role_balance_battery import build   # noqa: E402
from asym_penalty import pick_asym       # noqa: E402
from gap_constrained import fe_ols       # noqa: E402
from gap_diagnose import absences        # noqa: E402
import nulltest                          # noqa: E402

LAM = float(os.environ.get("LAM", 0.01)); SEEDS = int(os.environ.get("SEEDS", 24)); TOPK = int(os.environ.get("TOPK", 6)); NW = int(os.environ.get("NW", 10))
SP = float(os.environ.get("SHAPEPRIOR", 0.0))   # 포메이션 사전확률 가중 (팀 과거 모양 감쇠 빈도 + 전역 평활, 과거-only)
G_ = {}


def _init(B, rows, TAUS, PRI):
    G_["B"] = B; G_["rows"] = rows; G_["TAUS"] = TAUS; G_["PRI"] = PRI


def _select(args):
    i, sc, lam = args; B = G_["B"]; r = G_["rows"][i]; TAU = G_["TAUS"][i]; LIB = B["LIB"]; solve = B["solve"]
    LP = B["A"].LPk.iloc[0]
    # 1) 헝가리안(λ=0) 점수로 포메이션 순위 → 상위 TOPK 만 국소 탐색
    pre = []
    for G in LIB:
        xi, slot, v = solve(r, G, TAU, 0.0, sc); pre.append((v, G))
    pri = G_["PRI"].get(i, {})
    pre = [(v + SP * pri.get(tuple(np.asarray(G).ravel()), -6.0), G) for v, G in pre]
    pre.sort(key=lambda t: -t[0]); best = (None, -1e18, None)
    for _, G in pre[:TOPK]:
        xi, slot, v = solve(r, G, TAU, lam, sc); v = v + SP * pri.get(tuple(np.asarray(G).ravel()), -6.0)
        if v > best[1]: best = (xi, v, G, slot)
    return best[0], best[2], best[3]


SLOTS = []                                                            # 마지막 run_pass 의 칸 배정 (SAVEREC 용)
def run_pass(pool, rows, SC, lam):
    out = pool.map(_select, [(i, SC[i], lam) for i in range(len(rows))], chunksize=16)
    SLOTS[:] = [o[2] for o in out]
    return [o[0] for o in out], [o[1] for o in out]


def main():
    B = build(); A, rows, gpos, tau, CELL = (B[k] for k in ("A", "rows", "gpos", "tau", "CELL"))
    C = A[A.gid.isin([int(r.gid) for r in rows])].reset_index(drop=True); rows = list(C.itertuples(index=False))
    AB = absences(gpos); C["absent"] = [AB.get((int(r.gid), int(r.tid)), (np.nan,))[0] for r in rows]
    TAUS = [{p: tau(p, gpos[int(r.gid)]) for p in r.sc} for r in rows]
    SC = [r.sc for r in rows]; coach = [[int(p) for p in r.coach_xi] for r in rows]
    # 포메이션 사전확률: 팀별 과거 선언 모양(감쇠 반감기 10경기) + 전역 빈도 평활 → log P (과거-only)
    PRI = {}
    if SP > 0:
        LIBK = [tuple(np.asarray(G).ravel()) for G in B["LIB"]]; glob = Counter(tuple(G.ravel()) for G in B["FIT"].values()); gt = sum(glob.values())
        hist = sorted([(gpos[g_], t_, tuple(G.ravel())) for (g_, t_), G in CELL.items() if g_ in gpos])
        import math; lam_ = math.log(2) / 10; acc = {}
        by_team = {}
        for t_, tid, shp in hist: by_team.setdefault(tid, []).append((t_, shp))
        for i, r in enumerate(rows):
            tid, tt = int(r.tid), gpos[int(r.gid)]; cnt = Counter(); w = 0.0
            for t_, shp in by_team.get(tid, []):
                if t_ >= tt: break
                dw = math.exp(-lam_ * (tt - t_)); cnt[shp] += dw; w += dw
            PRI[i] = {k: math.log((cnt.get(k, 0.0) + 2.0 * glob.get(k, 0) / gt) / (w + 2.0)) for k in LIBK}
    gaps = lambda scores, recs: np.array([sum(scores[i][p] for p in recs[i]) - sum(scores[i][p] for p in coach[i]) for i in range(len(rows))])
    lab = lambda G: "-".join(str(int(x)) for x in np.asarray(G).sum(1) if x > 0)
    EV = int(os.environ.get("EVALSEASON", 0))
    with Pool(NW, initializer=_init, initargs=(B, rows, TAUS, PRI)) as pool:
        recs, shapes = run_pass(pool, rows, SC, LAM); g_obs = gaps(SC, recs)
        SC_full, rows_full, coach_full, keep = SC, rows, coach, np.ones(len(rows), bool)   # 귀무 패스는 워커가 든 전체 rows 와 정렬돼야 함
        gaps_full = lambda scores, recs_: np.array([sum(scores[i][p] for p in recs_[i]) - sum(scores[i][p] for p in coach_full[i]) for i in range(len(rows_full))])
        if EV:                                                        # 보류 시즌만 평가
            keep = np.array([int(r.season) == EV for r in rows])
            CAP = int(os.environ.get("EVALCAP", 0))                   # 보류 시즌의 시간순 앞 CAP 사례만 (부분 시즌 대조)
            if CAP: keep &= np.cumsum(keep) <= CAP
            print(f"평가 시즌 {EV}{f' (앞 {CAP} 사례)' if CAP else ''}: 사례 {int(keep.sum()):,}/{len(rows):,}", flush=True)
            C = C[keep].reset_index(drop=True); rows = [r for r, k in zip(rows, keep) if k]; recs = [x for x, k in zip(recs, keep) if k]; shapes = [x for x, k in zip(shapes, keep) if k]
            SC = [x for x, k in zip(SC, keep) if k]; coach = [x for x, k in zip(coach, keep) if k]; g_obs = g_obs[keep]
            SLOTS[:] = [x for x, k in zip(SLOTS, keep) if k]                  # SAVEREC 정렬 (칸 배정도 보류 시즌만)
        if os.environ.get("SAVEREC"):                                     # 추천 결과 덤프 (그림·사례용): gid, tid, 추천 XI, 모양, 감독 XI, 점수
            pd.to_pickle([dict(gid=int(r.gid), tid=int(r.tid), season=int(r.season), rec=list(recs[i]), slot=dict(SLOTS[i]), G=np.asarray(shapes[i]), coach=list(coach[i]), sc=SC[i]) for i, r in enumerate(rows)], os.environ["SAVEREC"])
            print(f"추천 덤프 → {os.environ['SAVEREC']}", flush=True)
        hit = np.mean([len(set(recs[i]) & set(coach[i])) for i in range(len(rows))])
        cs = Counter(lab(CELL[(int(r.gid), int(r.tid))]) for r in rows if (int(r.gid), int(r.tid)) in CELL); ms = Counter(lab(G) for G in shapes)
        same = np.mean([(int(r.gid), int(r.tid)) in CELL and np.array_equal(shapes[i], CELL[(int(r.gid), int(r.tid))]) for i, r in enumerate(rows)])
        g_sd = g_obs.std(); D0 = C.copy(); D0["gg"] = g_obs; b0, lo0, hi0, _ = fe_ols(D0, "gg")
        print(f"[λ={LAM} SP={SP} TOPK={TOPK}] 관측: 적중 {hit:.3f} · 동일모양 {same * 100:.1f}% · 격차 {b0 * g_sd:+.4f} [{lo0 * g_sd:+.4f},{hi0 * g_sd:+.4f}]", flush=True)
        n = len(rows); print(f"{'모양':14s} {'감독%':>7s} {'모델%':>7s}")
        for k, _ in cs.most_common(8): print(f"{k:14s} {100 * cs[k] / n:7.1f} {100 * ms[k] / n:7.1f}")
        fw2 = lambda c: 100 * sum(v for k, v in c.items() if int(k.split('-')[-1]) >= 2) / n
        print(f"{'최전방≥2 (투톱)':14s} {fw2(cs):7.1f} {fw2(ms):7.1f}", flush=True)
        # 외국인 동시 선발 (휴리스틱 플래그): 감독 vs 모델 분포 — 한도 위반 진단
        try:
            FRN = B["FRN"]
            fc = [sum(int(p) in FRN for p in coach[i]) for i in range(len(rows))]; fm = [sum(int(p) in FRN for p in recs[i]) for i in range(len(rows))]
            print(f"외국인 XI 수(공식 국적) — 감독 평균 {np.mean(fc):.2f} 최대 {max(fc)} · 모델 평균 {np.mean(fm):.2f} 최대 {max(fm)} · 모델 ≥5 비율 {np.mean(np.array(fm) >= 5) * 100:.1f}% (감독 {np.mean(np.array(fc) >= 5) * 100:.1f}%)", flush=True)
        except Exception as e_: print("외국인 진단 생략:", e_)
        nulls = []
        for s in range(SEEDS):
            s2 = nulltest.shuffled(SC_full, s); r2_, _ = run_pass(pool, rows_full, s2, LAM); nulls.append(gaps_full(s2, r2_)[keep])
            if (s + 1) % 6 == 0: print(f"  귀무 {s + 1}/{SEEDS} 시드", flush=True)
    if os.environ.get("SAVENULL"):                                        # 시즌별 실행을 합쳐 평가하기 위한 덤프 (pool_eval)
        Dd = C.copy(); Dd["gg"] = g_obs
        Dd["hit"] = [len(set(recs[i]) & set(coach[i])) for i in range(len(rows))]
        Dd["same"] = [(int(r.gid), int(r.tid)) in CELL and np.array_equal(shapes[i], CELL[(int(r.gid), int(r.tid))]) for i, r in enumerate(rows)]
        Dd["plc"] = [(sum(SC[i][p] for p in r.prev) - sum(SC[i][p] for p in coach[i])) if r.prev and set(r.prev) <= set(SC[i]) else np.nan for i, r in enumerate(rows)]
        pd.to_pickle(dict(D=Dd, nulls=np.array(nulls) if nulls else None, shapes=[lab(G) for G in shapes], cshape=[lab(CELL[(int(r.gid), int(r.tid))]) if (int(r.gid), int(r.tid)) in CELL else None for r in rows]), os.environ["SAVENULL"])
        print(f"평가 덤프 → {os.environ['SAVENULL']}", flush=True)
    strata = {"전체": C.absent.notna().to_numpy(), "결장 0명": C.absent.eq(0).to_numpy(), "결장 1명": C.absent.eq(1).to_numpy(), "결장 2명+": C.absent.ge(2).to_numpy()}
    print(f"\n칸 역할 균형 λ={LAM} · 귀무 {SEEDS}시드 · 팀-시즌 FE")
    print(f"  {'층':10s}{'n':>7s}{'효과':>10s}{'95% CI':>21s}{'귀무 평균±SD':>18s}{'z':>7s}")
    for name, mask in strata.items():
        m = np.asarray(mask, bool)
        if m.sum() < 40: continue
        D = C[m].copy(); D["gg"] = g_obs[m]; sd = D.gg.std(); b, lo, hi, nn_ = fe_ols(D, "gg")
        vals = []
        for gn in nulls:
            E = D.copy(); E["g2"] = gn[m]; vals.append(fe_ols(E, "g2")[0] * E.g2.std())
        mu, sg = float(np.mean(vals)), float(np.std(vals, ddof=1)); star = "*" if (lo > 0) == (hi > 0) else " "
        print(f"  {name:10s}{nn_:>7,}{b * sd:>10.4f} [{lo * sd:+.4f},{hi * sd:+.4f}]{star}  {mu:+.4f}±{sg:.4f}{(b * sd - mu) / max(sg, 1e-9):>7.2f}")
    plc = [(sum(SC[i][p] for p in r.prev) - sum(SC[i][p] for p in coach[i])) if r.prev and set(r.prev) <= set(SC[i]) else np.nan for i, r in enumerate(rows)]
    D = C.copy(); D["plc"] = plc; D = D.dropna(subset=["plc"]); sd = D.plc.std(); b, lo, hi, nn_ = fe_ols(D, "plc")
    print(f"  위약(직전 XI) {b * sd:+.4f} [{lo * sd:+.4f},{hi * sd:+.4f}] n={nn_:,}")


if __name__ == "__main__":
    main()
