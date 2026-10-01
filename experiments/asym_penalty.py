"""비대칭 배치 벌점 — 앞으로 미는 이동에만 추가 비용.

근거: `experiments/displace_cost.py` — 앞으로 1단 **-0.0193 (유의)**,
뒤로 1단 +0.0118. 현재 `-log q` 는 위아래를 대칭으로 벌해 형태가 틀렸다.

    cost = -sc[p] + β(-log q_k - log P(l|레인)) + **βf · max(0, k - 최빈단)**

βf 는 결과를 보고 고르지 않는다 — **감독의 전진 이동률에 맞추는** 규칙.
실행: python -m experiments.asym_penalty
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR  # noqa: E402
from gap_soft import QFLOOR  # noqa: E402
from gap_lane import declared_grid  # noqa: E402
from gap_diagnose import absences  # noqa: E402
import nulltest  # noqa: E402

BETA = float(os.environ.get("BETA", 0.05))
PKL = "outputs/gap_soft_fullonball_gk_resid_merged_pc20_all_npxg_nozvzn_fixsd.pkl"
GRID = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]

# 포화 자리 벌점(계단형) — 충분히 서본 자리(q ≥ τ)는 벌점 0, 그 아래는 종전
# -log q 그대로. 빈도 벌점은 "못 서는 자리"와 "여러 자리 이력"을 구별 못해
# 다재다능을 전 자리에서 과세했다(토마스 사례). 2026-09-04 배터리로 기본 채택:
# 격차 -0.0437* z=-3.08 · 동일모양 32.2% · 토마스 IN·권창훈 무연고 침투 차단.
# Q_TAU=0 으로 종전 동작 복원.
Q_TAU = float(os.environ.get("Q_TAU", "0.3"))
# 하드 컷: q < Q_MIN 인 (선수,단) 배정 금지 — 벌점(-log q)은 상한이 낮아
# 스타의 sc 프리미엄이 사버린다(이승우 LB: 0.191-0.15>최철순). 가격 대신 금지.
Q_MIN = float(os.environ.get("Q_MIN", "0.10"))
BIGPEN = 1e3


def qpen(qk):
    qk = max(float(qk), QFLOOR)
    if qk < Q_MIN:
        return BIGPEN                     # 무연고 단 금지 (배정·포메이션 비교 모두 전파)
    if Q_TAU > 0:
        # 계단형: 충분히 서본 자리(q≥τ)는 무벌점, 그 아래는 원래 벌점 그대로.
        # 연속형 -log(q/τ) 은 저확률 자리 억제력까지 -log τ 만큼 깎아서 기각.
        return 0.0 if qk >= Q_TAU else -np.log(qk)
    return -np.log(qk)


def pick_asym(sc, qmap, lane, G, gks, ps, LP, beta, bf):
    """pick_soft + 전진 이동 추가 벌점."""
    rec = [max(gks, key=lambda p: sc[p])] if gks else []
    slots = []
    for k in range(5):
        for l in range(3):
            slots += [(k, l)] * int(G[k, l])
    cand = [p for p in sc if p in ps and p not in set(rec)]
    if len(cand) < len(slots):
        return rec + sorted(cand, key=lambda p: -sc[p])[:len(slots)], {}
    cost = np.zeros((len(cand), len(slots)))
    for i, p in enumerate(cand):
        q = qmap.get(p); lp = LP[lane.get(p, 1)]
        mo = int(np.argmax(q)) if q is not None else 2
        for j, (k, l) in enumerate(slots):
            qk = float(q[k]) if q is not None else 0.2
            cost[i, j] = (-sc[p] + beta * (qpen(qk) - lp[l])
                          + bf * max(0, k - mo))
    ri, ci = linear_sum_assignment(cost)
    slot = {cand[i]: slots[j] for i, j in zip(ri, ci)}
    return rec + list(slot), slot


def main():
    C = pd.read_pickle(ROOT / PKL); LP = C.LPk.iloc[0]
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    AB = absences(gpos)
    C["absent"] = [AB.get((int(r.gid), int(r.tid)), (np.nan,))[0] for r in C.itertuples(index=False)]
    CELL, PER, _ = declared_grid()
    cn = Counter(tuple(np.asarray(G).ravel()) for G in CELL.values())
    LIB = [np.array(k, int).reshape(5, 3) for k, v in cn.items() if v >= 20]
    rows = list(C.itertuples(index=False))

    # ── 감독의 전진 이동률 (기준점)
    fw = bw = tot = 0
    for r in rows:
        for p in r.coach_xi:
            cell = PER.get((int(r.gid), int(r.tid), int(p))); q = r.qmap.get(p)
            if cell is None or q is None: continue
            d = cell[0] - int(np.argmax(q)); tot += 1
            fw += d > 0; bw += d < 0
    C_FW, C_BW = 100 * fw / tot, 100 * bw / tot
    print(f"감독 실제 배정 — 전진 {C_FW:.1f}% · 후진 {C_BW:.1f}% (n={tot:,})")

    def sel(r, sc, bf):
        best, bs, bsl = None, -1e18, None
        for F in LIB:
            R_, sl = pick_asym(sc, r.qmap, r.lane, F, list(r.gks), set(r.pool), LP, BETA, bf)
            s = sum(sc[p] for p in R_)
            if s > bs: bs, best, bsl = s, R_, sl
        return best, bsl

    print(f"\n  {'βf':>6s}{'전진%':>8s}{'후진%':>8s}{'적중':>8s}{'q<0.10':>9s}")
    print("  " + "-" * 42)
    stats = {}
    for bf in GRID:
        f_ = b_ = n_ = low = 0; hit = []
        for r in rows:
            R_, sl = sel(r, r.sc, bf)
            hit.append(len(set(R_) & set(r.coach_xi)))
            for p, (k, l) in sl.items():
                q = r.qmap.get(p)
                if q is None: continue
                d = k - int(np.argmax(q)); n_ += 1
                f_ += d > 0; b_ += d < 0; low += q[k] < 0.10
        stats[bf] = (100*f_/n_, 100*b_/n_, float(np.mean(hit)), 100*low/n_)
        print(f"  {bf:>6.2f}{stats[bf][0]:>7.1f}%{stats[bf][1]:>7.1f}%"
              f"{stats[bf][2]:>8.3f}{stats[bf][3]:>8.1f}%")
    pick = min(GRID, key=lambda b: abs(stats[b][0] - C_FW))
    print(f"\n  → 규칙: 감독 전진률 {C_FW:.1f}% 에 가장 가까운 βf = **{pick}**"
          f" (전진 {stats[pick][0]:.1f}%)")

    print(f"\n  결과 타당도 — βf=0 vs βf={pick}")
    SC = [r.sc for r in rows]
    st = {"전체": C.absent.notna().to_numpy(), "결장 0명": C.absent.eq(0).to_numpy()}
    for bf in (0.0, pick):
        res = nulltest.run(C, SC, lambda r, sc, b=bf: sel(r, sc, b)[0], st, verbose=False)
        nulltest.report(res, f"  βf = {bf}")


if __name__ == "__main__":
    main()
