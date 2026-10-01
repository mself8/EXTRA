"""TeamBuilder 식 ILP 추천 기준선 — 점수 합 최대화 + 하드 위치 커버리지, 모양은 목적값이 최대인 라이브러리 모양.

원 TeamBuilder 계열(VAEP 합 + 포지션 커버리지 정수계획)을 현재 프레임에 이식한 것:
  · 선수 점수 s_p (기본 = 과거 VAEP/90 합 pkl, SCORE 로 교체 가능 — 우리 최종 점수를 넣으면 '최적화기만 다른' 대조)
  · 하드 제약: 선수는 과거 출전 비율 q_k > τ 인 단(level)에만 배정, 단별 인원 = 모양 G 의 행 합, GK 1명(점수 최대)
  · 레인 자유 → 단 배정은 할당 문제(linear_sum_assignment)로 정확히 푼다. 모양은 라이브러리(감독 선언 20회 이상) 전체를 열거해 최대 목적값.
  · 역할 균형·모양 사전확률·외국인 상한 없음 (TeamBuilder 원형에 없음)
출력: role_balance_parallel SAVENULL 과 같은 덤프(D·shapes·cshape·nulls=None) → pool_eval 로 합산.
실행: PROD=<pkl> EVALSEASON=2024 OUT=outputs/eval_ilp_vaep_2024.pkl python -m experiments.ilp_baseline
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp_ref.parquet")
from config import VAEP_OUTPUT_DIR          # noqa: E402
from gap_diagnose import absences           # noqa: E402

PROD = ROOT / os.environ["PROD"]; EV = int(os.environ["EVALSEASON"]); OUT = ROOT / os.environ["OUT"]
TAU = float(os.environ.get("QTAU", 0.05))   # 단 적격 임계 (qmap 평활 바닥 0.05 초과 = 실제 출전 이력)
LIBSRC = ROOT / os.environ.get("LIBSRC", "outputs/gap_soft_fullonball_gk_resid_merged_defresp_ref_pc20_all_npxg_nozvzn_fixsd.pkl")
BIG = 1e6


def lab(G): return "-".join(str(int(x)) for x in np.asarray(G).sum(1) if x > 0)


FIXG = os.environ.get("FIXG", "0") == "1"   # 1 = 감독이 실제 공시한 모양에 고정 (그 경기 팀시트를 아는 오라클 조건)


def solve_case(r, LIB):
    if FIXG: LIB = [np.asarray(r.G)]
    sc = r.sc; gks = [p for p in r.gks if p in sc]; pool = [p for p in r.pool if p in sc]
    gk = max(gks, key=lambda p: sc[p]) if gks else None
    of = [p for p in pool if p not in set(r.gks)]
    best = None
    for G in LIB:
        need = np.asarray(G).sum(1)                                        # 단별 인원 (5)
        if need.sum() != 10 or len(of) < 10: continue
        slots = [k for k in range(5) for _ in range(int(need[k]))]
        C = np.full((len(of), 10), BIG)
        for i, p in enumerate(of):
            q = r.qmap.get(p)
            for j, k in enumerate(slots):
                if q is not None and q[k] > TAU: C[i, j] = -sc[p]
        ri, ci = linear_sum_assignment(C)
        if (C[ri, ci] >= BIG).any(): continue                                # 적격 선수로 못 채우는 모양
        val = -C[ri, ci].sum()
        if best is None or val > best[0]: best = (val, [of[i] for i in ri], np.asarray(G))
    if best is None: return None
    return [gk] + best[1] if gk is not None else best[1], best[2]


def main():
    A = pd.read_pickle(PROD); A = A[A.season == EV].reset_index(drop=True)
    F = pd.read_pickle(LIBSRC); cn = Counter(tuple(np.asarray(G).ravel()) for G in F.G); LIB = [np.array(k, int).reshape(5, 3) for k, v in cn.items() if v >= 20]
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date); g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}; AB = absences(gpos)
    recs, shapes, cshape, keep = [], [], [], []
    for r in A.itertuples(index=False):
        out = solve_case(r, LIB)
        if out is None: keep.append(False); continue
        keep.append(True); recs.append(out[0]); shapes.append(lab(out[1])); cshape.append(lab(r.G)); recs[-1] = (out[0], out[1])
    D = A[keep].reset_index(drop=True).copy(); coach = [[int(p) for p in r.coach_xi] for r in D.itertuples(index=False)]
    D["gg"] = [sum(r.sc[p] for p in recs[i][0]) - sum(r.sc[p] for p in coach[i]) for i, r in enumerate(D.itertuples(index=False))]
    D["hit"] = [len(set(recs[i][0]) & set(coach[i])) for i in range(len(D))]
    D["same"] = [np.array_equal(recs[i][1], np.asarray(r.G)) for i, r in enumerate(D.itertuples(index=False))]
    D["plc"] = [(sum(r.sc[p] for p in r.prev) - sum(r.sc[p] for p in coach[i])) if r.prev and set(r.prev) <= set(r.sc) else np.nan for i, r in enumerate(D.itertuples(index=False))]
    D["absent"] = [AB.get((int(r.gid), int(r.tid)), (np.nan,))[0] for r in D.itertuples(index=False)]
    pd.to_pickle(dict(D=D[["gid", "tid", "season", "y", "gg", "hit", "same", "plc", "absent"]], nulls=None, shapes=shapes, cshape=cshape), OUT)
    if os.environ.get("SAVEREC"):                                            # 추천 XI 덤프 (sub_alignment 용, role_balance_parallel SAVEREC 과 같은 형식)
        pd.to_pickle([dict(gid=int(r.gid), tid=int(r.tid), season=int(r.season), rec=[int(p) for p in recs[i][0]], G=np.asarray(recs[i][1]), coach=coach[i], sc=r.sc) for i, r in enumerate(D.itertuples(index=False))], os.environ["SAVEREC"])
    print(f"[ILP {PROD.name.split('_merged_')[1].split('_pc20')[0]} {EV}] 사례 {len(D):,}/{len(A):,} · 라이브러리 {len(LIB)} · 적중 {D.hit.mean():.3f} · 동일모양 {D.same.mean() * 100:.1f}% · 4-2-2-2 {np.mean(np.array(shapes) == '4-2-2-2') * 100:.1f}% → {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
