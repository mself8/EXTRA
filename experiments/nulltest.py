"""셔플 귀무 검정 — 공용 구현.

왜 필요한가 (2026-09-01)
  ① **중복 제거**: 귀무 선택은 층(전체·결장0명·시즌…)과 무관한데, 층 루프 안에서
     다시 뽑는 코드가 여럿 있었다. 층 수만큼 낭비된다.
  ② **규약 고정**: 귀무는 **선택도 채점도 셔플 점수로** 한다(`gap_soft` 규약).
     셔플로 뽑고 **실제** 점수로 격차를 재면 귀무 평균이 0 에서 크게 벗어나
     z 가 통째로 틀린다 — 한 번 그렇게 틀렸다(-0.0628 편향, z -1.40 vs -2.51).
  ③ **시드 하한**: 6시드 SD 는 3배까지 널뛴다. 기본 24.

비용: 시드 × 선택 1회 (층 수와 무관). 이전에는 시드 × 층.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_constrained import fe_ols  # noqa: E402

SEEDS = 24


def shuffled(SC, seed):
    """사례마다 점수를 후보 안에서 뒤섞는다 (gap_soft 와 동일)."""
    rng = np.random.default_rng(seed)
    out = []
    for d in SC:
        ks = list(d); vs = [d[k] for k in ks]; rng.shuffle(vs)
        out.append(dict(zip(ks, vs)))
    return out


def run(C, SC, select, strata, seeds=SEEDS, coach_col="coach_xi", verbose=True):
    """select(row, scores) -> XI · strata: {이름: 불리언 마스크}

    반환 {이름: dict(b, lo, hi, n, null_mu, null_sd, z)}
    """
    rows = list(C.itertuples(index=False))
    coach = [[int(p) for p in getattr(r, coach_col)] for r in rows]

    def gaps(scores, recs):
        return np.array([sum(scores[i][p] for p in recs[i])
                         - sum(scores[i][p] for p in coach[i]) for i in range(len(rows))])

    g_obs = gaps(SC, [select(r, SC[i]) for i, r in enumerate(rows)])
    # ── 귀무 선택은 층과 무관 → 한 번만
    nulls = []
    for s in range(seeds):
        s2 = shuffled(SC, s)
        nulls.append(gaps(s2, [select(r, s2[i]) for i, r in enumerate(rows)]))
    if verbose:
        print(f"  귀무 {seeds}시드 · 선택 {seeds + 1}회 (층 {len(strata)}개와 무관)")

    out = {}
    for name, mask in strata.items():
        m = np.asarray(mask, bool)
        if m.sum() < 40:
            out[name] = None; continue
        D = C[m].copy(); D["gg"] = g_obs[m]
        sd = D.gg.std(); b, lo, hi, n = fe_ols(D, "gg")
        vals = []
        for gn in nulls:
            E = D.copy(); E["g2"] = gn[m]
            vals.append(fe_ols(E, "g2")[0] * E.g2.std())
        mu, sg = float(np.mean(vals)), float(np.std(vals, ddof=1))
        out[name] = dict(b=b * sd, lo=lo * sd, hi=hi * sd, n=n,
                         null_mu=mu, null_sd=sg, z=(b * sd - mu) / max(sg, 1e-9))
    return out


def report(res, title=""):
    if title: print(f"\n{title}")
    print(f"  {'층':14s}{'n':>7s}{'효과':>10s}{'95% CI':>21s}{'귀무 평균±SD':>16s}{'z':>8s}")
    print("  " + "-" * 76)
    for k, v in res.items():
        if v is None:
            print(f"  {k:14s}   표본 부족"); continue
        star = "*" if (v["lo"] > 0) == (v["hi"] > 0) else " "
        print(f"  {k:14s}{v['n']:>7,}{v['b']:>10.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]{star}"
              f"{v['null_mu']:>+9.4f}±{v['null_sd']:.4f}{v['z']:>8.2f}")
