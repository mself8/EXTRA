"""시즌별 role_balance_parallel 실행(HOLDOUT=s EVALSEASON=s SAVENULL=...)의 덤프를 합쳐 평가 시즌(2024·2025) 풀 지표를 낸다.

격차 효과(팀-시즌 FE) · 24시드 귀무 z · 위약(직전 XI) · 적중 · 동일모양 · 4-2-2-2 비율 (전체 + 시즌별)
실행: python -m experiments.pool_eval outputs/eval_2024.pkl outputs/eval_2025.pkl
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from gap_constrained import fe_ols  # noqa: E402


def main():
    parts = [pd.read_pickle(f) for f in sys.argv[1:]]
    D = pd.concat([p["D"] for p in parts], ignore_index=True)
    nulls = np.concatenate([p["nulls"] for p in parts], axis=1) if all(p["nulls"] is not None for p in parts) else None
    shapes = sum([p["shapes"] for p in parts], []); cshape = sum([p["cshape"] for p in parts], [])
    def eff(Dx, col):
        Dx = Dx.dropna(subset=[col]); sd = Dx[col].std(); b, lo, hi, n = fe_ols(Dx, col); return b * sd, lo * sd, hi * sd, n
    def report(mask, name):
        Dx = D[mask].reset_index(drop=True); b, lo, hi, n = eff(Dx, "gg"); star = "*" if (lo > 0) == (hi > 0) else " "
        line = f"{name:14s} n={n:,} 격차 {b:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"
        if nulls is not None:
            vals = []
            for gn in nulls:
                E = Dx.copy(); E["g2"] = gn[np.asarray(mask)]; vals.append(fe_ols(E, "g2")[0] * E.g2.std())
            mu, sg = float(np.mean(vals)), float(np.std(vals, ddof=1)); line += f" · 귀무 {mu:+.4f}±{sg:.4f} · z={(b - mu) / max(sg, 1e-9):+.2f}"
        pb, plo, phi, pn = eff(Dx, "plc"); line += f" · 위약 {pb:+.4f} [{plo:+.4f},{phi:+.4f}] n={pn:,}"
        sh = np.array(shapes)[np.asarray(mask)]; cs = np.array(cshape, dtype=object)[np.asarray(mask)]
        line += f" · 적중 {Dx.hit.mean():.3f} · 동일모양 {Dx.same.mean() * 100:.1f}% · 4-2-2-2 모델 {np.mean(sh == '4-2-2-2') * 100:.1f}%(감독 {np.mean(cs == '4-2-2-2') * 100:.1f}%)"
        print(line, flush=True)
    report(np.ones(len(D), bool), "전체(평가 시즌)")
    for s in sorted(D.season.unique()): report((D.season == s).to_numpy(), f"시즌 {int(s)}")
    for name, m in (("결장 0명", D.absent.eq(0)), ("결장 1명", D.absent.eq(1)), ("결장 2명+", D.absent.ge(2))):
        if m.sum() >= 40: report(m.to_numpy(), name)


if __name__ == "__main__":
    main()
