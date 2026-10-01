"""2024-only 재학습(run_eventxi_2024only.sh) 평가 — v2(eval_arms.py)와 같은 2025 보류 R² · 경기 단위 짝 부트스트랩.

ssac24A/T: train 2024, history from 2021 · ssac24hA/T: train 2024, history from 2024 (abstract Table 1)
비교용 v2: ssacA / ssacT (학습 2021–24) 2단계 2025 행
실행  python tracking/eval_2024only.py
"""
from pathlib import Path
import numpy as np, pandas as pd

TB = Path(__file__).resolve().parent.parent / "outputs"; OUT = Path(__file__).resolve().parent.parent / "outputs"
NB = 2000
RUNS = {"v2 (2021-24)": ("ssn-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-ssac{}-s5", "A", "T"),
        "train 2024, history 2021-": ("ssn2025-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-ssac24{}-s5", "A", "T"),
        "train 2024, history 2024- (abstract)": ("ssn2025-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-ssac24h{}-s5", "A", "T")}


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    rows = []
    for name, (fmt, a, t) in RUNS.items():
        A = np.load(TB / f"player_encoder_set_{fmt.format(a)}.npz"); T = np.load(TB / f"player_encoder_set_{fmt.format(t)}.npz")
        y = A["y"]; gid = A["game_id"]; assert np.array_equal(gid, T["game_id"]) and np.allclose(y, T["y"])
        g = pd.read_csv(TB.parent / "vaep/output/games.csv", usecols=["game_id", "season"]).set_index("game_id").season
        m = (g.reindex(gid).to_numpy() == 2025) & np.isfinite(A["set"]) & np.isfinite(T["set"])
        yy, gg = y[m], gid[m]; pl, pa, pt = A["linear"][m], A["set"][m], T["set"][m]
        games = np.unique(gg); idx = {k: np.flatnonzero(gg == k) for k in games}
        rng = np.random.default_rng(0); B = [np.concatenate([idx[k] for k in rng.choice(games, len(games))]) for _ in range(NB)]
        d = np.array([r2(yy[b], pt[b]) - r2(yy[b], pa[b]) for b in B])
        rows.append(dict(run=name, n=len(yy), linear=r2(yy, pl), event=r2(yy, pa), tracking=r2(yy, pt), dR2=r2(yy, pt) - r2(yy, pa),
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), P_gt0=(d > 0).mean()))
    R = pd.DataFrame(rows); R.to_csv(OUT / "eval_2024only.csv", index=False)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
