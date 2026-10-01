"""Naive baselines under the abstract's protocol: train 2024, player history from 2024, test 2025.

Same scores as the EventXI paper (experiments/baseline_scores.py):
  context   season x venue intercept, no player information
  vaepsum   sum over the starting eleven of each player's past VAEP per 90 (45-day half-life, past-only)
  mins      sum over the starting eleven of each player's past minutes per match (same decay)
Each is a ridge regression on the npxG difference; alpha chosen on the last 20% of 2024 training rows.
Run: python tracking/baselines_2024only.py
"""
from pathlib import Path
import os, sys
import numpy as np, pandas as pd

TB = Path(os.environ.get("TB", Path(__file__).resolve().parent.parent))
sys.path[:0] = [str(TB / "gnn"), str(TB / "experiments")]
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp_ref.parquet")
os.chdir(TB)
import baseline_scores as BS                      # noqa: E402
from config import VAEP_OUTPUT_DIR                 # noqa: E402
from lineup_event_panel import build_event         # noqa: E402
from lineup_pipeline import AL                     # noqa: E402
from sklearn.linear_model import Ridge             # noqa: E402

HISTFROM, TRAIN, TEST = 2024, 2024, 2025
OUT = Path(__file__).resolve().parent.parent / "outputs"


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    hist, DAY = BS.player_histories()
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    cut_day = int((g[g.season == HISTFROM].game_date.min() - pd.Timestamp("2020-01-01")).days)
    hist = {p: (d[d >= cut_day], v[d >= cut_day], m[d >= cut_day]) for p, (d, v, m) in hist.items()}   # history from 2024 on
    R, F, BASE, IDX, cols = build_event("npxg"); y = R.y.to_numpy(float); sea = R.season.to_numpy(int)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])  # noqa: E712
    XI = {(int(a), int(b)): v for (a, b), v in st.items()}
    feats = {}
    for kind in ("vaepsum", "mins"):
        col = np.zeros(len(R))
        for i, r in enumerate(R.itertuples(index=False)):
            s = np.array([BS.score_at(hist, p, DAY[int(r.game_id)], kind) for p in XI.get((int(r.game_id), int(r.me)), [])], float)
            if len(s) and not np.isnan(s).all(): col[i] = np.nansum(s) + np.isnan(s).sum() * np.nanmean(s)
        feats[kind] = col
    tr = np.flatnonzero(sea == TRAIN); te = np.flatnonzero(sea == TEST); i1, i2 = tr[:int(.8 * len(tr))], tr[int(.8 * len(tr)):]
    rows = []
    for name, X in (("context", BASE), ("vaepsum", np.hstack([BASE, feats["vaepsum"][:, None]])), ("mins", np.hstack([BASE, feats["mins"][:, None]]))):
        Xs = X.copy()
        if X.shape[1] > BASE.shape[1]:
            mu, sd = X[i1, -1].mean(), X[i1, -1].std() + 1e-9; Xs[:, -1] = np.clip((X[:, -1] - mu) / sd, -5, 5)
        ba = max(AL, key=lambda a: r2(y[i2], Ridge(alpha=a).fit(Xs[i1], y[i1]).predict(Xs[i2])))
        p = Ridge(alpha=ba).fit(Xs[tr], y[tr]).predict(Xs[te])
        rows.append(dict(model=name, n_test=len(te), R2=r2(y[te], p), alpha=ba))
        np.save(OUT / f"baseline2024_{name}_pred.npy", np.column_stack([R.game_id.to_numpy()[te], R.home.to_numpy()[te] if "home" in R else np.zeros(len(te)), y[te], p]))
    D = pd.DataFrame(rows); D.to_csv(OUT / "baselines_2024only.csv", index=False); print(D.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
