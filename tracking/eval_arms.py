"""EventXI 세 팔(A 이벤트만 · T +페이즈 행 · P +합친 행)의 보류 예측을 시즌별로 평가.

입력  작업 트리 outputs/player_encoder_set_<tag>.npz (preds: linear·set, y, game_id — build_sides 행 순서)
평가  팀-경기 R² · 팀-시즌 안 R²(팀 평균 제거) · 팀×홈원정 안 R²(홈 이점까지 제거) · 경기 단위 짝 부트스트랩
주의  2024 폴드는 2021–23 으로만 학습해 트래킹 행을 학습 중 본 적이 없다 → 트래킹 팔의 2024 값은 비교에 쓰지 않는다
실행  python tracking/eval_arms.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

TB = Path(__file__).resolve().parent.parent; OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(TB / "gnn")); sys.path.insert(0, str(TB / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp_ref.parquet"); os.environ.setdefault("FAMMIX", "1")
NB = 2000
TAGS = {s: {a: f"ssn-set{0 if s == 1 else 1}-cross0-H16-L1-none-full{'-stage2' if s == 2 else ''}-days{'-warm' if s == 2 else ''}-fam5-ssac{a}-s5" for a in "ATP"} for s in (1, 2)}


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def within(y, p, grp):
    df = pd.DataFrame({"y": y, "p": p, "g": grp})
    yc = df.y - df.groupby("g").y.transform("mean"); pc = df.p - df.groupby("g").p.transform("mean")
    return np.corrcoef(yc, pc)[0, 1] ** 2


def main():
    from nonadditive_gate import build_sides
    R = build_sides()[0].reset_index(drop=True)
    g = pd.read_csv(TB / "vaep/output/games.csv", usecols=["game_id", "home_team_id", "away_team_id"])
    R = R.merge(g, on="game_id", how="left")
    R["team"] = np.where(R.home == 1, R.home_team_id, R.away_team_id)
    rows = []
    for stage, tags in TAGS.items():
        P = {a: np.load(TB / f"outputs/player_encoder_set_{t}.npz") for a, t in tags.items()}
        y = P["A"]["y"]; assert np.allclose(y, R.y.to_numpy()), "행 순서가 build_sides 와 다르다"
        for a in "TP": assert np.array_equal(P[a]["game_id"], P["A"]["game_id"])
        pred = {"ridge": P["A"]["linear"], **{a: P[a]["set"] for a in "ATP"}}
        for season in (2024, 2025):
            m = (R.season == season).to_numpy() & np.isfinite(pred["A"])
            yy = y[m]; gid = R.game_id.to_numpy()[m]; team = (R.team.astype(str) + "_" + str(season)).to_numpy()[m]; tv = (team + "_" + R.home.astype(str).to_numpy()[m])
            games = np.unique(gid); idx = {k: np.flatnonzero(gid == k) for k in games}
            rng = np.random.default_rng(0); B = [np.concatenate([idx[k] for k in rng.choice(games, len(games))]) for _ in range(NB)]
            for arm, pv in pred.items():
                p = pv[m]; row = dict(stage=stage, season=season, model=arm, n=len(yy), R2=r2(yy, p), within_team=within(yy, p, team), within_team_venue=within(yy, p, tv))
                if arm in ("T", "P"):
                    pa = pred["A"][m]; d = np.array([r2(yy[b], p[b]) - r2(yy[b], pa[b]) for b in B])
                    dw = np.array([within(yy[b], p[b], tv[b]) - within(yy[b], pa[b], tv[b]) for b in B])
                    row.update(dR2_vs_A=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), P_gt0=(d > 0).mean(),
                               d_within_tv=dw.mean(), dw_lo=np.percentile(dw, 2.5), dw_hi=np.percentile(dw, 97.5))
                if arm == "T":
                    pp_ = pred["P"][m]; d2 = np.array([r2(yy[b], p[b]) - r2(yy[b], pp_[b]) for b in B])
                    row.update(dR2_T_vs_P=d2.mean(), tp_lo=np.percentile(d2, 2.5), tp_hi=np.percentile(d2, 97.5))
                rows.append(row)
    Rz = pd.DataFrame(rows); Rz.to_csv(OUT / "eval_arms.csv", index=False)
    pd.set_option("display.width", 250)
    print(Rz.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
