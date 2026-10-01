"""논문 기준선(베이스라인) 점수 — 학습 없는 선수 점수로 예측·추천을 같은 규약에서 잰다.

기준선
  context   맥락만(시즌×홈 더미) — 선수 정보 없음
  vaepsum   선수별 과거 VAEP/90 (달력 반감기 45일 감쇠, 과거-only) 를 선발 11명 합산 → 릿지 1변수
  mins      선수별 과거 출전분/경기 (같은 감쇠) 합산 — "많이 뛰는 선수" 기준선

출력
  (1) 예측 R² — player_encoder_set 과 같은 R 프레임(build_event), 시즌 고정 폴드, 합산 R² 는 비가중(r2i 와 동일)
  (2) 추천용 pkl — ssn 프레임 pkl 의 sc 를 기준선 점수(시즌별 프로덕션 SD 로 재척도)로 바꿔 저장
      outputs/gap_soft_fullonball_gk_resid_merged_{name}_pc20_all_npxg_nozvzn_fixsd.pkl  (name = vaepsum25 / vaepsum26 / mins25 / mins26)
실행: python -m experiments.baseline_scores
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp_ref.parquet")
from config import VAEP_OUTPUT_DIR                  # noqa: E402
from lineup_event_panel import build_event          # noqa: E402
from lineup_pipeline import AL                      # noqa: E402
from sklearn.linear_model import Ridge              # noqa: E402

HL = float(os.environ.get("HL", 45.0))              # 달력 반감기(일) — 최종 모델과 동일
M0 = float(os.environ.get("M0", 450.0))            # 수축 사전 출전분(≈5경기)
MU = 0.0                                             # 리그 평균 분당 VAEP (player_histories 에서 채움)
PROTOCOLS = {"25": [2024, 2025], "26": [2025, 2026]}
PROD = {"25": ROOT / "outputs/gap_soft_fullonball_gk_resid_merged_defresp_ref_pc20_all_npxg_nozvzn_fixsd_ssn.pkl",
        "26": ROOT / "outputs/gap_soft_fullonball_gk_resid_merged_defresp_ref_pc20_all_npxg_nozvzn_fixsd_ssn20252026.pkl"}


def player_histories():
    """선수별 (날짜, VAEP 합, 출전분) 시계열 — 과거-only 점수용."""
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    DAY = {int(r.game_id): (r.game_date - pd.Timestamp("2020-01-01")).days for r in g.itertuples(index=False)}
    P = pd.read_parquet(ROOT / "outputs" / os.environ["ONBALL_FILE"])
    tv = [c for c in P.columns if c.startswith("tv_")]                      # SPADL 유형×결과×밴드 가치 = 온볼 VAEP 총합
    V = pd.DataFrame(dict(game_id=P.game_id.astype(int), player_id=P.player_id.astype(int), v=P[tv].sum(1).astype(float)))
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    M = pl.groupby(["game_id", "player_id"]).minutes_played.sum().rename("m").reset_index()
    M["game_id"] = M.game_id.astype(int); M["player_id"] = M.player_id.astype(int)
    H = V.merge(M, on=["game_id", "player_id"], how="left"); H["m"] = H.m.fillna(0.0).clip(0, 120)
    H["day"] = H.game_id.map(DAY); H = H.dropna(subset=["day"]).sort_values(["player_id", "day"])
    global MU; MU = float(H.v.sum() / max(H.m.sum(), 1.0)); print(f"  리그 평균 분당 VAEP {MU:.5f} (per-90 {MU * 90:.3f})")
    out = {}
    for p, gg in H.groupby("player_id", sort=False):
        out[int(p)] = (gg.day.to_numpy(float), gg.v.to_numpy(float), gg.m.to_numpy(float))
    return out, DAY


def score_at(hist, p, day, kind):
    """선수 p 의 day 직전 감쇠 점수. vaepsum: Σw·v / Σw·m × 90 (출전분 가중 per-90), mins: Σw·m / Σw."""
    h = hist.get(int(p))
    if h is None: return np.nan
    d, v, m = h; k = np.searchsorted(d, day, "left")                        # 같은 날 경기는 제외(과거-only)
    if k == 0: return np.nan
    w = 2.0 ** (-(day - d[:k]) / HL)
    if kind == "vaepsum":                                                   # 출전분 사전확률 M0 분으로 리그 평균(분당 MU)에 수축 — 소수 출전 선수의 per-90 폭주 방지
        den = (w * m[:k]).sum(); return float(((w * v[:k]).sum() + M0 * MU) / (den + M0) * 90.0)
    return float((w * m[:k]).sum() / w.sum())


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def prediction(hist, DAY):
    R, F, BASE, IDX, cols = build_event("npxg"); y = R.y.to_numpy(float)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])  # noqa: E712
    XI = {(int(a), int(b)): v for (a, b), v in st.items()}
    sea = R.season.to_numpy(int)
    feats = {}
    for kind in ("vaepsum", "mins"):
        col = np.zeros(len(R)); miss = 0
        for i, r in enumerate(R.itertuples(index=False)):
            xi = XI.get((int(r.game_id), int(r.me)), []); day = DAY[int(r.game_id)]
            s = np.array([score_at(hist, p, day, kind) for p in xi], float)
            if np.isnan(s).all(): miss += 1; continue
            col[i] = np.nansum(s) + np.isnan(s).sum() * np.nanmean(s)          # 이력 없는 선수는 팀 평균으로 채움
        feats[kind] = col; print(f"  {kind}: 전부 결측 {miss}")
    print("\n예측 R² (시즌 고정 폴드, 합산 비가중 · 시즌별)")
    for pk, TS in PROTOCOLS.items():
        for name, X in (("context", BASE), ("vaepsum", np.hstack([BASE, feats["vaepsum"][:, None]])), ("mins", np.hstack([BASE, feats["mins"][:, None]]))):
            pred = np.full(len(R), np.nan)
            for s_ in TS:
                tr = np.flatnonzero(sea < s_); te = np.flatnonzero(sea == s_); cut = int(0.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
                Xs = X.copy()
                if X.shape[1] > BASE.shape[1]:
                    mu, sd = X[i1, -1].mean(), X[i1, -1].std() + 1e-9; Xs[:, -1] = np.clip((X[:, -1] - mu) / sd, -5, 5)
                ba = max(AL, key=lambda a: r2(y[i2], Ridge(alpha=a).fit(Xs[i1], y[i1]).predict(Xs[i2])))
                pred[te] = Ridge(alpha=ba).fit(Xs[tr], y[tr]).predict(Xs[te])
            ix = np.flatnonzero(np.isfinite(pred))
            per = " · ".join(f"{s_}: {r2(y[sea == s_], pred[sea == s_]):+.4f}" for s_ in TS)
            print(f"  [{'·'.join(map(str, TS))}] {name:8s} R² {r2(y[ix], pred[ix]):+.4f}  ({per})", flush=True)


def battery_pkls(hist, DAY):
    for pk, path in PROD.items():
        A = pd.read_pickle(path)
        for kind in ("vaepsum", "mins"):
            B = A.copy(); new = []
            for r in B.itertuples(index=False):
                day = DAY[int(r.gid)]; s = {p: score_at(hist, p, day, kind) for p in r.sc}
                vals = np.array([v for v in s.values() if np.isfinite(v)]); fill = float(vals.mean()) if len(vals) else 0.0
                new.append({p: (v if np.isfinite(v) else fill) for p, v in s.items()})
            B["sc_b"] = new
            for s_ in sorted(B.season.unique()):                                    # 시즌별 프로덕션 SD 로 재척도 (set_score 와 동일 절차)
                ix = np.flatnonzero(B.season.to_numpy() == s_)
                p_all = np.array([v for i in ix for v in B.sc.iloc[i].values()]); b_all = np.array([v for i in ix for v in B.sc_b.iloc[i].values()])
                k = p_all.std() / max(b_all.std(), 1e-12); m = p_all.mean() - k * b_all.mean()
                for i in ix: B.sc_b.iloc[i].update({p: k * v + m for p, v in B.sc_b.iloc[i].items()})
            B["sc"] = B.sc_b; B.drop(columns=["sc_b"], inplace=True)
            B["coach_sc"] = [sum(r.sc[p] for p in r.coach_xi) for r in B.itertuples(index=False)]
            out = ROOT / f"outputs/gap_soft_fullonball_gk_resid_merged_{kind}{pk}_pc20_all_npxg_nozvzn_fixsd.pkl"; B.to_pickle(out)
            corr = np.corrcoef([v for r in A.itertuples(index=False) for v in r.sc.values()], [v for r in B.itertuples(index=False) for v in r.sc.values()])[0, 1]
            print(f"  {out.name}: 사례 {len(B):,} · 프로덕션 점수와 상관 {corr:+.3f}", flush=True)


def main():
    hist, DAY = player_histories(); print(f"선수 이력 {len(hist):,}명")
    prediction(hist, DAY)
    print("\n추천용 pkl"); battery_pkls(hist, DAY)


if __name__ == "__main__":
    main()
