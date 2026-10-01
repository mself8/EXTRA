"""격차 검정 약화가 역인과인가 잡음인가.

가설 A (잡음)     6시드 셔플 SD 추정 오차가 커서 z(-1.81 vs -1.43) 차이가 무의미
가설 B (역인과)   주전이 결장한 경기 → 감독 XI 점수가 낮고 → 원래 결과도 나쁘다.
                 모델은 결장자를 못 뽑으니 격차가 커진다. 즉 격차↔결과는 결장의 그림자.

검정
  ① 시드 24개로 셔플 귀무 SD 를 다시 추정 → z 안정화
  ② 결장 측도: 그 팀의 **최근 5경기 중 3번 이상 선발**한 선수 중 오늘 명단에 없는 수
     - 통제로 넣기
     - 결장 0명인 부분표본으로 제한

실행: python -m experiments.gap_diagnose <pkl>
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR
from gap_constrained import fe_ols  # noqa: E402
from gap_soft import pick_soft  # noqa: E402


def absences(gpos):
    """(gid,tid) -> 최근 5경기 중 3번 이상 선발한 선수 가운데 오늘 명단에 없는 수."""
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == 1].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])
    sq = pl.groupby(["game_id", "team_id"]).player_id.apply(lambda s: set(int(x) for x in s))
    rows = [(int(a), int(b)) for a, b in st.index]
    rows = sorted(rows, key=lambda k: gpos.get(k[0], 1 << 30))
    hist, out = defaultdict(list), {}
    for gid, tid in rows:
        h = hist[tid][-5:]
        if len(h) >= 3:
            c = defaultdict(int)
            for xs in h:
                for p in xs: c[p] += 1
            reg = [p for p, n in c.items() if n >= 3]
            today = sq.get((gid, tid), set())
            out[(gid, tid)] = (sum(1 for p in reg if p not in today), len(reg))
        hist[tid].append(list(st.loc[(gid, tid)]))
    return out


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("outputs/gap_soft_full*_pc20.pkl"))[-1]
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    C = pd.read_pickle(f)
    print(f"파일 {f}   BETA {beta}   사례 {len(C):,}")
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    AB = absences(gpos)
    C["absent"] = [AB.get((int(r.gid), int(r.tid)), (np.nan, np.nan))[0] for r in C.itertuples(index=False)]
    C["nreg"] = [AB.get((int(r.gid), int(r.tid)), (np.nan, np.nan))[1] for r in C.itertuples(index=False)]
    LP = C.LPk.iloc[0]
    rec = [pick_soft(r.sc, r.qmap, r.lane, r.G, list(r.gks), set(r.pool), LP, beta)[0]
           for r in C.itertuples(index=False)]
    C["rec"] = rec
    C["gap"] = [sum(r.sc[p] for p in r.rec) - r.coach_sc for r in C.itertuples(index=False)]
    C["hit"] = [len(set(r.rec) & set(r.coach_xi)) for r in C.itertuples(index=False)]
    print(f"적중 {C.hit.mean():.3f}/11   결장 평균 {C.absent.mean():.2f}명 "
          f"(정규 주전 {C.nreg.mean():.1f}명)   결장 0명 {int((C.absent==0).sum())}건")
    print(f"결장 ↔ 감독 XI 점수 r={C.absent.corr(C.coach_sc):+.3f}   "
          f"결장 ↔ 격차 r={C.absent.corr(C.gap):+.3f}   결장 ↔ 결과 r={C.absent.corr(C.y):+.3f}")

    sd = C.gap.std()
    print(f"\n=== ① 격차 효과 (1SD 당) ===")
    b, lo, hi, n = fe_ols(C, "gap")
    print(f"  기본                {b*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}] n={n}")
    b2, lo2, hi2, n2 = fe_ols(C, "gap", extra=["coach_sc"])
    print(f"  + 감독 XI 점수 통제   {b2*sd:+.4f} [{lo2*sd:+.4f},{hi2*sd:+.4f}] n={n2}")
    D = C.dropna(subset=["absent"])
    b3, lo3, hi3, n3 = fe_ols(D, "gap", extra=["absent"])
    print(f"  + 결장 수 통제        {b3*sd:+.4f} [{lo3*sd:+.4f},{hi3*sd:+.4f}] n={n3}")
    b4, lo4, hi4, n4 = fe_ols(D, "gap", extra=["absent", "coach_sc"])
    print(f"  + 둘 다 통제          {b4*sd:+.4f} [{lo4*sd:+.4f},{hi4*sd:+.4f}] n={n4}")
    Z = D[D.absent == 0]
    if len(Z) > 60:
        sdz = Z.gap.std(); b5, lo5, hi5, n5 = fe_ols(Z, "gap")
        print(f"  결장 0명 부분표본만    {b5*sdz:+.4f} [{lo5*sdz:+.4f},{hi5*sdz:+.4f}] n={n5}")
    Zp = D[D.absent >= 2]
    if len(Zp) > 60:
        sdp = Zp.gap.std(); b6, lo6, hi6, n6 = fe_ols(Zp, "gap")
        print(f"  결장 2명 이상만       {b6*sdp:+.4f} [{lo6*sdp:+.4f},{hi6*sdp:+.4f}] n={n6}")

    print(f"\n=== ② 셔플 귀무 24시드 ===")
    outs = []
    for seed in range(24):
        rng = np.random.default_rng(seed); gg = []
        for r in C.itertuples(index=False):
            ks = list(r.sc); vs = [r.sc[k] for k in ks]; rng.shuffle(vs); s2 = dict(zip(ks, vs))
            R_, _, _ = pick_soft(s2, r.qmap, r.lane, r.G, list(r.gks), set(r.pool), LP, beta)
            gg.append(sum(s2[p] for p in R_) - sum(s2[p] for p in r.coach_xi))
        C["gsh"] = gg
        s2d = C.gsh.std(); bb, _, _, _ = fe_ols(C, "gsh"); outs.append(bb * s2d)
    mu, sg = float(np.mean(outs)), float(np.std(outs, ddof=1))
    se_sd = sg / np.sqrt(2 * (len(outs) - 1))
    print(f"  평균 {mu:+.4f}  SD {sg:.4f} (±{se_sd:.4f})   →  z = {(b*sd-mu)/sg:+.2f}")
    print(f"  6시드만 썼을 때 SD 범위: " +
          " ".join(f"{np.std(outs[i:i+6], ddof=1):.4f}" for i in range(0, 24, 6)))

    print(f"\n=== ③ 결장 0명 부분표본에 같은 귀무 (24시드) ===")
    Z0 = C[C.absent == 0].copy()
    sdz = Z0.gap.std(); bz, loz, hiz, nz = fe_ols(Z0, "gap")
    o2 = []
    for seed in range(24):
        rng = np.random.default_rng(1000 + seed); gg = []
        for r in Z0.itertuples(index=False):
            ks = list(r.sc); vs = [r.sc[k] for k in ks]; rng.shuffle(vs); s2 = dict(zip(ks, vs))
            R_, _, _ = pick_soft(s2, r.qmap, r.lane, r.G, list(r.gks), set(r.pool), LP, beta)
            gg.append(sum(s2[p] for p in R_) - sum(s2[p] for p in r.coach_xi))
        Z0["gsh"] = gg
        s2d = Z0.gsh.std(); bb2, _, _, _ = fe_ols(Z0, "gsh"); o2.append(bb2 * s2d)
    m2, g2 = float(np.mean(o2)), float(np.std(o2, ddof=1))
    print(f"  효과 {bz*sdz:+.4f} [{loz*sdz:+.4f},{hiz*sdz:+.4f}] n={nz}")
    print(f"  셔플 {m2:+.4f} ± {g2:.4f}   →  z = {(bz*sdz-m2)/g2:+.2f}")


if __name__ == "__main__":
    main()
