"""출전분 축 제거 검정 — 격차가 '평소 안 쓰는 선수를 냈다' 를 재고 있는가.

① 통제:  추천 XI − 감독 XI 의 **과거 평균 출전분** 차이를 통제로 넣는다
② 위약:  점수를 **과거 평균 출전분**으로 바꿔 같은 절차로 추천 → 그 격차가
         결과를 예측하면 신호는 출전분이다

실행: python -m experiments.minutes_control <pkl> [beta]
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
from gap_diagnose import absences  # noqa: E402

HL = 60.0


def hist_minutes(gpos):
    """(gid,pid) -> 그 경기 **이전까지** 의 지수감쇠 평균 출전분."""
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    pl["ord"] = pl.game_id.map(gpos)
    pl = pl.dropna(subset=["ord"]).sort_values("ord")
    acc, out = defaultdict(lambda: [0.0, 0.0]), {}
    dec = 0.5 ** (1.0 / HL)
    for r in pl.itertuples(index=False):
        a = acc[int(r.player_id)]
        out[(int(r.game_id), int(r.player_id))] = a[0] / a[1] if a[1] > 0 else np.nan
        a[0] = a[0] * dec + float(r.minutes_played or 0); a[1] = a[1] * dec + 1
    return out


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("outputs/gap_soft_full*_all.pkl"))[-1]
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
    C = pd.read_pickle(f)
    print(f"파일 {Path(f).name}  BETA {beta}  n={len(C):,}")
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    HM = hist_minutes(gpos); AB = absences(gpos); LP = C.LPk.iloc[0]
    C["absent"] = [AB.get((int(r.gid), int(r.tid)), (99, 0))[0] for r in C.itertuples(index=False)]

    rows = []
    for r in C.itertuples(index=False):
        rec, _, _ = pick_soft(r.sc, r.qmap, r.lane, r.G, list(r.gks), set(r.pool), LP, beta)
        mm = {p: HM.get((int(r.gid), int(p)), np.nan) for p in r.pool}
        mm = {p: (0.0 if not np.isfinite(v) else v) for p, v in mm.items()}
        # 출전분을 점수로 삼은 추천 (같은 절차)
        recM, _, _ = pick_soft(mm, r.qmap, r.lane, r.G, list(r.gks), set(r.pool), LP, beta)
        mc = float(np.mean([mm.get(p, 0.0) for p in r.coach_xi]))
        mr = float(np.mean([mm.get(p, 0.0) for p in rec]))
        rows.append(dict(gid=r.gid, tid=r.tid, season=r.season, y=r.y, absent=r.absent,
                         gap=sum(r.sc[p] for p in rec) - r.coach_sc,
                         min_coach=mc, min_rec=mr, min_gap=mr - mc,
                         gapM=sum(mm.get(p, 0.0) for p in recM) - sum(mm.get(p, 0.0) for p in r.coach_xi),
                         hit=len(set(rec) & set(r.coach_xi)),
                         hitM=len(set(recM) & set(r.coach_xi))))
    D = pd.DataFrame(rows)
    Z = D[D.absent == 0]
    print(f"결장 0명 n={len(Z)}   적중: 점수기반 {Z.hit.mean():.2f} · 출전분기반 {Z.hitM.mean():.2f}/11")
    print(f"격차 ↔ 출전분격차 상관 r={Z.gap.corr(Z.min_gap):+.3f}   "
          f"격차 ↔ 출전분추천격차 r={Z.gap.corr(Z.gapM):+.3f}")
    print(f"\n{'명세':38s}{'계수 (1SD 당)':>24s}")
    for nm, col, extra in [
        ("격차 (기본)", "gap", []),
        ("격차 | 출전분 격차 통제", "gap", ["min_gap"]),
        ("격차 | 감독XI 평소출전분 통제", "gap", ["min_coach"]),
        ("격차 | 둘 다 통제", "gap", ["min_gap", "min_coach"]),
        ("── 위약 ──", None, None),
        ("출전분 격차 (추천−감독)", "min_gap", []),
        ("출전분으로 만든 추천의 격차", "gapM", []),
        ("감독 XI 의 평소 출전분", "min_coach", []),
    ]:
        if col is None:
            print(f"  {nm}"); continue
        s = Z.dropna(subset=[col]); sd = s[col].std()
        b, lo, hi, n = fe_ols(s, col, extra=extra)
        star = "*" if (lo > 0) == (hi > 0) else " "
        print(f"  {nm:36s}{b*sd:+.4f} [{lo*sd:+.3f},{hi*sd:+.3f}] {star}")


if __name__ == "__main__":
    main()
