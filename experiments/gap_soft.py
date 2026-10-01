"""연성 제약 — 좌우 교차와 단 이동을 **금지가 아니라 비용**으로.

엄격 격자(교차 금지)는 65% 사례에서 칸이 모자랐고, 대칭 완화(교차 무료)는
왼쪽 풀백 둘을 뽑아 하나를 오른쪽에 꽂는다. 실측은 그 사이다:

    과거 레인 → 그날 레인      좌→좌 76.2  좌→중 12.1  좌→우 11.7
                            중→좌  4.4  중→중 90.9  중→우  4.7
                            우→좌 11.8  우→중 13.1  우→우 75.1
    단        argmax 일치 83.9%  (동적 확률 q 가 나머지를 설명)

그래서 슬롯 (단 k, 레인 l) 배정 비용을

    cost(p → (k,l)) = -점수(p) + BETA * ( -log q_k(p) - log P(l | 과거레인 p) )

로 두고 10 슬롯 1:1 배정을 `linear_sum_assignment` 로 정확히 푼다.
BETA=0 이면 무제약, BETA→∞ 면 엄격 격자. 실측 교차율 13.5% 에 맞춰 보정한다.

실행: python -m experiments.gap_soft [BETA]
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
import os
from config import VAEP_OUTPUT_DIR
from gap_constrained import fe_ols, dyn_level  # noqa: E402
from gap_lane import declared_grid, past_lanes, LEFTPOS, RIGHTPOS  # noqa: E402
from gap_sym import dyn_q  # noqa: E402

QFLOOR = 1e-3
NPC = int(os.environ.get("NPC", 5))
TARGET = os.environ.get("TARGET", "xg")     # xg | npxg (기본은 기존 동작 보존)
SEEDS = int(os.environ.get("SEEDS", 6))     # 셔플 귀무 시드 수 (보고는 24 이상)
WITHG = os.environ.get("WITHGAMMA", "") == "1"   # 감독 모형 + 반분 점수 추가 산출
SEASONS = {int(x) for x in os.environ.get("SEASONS","").split(",") if x.strip()}
# 채점 표준화 수정 (2026-09-03): 릿지는 표준화 열 공간에서 적합되는데 채점은 원시
# 스냅샷에 내적했다 — 원시 SD가 큰 횟수열(tn 중앙 0.42 vs tv 0.007)만 ~60배 증폭돼
# 점수가 액션 볼륨에 지배됐다. FIXSD=1 이면 Wf 를 /σ 로 되돌리고, β·βf 의미 보존을
# 위해 점수 SD 를 종전과 같게 맞춘다. FIXSD=0 은 종전 동작 재현용.
FIXSD = os.environ.get("FIXSD", "1") == "1"
TAG = os.environ.get("ONBALL_FILE", "onball_vaep_event.parquet").replace("onball_vaep_event","").replace(".parquet","")


def lane_logp(PER, PLANE):
    """과거 레인 → 그날 레인 로그확률 (3x3). 데이터에서 직접."""
    M = np.ones((3, 3)) * 1.0
    for (gid, tid, p), (k, l) in PER.items():
        q = PLANE.get((gid, int(p)))
        if q is not None:
            M[q, l] += 1
    M /= M.sum(1, keepdims=True)
    return np.log(M)


def pick_soft(sc, qmap, lane, G, gks, ps, LP, beta):
    """슬롯 (단,레인) 1:1 배정 — 비용 = -점수 + beta*(-log q_k - log P(l|과거레인))."""
    rec = [max(gks, key=lambda p: sc[p])] if gks else []
    slots = []
    for k in range(5):
        for l in range(3):
            slots += [(k, l)] * int(G[k, l])
    cand = [p for p in sc if p in ps and p not in set(rec)]
    if len(cand) < len(slots):
        return rec + sorted(cand, key=lambda p: -sc[p])[:len(slots)], {}, 0
    cost = np.zeros((len(cand), len(slots)))
    for i, p in enumerate(cand):
        q = qmap.get(p); lp = LP[lane.get(p, 1)]
        for j, (k, l) in enumerate(slots):
            qk = float(q[k]) if q is not None else 0.2
            cost[i, j] = -sc[p] + beta * (-np.log(max(qk, QFLOOR)) - lp[l])
    ri, ci = linear_sum_assignment(cost)
    slot = {cand[i]: slots[j] for i, j in zip(ri, ci)}
    cross = sum(1 for p, (k, l) in slot.items()
                if lane.get(p, 1) != 1 and l != 1 and l != lane.get(p, 1))
    return rec + list(slot), slot, cross


def pick_soft2(sc, qmap, lane, G, gks, ps, LP, b_lv, b_ln):
    """단(깊이)과 레인(좌우)에 **다른 계수**를 준다.

    cost = -점수 + b_lv*(-log q_k) + b_ln*(-log P(l | 과거레인))

    실측 교차율이 중앙↔와이드 4.7% · 좌↔우 11.7% 로 2.5배 차이나는데
    단일 BETA 로는 둘을 따로 못 조인다. 로그확률에 그 비가 이미 담겨 있으므로
    b_ln 하나로 레인만 비싸게 매기면 비를 유지한 채 조여진다.
    """
    rec = [max(gks, key=lambda p: sc[p])] if gks else []
    slots = []
    for k in range(5):
        for l in range(3):
            slots += [(k, l)] * int(G[k, l])
    cand = [p for p in sc if p in ps and p not in set(rec)]
    if len(cand) < len(slots):
        return rec + sorted(cand, key=lambda p: -sc[p])[:len(slots)], {}, 0, 0
    cost = np.zeros((len(cand), len(slots)))
    for i, p in enumerate(cand):
        q = qmap.get(p); lp = LP[lane.get(p, 1)]
        for j, (k, l) in enumerate(slots):
            qk = float(q[k]) if q is not None else 0.2
            cost[i, j] = -sc[p] + b_lv * (-np.log(max(qk, QFLOOR))) + b_ln * (-lp[l])
    ri, ci = linear_sum_assignment(cost)
    slot = {cand[i]: slots[j] for i, j in zip(ri, ci)}
    cw = sum(1 for p, (k, l) in slot.items() if lane.get(p, 1) == 1 and l != 1)   # 중앙→와이드
    lr = sum(1 for p, (k, l) in slot.items()
             if lane.get(p, 1) != 1 and l != 1 and l != lane.get(p, 1))           # 좌↔우
    return rec + list(slot), slot, cw, lr


def pick_slot(sc, sp, G, gks, ps, beta):
    """학습된 15칸 확률로 배정. sp[p] = (15,) 칸 확률.

    cost = -점수 + beta * (-log P(칸))
    분해 추정(단 확률 × 레인 전이) 대신 칸을 통째로 배운 확률을 쓴다
    (표본외 칸 정확도 74.0% vs 휴리스틱 66.7%).
    """
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
        v = sp.get(p)
        for j, (k, l) in enumerate(slots):
            pr = float(v[k * 3 + l]) if v is not None else 1.0 / 15
            cost[i, j] = -sc[p] + beta * (-np.log(max(pr, QFLOOR)))
    ri, ci = linear_sum_assignment(cost)
    slot = {cand[i]: slots[j] for i, j in zip(ri, ci)}
    return rec + list(slot), slot


def build():
    sfx = (os.environ.get("SNAP_HL","60") + "_") + ("all" if not SEASONS else "-".join(str(x) for x in sorted(SEASONS)))
    sfx += "" if TARGET == "xg" else f"_{TARGET}"   # 목표별 캐시 분리
    _df = os.environ.get("DROPFAM", "").replace(",", "")
    sfx += f"_no{_df}" if _df else ""              # 열 계열 제외별 분리
    sfx += "_g" if WITHG else ""                   # 감독 모형 포함 여부
    sfx += "_fixsd" if FIXSD else ""               # 채점 표준화 수정본 캐시 분리
    sfx += "_red" if os.environ.get("REDCTRL", "0") == "1" else ""   # 퇴장(수적 열세) 라벨 통제
    sfx += {"1": "_p90", "2": "_p90b"}.get(os.environ.get("PER90", "0"), "")      # per-90 스냅샷 (1: 경기별 비율 평균, 2: 합의 비율)
    sfx += ("_ssn" + ("" if os.environ.get("TESTSEASONS", "2024,2025") == "2024,2025" else os.environ["TESTSEASONS"].replace(",", ""))) if os.environ.get("FOLDS", "block") == "season" else ""
    _np = os.environ.get("NOPCAFAM", "").replace(",", "")
    sfx += f"_raw{_np}" if _np else ""             # PCA 제외 가족별 캐시 분리 (재사용 오류 방지)
    cache = ROOT / "outputs" / f"gap_soft_base{TAG}_pc{NPC}_{sfx}.pkl"
    if cache.exists():
        C = pd.read_pickle(cache)
        print(f"캐시 사용 — 사례 {len(C):,}")
        return C, C.LPk.iloc[0]

    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from lineup_pipeline import r2w, AL
    from lineup_event_panel import _snapshots, build_event

    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    by, T_, dim, FC = _snapshots(gpos)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    POS = {}
    for pid, gg in pl.groupby("player_id"):
        m = gg["starting_position_name"].mode(); POS[int(pid)] = str(m.iloc[0]) if len(m) else "?"
    stat = pd.read_parquet(ROOT / "outputs" / "player_level.parquet")
    STAT = {int(r.player_id): int(r.level) for r in stat.itertuples(index=False)}
    st = pl[pl.is_starter == 1].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])
    st = st[st.str.len() == 11]
    squad = pl.groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])
    xg = (pd.read_parquet(ROOT / "outputs/team_npxg.parquet")
            .set_index(["game_id", "team_id"]).npxg if TARGET == "npxg" else
          pd.read_parquet(ROOT / "outputs/team_xg.parquet")
            .set_index(["game_id", "team_id"]).xg)

    R, F, BASE, IDX, cols = build_event(TARGET)
    M, B = F["vaepon"], BASE; y = R.y.to_numpy(); WT = R.nlvl.to_numpy(float)
    kx = np.array([i for i, c in enumerate(FC) if c in set(cols)])
    Wf, fold_of = {}, {}
    WH = {"A": {}, "B": {}}; Cf = {}
    # NOPCAFAM="cx": 해당 접두 열(맥락 스냅샷)은 PCA 에서 빼고 표준화 원값으로 릿지에 직접 넣는다
    #   (캠페인 발견: 맥락 21열을 가산으로 주면 딥러닝과 동률 — 20 성분 PCA 안에서는 희석되므로 분리)
    _nopca = tuple(x.strip() + "_" for x in os.environ.get("NOPCAFAM", "").split(",") if x.strip())
    idx_r = np.array([i for i, c in enumerate(cols) if c.startswith(_nopca)], int) if _nopca else np.array([], int)
    idx_p = np.array([i for i in range(len(cols)) if i not in set(idx_r.tolist())], int)
    if len(idx_r): print(f"  PCA 제외 열 {len(idx_r)} ({','.join(_nopca)}) — 원값 직접 투입")
    def _wmap(pc, zsd, coef):
        w = np.zeros(len(cols)); w[idx_p] = (pc.components_.T / zsd) @ coef[:NPC]; w[idx_r] = coef[NPC:]; return w
    if os.environ.get("FOLDS", "block") == "season":                  # 평가 시즌 고정: 2024(학습 <2024) · 2025(학습 <2025) — 그 외 시즌은 채점 안 함
        _SEA = R.season.to_numpy().astype(int); _TS = [int(x) for x in os.environ.get("TESTSEASONS", "2024,2025").split(",")]
        FOLDLIST = [(f, np.sort(np.flatnonzero(_SEA < s_)), np.sort(np.flatnonzero(_SEA == s_))) for f, s_ in enumerate(_TS, 1)]
        print(f"  시즌 폴드: {[(f, int((_SEA < s_).sum()), int((_SEA == s_).sum())) for (f, _, _), s_ in zip(FOLDLIST, _TS)]}")
    else:
        FOLDLIST = [(f, np.concatenate(IDX[:f]), IDX[f]) for f in range(1, 5)]
    for f, tr, te_ in FOLDLIST:
        cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
        pc = PCA(n_components=NPC, random_state=0).fit(M[i1][:, idx_p])
        Z = pc.transform(M[:, idx_p]); zmu, zsd = Z[i1].mean(0), Z[i1].std(0).clip(1e-9)
        X = np.hstack([B, (Z - zmu) / zsd] + ([M[:, idx_r]] if len(idx_r) else []))
        ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(X[i1], y[i1], sample_weight=WT[i1]).predict(X[i2]), WT[i2]))
        mm = Ridge(alpha=ba).fit(X[tr], y[tr], sample_weight=WT[tr])
        Wf[f] = _wmap(pc, zsd, mm.coef_[B.shape[1]:])
        if WITHG:                       # 정직이득용 반분 — A 로 고르고 B 로 채점
            h = len(tr) // 2
            for tag, sub in (("A", tr[:h]), ("B", tr[h:])):
                m2 = Ridge(alpha=ba).fit(X[sub], y[sub], sample_weight=WT[sub])
                WH[tag][f] = _wmap(pc, zsd, m2.coef_[B.shape[1]:])
        for i in te_: fold_of[int(R.game_id.iloc[i])] = f

    if FIXSD:
        # 적합 공간(표준화 열)과 채점 공간(원시 스냅샷)을 일치시키고,
        # 폴드별로 점수 SD 를 종전 스케일에 맞춰 β·βf 의미를 보존한다.
        sd_raw = F["raw"].std(0)
        V = np.stack([e[1][kx] for lst in by.values() for e in lst[::3]])
        for dd in (Wf, WH["A"], WH["B"]):
            for k_ in list(dd):
                w0 = dd[k_]
                K_ = (V @ w0).std() / max((V @ (w0 / sd_raw)).std(), 1e-12)
                dd[k_] = w0 / sd_raw * K_

    def snap(p, tt_):
        a = T_.get(int(p))
        if a is None: return None
        j = np.searchsorted(a, tt_, "right") - 1
        return by[int(p)][j][1][kx] if j >= 0 else None

    if WITHG:                       # 감독 모형 — 같은 피처로 '선발 여부' 예측 (선형 유지)
        from sklearn.linear_model import LogisticRegression
        tof = {int(r.game_id): gpos[int(r.game_id)] for r in g.itertuples()
               if int(r.game_id) in gpos}
        gt = {}
        for r in g.itertuples():
            gt.setdefault(int(r.game_id), []).extend(
                [int(r.home_team_id), int(r.away_team_id)])
        for f in range(1, 5):
            gids = sorted({int(R.game_id.iloc[i]) for i in np.concatenate(IDX[:f])})
            Xs, ys = [], []
            for gid in gids:
                tt_ = tof.get(gid)
                if tt_ is None: continue
                for tid in gt.get(gid, []):
                    if (gid, tid) not in st.index or (gid, tid) not in squad.index: continue
                    starters = {int(x) for x in st.loc[(gid, tid)]}
                    for p in squad.loc[(gid, tid)]:
                        v = snap(int(p), tt_)
                        if v is None: continue
                        Xs.append(v); ys.append(1 if int(p) in starters else 0)
            Xs = np.asarray(Xs, np.float64); ys = np.asarray(ys)
            pcc = PCA(n_components=NPC, random_state=0).fit(Xs)
            Zc = pcc.transform(Xs); cmu, csd = Zc.mean(0), Zc.std(0).clip(1e-9)
            lr = LogisticRegression(max_iter=3000).fit((Zc - cmu) / csd, ys)
            Cf[f] = (pcc.components_.T / csd) @ lr.coef_[0]
            print(f"  감독 모형 폴드 {f}: 표본 {len(ys):,} · 선발률 {ys.mean():.3f}")

    CELL, PER, HIST = declared_grid()
    PLANE = past_lanes(HIST, gpos)
    LP = lane_logp(PER, PLANE)
    print("과거레인→그날레인 확률:")
    for i, nm in enumerate("좌중우"):
        print("  " + nm + " → " + "  ".join(f"{100*np.exp(LP[i,j]):5.1f}%" for j in range(3)))

    def lane_fb(p):
        q = POS.get(int(p), "?")
        return 0 if q in LEFTPOS else (2 if q in RIGHTPOS else 1)

    prev_xi, lastxi = {}, {}
    for gid in sorted(set(k[0] for k in CELL), key=lambda z: gpos.get(z, 1 << 30)):
        for (gg, tt) in [k for k in CELL if k[0] == gid]:
            if (gg, tt) in st.index:
                prev_xi[(gg, tt)] = lastxi.get(tt); lastxi[tt] = list(st.loc[(gg, tt)])

    rows = []
    for r in g.itertuples():
        gid = int(r.game_id)
        if gid not in fold_of: continue
        if SEASONS and int(r.season) not in SEASONS: continue
        w = Wf[fold_of[gid]]; tt_ = gpos[gid]
        for tid, oid in ((int(r.home_team_id), int(r.away_team_id)),
                         (int(r.away_team_id), int(r.home_team_id))):
            if (gid, tid) not in st.index or (gid, tid) not in CELL: continue
            pool = [int(p) for p in squad.loc[(gid, tid)]]
            pv = prev_xi.get((gid, tid)) or []
            sc, csc, scA, scB = {}, {}, {}, {}
            for p in list(pool) + list(pv):
                v = snap(p, tt_)
                if v is None: continue
                pi = int(p)
                sc.setdefault(pi, float(w @ v))
                if WITHG:
                    csc.setdefault(pi, float(Cf[fold_of[gid]] @ v))
                    scA.setdefault(pi, float(WH["A"][fold_of[gid]] @ v))
                    scB.setdefault(pi, float(WH["B"][fold_of[gid]] @ v))
            coach = list(st.loc[(gid, tid)])
            if len(sc) < 12 or not set(coach) <= set(sc): continue
            yv = xg.get((gid, tid), np.nan) - xg.get((gid, oid), np.nan)
            if not np.isfinite(yv): continue
            ps = set(pool)
            gks = [p for p in sc if p in ps and POS.get(p) == "GK"]
            qmap, lane = {}, {}
            for p in sc:
                if p in gks: continue
                qq = dyn_q(gid, p)
                if qq is None:
                    qq = np.full(5, .05); qq[STAT.get(int(p), 2)] = .8
                qmap[p] = np.asarray(qq, float)
                l = PLANE.get((gid, int(p)))
                lane[p] = int(l) if l is not None else lane_fb(p)
            # 감독 XI 의 실제 교차 여부 (보정 기준)
            ccross = sum(1 for p in coach if (gid, tid, p) in PER
                         and lane.get(p, 1) != 1 and PER[(gid, tid, p)][1] != 1
                         and PER[(gid, tid, p)][1] != lane.get(p, 1))
            cwide = sum(1 for p in coach if (gid, tid, p) in PER
                        and lane.get(p, 1) != 1 and PER[(gid, tid, p)][1] != 1)
            rows.append(dict(gid=gid, tid=tid, oid=oid, season=int(r.season),
                             date=str(r.game_date.date()), y=yv, sc=sc, qmap=qmap,
                             lane=lane, G=CELL[(gid, tid)], gks=tuple(gks), pool=tuple(pool),
                             coach_xi=coach, coach_sc=sum(sc[p] for p in coach),
                             prev=prev_xi.get((gid, tid)), LPk=LP,
                             ccross=ccross, cwide=cwide,
                             csc=csc, scA=scA, scB=scB))
    C = pd.DataFrame(rows)
    C.to_pickle(cache)
    print(f"\n사례 {len(C):,}   감독 XI 실측 좌우교차 {C.ccross.sum()}/{C.cwide.sum()} "
          f"= {100*C.ccross.sum()/max(C.cwide.sum(),1):.1f}%")
    return C, LP


def evaluate(C, LP, beta):
    rec, cross, wide = [], 0, 0
    for r in C.itertuples(index=False):
        R_, slot, cr = pick_soft(r.sc, r.qmap, r.lane, r.G, list(r.gks), set(r.pool), LP, beta)
        rec.append(R_)
        cross += cr
        wide += sum(1 for p, (k, l) in slot.items() if r.lane.get(p, 1) != 1 and l != 1)
    return rec, (100 * cross / max(wide, 1))


def main():
    C, LP = build()
    C["gap_placebo"] = [sum(r.sc[p] for p in r.prev) - r.coach_sc
                        if r.prev and set(r.prev) <= set(r.sc) else np.nan
                        for r in C.itertuples(index=False)]
    target = 100 * C.ccross.sum() / max(C.cwide.sum(), 1)
    print(f"\n=== BETA 보정 (목표 좌우교차율 {target:.1f}%) ===")
    grid = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    res = {}
    for b in grid:
        rec, cr = evaluate(C, LP, b)
        hit = float(np.mean([len(set(R_) & set(r.coach_xi))
                             for R_, r in zip(rec, C.itertuples(index=False))]))
        C["tmpr"] = rec
        gp = np.array([sum(r.sc[p] for p in r.tmpr) - r.coach_sc for r in C.itertuples(index=False)])
        C["tmpg"] = gp; sd = gp.std(); bb, lo, hi, n = fe_ols(C, "tmpg")
        res[b] = (rec, cr, hit, bb * sd, lo * sd, hi * sd)
        print(f"  BETA {b:5.3f}   좌우교차 {cr:5.1f}%   적중 {hit:.3f}/11   "
              f"격차효과 {bb*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}]")
    # 교차율이 감독 실측 이하인 것 중 적중 최대
    ok = [b for b in grid if res[b][1] <= target]
    best = max(ok, key=lambda b: res[b][2]) if ok else grid[-1]
    print(f"  → 채택 BETA = {best}  (교차율 ≤ {target:.1f}% 중 적중 최대)")
    C.drop(columns=["tmpr", "tmpg"], errors="ignore", inplace=True)
    C["rec_s"] = res[best][0]
    C["gap_s"] = [sum(r.sc[p] for p in r.rec_s) - r.coach_sc for r in C.itertuples(index=False)]
    C["hit_s"] = [len(set(r.rec_s) & set(r.coach_xi)) for r in C.itertuples(index=False)]
    print(f"\n=== 격차 → xG 차 (BETA={best}) ===")
    for nm, col in [("연성 제약", "gap_s"), ("위약 직전 XI", "gap_placebo")]:
        sd = C[col].std(); bb, lo, hi, n = fe_ols(C, col)
        print(f"  {nm:14s} {bb*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}] n={n:4d} "
              f"{'*' if (lo>0)==(hi>0) else ' '}")
    sd = C.gap_s.std(); bb, lo, hi, n = fe_ols(C, "gap_s", extra=["coach_sc"])
    print(f"  {'감독점수 통제':14s} {bb*sd:+.4f} [{lo*sd:+.4f},{hi*sd:+.4f}] "
          f"{'*' if (lo>0)==(hi>0) else ' '}")
    outs = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed); gg2 = []
        for r in C.itertuples(index=False):
            ks = list(r.sc); vs = [r.sc[k] for k in ks]; rng.shuffle(vs); s2 = dict(zip(ks, vs))
            R_, _, _ = pick_soft(s2, r.qmap, r.lane, r.G, list(r.gks), set(r.pool), LP, best)
            gg2.append(sum(s2[p] for p in R_) - sum(s2[p] for p in r.coach_xi))
        C["gap_sh"] = gg2
        s2d = C.gap_sh.std(); b2, _, _, _ = fe_ols(C, "gap_sh"); outs.append(b2 * s2d)
    mu, sg = float(np.mean(outs)), float(np.std(outs))
    print(f"  셔플 위약 {mu:+.4f} ± {sg:.4f}   →  z = {(bb*sd-mu)/max(sg,1e-9):+.2f}")
    _dfo = os.environ.get("DROPFAM", "").replace(",", "")
    out = ROOT / "outputs" / (f"gap_soft_full{TAG}_pc{NPC}_"
          f"{('all' if not SEASONS else '-'.join(str(x) for x in sorted(SEASONS)))}"
          f"{'' if TARGET == 'xg' else '_' + TARGET}{'_no' + _dfo if _dfo else ''}"
          f"{'_fixsd' if FIXSD else ''}{'_red' if os.environ.get('REDCTRL', '0') == '1' else ''}{ {'1': '_p90', '2': '_p90b'}.get(os.environ.get('PER90', '0'), '') }{('_raw' + os.environ.get('NOPCAFAM', '').replace(',', '')) if os.environ.get('NOPCAFAM') else ''}{('_ssn' + ('' if os.environ.get('TESTSEASONS', '2024,2025') == '2024,2025' else os.environ['TESTSEASONS'].replace(',', ''))) if os.environ.get('FOLDS', 'block') == 'season' else ''}.pkl")
    C.drop(columns=["gap_sh"], errors="ignore").to_pickle(out)
    print(f"\n저장 {out.name}")


if __name__ == "__main__":
    main()
