"""라인업 추천 파이프라인 — EPV 를 스위치로 둔 절제(ablation) 구조.

목표변수 (동점 프레임만 집계)
  비EPV   tilt(영역지배) · ballx(볼 전진) · poss(점유) · xg · shots
  EPV     apv(프레임 점유가치)
선수 피처
  비EPV   pos  (점유상태×24존 + 속도·볼거리, 78차원)
          pos2 (국면×24존 + 국면별 상대높이, 170차원)
  EPV     epv  (액션 EPV 존·이벤트별 480차원)

같은 구간·같은 검증 규약을 공유하므로, 목표와 피처 양쪽에서 EPV 를 빼고 넣어
그 효과만 분리할 수 있다.

검증: 전방 연쇄 4폴드(폴드는 **경기** 단위), 블록가중치×알파는 내부 분할에서 선택,
      귀무는 XI 피처 행 셔플 — 설정 선택까지 귀무에 포함한다.
"""
import numpy as np, pandas as pd, argparse
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# APV(λ) = 창출 − λ·억제. λ 로 위험 선호를 조절한다.
#   λ=0 창출만(공격적) · λ=1 대칭(기존 apv) · λ=2 억제 강조(수비적)
_PARTS = None
def _apv_lam(d, lam):
    global _PARTS
    if _PARTS is None:
        _PARTS = pd.read_parquet("outputs/segment_apv_parts.parquet").set_index("seg_key")
    mk = d.seg_key.map(_PARTS.make_a); st = d.seg_key.map(_PARTS.stop_a)
    return (mk - lam * st).to_numpy()

def _mix(d, w):
    """z(APV) + w·z(xG). 표준화는 전체 표본 기준(목표 정의의 일부이므로 누수 아님)."""
    a = _apv_lam(d, 1.0)
    x = ((d.xg_a - d.xg_b) / d.dur * 90).to_numpy()
    za = (a - np.nanmean(a)) / np.nanstd(a)
    zx = (x - np.nanmean(x)) / np.nanstd(x)
    return za + w * zx

TARGETS = {
    "apv":   ("EPV",   lambda d: d.apv_a.to_numpy()),
    "tilt":  ("비EPV", lambda d: (d.tilt_a - d.tilt_b).to_numpy()),
    "ballx": ("비EPV", lambda d: d.ballx_a.to_numpy()),
    "poss":  ("비EPV", lambda d: (d.poss_a - d.poss_b).to_numpy()),
    "xg":    ("비EPV", lambda d: ((d.xg_a - d.xg_b) / d.dur * 90).to_numpy()),
    "shots": ("비EPV", lambda d: ((d.sh_a - d.sh_b) / d.dur * 90).to_numpy()),
    "vaep":  ("모델",  lambda d: ((d.vaep_a - d.vaep_b) / d.dur * 90).to_numpy()),
    "vaepns":("모델",  lambda d: ((d.vns_a - d.vns_b) / d.dur * 90).to_numpy()),
    **{f"apv{int(l*10):02d}": ("EPV", (lambda L: (lambda d: _apv_lam(d, L)))(l))
       for l in (0.0, 0.5, 1.0, 1.5, 2.0)},
    # 복합 목표: z(APV) + w·z(xG). w 는 **실제 득실차 예측을 최대화**하도록 고른 값(1.70).
    # 근거: 두 지표는 다른 것을 잰다(상관 0.30, 각자의 잔차가 득실차를 설명) →
    #       신뢰도 역가중은 전제가 깨지므로 외적 기준으로 정한다.
    **{f"mix{int(w*100):03d}": ("복합", (lambda W: (lambda d: _mix(d, W)))(w))
       for w in (0.0, 0.5, 1.0, 1.7, 2.5)},
}
# ── 피처 두 부류를 구분한다. 이 구분이 해석을 좌우한다.
#   사전 결정  선수 피처(pos2/vaepon/space) — 전부 과거-only, 구간 시작 전에 정해진다
#   동시 관측  배치 피처(arr/arrdev) — 그 구간에서 관측된다. 결과와 동시대라
#             역인과가 섞인다. 실제로 시간을 가르면(앞절반 배치→뒤절반 결과)
#             R² 기여가 0 이 된다. 최종 표에서 **반드시 분리 표기**할 것.
SEGF = {                      # 구간-팀 수준 피처 — 선수 이력이 아니라 그 구간의 배치
    "arr":    ("비EPV", ["cx", "cy", "sx", "sy", "width", "depth", "compact",
                        "h_GK", "h_DF", "h_MF", "h_FW", "w_DF", "w_MF", "w_FW"]),
    "arrdev": ("비EPV", ["dx_all", "dy_all", "dsx", "dx_DF", "dy_DF",
                        "dx_MF", "dy_MF", "dx_FW", "dy_FW", "spr", "v"]),
}
FEATS = {
    "pos":  ("비EPV", "outputs/frame_player_match.parquet", ("game_id", "pid", "team_id", "nfr")),
    "pos2": ("비EPV", "outputs/frame_player_match2.parquet", ("game_id", "pid", "team_id", "nfr")),
    "epv":  ("EPV",   "outputs/epv_player_match_zone.parquet", ("game_id", "pid")),
    "onball": ("EPV", "outputs/onball_player_type.parquet", ("game_id", "pid")),
    "space": ("비EPV", "outputs/frame_player_space", ("game_id", "pid", "n")),
    "vaepon": ("모델", "outputs/onball_vaep_player.parquet", ("game_id", "pid")),
    "epvon":  ("EPV",  "outputs/onball_epv_player.parquet", ("game_id", "pid")),
    "epvoff": ("EPV",  "outputs/offball_epv_player.parquet", ("game_id", "pid", "team_id", "n")),
    # GK 슛 스토핑 — 유효슈팅 조건부 기대실점 대비 실제 실점.
    # 유효슈팅 판정은 SPADL 의 키퍼 액션(세이브·캐치·펀칭)이 슛 직후 오는지로.
    # 사전 xG 를 그대로 쓰면 척도가 안 맞아(기대 0.19 vs 실점 0.45) 유효슈팅만으로
    # 위치·바디파트에 재적합했다. 반분 신뢰도 SB +0.41 (사전 xG 판 +0.26).
    "gkstop": ("비EPV", "outputs/gk_psxg.parquet", ("game_id", "pid")),
}
AL = [3, 10, 30, 100, 300, 1000, 3000, 10000]
WG = [0., .05, .1, .2, .4, .8, 1.5]


def load_snapshots(path, drop, gpos, hist=1.0, hl=60):
    """선수-경기 표 → 과거-only 지수감쇠 스냅샷 (pid → 시각배열, 값배열)."""
    P = pd.read_parquet(path)
    FC = [c for c in P.columns if c not in drop]
    P = P[P.game_id.isin(gpos)].copy(); P["t"] = P.game_id.map(gpos)
    P = P.sort_values(["pid", "t"]).reset_index(drop=True)
    X = P[FC].to_numpy(np.float32); lam = np.log(2) / hl
    by = {}; cur = None; a = None; w = 0.; last = None
    for i, (pid_, t_) in enumerate(zip(P.pid.to_numpy(), P.t.to_numpy())):
        if pid_ != cur: cur = pid_; a = np.zeros(X.shape[1], np.float32); w = 0.; last = None
        if last is not None:
            dd = np.exp(-lam * (t_ - last)); a *= dd; w *= dd
        if w >= hist: by.setdefault(int(pid_), []).append((int(t_), a / w))
        a = a + X[i]; w += 1.; last = t_
    T = {p: np.array([z[0] for z in v]) for p, v in by.items()}
    return by, T, len(FC)


def build(target, feats, mindur=3.0, minpl=8, hist=1.0, team_dummies=False):
    """team_dummies: 기본 False.

    팀 정체성은 곧 그 선수 구성이라, 더미로 통제하면 라인업 효과가 정의상 0 이 된다
    (사다리 실측: 팀상태 통제 시 기여 +0.1065 → 팀더미 추가 시 +0.0006).
    게다가 팀더미 60여 개는 표본외에서 과적합이라 기저 R² 자체가 낮아진다
    (+0.1159 → +0.0556). 통제로도 나쁘고 재려는 것도 죽인다.
    대신 **팀 상태**(순위·승점·득실·폼·휴식)로 전력을 통제한다 — 전력은 담되
    정체성은 아니다."""
    D = pd.read_parquet("outputs/segment_targets.parquet")
    gm = pd.read_csv("vaep/output/games.csv"); gm["game_date"] = pd.to_datetime(gm.game_date)
    gm = gm.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in gm.iterrows()}
    sea = {int(r.game_id): int(r.season) for r in gm.itertuples()}
    home = {int(r.game_id): float(r.home_team_id) for r in gm.itertuples()}
    D = D[D.game_id.isin(gpos) & (D.dur >= mindur)].copy()
    ylab, fn = TARGETS[target]
    D["yv"] = fn(D)
    D = D[np.isfinite(D.yv)]
    segk = [k for k in feats if k in SEGF]
    plk = [k for k in feats if k not in SEGF]
    SRC = {k: load_snapshots(FEATS[k][1], FEATS[k][2], gpos, hist) for k in plk}
    AR = None
    if segk:
        AR = pd.read_parquet("outputs/segment_arrangement.parquet")
        AR = AR.set_index(["seg_key", "team_id"])
    TS = pd.read_parquet("outputs/team_state.parquet").set_index(["game_id", "team_id"])
    SFE = ["rank_pct", "ppg", "gd", "form5", "rest"]

    def fx(k, ids, tt):
        by, T, dim = SRC[k]
        v = np.zeros(dim, np.float32); c = 0
        for p in ids:
            arr = T.get(p)
            if arr is None: continue
            j = np.searchsorted(arr, tt, "right") - 1
            if j < 0: continue
            v += by[p][j][1]; c += 1
        return v, c

    rows, FT = [], {k: [] for k in feats}
    for r in D.itertuples():
        gid_ = int(r.game_id); tt = gpos[gid_]
        xa = [int(x) for x in r.xi_a.split(",")]; xb = [int(x) for x in r.xi_b.split(",")]
        got = {}
        ok = True
        if segk:
            try:
                aa = AR.loc[(r.seg_key, float(r.team_a))]
                ab = AR.loc[(r.seg_key, float(r.team_b))]
            except KeyError:
                continue
            for k in segk:
                va = aa[SEGF[k][1]].to_numpy(float); vb = ab[SEGF[k][1]].to_numpy(float)
                if not (np.isfinite(va).all() and np.isfinite(vb).all()): ok = False; break
                got[k] = (va, vb)
            if not ok: continue
        for k in plk:
            va, ca = fx(k, xa, tt); vb, cb = fx(k, xb, tt)
            if min(ca, cb) < minpl: ok = False; break
            got[k] = (va, vb)
        if not ok: continue
        for me, opp, sgn in ((r.team_a, r.team_b, +1.), (r.team_b, r.team_a, -1.)):
            st = {}
            for c in SFE:
                try: st[c] = float(TS.loc[(gid_, int(me)), c]) - float(TS.loc[(gid_, int(opp)), c])
                except KeyError: st[c] = np.nan
            rows.append({"game_id": gid_, "t": tt, "season": sea[gid_], "me": int(me), "opp": int(opp),
                         "home": 1. if float(me) == home.get(gid_, -1) else 0., "dur": r.dur,
                         "nlvl": r.nlvl, "period": r.period, "t0": r.t0 / 60, "y": sgn * r.yv,
                         "seg_key": r.seg_key, **{f"s_{c}": st[c] for c in SFE}})
            for k in feats:
                va, vb = got[k]
                FT[k].append((va - vb) if sgn > 0 else (vb - va))
    R = pd.DataFrame(rows)
    def prep(M):
        M = np.vstack(M).astype(np.float64)
        keep = M.std(0) > 1e-9; M = M[:, keep]
        mu, sd = M.mean(0), M.std(0)
        return (M - mu) / sd, (keep, mu, sd)
    F, PRE = {}, {}
    for k, v in FT.items():
        F[k], PRE[k] = prep(v)
    SEA = pd.get_dummies(R.season.astype(str)).to_numpy(float)
    ST = R[[f"s_{c}" for c in SFE]].to_numpy(float)
    ST = np.nan_to_num((ST - np.nanmean(ST, 0)) / np.nanstd(ST, 0))
    CTX = np.column_stack([R.home, np.log(R.dur), R.t0 / 45, R.period == 2])
    parts = [CTX, SEA, ST]
    if team_dummies:
        parts.append(pd.get_dummies(R.me.astype(str), prefix="t").join(
            pd.get_dummies(R.opp.astype(str), prefix="o")).to_numpy(float))
    BASE = np.hstack(parts)
    BASE = (BASE - BASE.mean(0)) / np.where(BASE.std(0) == 0, 1, BASE.std(0))
    gord = R.groupby("game_id").t.first().sort_values().index.to_numpy()
    IDX = [np.flatnonzero(R.game_id.isin(set(x)).to_numpy()) for x in np.array_split(gord, 5)]
    return R, F, BASE, IDX, SRC, PRE


def baseA(BASE, R):
    """**기저 A** — 상황(홈·길이·시각·피리어드) + 시즌 더미만 남긴다.

    `build` 가 만드는 기저에는 팀 전력 스칼라 차이(순위·승점·득실·최근5·휴식) 5열이
    들어 있다. 그런데 승점·득실·최근폼은 **선수 구성의 누적 결과**라, 통제하면
    라인업이 설명할 몫을 스스로 깎는다 — 모델 기여가 +0.094 → +0.042 로 반토막 나고
    시즌 막바지 폴드에서는 음수가 된다(그 구간에서 전력 지표가 가장 잘 맞으므로).
    (팀 정체성 더미는 `team_dummies=False` 가 기본이라 애초에 없다.)

    A 를 주 기저로, 전력 스칼라를 넣은 것을 기저 B(보수 대조)로 병기한다.
    """
    a = 4 + R.season.nunique()          # CTX 4열 + 시즌 더미
    return np.hstack([BASE[:, :a], BASE[:, a + 5:]])


def r2w(a, p, w):
    mu = np.average(a, weights=w); return 1 - (w * (a - p) ** 2).sum() / (w * (a - mu) ** 2).sum()


def cv(R, F, BASE, IDX, keys, nc, minlen, perm=None, ret_model=False):
    y = R.y.to_numpy(); WT = R.nlvl.to_numpy(float)
    sel = R.dur.to_numpy() >= minlen
    rs, last = [], None
    for f in range(1, 5):
        tr = np.concatenate(IDX[:f]); te = IDX[f]
        tr = tr[sel[tr]]; te = te[sel[te]]
        if len(tr) < 60 or len(te) < 20: return (np.nan, None) if ret_model else np.nan
        cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
        bl, meta = [BASE], []
        for k in keys:
            M = F[k] if perm is None else F[k][perm]
            if nc:
                pc = PCA(n_components=min(nc, len(i1) - 1, M.shape[1]), random_state=0).fit(M[i1])
                Z = pc.transform(M); mu, sd = Z[i1].mean(0), Z[i1].std(0).clip(1e-9)
                bl.append((Z - mu) / sd); meta.append((k, pc, mu, sd))
            else:
                bl.append(np.column_stack([M.sum(1)])); meta.append((k, None, None, None))
        nb = len(bl) - 1; ws = [0.] * nb
        def Z_(w_):
            u = [b * ww for b, ww in zip(bl[1:], w_) if ww > 0]
            return np.hstack([bl[0]] + u) if u else bl[0]
        def sc(w_, a_):
            z = Z_(w_); return r2w(y[i2], Ridge(alpha=a_).fit(z[i1], y[i1], sample_weight=WT[i1]).predict(z[i2]), WT[i2])
        cur, ba = max((sc(ws, a), a) for a in AL)
        for _ in range(2):
            for b in range(nb):
                for ww in WG:
                    w2 = ws.copy(); w2[b] = ww
                    for a_ in AL:
                        s = sc(w2, a_)
                        if s > cur: cur, ws, ba = s, w2.copy(), a_
        z = Z_(ws)
        m = Ridge(alpha=ba).fit(z[tr], y[tr], sample_weight=WT[tr])
        rs.append(r2w(y[te], m.predict(z[te]), WT[te]))
        last = (m, meta, ws, ba, tr, te)
    return (np.mean(rs), last) if ret_model else np.mean(rs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="tilt,ballx,poss,xg")
    ap.add_argument("--featsets", default="pos|pos2|pos,epv|epv")
    ap.add_argument("--pcas", default="5,10,20")
    ap.add_argument("--mindur", type=float, default=15.0)
    ap.add_argument("--nperm", type=int, default=6)
    a = ap.parse_args()
    PC = [int(x) for x in a.pcas.split(",")]
    FS = [tuple(s.split(",")) for s in a.featsets.split("|")]
    print(f"{'목표':<10}{'계열':<7}{'n':>6}{'기저':>9}   " + "  ".join(f"{'+'.join(fs):>12}" for fs in FS) + "   최고기여 / 귀무")
    for tg in a.targets.split(","):
        lab, _ = TARGETS[tg]
        need = sorted({k for fs in FS for k in fs})
        R, F, BASE, IDX, SRC, PRE = build(tg, need, mindur=a.mindur)
        n = int((R.dur >= a.mindur).sum())
        b0 = cv(R, F, BASE, IDX, [], 0, a.mindur)
        cells, best = [], (-9, None, None)
        for fs in FS:
            v = max((cv(R, F, BASE, IDX, list(fs), nc, a.mindur), nc) for nc in PC)
            cells.append(v[0] - b0)
            if v[0] - b0 > best[0]: best = (v[0] - b0, fs, v[1])
        rng = np.random.default_rng(1); nl = []
        for _ in range(a.nperm):
            pm = rng.permutation(len(R))
            nl.append(max(cv(R, F, BASE, IDX, list(fs), nc, a.mindur, perm=pm) - b0
                          for fs in FS for nc in PC))
        nl = np.array(nl)
        z = (best[0] - nl.mean()) / max(nl.std(), 1e-9)
        print(f"{tg:<10}{lab:<7}{n:>6}{b0:>+9.4f}   " + "  ".join(f"{c:>+12.4f}" for c in cells) +
              f"   {best[0]:+.4f} ({'+'.join(best[1])},PCA{best[2]}) | 귀무 {nl.mean():+.4f}±{nl.std():.4f} z={z:+.1f}")
