"""우리 점수 모델(확장 채널 CNN + 이력 Transformer, 시즌 폴드)의 선수 벡터 위에 두 예측 헤드를 얹는다.

  ① 선발 XI 예측 헤드: 선수 벡터 v_p(16) + 가산 점수 + 로스터 사용 피처(BASE_F 32) → 선발 확률 (MLP, BCE)
       → 공시 격자 제약 하 11명 선택(pick_soft), 기존 평가와 동일 지표(적중/11, 팀-경기 단위)
       비교: v2 로지스틱(32피처) · v3 부스팅(전 피처) · 부스팅+선수벡터(하이브리드) · 과거 평균 출전분
  ② 포메이션 예측 헤드: 로스터 선수 벡터 풀링(평균·최대) + 모양 이력(S) + 성적(R) → 병합 모양 다중분류 (MLP)
       비교: 지속성 · 다항 LR(S+R). 지표 = 정확일치.
규약: 시즌 폴드 — 2025 (학습 ≤2024, 인코더 fold1) · 2026 (학습 ≤2025, 인코더 fold2). 인코더는 고정(특징 추출기).
실행: TAG=ssn20252026-set0-cross0-H16-L1-none-full-tf1-s5 SCPKL=<점수 pkl> python -m experiments.multitask_heads
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
import formation_predict4 as FP                      # noqa: E402  (모듈 import 시 ONBALL_FILE 을 덮어쓰므로 아래서 되돌림)
os.environ["ONBALL_FILE"] = "onball_gk_resid_merged_defresp_ref.parquet"
TAG = os.environ.get("TAG", "ssn20252026-set0-cross0-H16-L1-none-full-tf1-s5")
os.environ["SET"] = "1" if TAG.split("-")[1].startswith("set1") or "-set1" in TAG else "0"; os.environ["CROSS"] = "0"
os.environ["H"] = TAG.split("-H")[1].split("-")[0]; os.environ["LAYERS"] = TAG.split("-L")[1].split("-")[0]; os.environ["LANE"] = "none"
os.environ["FULLCH"] = "1" if "-full" in TAG else "0"; os.environ["TOKCOV"] = "1" if "-cov" in TAG else "0"
import re as _re; _m = _re.search(r"-tf(\d+)", TAG); os.environ["MATCHTF"] = _m.group(1) if _m else "0"; os.environ["SLOTEMB"] = "1" if "-slot" in TAG else "0"; os.environ["DTDAYS"] = "1" if "-days" in TAG else "0"
import torch, torch.nn as nn                          # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from player_encoder_set import SetNet, load_structured   # noqa: E402
from player_encoder import pack, gather, K, DEV      # noqa: E402
from gap_soft import pick_soft                       # noqa: E402
from xi_predict3 import BASE_F                       # noqa: E402
from minutes_control import hist_minutes             # noqa: E402
from gap_lane import declared_grid                   # noqa: E402
from config import VAEP_OUTPUT_DIR                   # noqa: E402

TESTS = [int(x) for x in os.environ.get("TESTSEASONS", "2025,2026").split(",")]
SCPKL = ROOT / os.environ.get("SCPKL", "outputs/gap_soft_fullonball_gk_resid_merged_tf26_ssn26_pc20_all_npxg_nozvzn_fixsd.pkl")
BETA = 0.04; NS = 5


def embed_all(D, g, gpos):
    """opp_panel 의 (game, player) 행마다 인코더 선수 벡터 v(16)·가산 점수. 시즌 s 행은 s 를 평가 시즌으로 둔 폴드(학습 < s) 인코더로 — s < 첫 평가 시즌이면 fold1."""
    _, _, X, pid, t_arr, gid_arr, bounds, tidx, zidx, sidx = load_structured()
    F = {}
    for f in range(1, 5):
        fp = ROOT / f"outputs/player_encoder_set_{TAG}_fold{f}.pt"
        if fp.exists(): F[f] = torch.load(fp, weights_only=False, map_location="cpu")
    fold_for = {}
    for f, s in enumerate(TESTS, 1): fold_for[s] = f
    def fold_of_season(s): return fold_for.get(int(s), 1 if int(s) < TESTS[0] else max(F))
    V = np.zeros((len(D), int(os.environ["H"])), np.float32); ADD = np.zeros(len(D), np.float32)
    for f in F:
        rows_f = np.flatnonzero([fold_of_season(s) == f for s in D.season.to_numpy()])
        if not len(rows_f): continue
        d = F[f]; Xg = torch.as_tensor(((X - d["mu"]) / d["sd"]).clip(-5, 5).astype(np.float32), device=DEV)
        nets = []
        for sdict in d["nets"]:
            net = SetNet(*d["S"]).to(DEV); net.load_state_dict(sdict); net.eval(); nets.append(net)
        pairs = [(int(D.player_id.iloc[i]), gpos[int(D.game_id.iloc[i])]) for i in rows_f]
        for s0 in range(0, len(pairs), 512):
            chunk = pairs[s0:s0 + 512]; idx_, dt_ = pack(chunk, bounds, t_arr)
            with torch.no_grad():
                Xs, dts, ms = gather(Xg, torch.as_tensor(idx_, device=DEV), torch.as_tensor(dt_, device=DEV))
                outs = [n_.player(Xs, dts, ms) for n_ in nets]
                add = np.mean([o[0].cpu().numpy() for o in outs], 0); v = np.mean([o[1].cpu().numpy() for o in outs], 0)
            has = (idx_ >= 0).any(1); ii = rows_f[s0:s0 + 512]
            V[ii] = np.where(has[:, None], v, 0.0); ADD[ii] = np.where(has, add, 0.0)
        print(f"  임베딩 폴드 {f}: {len(rows_f):,} 행", flush=True)
    return V, ADD


class Head(nn.Module):
    def __init__(self, d, h=32, drop=0.3, out=1):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, out))
    def forward(self, x): return self.net(x)


class FormSetHead(nn.Module):
    """로스터 선수 토큰 [B, P, d_tok] (패딩 마스크) → SAB → PMA → 로스터 요약 ⊕ 모양 이력·성적 [B, d_s] → 모양 클래스."""
    def __init__(self, d_tok, d_s, n_out, h=32, heads=4, drop=0.3):
        super().__init__()
        from player_encoder_set import SAB, PMA
        self.inp = nn.Sequential(nn.Linear(d_tok, h), nn.ReLU(), nn.Dropout(drop)); self.sab = SAB(h, heads, drop); self.pma = PMA(h, heads, drop)
        self.out = nn.Sequential(nn.Linear(h + d_s, h), nn.ReLU(), nn.Dropout(drop), nn.Linear(h, n_out))
    def forward(self, T, pad, S):
        X = self.inp(T); X = self.sab(X, pad); z = self.pma(X, pad); return self.out(torch.cat([z, S], -1))


def fit_set_head(T, PAD, S, y, tr, va, n_out, seeds=NS, epochs=200, lr=1e-3, wd=1e-3):
    T_ = torch.as_tensor(T, dtype=torch.float32, device=DEV); P_ = torch.as_tensor(PAD, dtype=torch.bool, device=DEV); S_ = torch.as_tensor(S, dtype=torch.float32, device=DEV); y_ = torch.as_tensor(y, dtype=torch.long, device=DEV)
    tr_t = torch.as_tensor(tr, device=DEV); va_t = torch.as_tensor(va, device=DEV); lossf = nn.CrossEntropyLoss(); nets = []
    for sd in range(seeds):
        torch.manual_seed(sd); net = FormSetHead(T.shape[2], S.shape[1], n_out).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, state, pat = 1e9, None, 0
        for ep in range(epochs):
            net.train(); perm = tr_t[torch.randperm(len(tr_t), device=DEV)]
            for s0 in range(0, len(perm), 128):
                b = perm[s0:s0 + 128]; opt.zero_grad(); loss = lossf(net(T_[b], P_[b], S_[b]), y_[b]); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            with torch.no_grad(): v = lossf(net(T_[va_t], P_[va_t], S_[va_t]), y_[va_t]).item()
            if np.isfinite(v) and v < best - 1e-5: best, state, pat = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 15: break
        if state is not None: net.load_state_dict(state)
        net.eval(); nets.append(net)
    def pred(ix):
        ix = torch.as_tensor(ix, device=DEV)
        with torch.no_grad(): return np.mean([torch.softmax(n_(T_[ix], P_[ix], S_[ix]), -1).cpu().numpy() for n_ in nets], 0)
    return pred


def fit_head(Xtr, ytr, Xva, yva, out=1, seeds=NS, epochs=200, lr=1e-3, wd=1e-3):
    """MLP 헤드 (BCE 또는 CE), 검증 조기 종료, 시드 평균 예측 함수 반환."""
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV); ytr_t = torch.as_tensor(ytr, dtype=torch.float32 if out == 1 else torch.long, device=DEV)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=DEV); yva_t = torch.as_tensor(yva, dtype=torch.float32 if out == 1 else torch.long, device=DEV)
    lossf = nn.BCEWithLogitsLoss() if out == 1 else nn.CrossEntropyLoss()
    nets = []
    for sd in range(seeds):
        torch.manual_seed(sd); net = Head(Xtr.shape[1], out=out).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, state, pat = 1e9, None, 0
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Xtr_t), device=DEV)
            for s in range(0, len(perm), 256):
                b = perm[s:s + 256]; opt.zero_grad(); o = net(Xtr_t[b]); loss = lossf(o.squeeze(-1) if out == 1 else o, ytr_t[b]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): o = net(Xva_t); v = lossf(o.squeeze(-1) if out == 1 else o, yva_t).item()
            if np.isfinite(v) and v < best - 1e-5: best, state, pat = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 15: break
        if state is not None: net.load_state_dict(state)
        net.eval(); nets.append(net)
    def pred(Xte):
        Xt = torch.as_tensor(Xte, dtype=torch.float32, device=DEV)
        with torch.no_grad():
            if out == 1: return np.mean([torch.sigmoid(n_(Xt).squeeze(-1)).cpu().numpy() for n_ in nets], 0)
            return np.mean([torch.softmax(n_(Xt), -1).cpu().numpy() for n_ in nets], 0)
    return pred


def main():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date); g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    D = pd.read_parquet(ROOT / "outputs" / "opp_panel_v2.parquet"); D = D[D.game_id.isin(gpos)].reset_index(drop=True)
    C = pd.read_pickle(SCPKL); LP = C.LPk.iloc[0]
    SC, CELL, NEED, POOL = {}, {}, {}, {}
    for r in C.itertuples(index=False):
        k = (int(r.gid), int(r.tid)); SC[k] = r.sc; POOL[k] = set(r.pool)
        CELL[k] = {p: (int(np.argmax(r.qmap[p])) if r.qmap.get(p) is not None else 2, int(r.lane.get(p, 1))) for p in r.pool}; NEED[k] = r.G
    D["key"] = list(zip(D.game_id.astype(int), D.team_id.astype(int)))
    print(f"패널 {len(D):,} 행 · 평가 시즌 {TESTS} · 인코더 {TAG}", flush=True)
    V, ADD = embed_all(D, g, gpos)
    for j in range(V.shape[1]): D[f"v{j}"] = V[:, j]
    D["dl_sc"] = ADD
    # ── ① 선발 XI 예측 ────────────────────────────────────────────
    SKIPXI = os.environ.get("SKIPXI", "0") == "1"
    HM = hist_minutes(gpos) if not SKIPXI else None
    _pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv"); _pos = _pl.groupby("player_id").starting_position_name.agg(lambda z: z.mode().iloc[0] if len(z.mode()) else "?")
    ISGK = {int(k): (v == "GK") for k, v in _pos.items()}
    E = D[D.key.isin(set(SC))].reset_index(drop=True) if not SKIPXI else D.iloc[:0]
    sc, hm, lvl, lane, need, ncand = [], [], [], [], [], []
    for r in E.itertuples(index=False):
        k = r.key; p = int(r.player_id); sc.append(SC[k].get(p, np.nan)); v = HM.get((int(r.game_id), p)); hm.append(np.nan if v is None else v)
        c = CELL[k].get(p, (2, 1)); lvl.append(c[0]); lane.append(c[1]); need.append(float(NEED[k][c[0], c[1]])); ncand.append(float(sum(1 for q in POOL[k] if CELL[k].get(q, (2, 1)) == c)))
    E["is_gk"] = [1.0 if ISGK.get(int(p), False) else 0.0 for p in E.player_id]; E["our_sc"] = sc; E["hist_min"] = hm; E["cell_lv"] = lvl; E["cell_ln"] = lane; E["cell_need"] = need; E["cell_n"] = ncand; E["cell_ratio"] = E.cell_need / E.cell_n.clip(lower=1)
    for c, nm in [("our_sc", "r_sc"), ("hist_min", "r_min"), ("st_rate", "r_st")]:
        E[nm] = E.groupby(["key", "cell_lv", "cell_ln"])[c].rank(ascending=False, pct=True); E[nm + "_t"] = E.groupby("key")[c].rank(ascending=False, pct=True)
    F3 = BASE_F + ["our_sc", "hist_min", "cell_need", "cell_n", "cell_ratio", "r_sc", "r_min", "r_st", "r_sc_t", "r_min_t", "r_st_t"]
    VC = [f"v{j}" for j in range(V.shape[1])] + ["dl_sc"]
    E[F3 + VC] = E[F3 + VC].astype(float)
    if os.environ.get("DUMP_E"):                                            # 외부 기준선(DraftRec 등)용 프레임 덤프: 사례 키·선수·피처·선수벡터·정답
        _cols = list(dict.fromkeys(["game_id", "team_id", "player_id", "season", "y", "is_gk"] + F3 + VC)); E2 = E[_cols].copy(); E2.to_parquet(os.environ["DUMP_E"], index=False)
        C[["gid", "tid", "oid", "season", "y", "date"]].to_parquet(os.environ["DUMP_E"].replace(".parquet", "_cases.parquet"), index=False)
        print(f"  프레임 덤프 → {os.environ['DUMP_E']} ({len(E2):,} 행)", flush=True)
    for col in ("p_mlp", "p_v3", "p_hyb", "p_lr"): E[col] = np.nan
    E["p_prev"] = np.nan_to_num(E.st_last.to_numpy(float)) + 1e-3 * np.nan_to_num(E.st_rate.to_numpy(float)); E["p_strate"] = np.nan_to_num(E.st_rate.to_numpy(float))   # 기준선: 직전 XI 지속 · 선발률
    for s in TESTS:
        tr = E[E.season < s]; te = (E.season == s).to_numpy()
        if te.sum() == 0: continue
        cut = int(.85 * len(tr)); Xall = np.nan_to_num(E[BASE_F + VC].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0); mu, sd = Xall[tr.index[:cut]].mean(0), Xall[tr.index[:cut]].std(0).clip(1e-6); Xz = np.clip((Xall - mu) / sd, -10, 10)
        pred = fit_head(Xz[tr.index[:cut]], tr.y.to_numpy()[:cut], Xz[tr.index[cut:]], tr.y.to_numpy()[cut:])
        E.loc[te, "p_mlp"] = pred(Xz[te])
        # GBSEEDS>1: 부스팅도 시드를 흔들어 평균 — 드래프트 정책(시드 평균)과 같은 조건으로 비교하기 위함.
        _GBS = int(os.environ.get("GBSEEDS", 1))
        def _gb(cols, seed):
            return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                                  l2_regularization=1.0, random_state=seed)
        for _tag, _cols in (("p_v3", F3), ("p_hyb", F3 + VC)):
            _ps = [ _gb(_cols, sd).fit(tr[_cols].to_numpy(float), tr.y.to_numpy())
                        .predict_proba(E.loc[te, _cols].to_numpy(float))[:, 1] for sd in range(_GBS) ]
            E.loc[te, _tag] = np.mean(_ps, 0)
            if _GBS > 1:
                for sd in range(_GBS): E.loc[te, f"{_tag}_s{sd}"] = _ps[sd]
        lr = LogisticRegression(max_iter=3000).fit(np.nan_to_num(tr[BASE_F].to_numpy(float)), tr.y.to_numpy()); E.loc[te, "p_lr"] = lr.predict_proba(np.nan_to_num(E.loc[te, BASE_F].to_numpy(float)))[:, 1]
        print(f"  선발 헤드 학습 완료 · 평가 {s}", flush=True)
    T = E.dropna(subset=["p_v3"]).copy() if not SKIPXI else E; res = {}; KEYS = {}
    _per_seed = [(f"  v3 부스팅 시드 {sd}", f"p_v3_s{sd}") for sd in range(int(os.environ.get("GBSEEDS", 1)))
                 if f"p_v3_s{sd}" in E.columns]
    for nm, col in ([] if SKIPXI else [("MLP 헤드 (선수벡터+점수+32피처) [DL]", "p_mlp"), ("부스팅 + 선수벡터 (하이브리드) [DL]", "p_hyb"), ("v3 부스팅 (전 피처)", "p_v3")] + _per_seed + [ ("v2 로지스틱 (32피처)", "p_lr"), ("직전 XI 지속 (st_last)", "p_prev"), ("과거 선발률 (st_rate)", "p_strate"), ("우리 점수 상위", "our_sc"), ("과거 평균 출전분", "hist_min")]):
        hits = []; per = defaultdict(list); keys_ = []
        for k, sub in T.groupby("key"):
            truth = set(sub[sub.y == 1].player_id.astype(int))
            if len(truth) != 11: continue
            keys_.append(k)
            s_ = {int(p): (0.0 if not np.isfinite(v) else float(v)) for p, v in zip(sub.player_id, sub[col])}
            gk_pool = [int(p) for p in sub.player_id if ISGK.get(int(p), False)]
            qmap = {int(p): np.eye(5)[CELL[k].get(int(p), (2, 1))[0]] for p in sub.player_id}; lane_ = {int(p): CELL[k].get(int(p), (2, 1))[1] for p in sub.player_id}
            rec, _, _ = pick_soft(s_, qmap, lane_, NEED[k], gk_pool, set(s_), LP, BETA); h = len(truth & set(rec)); hits.append(h); per[int(sub.season.iloc[0])].append(h)
        res[nm] = np.array(hits, float); KEYS.setdefault(nm, keys_)
        print(f"  {nm:38s} {res[nm].mean():.3f}/11  n={len(hits):,}  " + " · ".join(f"{s}: {np.mean(v):.3f}" for s, v in sorted(per.items())), flush=True)
    if os.environ.get("DUMP_HITS") and res: pd.to_pickle({nm: dict(zip(KEYS[nm], res[nm])) for nm in res}, os.environ["DUMP_HITS"]); print(f"  사례별 적중 덤프 → {os.environ['DUMP_HITS']}", flush=True)
    rng = np.random.default_rng(0); a = res.get("MLP 헤드 (선수벡터+점수+32피처) [DL]"); hh = res.get("부스팅 + 선수벡터 (하이브리드) [DL]")
    for nm2 in ([] if SKIPXI else ("v3 부스팅 (전 피처)", "v2 로지스틱 (32피처)", "과거 평균 출전분")):
        b = res[nm2]; n = min(len(a), len(b))
        for lab, x in (("MLP 헤드", a), ("하이브리드", hh)):
            d = x[:n] - b[:n]; bs = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
            print(f"  {lab} − {nm2}: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]", flush=True)
    # ── ② 포메이션 예측 ─────────────────────────────────────────
    NEW = FP.merged_labels(); classes = Counter(NEW.values()); top = [c for c, n in classes.most_common(8)]; cls = {c: i for i, c in enumerate(top)}; OTH = len(top)
    lab = lambda s_: cls.get(s_, OTH)
    KH = int(os.environ.get("FORMK", 5)); NPREV = int(os.environ.get("FORMPREV", 3))                       # 모양 이력 길이 · 원핫으로 넣는 직전 모양 수
    MGR = os.environ.get("FORMMGR", "0") == "1"                                                              # 감독 블록: 재임 경기 수·신임 여부·감독의 과거 모양 분포
    if MGR:
        MG = pd.read_parquet(ROOT / "outputs/game_manager.parquet"); MG = MG[MG.role.astype(str).str.contains("감독", na=False)]
        MGK = {(int(r.game_id), int(r.team_id)): (str(r.manager), float(r.mgr_game_no or 0)) for r in MG.itertuples(index=False)}
        mgr_hist = defaultdict(list)
    SEA = dict(zip(g.game_id, g.season)); res_t = defaultdict(list); shp_t = defaultdict(list); rows = []
    sq = D.groupby("key").indices
    TOKSET = os.environ.get("TOKSET", "squad")        # squad | xi | shuffle(위약: 토큰을 경기 간에 섞음)
    TOKOFF = os.environ.get("TOKOFF", "0") == "1"     # 오프더볼 원 열(VERSA+ 상대 벌점)을 토큰에 직접 추가
    if TOKOFF:                                        #   인코더의 16차원 병목을 거치지 않고 들어가는지 보는 검정
        _VP = pd.read_parquet(ROOT / "outputs/versaplus.parquet")
        _vc = [c for c in _VP.columns if c.startswith(("bv_opp", "bn_opp"))]
        _VPI = _VP.set_index(["game_id", "player_id"])[_vc]
        _OFF = np.nan_to_num(_VPI.reindex(pd.MultiIndex.from_arrays(
            [D.game_id.to_numpy(), D.player_id.to_numpy()])).to_numpy(np.float32), nan=0.0)
        _OFF = np.clip((_OFF - _OFF.mean(0)) / _OFF.std(0).clip(1e-6), -10, 10)
        print(f"  TOKOFF=1: 오프더볼 원 열 {len(_vc)} 을 토큰에 직접 추가", flush=True)
    TOKPOS = os.environ.get("TOKPOS", "0") == "1"     # 추천기가 쓰는 위치 입력을 토큰에 추가:
    if TOKPOS:                                        #   동적 단 분포 q0..q4(직전 경기까지) + 과거 레인 원핫
        from gap_sym import dyn_q
        from gap_lane import past_lanes, declared_grid as _dg
        _, _, _HIST = _dg(); _PLANE = past_lanes(_HIST, gpos)
        print("  TOKPOS=1: 동적 단 분포 q(5) + 과거 레인(3) 을 토큰에 추가", flush=True)
    YST = D["y"].to_numpy(float) if "y" in D.columns else np.zeros(len(D))
    UF = np.nan_to_num(D[BASE_F].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0); UF = np.clip((UF - UF.mean(0)) / UF.std(0).clip(1e-6), -10, 10).astype(np.float32)
    for r in g.itertuples(index=False):
        gid = int(r.game_id)
        pair = ((int(r.home_team_id), r.home_score, r.away_score), (int(r.away_team_id), r.away_score, r.home_score))
        snap = {tid: (list(shp_t[tid][-KH:]), list(res_t[tid][-KH:])) for tid, _, _ in pair}
        for tid, gf, ga in pair:
            s_ = NEW.get((gid, tid)); pts = 3 if gf > ga else (1 if gf == ga else 0); h5, r5 = snap[tid]
            if s_ is not None and len(h5) >= 3 and (gid, tid) in sq:
                x = np.zeros(NPREV * (OTH + 1) + 1 + 3 + (2 + OTH + 1 if MGR else 0))
                for j, prev in enumerate(h5[::-1][:NPREV]): x[j * (OTH + 1) + lab(prev)] = 1.0
                x[NPREV * (OTH + 1)] = Counter(h5).most_common(1)[0][1] / len(h5); x[-3 - (2 + OTH + 1 if MGR else 0)] = sum(r5[-3:]) / 9.0; x[-2 - (2 + OTH + 1 if MGR else 0)] = sum(r5) / (3 * len(r5)); x[-1 - (2 + OTH + 1 if MGR else 0)] = 1.0 if tid == int(r.home_team_id) else 0.0
                if MGR:
                    mn, gno = MGK.get((gid, tid), ("?", 0.0)); hh = mgr_hist[mn]
                    x[-(2 + OTH + 1)] = np.log1p(gno) / 5.0; x[-(1 + OTH + 1)] = 1.0 if gno <= 5 else 0.0
                    if hh:
                        c_ = Counter(hh)
                        for k_, v_ in c_.items(): x[-(OTH + 1) + k_] = v_ / len(hh)          # 감독의 과거 모양 분포(경기 이전만)
                idx = sq[(gid, tid)]
                if TOKSET == "xi":                    # 선발 XI 만 토큰으로 (오라클 진단: 모양이 인적 구성으로 정해지는가)
                    _st = idx[YST[idx] > 0.5]
                    if len(_st) >= 7: idx = _st
                vv = V[idx]; emb = np.concatenate([vv.mean(0), vv.max(0), [ADD[idx].mean(), ADD[idx].max()]])
                tok = np.hstack([vv, ADD[idx][:, None], UF[idx]])
                if TOKOFF:
                    tok = np.hstack([tok, _OFF[idx]])
                    emb = np.concatenate([emb, _OFF[idx].mean(0), _OFF[idx].sum(0)])
                if TOKPOS:
                    _pid = D["player_id"].to_numpy()[idx]
                    _q = np.zeros((len(idx), 5), np.float32); _ln = np.zeros((len(idx), 3), np.float32)
                    for _j, _p in enumerate(_pid):
                        _qq = dyn_q(gid, int(_p))
                        if _qq is not None: _q[_j] = np.asarray(_qq, np.float32)
                        _l = _PLANE.get((gid, int(_p)))
                        if _l is not None: _ln[_j, int(_l)] = 1.0
                    tok = np.hstack([tok, _q, _ln])
                    emb = np.concatenate([emb, _q.mean(0), _q.sum(0), _ln.sum(0)])   # 팀 수준: 밴드별 예상 인원
                rows.append((gid, tid, int(SEA[gid]), lab(s_), lab(h5[-1]), x, emb, tok, lab(Counter(h5).most_common(1)[0][0]), [lab(h) for h in h5[-KH:]]))
            if s_ is not None:
                shp_t[tid].append(s_); res_t[tid].append(pts)
                if MGR: mgr_hist[MGK.get((gid, tid), ("?", 0.0))[0]].append(lab(s_))
    Xs = np.array([r_[5] for r_ in rows]); Xe = np.array([r_[6] for r_ in rows]); y = np.array([r_[3] for r_ in rows]); pers = np.array([r_[4] for r_ in rows]); sea = np.array([r_[2] for r_ in rows])
    if os.environ.get("TEAMSTYLE", "0") == "1":                                                        # 팀 단위 스타일(team_stats): 직전 5경기 평균 — 포메이션과 상관될 수 있는 플레이 방식
        TS = pd.read_parquet(ROOT / "outputs/teamstats.parquet"); tcols = [c for c in TS.columns if c.startswith("ts_")]
        TS = TS.set_index(["game_id", "team_id"]); TSv = TS[tcols].to_numpy(np.float32)
        pos = {k: i for i, k in enumerate(TS.index)}
        pv = TSv[:, tcols.index("ts_pass")].clip(1, None) if "ts_pass" in tcols else np.ones(len(TSv), np.float32)
        TSr = TSv / pv[:, None]                                                                            # 패스 수로 나눠 볼륨이 아니라 성향을 남김
        histT = defaultdict(list); ST = np.zeros((len(rows), len(tcols)), np.float32); okT = np.zeros(len(rows), bool)
        order = {int(r.game_id): i for i, r in enumerate(g.itertuples(index=False))}
        rr = sorted(range(len(rows)), key=lambda i: order.get(rows[i][0], 0))
        seen = set()
        for i in rr:
            gid_, tid_ = rows[i][0], rows[i][1]; h = histT[tid_]
            if len(h) >= 3: ST[i] = np.mean(h[-5:], 0); okT[i] = True
            k = (gid_, tid_)
            if k in pos and k not in seen: seen.add(k); histT[tid_].append(TSr[pos[k]])
        mu_ = ST[okT].mean(0); sd_ = ST[okT].std(0).clip(1e-6); ST = np.clip((ST - mu_) / sd_, -5, 5) * okT[:, None]
        Xs = np.hstack([Xs, ST, okT[:, None].astype(np.float32)]); print(f"팀 스타일 {len(tcols)} 열 추가 · 이력 있음 {okT.mean() * 100:.1f}%", flush=True)
    tmode = np.array([r_[8] for r_ in rows]); SEQ = np.full((len(rows), KH), OTH + 1, int)                   # 팀 최빈(최근 5) · GRU 용 시퀀스(패딩 = OTH+1)
    for i, r_ in enumerate(rows): q = r_[9]; SEQ[i, KH - len(q):] = q
    def fit_gru(itr_, iva_, seeds=3, epochs=150):
        """GRU over the last-5 shape sequence (+ S·R 스칼라) → 다중분류, 검증 조기 종료, 시드 평균."""
        class GRUHead(nn.Module):
            def __init__(self, ncls, d=32):
                super().__init__(); self.emb = nn.Embedding(ncls + 1, d); self.gru = nn.GRU(d, d, batch_first=True); self.fc = nn.Linear(d + Xs.shape[1], ncls)
            def forward(self, sq, xs): _, h = self.gru(self.emb(sq)); return self.fc(torch.cat([h[-1], xs], 1))
        S_t = torch.as_tensor(SEQ, device=DEV); X_t = torch.as_tensor(Xs, dtype=torch.float32, device=DEV); y_t = torch.as_tensor(y, device=DEV); lossf = nn.CrossEntropyLoss(); nets = []
        for sd in range(seeds):
            torch.manual_seed(sd); net = GRUHead(OTH + 1).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3); best, state, pat = 1e9, None, 0
            for ep in range(epochs):
                net.train(); perm = torch.as_tensor(np.random.default_rng(sd * 1000 + ep).permutation(itr_), device=DEV)
                for s0 in range(0, len(perm), 256):
                    b = perm[s0:s0 + 256]; opt.zero_grad(); loss = lossf(net(S_t[b], X_t[b]), y_t[b]); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): v = lossf(net(S_t[iva_], X_t[iva_]), y_t[iva_]).item()
                if np.isfinite(v) and v < best - 1e-5: best, state, pat = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    pat += 1
                    if pat >= 15: break
            if state is not None: net.load_state_dict(state)
            net.eval(); nets.append(net)
        def pred(ix):
            with torch.no_grad(): return np.mean([torch.softmax(n_(S_t[ix], X_t[ix]), 1).cpu().numpy() for n_ in nets], 0)
        return pred
    def fit_tf(itr_, iva_, seeds=3, epochs=150):
        """모양 시퀀스(길이 5) 위 Transformer 인코더 + S·R 스칼라 → 다중분류. GRU 와 동일 규약(검증 조기 종료·시드 평균)."""
        class TFHead(nn.Module):
            def __init__(self, ncls, d=32, heads=4, layers=1):
                super().__init__()
                self.emb = nn.Embedding(ncls + 1, d); self.pos = nn.Parameter(torch.randn(1, KH, d) * 0.02)
                self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True), layers)
                self.fc = nn.Linear(d + Xs.shape[1], ncls)
            def forward(self, sq, xs):
                h = self.tf(self.emb(sq) + self.pos)                                  # 패딩은 클래스 ncls 로 표시되어 임베딩이 구분한다
                return self.fc(torch.cat([h[:, -1], xs], 1))
        S_t = torch.as_tensor(SEQ, device=DEV); X_t = torch.as_tensor(Xs, dtype=torch.float32, device=DEV); y_t = torch.as_tensor(y, device=DEV); lossf = nn.CrossEntropyLoss(); nets = []
        for sd in range(seeds):
            torch.manual_seed(sd); net = TFHead(OTH + 1).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3); best, state, pat = 1e9, None, 0
            for ep in range(epochs):
                net.train(); perm = torch.as_tensor(np.random.default_rng(sd * 1000 + ep).permutation(itr_), device=DEV)
                for s0 in range(0, len(perm), 256):
                    b = perm[s0:s0 + 256]; opt.zero_grad(); loss = lossf(net(S_t[b], X_t[b]), y_t[b]); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad(): v = lossf(net(S_t[iva_], X_t[iva_]), y_t[iva_]).item()
                if np.isfinite(v) and v < best - 1e-5: best, state, pat = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    pat += 1
                    if pat >= 15: break
            if state is not None: net.load_state_dict(state)
            net.eval(); nets.append(net)
        def pred(ix):
            with torch.no_grad(): return np.mean([torch.softmax(n_(S_t[ix], X_t[ix]), 1).cpu().numpy() for n_ in nets], 0)
        return pred

    PMAX = max(len(r_[7]) for r_ in rows); dtok = rows[0][7].shape[1]; TOK = np.zeros((len(rows), PMAX, dtok), np.float32); PAD = np.ones((len(rows), PMAX), bool)
    for i, r_ in enumerate(rows): n_ = len(r_[7]); TOK[i, :n_] = r_[7]; PAD[i, :n_] = False
    if TOKSET == "shuffle":                            # 위약 — 구조·S+R 은 그대로, 로스터↔경기 대응만 파괴
        _p = np.random.default_rng(0).permutation(len(rows)); TOK = TOK[_p]; PAD = PAD[_p]
        print("  TOKSET=shuffle: 로스터 토큰을 경기 간에 섞음(위약)", flush=True)
    _nev = int(np.isin(sea, TESTS).sum())
    print(f"\n포메이션: 표본 {len(rows):,} · 평가(보류 시즌) {_nev:,} · 클래스 {OTH}+other", flush=True)
    acc = defaultdict(list); accs = defaultdict(lambda: defaultdict(list))
    for s in TESTS:
        tr = sea < s; te = sea == s
        if te.sum() == 0: continue
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(Xs[tr], y[tr]); p_lr = lr.predict(Xs[te])
        XX = np.hstack([Xs, (Xe - Xe[tr].mean(0)) / Xe[tr].std(0).clip(1e-6)]); cut = int(.85 * tr.sum()); itr = np.flatnonzero(tr)
        pred = fit_head(XX[itr[:cut]], y[itr[:cut]], XX[itr[cut:]], y[itr[cut:]], out=OTH + 1); p_mlp = pred(XX[te]).argmax(1)
        lr2 = LogisticRegression(max_iter=3000, C=1.0).fit(XX[tr], y[tr]); p_lr2 = lr2.predict(XX[te])
        preds_ = fit_set_head(TOK, PAD, Xs, y, itr[:cut], itr[cut:], OTH + 1); p_set = preds_(np.flatnonzero(te)).argmax(1)
        p_maj = np.full(te.sum(), Counter(y[tr].tolist()).most_common(1)[0][0]); p_tm = tmode[te]
        TM = np.ones((OTH + 1, OTH + 1)); [TM.__setitem__((a_, b_), TM[a_, b_] + 1) for a_, b_ in zip(pers[tr], y[tr])]; p_mk = TM[pers[te]].argmax(1)          # 마르코프 전이(직전 모양 → 다음 모양, 라플라스 평활)
        p_gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0).fit(Xs[tr], y[tr]).predict(Xs[te])
        p_gru = fit_gru(itr[:cut], itr[cut:])(np.flatnonzero(te)).argmax(1)
        p_tf = fit_tf(itr[:cut], itr[cut:])(np.flatnonzero(te)).argmax(1)
        for nm, p_ in (("다수 클래스", p_maj), ("팀 최빈 모양(최근 5)", p_tm), ("마르코프 전이", p_mk), ("부스팅 (S+R)", p_gb), ("GRU 모양 시퀀스 + S+R [DL]", p_gru), ("Transformer 모양 시퀀스 + S+R [DL]", p_tf)):
            ok = (p_ == y[te]).astype(float); acc[nm].extend(ok.tolist()); accs[nm][s] = ok.mean()
        for nm, p_ in (("지속성", pers[te]), ("다항 LR (S+R)", p_lr), ("다항 LR + 선수벡터 [DL]", p_lr2), ("MLP 헤드 (S+R+선수벡터) [DL]", p_mlp), ("Set Transformer 헤드 (로스터 토큰 SAB→PMA + S+R) [DL]", p_set)):
            ok = (p_ == y[te]).astype(float); acc[nm].extend(ok.tolist()); accs[nm][s] = ok.mean()
    for nm in acc:
        a_ = np.array(acc[nm]); print(f"  {nm:30s} {a_.mean():.3f}  " + " · ".join(f"{s}: {v:.3f}" for s, v in sorted(accs[nm].items())), flush=True)
    if os.environ.get("DUMP_FHITS"):                                         # 포메이션 사례별 적중 덤프 — 변형 간 짝 검정용
        pd.to_pickle({nm: np.array(v) for nm, v in acc.items()}, os.environ["DUMP_FHITS"])
        print(f"  포메이션 적중 덤프 → {os.environ['DUMP_FHITS']}", flush=True)
    base = np.array(acc["지속성"])
    for nm in ("Transformer 모양 시퀀스 + S+R [DL]", "다항 LR (S+R)", "다항 LR + 선수벡터 [DL]", "MLP 헤드 (S+R+선수벡터) [DL]", "Set Transformer 헤드 (로스터 토큰 SAB→PMA + S+R) [DL]", "다수 클래스", "팀 최빈 모양(최근 5)", "마르코프 전이", "부스팅 (S+R)", "GRU 모양 시퀀스 + S+R [DL]"):
        d = np.array(acc[nm]) - base; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"  {nm} − 지속성: {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]", flush=True)


if __name__ == "__main__":
    main()
