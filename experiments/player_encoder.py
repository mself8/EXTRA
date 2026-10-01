"""선수 표현 학습 — 손으로 만든 스냅샷(지수감쇠 평균) 대신 경기 시퀀스에서 임베딩을 배운다.

지금까지(v1~v4): 입력 = 682열 지수감쇠 스냅샷이면 어떤 머리도 릿지를 못 넘는다.
여기서는 **입력**을 바꾼다.
  · 선수-경기 원시 벡터 시퀀스(과거 K경기) → 어텐션 풀링 (감쇠 가중을 학습)
  · 지수감쇠 평균을 skip 으로 동봉 (현행 표현을 정확히 재현할 수 있게)
  · 보조 과제: 선수-경기 5만 행으로 "다음 경기 요약(PCA-8)" 예측 → 표현을 가르치는 감독 신호
  · 팀 머리는 가산 그대로: Σ우리 s_p − Σ상대 s_p + 기저 (교차항은 9단 귀무)
변형   att      어텐션 풀링 + skip, 보조 과제 없음
       att+aux  + 보조 과제 (λ_aux=1)
       tf+aux   + 트랜스포머 1층
판정   경기 단위 보류 R² Δ vs 현행 선형(차분 PCA-20 릿지), 경기 블록 부트스트랩.
       보조 과제 R²(시퀀스 vs 감쇠 평균)도 보고 — 가중 학습이 고정 감쇠를 이기는가.
산출   outputs/player_encoder_scores.pkl — 프로덕션 pkl 의 (gid,tid) 별 {pid: s} (배터리용)

실행: python -m experiments.player_encoder
"""
from __future__ import annotations
import os, sys, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp.parquet")
os.environ.setdefault("DROPFAM", "zv,zn")
import torch, torch.nn as nn                        # noqa: E402
from sklearn.linear_model import Ridge              # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from lineup_pipeline import r2w, AL                 # noqa: E402
from nonadditive_gate import build_sides            # noqa: E402
from lineup_event_panel import FEAT_FILE, HL        # noqa: E402
from config import VAEP_OUTPUT_DIR                  # noqa: E402

K = int(os.environ.get("K", 32)); D = 64; NPC = 20; NAUX = 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2)
PROD = ROOT / "outputs/gap_soft_fullonball_gk_resid_merged_mix25d_pc20_all_npxg_nozvzn_fixsd.pkl"
VARIANTS = {"att": dict(aux=0.0, tf=False), "att+aux": dict(aux=1.0, tf=False), "tf+aux": dict(aux=1.0, tf=True)}


# ── 데이터 ────────────────────────────────────────────────────────────────
def load_rows():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    P = pd.read_parquet(FEAT_FILE)
    dp = tuple(x.strip() + "_" for x in os.environ.get("DROPFAM", "").split(",") if x.strip())
    FC = [c for c in P.columns if c not in ("game_id", "player_id") and not c.startswith(dp)]
    P = P[P.game_id.isin(gpos)].copy(); P["t"] = P.game_id.map(gpos)
    P = P.sort_values(["player_id", "t"]).reset_index(drop=True)
    X = P[FC].to_numpy(np.float32)
    pid = P.player_id.to_numpy(int); t = P.t.to_numpy(int); gid = P.game_id.to_numpy(int)
    start = {};
    for i, p in enumerate(pid):
        if p not in start: start[p] = i
    bounds = {}
    ps = list(start);
    for j, p in enumerate(ps): bounds[p] = (start[p], start[ps[j + 1]] if j + 1 < len(ps) else len(pid))
    return g, gpos, X, pid, t, gid, bounds


def seq_for(p, tt, bounds, t_arr):
    """선수 p 의 t<tt 행 인덱스 최근 K개 (오래된→최근), 없으면 빈 배열."""
    if p not in bounds: return np.empty(0, int)
    a, b = bounds[p]; ts = t_arr[a:b]
    j = np.searchsorted(ts, tt, "left")          # ts[j-1] < tt
    lo = max(a, a + j - K); return np.arange(lo, a + j)


DAYIDX = None          # DTDAYS=1: 경기 순번 → 날짜(일) 배열 (load_structured 가 채움) — dt 를 달력 일수로
def pack(pairs, bounds, t_arr):
    """(p, tt) 목록 → idx [n,K] (−1 패딩), dt [n,K] (경기 순번 차 또는 DTDAYS=1 이면 달력 일수 차)."""
    idx = np.full((len(pairs), K), -1, np.int64); dt = np.zeros((len(pairs), K), np.float32)
    use_days = os.environ.get("DTDAYS", "0") == "1" and DAYIDX is not None
    for r, (p, tt) in enumerate(pairs):
        s = seq_for(p, tt, bounds, t_arr)
        if len(s): idx[r, K - len(s):] = s; dt[r, K - len(s):] = (DAYIDX[tt] - DAYIDX[t_arr[s]]) if use_days else (tt - t_arr[s])
    return idx, dt


# ── 모델 ──────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, d_in, tf=False, drop=0.1):
        super().__init__()
        self.tok = nn.Linear(d_in, D); self.dt = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, D))
        self.skip = nn.Linear(d_in, D)
        self.tf = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=drop, batch_first=True) if tf else None
        self.att = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, 1))
        self.drop = nn.Dropout(drop); self.fdrop = nn.Dropout(0.2); self.norm = nn.LayerNorm(D)
        self.lam = math.log(2) / HL

    def forward(self, Xs, dt, mask):
        # Xs [n,K,d_in]  dt [n,K]  mask [n,K] (True=유효)
        w = torch.exp(-self.lam * dt) * mask.float(); w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
        dec = (Xs * w.unsqueeze(-1)).sum(1)                       # 지수감쇠 평균 (현행 표현)
        Xs = self.fdrop(Xs)                                        # 피처 드롭아웃 (682열 과적합 억제)
        tok = self.drop(self.tok(Xs) + self.dt(torch.log1p(dt).unsqueeze(-1)))
        if self.tf is not None: tok = self.tf(tok, src_key_padding_mask=~mask)
        sc = self.att(tok).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        a = (tok * sc.unsqueeze(-1)).sum(1) * (mask.any(1, keepdim=True).float())
        return self.norm(self.skip(dec) + a)                       # [n,D]


class Model(nn.Module):
    def __init__(self, d_in, d_base, tf=False):
        super().__init__()
        self.enc = Encoder(d_in, tf); self.head = nn.Linear(D, 1); self.base = nn.Linear(d_base, 1)
        self.aux = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, NAUX))
        for lin in (self.head, self.base):                          # 초기 예측 = 0 (평균) 에서 출발
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def player_score(self, Xs, dt, mask): return self.head(self.enc(Xs, dt, mask)).squeeze(-1)

    def team(self, Xa, dta, ma, Xb, dtb, mb, base):
        B = base.shape[0]
        sa = self.player_score(Xa, dta, ma).view(B, 11); sb = self.player_score(Xb, dtb, mb).view(B, 11)
        return sa.sum(1) - sb.sum(1) + self.base(base).squeeze(-1)


def gather(Xg, idx, dt):
    """idx [n,K] (GPU long) → Xs [n,K,d], dt, mask."""
    mask = idx >= 0
    empty = ~mask.any(1)                                           # 이력 없는 선수: 더미 토큰 하나 (NaN 방지)
    if empty.any(): mask = mask.clone(); mask[empty, -1] = True
    Xs = Xg[idx.clamp(min=0)] * (idx >= 0).unsqueeze(-1).float()
    return Xs, dt, mask


def train_fold(Xg, TG, AUX, cfg, tr_i, va_i, seed, epochs=None, d_base=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Model(Xg.shape[1], d_base, tf=cfg["tf"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    Ia, Da, Ib, Db, BASE, y = TG; Ai, Ad, Az, Ag, _ = AUX
    def team_loss(i):
        b = torch.as_tensor(i, device=DEV)
        Xa, dta, ma = gather(Xg, Ia[b].view(-1, K), Da[b].view(-1, K)); Xb, dtb, mb = gather(Xg, Ib[b].view(-1, K), Db[b].view(-1, K))
        return ((net.team(Xa, dta, ma, Xb, dtb, mb, BASE[b]) - y[b]) ** 2).mean()
    def aux_loss(j):
        b = torch.as_tensor(j, device=DEV); Xs, dts, ms = gather(Xg, Ai[b], Ad[b])
        return ((net.aux(net.enc(Xs, dts, ms)) - Az[b]) ** 2).mean()
    best, best_ep, pat, state = 1e9, 0, 0, None
    n_ep = epochs or 60
    aux_pool = AUX[4]                                           # 학습 가능한 보조 행 인덱스
    for ep in range(n_ep):
        net.train(); perm = np.random.permutation(tr_i); nb = 0
        for s in range(0, len(perm), 64):
            i = perm[s:s + 64]; opt.zero_grad(); loss = team_loss(i)
            if cfg["aux"] > 0 and len(aux_pool):
                j = np.random.choice(aux_pool, 512, replace=len(aux_pool) < 512); loss = loss + cfg["aux"] * aux_loss(j)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); nb += 1
        if va_i is None: continue
        net.eval()
        with torch.no_grad(): v = float(np.mean([team_loss(va_i[s:s + 256]).item() for s in range(0, len(va_i), 256)]))
        if v < best - 1e-6: best, best_ep, pat, state = v, ep, 0, {k: t_.clone() for k, t_ in net.state_dict().items()}
        else:
            pat += 1
            if pat >= 8: break
    if state is not None: net.load_state_dict(state)
    net.eval(); return net, best, best_ep + 1


def main():
    g, gpos, X, pid, t_arr, gid_arr, bounds = load_rows()
    R, A, Bm, BASE, IDX = build_sides()                     # 사례·폴드·기저는 현행과 동일
    y = R.y.to_numpy(); n = len(R)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])  # noqa: E712
    hm = g.set_index("game_id"); me = np.where(R.home.to_numpy() == 1, hm.home_team_id.reindex(R.game_id), hm.away_team_id.reindex(R.game_id)).astype(int)
    op = np.where(R.home.to_numpy() == 1, hm.away_team_id.reindex(R.game_id), hm.home_team_id.reindex(R.game_id)).astype(int)
    tt = R.t.to_numpy()
    pairs_a = [(p, tt[i]) for i in range(n) for p in st.loc[(int(R.game_id.iloc[i]), int(me[i]))][:11]]
    pairs_b = [(p, tt[i]) for i in range(n) for p in st.loc[(int(R.game_id.iloc[i]), int(op[i]))][:11]]
    Ia, Da = pack(pairs_a, bounds, t_arr); Ib, Db = pack(pairs_b, bounds, t_arr)
    Ia, Da, Ib, Db = Ia.reshape(n, 11, K), Da.reshape(n, 11, K), Ib.reshape(n, 11, K), Db.reshape(n, 11, K)
    print(f"사례 {n:,} · 선수-경기 행 {len(X):,} · K={K} · 장치 {DEV}", flush=True)
    # 보조 과제 행: 이력 ≥1 인 선수-경기 (현재 경기 요약을 과거 시퀀스로 예측)
    aux_pairs = [(int(pid[i]), int(t_arr[i])) for i in range(len(pid))]
    Ai_all, Ad_all = pack(aux_pairs, bounds, t_arr); has = Ai_all[:, -1] >= 0
    aux_rows = np.flatnonzero(has); print(f"보조 과제 행 {len(aux_rows):,}", flush=True)

    D_ = A - Bm; Ds = (D_ - D_.mean(0)) / D_.std(0).clip(1e-9)
    names = ["linear"] + list(VARIANTS)
    preds = {k: np.full(n, np.nan) for k in names}; auxr2 = {k: [] for k in VARIANTS}
    SC = {}                                              # (gid,tid) → {pid: s}  (프로덕션 pkl 풀)
    prod = pd.read_pickle(PROD); prod_rows = {(int(r.gid), int(r.tid)): list(r.sc) for r in prod.itertuples(index=False)}
    game_fold = {}
    for f in range(1, 5):
        tr = np.concatenate(IDX[:f]); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]; te = IDX[f]
        for i in te: game_fold[int(R.game_id.iloc[i])] = f
        # 현행 선형
        pc = PCA(n_components=NPC, random_state=0).fit(Ds[i1]); Zd = pc.transform(Ds); Zd = (Zd - Zd[i1].mean(0)) / Zd[i1].std(0).clip(1e-9)
        Xl = np.hstack([BASE, Zd])
        ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(Xl[i1], y[i1]).predict(Xl[i2]), np.ones(len(i2))))
        preds["linear"][te] = Ridge(alpha=ba).fit(Xl[tr], y[tr]).predict(Xl[te])
        # 표준화(학습 경기 행 기준) · 보조 목표 PCA-8
        tr_games = set(int(x) for x in R.game_id.iloc[tr]); i1_games = set(int(x) for x in R.game_id.iloc[i1])
        rows_tr = np.flatnonzero(np.isin(gid_arr, list(i1_games)))
        mu, sd = X[rows_tr].mean(0), X[rows_tr].std(0).clip(1e-6)
        Xs_np = ((X - mu) / sd).clip(-5, 5).astype(np.float32)
        Xg = torch.as_tensor(Xs_np, device=DEV)
        pca8 = PCA(n_components=NAUX, random_state=0).fit(Xs_np[rows_tr]); Z8 = pca8.transform(Xs_np); Z8 = (Z8 - Z8[rows_tr].mean(0)) / Z8[rows_tr].std(0).clip(1e-9)
        aux_ok = aux_rows[np.isin(gid_arr[aux_rows], list(i1_games))]
        AUX = (torch.as_tensor(Ai_all, device=DEV), torch.as_tensor(Ad_all, device=DEV), torch.as_tensor(Z8.astype(np.float32), device=DEV), gid_arr, aux_ok)
        TG = (torch.as_tensor(Ia, device=DEV), torch.as_tensor(Da, device=DEV), torch.as_tensor(Ib, device=DEV), torch.as_tensor(Db, device=DEV),
              torch.as_tensor(BASE.astype(np.float32), device=DEV), torch.as_tensor(y.astype(np.float32), device=DEV))
        for name, cfg in VARIANTS.items():
            _, v, ep = train_fold(Xg, TG, AUX, cfg, i1, i2, 0, d_base=BASE.shape[1])
            aux_full = aux_rows[np.isin(gid_arr[aux_rows], list(tr_games))]
            AUXf = AUX[:4] + (aux_full,)
            pr = np.zeros(len(te)); nets = []
            for sd_ in SEEDS:
                net, _, _ = train_fold(Xg, TG, AUXf, cfg, tr, None, sd_, epochs=ep, d_base=BASE.shape[1]); nets.append(net)
                with torch.no_grad():
                    for s in range(0, len(te), 256):
                        b = torch.as_tensor(te[s:s + 256], device=DEV)
                        Xa, dta, ma = gather(Xg, TG[0][b].view(-1, K), TG[1][b].view(-1, K)); Xb, dtb, mb = gather(Xg, TG[2][b].view(-1, K), TG[3][b].view(-1, K))
                        pr[s:s + 256] += net.team(Xa, dta, ma, Xb, dtb, mb, TG[4][b]).cpu().numpy() / len(SEEDS)
            preds[name][te] = pr
            # 보조 과제 R² (보류 경기 행): 시퀀스 모델 vs 감쇠 평균의 선형 예측
            te_games = set(int(x) for x in R.game_id.iloc[te]); ar = aux_rows[np.isin(gid_arr[aux_rows], list(te_games))]
            if cfg["aux"] > 0 and len(ar):
                with torch.no_grad():
                    b = torch.as_tensor(ar, device=DEV); Xs_, d_, m_ = gather(Xg, AUX[0][b], AUX[1][b])
                    pz = np.mean([nn_.aux(nn_.enc(Xs_, d_, m_)).cpu().numpy() for nn_ in nets], 0)
                zt = Z8[ar]; auxr2[name].append(1 - ((zt - pz) ** 2).sum() / ((zt - zt.mean(0)) ** 2).sum())
                if name == "att+aux":                     # 대조: 감쇠 평균 → 릿지 (고정 가중)
                    def decmean(rows_):
                        idx_, dt_ = Ai_all[rows_], Ad_all[rows_]; w_ = np.exp(-math.log(2) / HL * dt_) * (idx_ >= 0)
                        w_ = w_ / w_.sum(1, keepdims=True).clip(1e-6)
                        return np.einsum("nk,nkd->nd", w_, Xs_np[idx_.clip(min=0)])
                    Dm_tr, Dm_te = decmean(aux_full), decmean(ar)
                    rr = Ridge(alpha=10.0).fit(Dm_tr, Z8[aux_full]); pz0 = rr.predict(Dm_te)
                    auxr2["_dec"] = auxr2.get("_dec", []) + [1 - ((zt - pz0) ** 2).sum() / ((zt - zt.mean(0)) ** 2).sum()]
            # 배터리용 선수 점수 (프로덕션 풀)
            if name == "att+aux":
                for i in te:
                    key = (int(R.game_id.iloc[i]), int(me[i]))
                    if key not in prod_rows: continue
                    pids = prod_rows[key]; idx_, dt_ = pack([(p, tt[i]) for p in pids], bounds, t_arr)
                    with torch.no_grad():
                        Xs_, d_, m_ = gather(Xg, torch.as_tensor(idx_, device=DEV), torch.as_tensor(dt_, device=DEV))
                        s_ = np.mean([nn_.player_score(Xs_, d_, m_).cpu().numpy() for nn_ in nets], 0)
                    SC[key] = {p: float(v_) for p, v_ in zip(pids, s_)}
            print(f"  폴드 {f} {name}: ep={ep} val={v:.4f}", flush=True)
        r2 = lambda k: 1 - ((y[te] - preds[k][te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
        print(f"폴드 {f}: " + " · ".join(f"{k} {r2(k):+.4f}" for k in names), flush=True)

    ix0 = np.flatnonzero(np.isfinite(preds["linear"]))
    gids = R.game_id.to_numpy(); G = {}
    for i in ix0: G.setdefault(int(gids[i]), []).append(i)
    keys = list(G); rng = np.random.default_rng(0)
    def r2i(pv, ix): return 1 - ((y[ix] - pv[ix]) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    print(f"\n기준(차분 선형) R² {r2i(preds['linear'], ix0):+.4f}")
    if auxr2.get("_dec"): print(f"보조 과제 대조 — 감쇠 평균+릿지 R² {np.mean(auxr2['_dec']):+.3f}")
    for k in VARIANTS:
        ds = []
        for _ in range(2000):
            pick = rng.choice(len(keys), len(keys), replace=True); ix = np.concatenate([G[keys[j]] for j in pick])
            ds.append(r2i(preds[k], ix) - r2i(preds["linear"], ix))
        ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
        ax = f" · 보조 R² {np.mean(auxr2[k]):+.3f}" if auxr2[k] else ""
        print(f"  {k:8s} R² {r2i(preds[k], ix0):+.4f}  Δ={ds.mean():+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ>0)={np.mean(ds > 0):.3f}{ax}")
    pd.to_pickle(SC, ROOT / "outputs/player_encoder_scores.pkl")
    np.savez(ROOT / "outputs/player_encoder_preds.npz", **preds, y=y, game_id=gids)
    print(f"저장 player_encoder_scores.pkl ({len(SC):,} 팀-경기)")


if __name__ == "__main__":
    main()
