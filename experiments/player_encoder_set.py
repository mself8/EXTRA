"""단일 모델: 선형 레인(릿지 내장) + 구조 CNN 레인 + 선수 집합 위 Set Transformer — 한 손실로 끝까지 함께 학습.

선수 p (이력 K경기 토큰 Xs):
  선형 레인  lin_p = w · dec_p           dec_p = 표준화 원공간 감쇠 평균 (PCA 없음). w 는 폴드별 닫힌 해 릿지(α 격자 검증)로 **초기화**하고
                                          손실에 (α/n)·‖w‖² 를 명시 — 릿지가 모델 안에 들어 있고, 이후 나머지와 함께 미세조정된다.
  CNN 레인   v_p = MatchEncoder2 토큰 → 고정 감쇠 평균 + 어텐션 풀링 (h 차원)      f_p = head(v_p)
팀 XI (11명):
  가산부     Σ_p (lin_p + f_p)
  집합부     g(XI) = MLP(PMA(SAB^L({v_p})))  — Set Transformer(SAB: 멀티헤드 자기어텐션 + rFF, PMA: 시드 1 풀링), 출력층 0 초기화 → 가산에서 출발
  y = team_me − team_op + base(시즌·홈)
SET=0 이면 집합부 제거(가산 단일 모델) — 어텐션의 기여를 절제로 본다. CROSS=1 이면 상대 XI 토큰까지 한 집합에 넣고 팀 구분 임베딩(상대 조건화).
정규화: lr 1e-4 · wd 5e-2 · drop 0.4 · 조기 종료(검증) · 5 시드 평균 — cnn2 잔차 재학습에서 검증된 레시피.
판정: 보류 R² Δ vs 프로덕션 릿지(PCA-20), 경기 블록 부트스트랩. SAVE=1 이면 폴드 모델 저장(배터리 채점·집합 목적함수용).
실행: SET=1 H=16 SEEDS=5 SAVE=1 python -m experiments.player_encoder_set
"""
from __future__ import annotations
import os, sys, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp_ref.parquet")
os.environ["ZONES"] = "0"; os.environ["RESID"] = "0"; os.environ["SKIP"] = "0"; os.environ.setdefault("K", "96")
os.environ.setdefault("DROP", "0.4")
import torch, torch.nn as nn, torch.nn.functional as F   # noqa: E402
from sklearn.linear_model import Ridge              # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from lineup_pipeline import r2w, AL                 # noqa: E402
from lineup_event_panel import HL                   # noqa: E402
from player_encoder import pack, gather, K, DEV     # noqa: E402
from player_encoder_cnn2 import MatchEncoder2, load_structured, NPC   # noqa: E402
os.environ["DROPFAM"] = "zv,zn"
from nonadditive_gate import build_sides            # noqa: E402
from config import VAEP_OUTPUT_DIR                  # noqa: E402

H = int(os.environ.get("H", 16)); SET = os.environ.get("SET", "1") == "1"; CROSS = os.environ.get("CROSS", "0") == "1"
NS = int(os.environ.get("SEEDS", 5)); L = int(os.environ.get("LAYERS", 1)); HEADS = int(os.environ.get("HEADS", 4))
LR = float(os.environ.get("LR", 1e-4)); WD = float(os.environ.get("WD", 5e-2)); DROP = float(os.environ.get("DROP", 0.4))
LANE = os.environ.get("LANE", "raw")   # raw: 표준화 원공간 감쇠 평균 위 선형 | pca: 프로덕션 릿지와 동치(원 단위 감쇠 평균 → PCA-20 고정 기저 → β), 계수 20 개


class SAB(nn.Module):
    """Set Attention Block: X + MHA(X, X) → + rFF (Lee et al. 2019)."""
    def __init__(self, d, heads, drop):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)

    def forward(self, X, pad=None):
        A, _ = self.mha(X, X, X, key_padding_mask=pad); X = self.n1(X + A)
        return self.n2(X + self.ff(X))


class PMA(nn.Module):
    """Pooling by Multihead Attention: 학습 시드 1개가 집합을 요약."""
    def __init__(self, d, heads, drop):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.1); self.mha = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True); self.n = nn.LayerNorm(d)

    def forward(self, X, pad=None):
        S = self.seed.expand(X.shape[0], -1, -1); A, _ = self.mha(S, X, X, key_padding_mask=pad); return self.n(S + A).squeeze(1)


class SetNet(nn.Module):
    def __init__(self, tidx, zidx, sidx, d_in, d_base, h=H):
        super().__init__()
        self.menc = MatchEncoder2(tidx, zidx, sidx, h, zones=os.environ.get("ZONEMAP", "0") == "1"); self.dt = nn.Linear(1, h)   # ZONEMAP=1: 범주별 2-D 존 지도 분기
        MTF = int(os.environ.get("MATCHTF", 0))                        # 이력 시퀀스 Transformer 인코더 층 수 (0 = 감쇠 평균 + 가산 어텐션만)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, HEADS, 2 * h, DROP, batch_first=True), MTF) if MTF else None
        self.tenc = nn.Sequential(nn.Linear(2, h), nn.Tanh()) if MTF else None    # 시간 임베딩: [log1p dt, 최근 여부]
        self.att = nn.Sequential(nn.Tanh(), nn.Linear(h, 1)); self.head = nn.Linear(h, 1)
        self.lin = nn.Linear(d_in, 1)                                 # 선형 레인 (릿지 내장): 표준화 원공간 감쇠 평균 → 스칼라 (LANE=raw)
        RK = int(os.environ.get("RANK", 20))
        self.lowrank = nn.Sequential(nn.Dropout(DROP), nn.Linear(d_in, RK, bias=False), nn.Linear(RK, 1)) if LANE == "lowrank" else None   # 학습 저랭크 선형 레인 (PCA 없음)
        self.P = None                                                 # LANE=pca: 고정 기저 [20, 682] 와 β (set_pca 에서 생성)
        self.base = nn.Linear(d_base, 1)
        DAYS = os.environ.get("DTDAYS", "0") == "1"; self.lam = math.log(2) / (float(os.environ.get("HL_DAYS", 45)) if DAYS else HL); self.recent = 14.0 if DAYS else 5.0   # 감쇠 반감기: 일수(DTDAYS) 또는 경기 순번
        self.set = SET
        self.cellemb = nn.Embedding(17, h) if (SET and os.environ.get("SLOTEMB", "0") == "1") else None   # 15칸 + GK(15) + 미상(16): 자리 인지 집합 블록
        if SET:
            self.side = nn.Embedding(2, h) if CROSS else None
            self.sabs = nn.ModuleList([SAB(h, HEADS, DROP) for _ in range(L)]); self.pma = PMA(h, HEADS, DROP)
            self.g = nn.Sequential(nn.Dropout(DROP), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
            nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)   # 가산에서 출발
        for lin in (self.head, self.base, self.lin): nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def set_pca(self, Vt, Dsd, zsd, keep):
        del self.P; self.register_buffer("P", torch.as_tensor((Vt / Dsd[None, :] / zsd[:, None]).astype(np.float32))); self.register_buffer("keep", torch.as_tensor(np.flatnonzero(keep)))
        self.beta = nn.Parameter(torch.zeros(Vt.shape[0]))

    def player(self, Xs, dt, mask, Xr=None):
        """→ (가산 점수 [n], 벡터 v_p [n,h]). Xr: 원 단위 이력 [n,K,d] (LANE=pca)."""
        n, k, d = Xs.shape
        tok = self.menc(Xs.reshape(n * k, d)).view(n, k, -1)
        if self.tf is not None:                                       # 시간 인지 Transformer: 토큰 + 시간 임베딩, 패딩 마스크
            te = self.tenc(torch.stack([torch.log1p(dt), (dt < self.recent).float()], -1))
            tok = tok + self.tf(tok + te, src_key_padding_mask=~mask)
        w = torch.exp(-self.lam * dt) * mask.float(); w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
        dec_tok = (tok * w.unsqueeze(-1)).sum(1)
        sc = self.att(tok + self.dt(torch.log1p(dt).unsqueeze(-1))).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        v = dec_tok + (tok * sc.unsqueeze(-1)).sum(1)
        if LANE == "none":                                            # 선형 레인 없음 — CNN 레인만 (PCA 없음)
            add = self.head(v).squeeze(-1)
        elif LANE == "lowrank":                                       # 선수별 가공 피처(표준화 감쇠 평균) → 학습 저랭크 선형 레인
            dec_raw = (Xs * w.unsqueeze(-1)).sum(1)
            add = self.head(v).squeeze(-1) + self.lowrank(dec_raw).squeeze(-1)
        elif self.P is None:
            dec_raw = (Xs * w.unsqueeze(-1)).sum(1)                   # 표준화 원공간 감쇠 평균 (선형 레인 입력)
            add = self.head(v).squeeze(-1) + self.lin(dec_raw).squeeze(-1)
        else:                                                         # 프로덕션 릿지 동치: 원 단위 감쇠 평균 → 고정 PCA-20 기저 → β
            dec = (Xr * w.unsqueeze(-1)).sum(1)[:, self.keep]
            add = self.head(v).squeeze(-1) + (dec @ self.P.T) @ self.beta
        return add, v

    def setterm(self, Va, Vb, Ca=None, Cb=None):
        """Va, Vb [B,11,h] → g_me − g_op (CROSS=0) 또는 g(me∪op) − g(op∪me) (CROSS=1, 팀 구분 임베딩). Ca, Cb [B,11] 칸 인덱스(SLOTEMB)."""
        if not self.set: return 0.0
        if self.cellemb is not None and Ca is not None: Va = Va + self.cellemb(Ca); Vb = Vb + self.cellemb(Cb)
        def enc(X, pad=None):
            for s in self.sabs: X = s(X, pad)
            return self.g(self.pma(X, pad)).squeeze(-1)
        if not CROSS: return enc(Va) - enc(Vb)
        e = self.side.weight
        Xm = torch.cat([Va + e[0], Vb + e[1]], 1); Xo = torch.cat([Vb + e[0], Va + e[1]], 1)
        return enc(Xm) - enc(Xo)

    def team(self, Xa, dta, ma, Xb, dtb, mb, base, Ra=None, Rb=None, Ca=None, Cb=None):
        B = base.shape[0]
        sa, va = self.player(Xa, dta, ma, Ra); sb, vb = self.player(Xb, dtb, mb, Rb)
        out = sa.view(B, 11).sum(1) - sb.view(B, 11).sum(1) + self.base(base).squeeze(-1)
        return out + self.setterm(va.view(B, 11, -1), vb.view(B, 11, -1), Ca, Cb)

    def l2(self): return ((self.beta ** 2).sum() if self.P is not None else (self.lin.weight ** 2).sum()) + (self.base.weight ** 2).sum() + ((self.lowrank[1].weight ** 2).sum() if self.lowrank is not None else 0.0)


def dec_features(Xg, I, D, n):
    """XI 합의 표준화 감쇠 평균 [n, d] (선형 레인 초기화용, no grad)."""
    lam = math.log(2) / HL; out = []
    with torch.no_grad():
        for s in range(0, n, 128):
            b = torch.arange(s, min(n, s + 128), device=DEV)
            Xs, dt, mask = gather(Xg, I[b].view(-1, K), D[b].view(-1, K))
            w = torch.exp(-lam * dt) * mask.float(); w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
            out.append((Xs * w.unsqueeze(-1)).sum(1).view(len(b), 11, -1).sum(1))
    return torch.cat(out).cpu().numpy().astype(np.float64)


STAGE2 = os.environ.get("SETSTAGE", "0") == "2"; WARMTAG = os.environ.get("WARMTAG", "")   # 2단계: 가산 모델 워밍 스타트 + 집합 블록만 학습
AUGHIST = float(os.environ.get("AUGHIST", 0))   # 이력 절단 증강: 확률 AUGHIST 로 선수 이력을 최근 m 경기(m~U[1,n])만 남긴다 — 짧은 이력 선수의 점수 편향(과소평가) 교정용


def _aug(idx):
    """학습 시에만: idx [n,K] (−1 패딩, 최근이 뒤) → 일부 행의 앞쪽 토큰을 −1 로 지운다."""
    if AUGHIST <= 0: return idx
    n = (idx >= 0).sum(1); pick = (torch.rand(len(idx), device=idx.device) < AUGHIST) & (n > 1)
    if not pick.any(): return idx
    m = (torch.rand(len(idx), device=idx.device) * n.clamp(min=1).float()).floor().long().clamp(min=1)   # 남길 최근 경기 수 m ~ U[1, n]
    cut = (idx.shape[1] - m).unsqueeze(1); pos = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
    return torch.where(pick.unsqueeze(1) & (pos < cut), torch.full_like(idx, -1), idx)


def fit(Xg, TG, S, alpha, n_tr, tr_i, va_i, seed, epochs=None, init=None, pca=None, Xr=None, warm=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Ia, Da, Ib, Db, BASE, y = TG[:6]; CA, CB = (TG[6], TG[7]) if len(TG) > 6 else (None, None)
    net = SetNet(*S)
    if pca is not None: net.set_pca(*pca)
    net = net.to(DEV)
    if warm is not None:                                              # 가산 모델(SET=0) 가중치로 인코더·헤드·기저 초기화
        net.load_state_dict(warm, strict=False)
        if STAGE2:
            for n_, p_ in net.named_parameters():
                p_.requires_grad_(n_.startswith(("sabs", "pma", "g.", "side")))   # 집합 블록만 학습
    if init is not None:                                              # 선형 레인 = 릿지 닫힌 해로 초기화
        coef, icpt, d_base = init
        with torch.no_grad():
            if pca is not None: net.beta.copy_(torch.as_tensor(coef[d_base:], dtype=torch.float32))
            elif len(coef) > d_base: net.lin.weight.copy_(torch.as_tensor(coef[d_base:], dtype=torch.float32)[None, :])
            net.base.weight.copy_(torch.as_tensor(coef[:d_base], dtype=torch.float32)[None, :]); net.base.bias.fill_(float(icpt))
    lin_params = (list(net.lin.parameters()) if pca is None else [net.beta]) + list(net.base.parameters()); lin_ids = {id(p) for p in lin_params}
    def gR(b, I, D):
        if Xr is None: return None
        R_, _, _ = gather(Xr, I[b].view(-1, K), D[b].view(-1, K)); return R_
    lrk_params = [p for p in (net.lowrank.parameters() if net.lowrank is not None else []) if p.requires_grad]; lrk_ids = {id(p) for p in lrk_params}   # 저랭크 선형 레인: 별도 lr·강한 감쇠 (LRK_LR·LRK_WD)
    ENC = ("menc.", "dt.", "att.", "tf.", "tenc."); ENC_LR = os.environ.get("ENC_LR")                       # 사전학습 인코더(pretrain_encoder WARMTAG): 동결(FREEZE_ENC=1) 또는 별도 lr(ENC_LR)
    if warm is not None and os.environ.get("FREEZE_ENC", "0") == "1":
        for n_, p_ in net.named_parameters():
            if n_.startswith(ENC): p_.requires_grad_(False)
    enc_params = [p_ for n_, p_ in net.named_parameters() if n_.startswith(ENC) and p_.requires_grad and id(p_) not in lin_ids and id(p_) not in lrk_ids] if ENC_LR else []; enc_ids = {id(p) for p in enc_params}
    rest = [p for p in net.parameters() if id(p) not in lin_ids and id(p) not in lrk_ids and id(p) not in enc_ids and p.requires_grad]; lin_params = [p for p in lin_params if p.requires_grad]
    groups = ([{"params": lin_params, "lr": float(os.environ.get("LIN_LR", 1e-4)), "weight_decay": 0.0}] if lin_params else []) + ([{"params": lrk_params, "lr": float(os.environ.get("LRK_LR", LR)), "weight_decay": float(os.environ.get("LRK_WD", WD))}] if lrk_params else []) + ([{"params": enc_params, "lr": float(ENC_LR), "weight_decay": WD}] if enc_params else []) + [{"params": rest, "lr": LR, "weight_decay": WD}]
    opt = torch.optim.AdamW(groups)
    def pred(ix):
        out = []
        for s in range(0, len(ix), 64):
            b = torch.as_tensor(ix[s:s + 64], device=DEV)
            Xa, dta, ma = gather(Xg, Ia[b].view(-1, K), Da[b].view(-1, K)); Xb, dtb, mb = gather(Xg, Ib[b].view(-1, K), Db[b].view(-1, K))
            out.append(net.team(Xa, dta, ma, Xb, dtb, mb, BASE[b], gR(b, Ia, Da), gR(b, Ib, Db), None if CA is None else CA[b], None if CB is None else CB[b]))
        return torch.cat(out)
    best, best_ep, pat, state, hist = 1e9, 0, 0, None, []
    for ep in range(epochs or 60):
        net.train(); perm = np.random.permutation(tr_i)
        for s in range(0, len(perm), 32):
            b = torch.as_tensor(perm[s:s + 32], device=DEV); opt.zero_grad()
            Xa, dta, ma = gather(Xg, _aug(Ia[b].view(-1, K)), Da[b].view(-1, K)); Xb, dtb, mb = gather(Xg, _aug(Ib[b].view(-1, K)), Db[b].view(-1, K))
            loss = ((net.team(Xa, dta, ma, Xb, dtb, mb, BASE[b], gR(b, Ia, Da), gR(b, Ib, Db), None if CA is None else CA[b], None if CB is None else CB[b]) - y[b]) ** 2).mean() + (alpha / n_tr) * net.l2()
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if va_i is None: continue
        net.eval()
        with torch.no_grad():
            v = ((pred(va_i) - y[torch.as_tensor(va_i, device=DEV)]) ** 2).mean().item()
            tl = ((pred(tr_i[:len(va_i)]) - y[torch.as_tensor(tr_i[:len(va_i)], device=DEV)]) ** 2).mean().item()
        hist.append((tl, v))
        if v < best - 1e-6: best, best_ep, pat, state = v, ep, 0, {k: t.clone() for k, t in net.state_dict().items()}
        else:
            pat += 1
            if pat >= 10: break
    if state is not None: net.load_state_dict(state)
    net.eval(); return net, best, best_ep + 1, pred, hist


def main():
    g, gpos, X, pid, t_arr, gid_arr, bounds, tidx, zidx, sidx = load_structured()
    R, A, Bm, BASE, IDX = build_sides(); y = R.y.to_numpy(); n = len(R)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])  # noqa: E712
    hm = g.set_index("game_id"); me = np.where(R.home.to_numpy() == 1, hm.home_team_id.reindex(R.game_id), hm.away_team_id.reindex(R.game_id)).astype(int)
    op = np.where(R.home.to_numpy() == 1, hm.away_team_id.reindex(R.game_id), hm.home_team_id.reindex(R.game_id)).astype(int)
    tt = R.t.to_numpy()
    Ia, Da = pack([(p, tt[i]) for i in range(n) for p in st.loc[(int(R.game_id.iloc[i]), int(me[i]))][:11]], bounds, t_arr)
    Ib, Db = pack([(p, tt[i]) for i in range(n) for p in st.loc[(int(R.game_id.iloc[i]), int(op[i]))][:11]], bounds, t_arr)
    if os.environ.get("HISTFROM"):                                    # 이력 토큰을 HISTFROM 시즌 이후 경기로 제한 (그 전 경기는 패딩 −1)
        _sea = g.set_index("game_id").season.reindex(gid_arr).to_numpy().astype(int); _cut = int(os.environ["HISTFROM"])
        for _I in (Ia, Ib): _I[(_I >= 0) & (_sea[np.clip(_I, 0, None)] < _cut)] = -1
        print(f"HISTFROM={_cut}: 이력 토큰 {(Ia >= 0).mean() * 100:.1f}% 남음", flush=True)
    Ia, Da, Ib, Db = [torch.as_tensor(z.reshape(n, 11, K), device=DEV) for z in (Ia, Da, Ib, Db)]
    D_ = A - Bm; Dmu, Dsd = D_.mean(0), D_.std(0).clip(1e-9); Ds = (D_ - Dmu) / Dsd
    Xr_t, keep = None, None
    if LANE == "pca":
        from lineup_event_panel import MINPL, _snapshots
        by, T, dim, FC = _snapshots(gpos)
        P_raw = pd.read_parquet(ROOT / "outputs" / os.environ["ONBALL_FILE"]); P_raw = P_raw[P_raw.game_id.isin(gpos)].copy(); P_raw["t"] = P_raw.game_id.map(gpos)
        P_raw = P_raw.sort_values(["player_id", "t"]).reset_index(drop=True); Xraw = P_raw[FC].to_numpy(np.float32); assert len(Xraw) == len(X)
        st11 = st[st.str.len() == 11]
        def fx(ids, tt_):
            v = np.zeros(dim, np.float32); c = 0
            for p in ids:
                a = T.get(p)
                if a is None: continue
                j = np.searchsorted(a, tt_, "right") - 1
                if j < 0: continue
                v += by[p][j][1]; c += 1
            return v, c
        diffs = []
        for r in g.itertuples(index=False):
            gid = int(r.game_id); ha, aw = int(r.home_team_id), int(r.away_team_id)
            if (gid, ha) not in st11.index or (gid, aw) not in st11.index: continue
            va_, ca = fx(st11.loc[(gid, ha)], gpos[gid]); vb_, cb = fx(st11.loc[(gid, aw)], gpos[gid])
            if min(ca, cb) < MINPL: continue
            diffs.append(va_ - vb_)
        keep = np.vstack(diffs).astype(np.float64).std(0) > 1e-9; assert keep.sum() == A.shape[1]
        Xr_t = torch.as_tensor(Xraw, device=DEV)
    tag = f"{('ssn' + ('' if os.environ.get('TESTSEASONS', '2024,2025') == '2024,2025' else os.environ['TESTSEASONS'].replace(',', '')) + '-') if os.environ.get('FOLDS', 'block') == 'season' else ''}set{int(SET)}-cross{int(CROSS)}-H{H}-L{L}-{LANE}{'-full' if os.environ.get('FULLCH', '0') == '1' else ''}{'-bio' if 'bio' in os.environ['ONBALL_FILE'] else ''}{'-cov' if os.environ.get('TOKCOV', '0') == '1' else ''}{'-tf' + os.environ['MATCHTF'] if os.environ.get('MATCHTF', '0') != '0' else ''}{'-stage2' if STAGE2 else ''}{'-slot' if os.environ.get('SLOTEMB', '0') == '1' else ''}{'-days' if os.environ.get('DTDAYS', '0') == '1' else ''}{('-aug' + os.environ['AUGHIST']) if float(os.environ.get('AUGHIST', 0)) > 0 else ''}{('-lrkwd' + os.environ['LRK_WD']) if os.environ.get('LRK_WD') else ''}{'-zmap' if os.environ.get('ZONEMAP', '0') == '1' else ''}{('-warm' + ('F' if os.environ.get('FREEZE_ENC', '0') == '1' else ('E' + os.environ['ENC_LR'] if os.environ.get('ENC_LR') else ''))) if WARMTAG else ''}{os.environ.get('TUNETAG', '')}-s{NS}"
    n_par = sum(p.numel() for p in SetNet(tidx, zidx, sidx, X.shape[1], BASE.shape[1]).parameters())
    print(f"[{tag}] 사례 {n:,} · 파라미터 {n_par:,} · 입력 열 {X.shape[1]} · lr {LR} wd {WD} drop {DROP} · 장치 {DEV}", flush=True)
    preds = {"linear": np.full(n, np.nan), "set": np.full(n, np.nan)}
    r2 = lambda p, ix: 1 - ((y[ix] - p) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    SEA = R.season.to_numpy().astype(int)
    if os.environ.get("FOLDS", "block") == "season":                  # 평가 시즌 고정: 2024 (학습 ≤2023) · 2025 (학습 ≤2024). 학습 내 검증 = 시간순 마지막 20%
        _tf = int(os.environ.get("TRAINFROM", "0"))                    # TRAINFROM: 학습 대상 경기의 첫 시즌 (기본 0 = 전부)
        FOLDLIST = [(f, np.flatnonzero((SEA < s) & (SEA >= _tf)), np.flatnonzero(SEA == s)) for f, s in enumerate([int(x) for x in os.environ.get("TESTSEASONS", "2024,2025").split(",")], 1)]
    else:
        FOLDLIST = [(f, np.concatenate(IDX[:f]), IDX[f]) for f in range(1, 5)]
    for f, tr, te in FOLDLIST:
        tr = np.sort(tr); te = np.sort(te); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]
        # 기준선: 프로덕션 릿지 (PCA-20)
        pc = PCA(n_components=NPC, random_state=0).fit(Ds[i1]); Zd = pc.transform(Ds); Zd_sd = Zd[i1].std(0).clip(1e-9); Zd = (Zd - Zd[i1].mean(0)) / Zd_sd
        Xl = np.hstack([BASE, Zd]); ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(Xl[i1], y[i1]).predict(Xl[i2]), np.ones(len(i2))))
        preds["linear"][te] = Ridge(alpha=ba).fit(Xl[tr], y[tr]).predict(Xl[te]); lin_val = r2(Ridge(alpha=ba).fit(Xl[i1], y[i1]).predict(Xl[i2]), i2)
        g1 = set(int(x) for x in R.game_id.iloc[i1]); rows1 = np.flatnonzero(np.isin(gid_arr, list(g1)))
        mu, sd = X[rows1].mean(0), X[rows1].std(0).clip(1e-6); Xg = torch.as_tensor(((X - mu) / sd).clip(-5, 5).astype(np.float32), device=DEV)
        # 선형 레인 초기화 = 모델 안 릿지의 닫힌 해. LANE=raw: 표준화 원공간 감쇠 평균의 XI 합차 위 릿지 / LANE=pca: 프로덕션 릿지(PCA-20) 그대로
        if LANE == "pca":
            Xr = Xl; br = ba; pca_args = (pc.components_, Dsd, Zd_sd, keep)
        elif LANE in ("none", "lowrank"):
            Xr = BASE.astype(np.float64); br = ba; pca_args = None      # 기저(시즌·홈)만 릿지로 초기화, 선수 레인은 학습
        else:
            Dd = dec_features(Xg, Ia, Da, n) - dec_features(Xg, Ib, Db, n); Xr = np.hstack([BASE, Dd]); pca_args = None
            br = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(Xr[i1], y[i1]).predict(Xr[i2]), np.ones(len(i2))))
        rr1 = Ridge(alpha=br).fit(Xr[i1], y[i1]); rrt = Ridge(alpha=br).fit(Xr[tr], y[tr]); raw_val = r2(rr1.predict(Xr[i2]), i2); raw_te = r2(rrt.predict(Xr[te]), te)
        Bg = torch.as_tensor(BASE.astype(np.float32), device=DEV); yg = torch.as_tensor(y.astype(np.float32), device=DEV)
        S = (tidx, zidx, sidx, X.shape[1], BASE.shape[1]); TG = (Ia, Da, Ib, Db, Bg, yg)
        if os.environ.get("SLOTEMB", "0") == "1":
            if "CELLIDX" not in globals():
                from gap_lane import declared_grid
                _, PER_, _ = declared_grid(); GKS_ = set(pl[pl.starting_position_name == "GK"].player_id.astype(int))
                def cidx(gid_, tid_, p_):
                    c = PER_.get((int(gid_), int(tid_), int(p_)))
                    return (c[0] * 3 + c[1]) if c is not None else (15 if int(p_) in GKS_ else 16)
                CA_ = np.array([[cidx(R.game_id.iloc[i], me[i], p) for p in st.loc[(int(R.game_id.iloc[i]), int(me[i]))][:11]] for i in range(n)])
                CB_ = np.array([[cidx(R.game_id.iloc[i], op[i], p) for p in st.loc[(int(R.game_id.iloc[i]), int(op[i]))][:11]] for i in range(n)])
                globals()["CELLIDX"] = (torch.as_tensor(CA_, device=DEV), torch.as_tensor(CB_, device=DEV)); print(f"칸 임베딩: 미상 비율 {np.mean(CA_ == 16):.3f}", flush=True)
            TG = TG + CELLIDX
        warm_f = None
        if WARMTAG:
            wf = ROOT / f"outputs/player_encoder_set_{WARMTAG}_fold{f}.pt"
            if wf.exists(): warm_f = torch.load(wf, weights_only=False, map_location="cpu")["nets"]; print(f"  워밍 스타트 {wf.name} ({len(warm_f)} 시드)", flush=True)
        _, v, ep, pr, hist = fit(Xg, TG, S, float(br), len(i1), i1, i2, 0, init=(rr1.coef_, rr1.intercept_, BASE.shape[1]), pca=pca_args, Xr=Xr_t, warm=(warm_f[0] if warm_f else None))
        with torch.no_grad(): vr2 = r2(pr(i2).cpu().numpy(), i2)
        pt = np.zeros(len(te)); nets = []
        for sd_ in range(NS):
            net, _, _, pr, _ = fit(Xg, TG, S, float(br), len(tr), tr, None, sd_, epochs=ep, init=(rrt.coef_, rrt.intercept_, BASE.shape[1]), pca=pca_args, Xr=Xr_t, warm=(warm_f[sd_ % len(warm_f)] if warm_f else None))
            with torch.no_grad(): pt += pr(te).cpu().numpy() / NS
            nets.append({k: t.detach().cpu() for k, t in net.state_dict().items()})
        if os.environ.get("SAVE", "0") == "1":
            torch.save(dict(nets=nets, mu=mu, sd=sd, S=S, pca=pca_args, te_games=sorted(set(int(x) for x in R.game_id.iloc[te])), tag=tag, cfg=dict(SET=SET, CROSS=CROSS, H=H, L=L, HEADS=HEADS, DROP=DROP, LANE=LANE)),
                       ROOT / f"outputs/player_encoder_set_{tag}_fold{f}.pt")
        preds["set"][te] = pt
        curve = " ".join(f"{a:.2f}/{b:.2f}" for a, b in hist[:8]) + " …"
        print(f"폴드 {f}: 릿지(PCA) val {lin_val:+.3f}/test {r2(preds['linear'][te], te):+.3f} · 원공간 릿지(초기값) val {raw_val:+.3f}/test {raw_te:+.3f} · 단일모델 ep={ep} val {vr2:+.3f}/test {r2(pt, te):+.3f} · train/val {curve}", flush=True)
    ix0 = np.flatnonzero(np.isfinite(preds["linear"])); gids = R.game_id.to_numpy(); G = {}
    for i in ix0: G.setdefault(int(gids[i]), []).append(i)
    keys = list(G); rng = np.random.default_rng(0)
    def r2i(pv, ix): return 1 - ((y[ix] - pv[ix]) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    ds = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys), replace=True); ix = np.concatenate([G[keys[j]] for j in pick]); ds.append(r2i(preds["set"], ix) - r2i(preds["linear"], ix))
    ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"\n[{tag}] 기준(릿지 PCA-20) R² {r2i(preds['linear'], ix0):+.4f} · 단일모델 R² {r2i(preds['set'], ix0):+.4f}  Δ={ds.mean():+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ>0)={np.mean(ds > 0):.3f}", flush=True)
    np.savez(ROOT / f"outputs/player_encoder_set_{tag}.npz", **preds, y=y, game_id=gids)


if __name__ == "__main__":
    main()
