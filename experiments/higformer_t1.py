"""HIGFormer (Wang et al., KDD'25) 재현 — 현재 프레임(시즌 고정 규약)의 T1 결과 예측 + T2 교체 탐색 추천.

원 논문에서 그대로 옮긴 것
  · 경기별 이종 선수 상호작용 그래프: 노드 = 출전 선수(양 팀 선발 11+11), 노드 피처 = 핵심 이벤트 카운트(10종),
    엣지 = 패스 계열(u→v 성공 패스 횟수, 같은 팀) · 수비 계열(u 의 태클/인터셉트/파울/클리어 직전 행위자 v, 상대 팀), 엣지 피처 = 횟수
  · Heter. Transformer (TokenGT): |V| 노드 토큰 + |E| 엣지 토큰 + [class], 라플라시안 고유벡터 노드 식별자(P_u, P_v; 엣지는 P_u−P_v 로 방향),
    노드·엣지 타입 임베딩, 표준 Transformer 3층 d=64 → 노드 임베딩 16
  · Heter. GCN: 엣지 타입별 가중치 W^{l,k} 를 갖는 GAT 3층 + self-loop → 노드 임베딩 16
  · 2단계 학습: 1단계 — 두 전문가를 경기 그래프 → 결과 손실로 각각 사전학습(class 토큰 / 평균 풀링). 2단계 — 선수 임베딩은 직전 T=10 경기 평균(동결),
    MoE 게이트(선수 피처 → 전문가 가중)로 global·local 융합, Team Interaction Network(과거 승률 방향 그래프 GAT + 학습 팀 임베딩),
    Match Comparison Transformer(1층) → 팀별 평균 풀링 → MLP(r − b)
  · 결과 손실 MSE
바꾼 것(논문에 명시)
  · 결과 = 무페널티 xG 차(회귀) — 본 논문의 모든 T1 모델과 동일 타깃 · 평가 = 시즌 고정 보류 R² (팀-경기)
  · 사전 정보만: 2단계의 선수 임베딩·게이트 입력은 해당 경기 이전 경기들에서만, 상대 XI 도 실제 선발(T1 은 양 팀 XI 조건부이므로 우리 T1 과 동일)
  · 노드는 23명이 아니라 선발 11명(T1 이 XI 조건부이므로)
T2: 감독 XI 에서 시작해 자격(과거 출전 단) 있는 벤치 선수와 1명씩 교체하며 HIGFormer 예측을 최대화(§4.5 교체 재채점의 탐색판, 최대 3회).
    격차 = 예측(추천) − 예측(감독), 위약 = 예측(직전 XI) − 예측(감독), 모양은 감독 선언 유지.
실행: TESTSEASONS=2024,2025 python -m experiments.higformer_t1   (T2=1 이면 추천 덤프도)
"""
from __future__ import annotations
import os, sys, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TESTS = [int(x) for x in os.environ.get("TESTSEASONS", "2024,2025").split(",")]
SEEDS = int(os.environ.get("SEEDS", 5)); T_HIST = int(os.environ.get("THIST", 10)); D = 64; DO = 16; NF = 10; KE = 2; KEIG = 4
PAT1 = int(os.environ.get("PAT1", 12)); LR2 = float(os.environ.get("LR2", 1e-3)); WD2 = float(os.environ.get("WD2", 1e-2)); DROP2 = float(os.environ.get("DROP2", 0.3))   # 2025 폴드 검증 R² 로 선택(테스트 미참조)
CACHE = ROOT / "outputs/higformer_graphs.pkl"
# SPADL 유형 → 핵심 이벤트 10종 (pass, cross, set-piece pass, dribble/take-on, foul, tackle, interception, shot, keeper, clearance/bad touch)
GRP = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 22: 2, 7: 3, 21: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 7, 13: 7, 14: 8, 15: 8, 16: 8, 17: 8, 18: 9, 19: 9}
PASS = {0, 1, 2, 3, 4, 5, 6, 22}; DEF = {8, 9, 10, 18}


# ── 그래프 구축 ────────────────────────────────────────────────────────────────
def build_graphs():
    if CACHE.exists(): return pd.read_pickle(CACHE)
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    M = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_merge.json")).items()}
    M26 = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_merge_2026.json")).items()}
    B26 = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_bridge_2026.json")).items()}
    G26 = set(g[g.season >= 2026].game_id.astype(int))
    def cid(p, gid):
        if gid in G26: q = B26.get(p, p); return M26.get(q, q)
        return M.get(p, p)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv"); st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])  # noqa: E712
    npx = pd.read_parquet(ROOT / "outputs/team_npxg.parquet").set_index(["game_id", "team_id"]).npxg
    S = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet", columns=["game_id", "period_id", "time_seconds", "action_id", "team_id", "player_id", "type_id", "result_id"]).dropna(subset=["player_id"])
    S = S.sort_values(["game_id", "period_id", "time_seconds", "action_id"]); out = {}
    for gid, A in S.groupby("game_id", sort=False):
        gid = int(gid); r = g[g.game_id == gid]
        if not len(r): continue
        r = r.iloc[0]; h, a = int(r.home_team_id), int(r.away_team_id)
        if (gid, h) not in st.index or (gid, a) not in st.index or len(st[(gid, h)]) != 11 or len(st[(gid, a)]) != 11: continue
        if (gid, h) not in npx.index or (gid, a) not in npx.index: continue
        nodes = list(st[(gid, h)]) + list(st[(gid, a)]); pos = {p: i for i, p in enumerate(nodes)}
        X = np.zeros((22, NF), np.float32); ep = defaultdict(float); ed = defaultdict(float)
        pid = np.array([cid(int(p), gid) for p in A.player_id]); tid = A.team_id.to_numpy(int); typ = A.type_id.to_numpy(int); res = A.result_id.to_numpy(int)
        for i in range(len(pid)):
            u = pos.get(int(pid[i]))
            if u is not None and int(typ[i]) in GRP: X[u, GRP[int(typ[i])]] += 1
            if i + 1 < len(pid) and int(typ[i]) in PASS and int(res[i]) == 1 and tid[i + 1] == tid[i]:
                v = pos.get(int(pid[i + 1]))
                if u is not None and v is not None and u != v: ep[(u, v)] += 1
            if i > 0 and int(typ[i]) in DEF and tid[i - 1] != tid[i]:
                v = pos.get(int(pid[i - 1]))
                if u is not None and v is not None: ed[(u, v)] += 1
        E = [(u, v, 0, c) for (u, v), c in ep.items()] + [(u, v, 1, c) for (u, v), c in ed.items()]
        out[gid] = dict(nodes=nodes, home=h, away=a, X=np.log1p(X), E=E, y=float(npx[(gid, h)] - npx[(gid, a)]), date=r.game_date, season=int(r.season))
    pd.to_pickle(out, CACHE); print(f"그래프 {len(out):,} 경기 캐시 → {CACHE.name}", flush=True); return out


def eig_ids(E, n=22, k=KEIG):
    A = np.zeros((n, n))
    for u, v, _, c in E: A[u, v] += c; A[v, u] += c
    d = A.sum(1); Dm = np.diag(1 / np.sqrt(np.maximum(d, 1e-6))); L = np.eye(n) - Dm @ A @ Dm
    w, V = np.linalg.eigh(L); return V[:, 1:k + 1].astype(np.float32)


def tensors(gs, keys, Emax=None):
    """경기 목록 → 패딩 텐서 (X, ntype, eig, edge tokens, dense adjacency per type, y)"""
    B = len(keys); Emax = Emax or max(1, max(len(gs[k]["E"]) for k in keys))
    X = np.zeros((B, 22, NF), np.float32); P = np.zeros((B, 22, KEIG), np.float32); NT = np.zeros((B, 22), np.int64); NT[:, 11:] = 1
    EU = np.zeros((B, Emax), np.int64); EV = np.zeros((B, Emax), np.int64); ET = np.zeros((B, Emax), np.int64); EW = np.zeros((B, Emax), np.float32); EM = np.ones((B, Emax), bool)
    AD = np.zeros((B, KE, 22, 22), np.float32); y = np.zeros(B, np.float32)
    for b, k in enumerate(keys):
        d = gs[k]; X[b] = d["X"]; P[b] = d.get("eig") if d.get("eig") is not None else eig_ids(d["E"]); y[b] = d["y"]
        for j, (u, v, t, c) in enumerate(d["E"][:Emax]): EU[b, j] = u; EV[b, j] = v; ET[b, j] = t; EW[b, j] = math.log1p(c); EM[b, j] = False; AD[b, t, v, u] = math.log1p(c)   # 메시지 u→v: 수신 v 행
    T = lambda a: torch.as_tensor(a, device=DEV)  # noqa: E731
    return T(X), T(NT), T(P), T(EU), T(EV), T(ET), T(EW), T(EM), T(AD), T(y)


# ── 1단계 전문가 ───────────────────────────────────────────────────────────────
class HeterTransformer(nn.Module):
    """TokenGT 식: 노드 토큰 + 엣지 토큰 + class 토큰, 라플라시안 식별자·타입 임베딩."""
    def __init__(self, nf=NF, d=D, do=DO, layers=3, heads=4, drop=0.1):
        super().__init__(); self.xn = nn.Linear(nf, d); self.xe = nn.Linear(1, d); self.pid = nn.Linear(KEIG, d); self.tn = nn.Embedding(2, d); self.te = nn.Embedding(KE, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.cls, std=.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True), layers); self.out = nn.Linear(d, do); self.head = nn.Linear(d, 1)
    def forward(self, X, NT, P, EU, EV, ET, EW, EM, rand_sign=False):
        B = X.shape[0]; Pp = P * (torch.randint(0, 2, (B, 1, KEIG), device=P.device) * 2 - 1) if rand_sign else P; Pn = self.pid(Pp)
        hn = self.xn(X) + Pn + self.tn(NT); idx = lambda M: torch.gather(Pn, 1, M.unsqueeze(-1).expand(-1, -1, Pn.shape[-1]))  # noqa: E731
        he = self.xe(EW.unsqueeze(-1)) + idx(EU) - idx(EV) + self.te(ET)
        h = torch.cat([self.cls.expand(B, 1, -1), hn, he], 1); m = torch.cat([torch.zeros(B, 1 + 22, dtype=torch.bool, device=X.device), EM], 1)
        h = self.enc(h, src_key_padding_mask=m); return self.out(h[:, 1:23]), self.head(h[:, 0]).squeeze(-1)


class HeterGATLayer(nn.Module):
    def __init__(self, din, dout, heads=4):
        super().__init__(); self.W = nn.ModuleList([nn.Linear(din, dout, bias=False) for _ in range(KE)]); self.a1 = nn.ModuleList([nn.Linear(dout, heads, bias=False) for _ in range(KE)]); self.a2 = nn.ModuleList([nn.Linear(dout, heads, bias=False) for _ in range(KE)])
        self.wb = nn.Parameter(torch.zeros(KE, heads)); self.self = nn.Linear(din, dout); self.h = heads; self.dout = dout
    def forward(self, x, AD):
        out = self.self(x)
        for k in range(KE):
            h = self.W[k](x); s = self.a1[k](h).unsqueeze(2) + self.a2[k](h).unsqueeze(1)                        # (B,N,N,H): 수신 i, 송신 j
            s = torch.nn.functional.leaky_relu(s, 0.2) + AD[:, k].unsqueeze(-1) * self.wb[k]; mask = AD[:, k] > 0
            s = s.masked_fill(~mask.unsqueeze(-1), -1e9); al = torch.softmax(s, 2) * mask.unsqueeze(-1)            # 이웃 없으면 0
            hh = h.view(h.shape[0], h.shape[1], self.h, -1)                                                          # (B,N,H,d/H)
            out = out + torch.einsum("bijh,bjhc->bihc", al, hh).reshape(h.shape)
        return torch.nn.functional.elu(out)


class HeterGCN(nn.Module):
    def __init__(self, nf=NF, d=D, do=DO, layers=3):
        super().__init__(); self.inp = nn.Linear(nf, d); self.layers = nn.ModuleList([HeterGATLayer(d, d) for _ in range(layers)]); self.out = nn.Linear(d, do); self.head = nn.Linear(d, 1); self.drop = nn.Dropout(0.1)
    def forward(self, X, NT, AD):
        h = self.inp(X)
        for l in self.layers: h = self.drop(l(h, AD))
        return self.out(h), self.head(h.mean(1)).squeeze(-1)


def fit_stage1(gs, tr, va, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); Emax = max(len(gs[k]["E"]) for k in tr + va)
    ym = np.mean([gs[k]["y"] for k in tr]); ys = np.std([gs[k]["y"] for k in tr]) + 1e-6
    models = {}
    for name in ("glo", "loc"):
        net = (HeterTransformer() if name == "glo" else HeterGCN()).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); best, state, pat = 1e9, None, 0
        VA = tensors(gs, va, Emax)
        for ep in range(60):
            net.train(); perm = rng.permutation(len(tr))
            for s in range(0, len(perm), 32):
                keys = [tr[i] for i in perm[s:s + 32]]; X, NT, P, EU, EV, ET, EW, EM, AD, y = tensors(gs, keys, Emax)
                _, pr = net(X, NT, P, EU, EV, ET, EW, EM, rand_sign=True) if name == "glo" else net(X, NT, AD)
                loss = ((pr - (y - ym) / ys) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                X, NT, P, EU, EV, ET, EW, EM, AD, y = VA; _, pr = net(X, NT, P, EU, EV, ET, EW, EM) if name == "glo" else net(X, NT, AD); v = ((pr - (y - ym) / ys) ** 2).mean().item()
            if v < best - 1e-5: best, state, pat = v, {k_: t.clone() for k_, t in net.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= PAT1: break
        net.load_state_dict(state); net.eval(); models[name] = net
    return models


@torch.no_grad()
def embed_matches(gs, keys, models):
    """경기별 (선수 → glo·loc 임베딩) — 동결 1단계 전문가"""
    Z = {}; Emax = max(len(gs[k]["E"]) for k in keys)
    for s in range(0, len(keys), 64):
        kk = keys[s:s + 64]; X, NT, P, EU, EV, ET, EW, EM, AD, _ = tensors(gs, kk, Emax)
        zg, _ = models["glo"](X, NT, P, EU, EV, ET, EW, EM); zl, _ = models["loc"](X, NT, AD)
        for b, k in enumerate(kk): Z[k] = (zg[b].cpu().numpy(), zl[b].cpu().numpy())
    return Z


# ── 2단계 ─────────────────────────────────────────────────────────────────────
class TeamNet(nn.Module):
    """Team Interaction Network: 학습 팀 임베딩 + 승률 방향 그래프 GAT(1층)"""
    def __init__(self, nteam, d=D, do=DO):
        super().__init__(); self.emb = nn.Embedding(nteam + 1, d); self.W = nn.Linear(d, d, bias=False); self.a1 = nn.Linear(d, 1, bias=False); self.a2 = nn.Linear(d, 1, bias=False); self.wb = nn.Parameter(torch.zeros(1)); self.out = nn.Linear(d, do)
    def forward(self, AT):
        x = self.emb.weight; h = self.W(x); s = torch.nn.functional.leaky_relu(self.a1(h) + self.a2(h).T, 0.2) + AT * self.wb; s = s.masked_fill(AT <= 0, -1e9)
        al = torch.softmax(s, 1) * (AT > 0); return self.out(torch.nn.functional.elu(x + al @ h))


class Stage2(nn.Module):
    def __init__(self, nteam, d=32):
        super().__init__(); self.gate = nn.Sequential(nn.Linear(NF, 32), nn.ReLU(), nn.Linear(32, 2)); self.team = TeamNet(nteam); self.proj = nn.Linear(DO, d); self.tproj = nn.Linear(DO, d); self.side = nn.Embedding(2, d); self.miss = nn.Parameter(torch.zeros(DO))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, DROP2, batch_first=True, norm_first=True), 1); self.mlp = nn.Sequential(nn.Dropout(DROP2), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
    def forward(self, ZG, ZL, F, HAS, TI, AT, NT):
        w = torch.softmax(self.gate(F), -1); z = w[..., :1] * ZG + w[..., 1:] * ZL; z = torch.where(HAS.unsqueeze(-1), z, self.miss.expand_as(z))
        te = self.team(AT)[TI]                                                                                  # (B,22,DO)
        h = self.enc(self.proj(z) + self.tproj(te) + self.side(NT)); r = h[:, :11].mean(1); b = h[:, 11:].mean(1); return self.mlp(r - b).squeeze(-1)


def team_graph(gs, tr_keys, tid2i):
    n = len(tid2i) + 1; W = np.zeros((n, n)); C = np.zeros((n, n))
    for k in tr_keys:
        d = gs[k]; i, j = tid2i[d["home"]], tid2i[d["away"]]; C[i, j] += 1; C[j, i] += 1
        if d["y"] > 0: W[i, j] += 1
        elif d["y"] < 0: W[j, i] += 1
        else: W[i, j] += .5; W[j, i] += .5
    R = np.where(C > 0, W / np.maximum(C, 1), 0); AT = np.where(R > 0.5, R, 0.0); return torch.as_tensor(AT, dtype=torch.float32, device=DEV)


class HistIndex:
    """선수별 (날짜, 경기키) 시계열 → 직전 T 경기 임베딩 평균·피처 평균"""
    def __init__(self, gs, Z, keys):
        self.by = defaultdict(list); self.Z = Z; self.gs = gs
        for k in sorted(keys, key=lambda k: gs[k]["date"]):
            for i, p in enumerate(gs[k]["nodes"]): self.by[p].append((gs[k]["date"], k, i))
        self.dates = {p: np.array([d for d, _, _ in v], dtype="datetime64[ns]") for p, v in self.by.items()}
    def get(self, p, date):
        v = self.by.get(p)
        if not v: return None
        j = np.searchsorted(self.dates[p], np.datetime64(date), "left")
        if j == 0: return None
        sel = v[max(0, j - T_HIST):j]; zg = np.mean([self.Z[k][0][i] for _, k, i in sel if k in self.Z], 0); zl = np.mean([self.Z[k][1][i] for _, k, i in sel if k in self.Z], 0); f = np.mean([self.gs[k]["X"][i] for _, k, i in sel], 0)
        return zg, zl, f


def stage2_inputs(gs, keys, H, tid2i, lineup=None):
    B = len(keys); ZG = np.zeros((B, 22, DO), np.float32); ZL = np.zeros((B, 22, DO), np.float32); F = np.zeros((B, 22, NF), np.float32); HAS = np.zeros((B, 22), bool); TI = np.zeros((B, 22), np.int64); NT = np.zeros((B, 22), np.int64); NT[:, 11:] = 1; y = np.zeros(B, np.float32)
    for b, k in enumerate(keys):
        d = gs[k]; nodes = lineup[b] if lineup is not None else d["nodes"]; y[b] = d["y"]
        for i, p in enumerate(nodes):
            TI[b, i] = tid2i.get(d["home"] if i < 11 else d["away"], len(tid2i)); h = H.get(p, d["date"])
            if h is not None: ZG[b, i], ZL[b, i], F[b, i] = h; HAS[b, i] = True
    T = lambda a: torch.as_tensor(a, device=DEV)  # noqa: E731
    return T(ZG), T(ZL), T(F), T(HAS), T(TI), T(NT), T(y)


def fit_stage2(gs, tr, va, H, tid2i, AT, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = Stage2(len(tid2i)).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=LR2, weight_decay=WD2)
    ym = np.mean([gs[k]["y"] for k in tr]); ys = np.std([gs[k]["y"] for k in tr]) + 1e-6; TR = stage2_inputs(gs, tr, H, tid2i); VA = stage2_inputs(gs, va, H, tid2i); best, state, pat = 1e9, None, 0
    for ep in range(100):
        net.train(); perm = torch.as_tensor(rng.permutation(len(tr)), device=DEV)
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]; pr = net(TR[0][b], TR[1][b], TR[2][b], TR[3][b], TR[4][b], AT, TR[5][b]); loss = ((pr - (TR[6][b] - ym) / ys) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = ((net(VA[0], VA[1], VA[2], VA[3], VA[4], AT, VA[5]) - (VA[6] - ym) / ys) ** 2).mean().item()
        if v < best - 1e-5: best, state, pat = v, {k_: t.clone() for k_, t in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 10: break
    net.load_state_dict(state); net.eval(); return net, ym, ys


@torch.no_grad()
def predict(net, ym, ys, gs, keys, H, tid2i, AT, lineup=None):
    out = []
    for s in range(0, len(keys), 256):
        I = stage2_inputs(gs, keys[s:s + 256], H, tid2i, None if lineup is None else lineup[s:s + 256]); out.append((net(I[0], I[1], I[2], I[3], I[4], AT, I[5]) * ys + ym).cpu().numpy())
    return np.concatenate(out)


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    gs = build_graphs(); keys_all = sorted(gs, key=lambda k: gs[k]["date"]); sea = {k: gs[k]["season"] for k in keys_all}
    teams = sorted({gs[k]["home"] for k in keys_all} | {gs[k]["away"] for k in keys_all}); tid2i = {t: i for i, t in enumerate(teams)}
    print(f"경기 {len(keys_all):,} · 시즌 {sorted(set(sea.values()))} · 팀 {len(teams)} · 테스트 {TESTS}", flush=True)
    preds = {}; models_by_S = {}
    for S in TESTS:
        tr_all = [k for k in keys_all if sea[k] < S]; te = [k for k in keys_all if sea[k] == S]; cut = int(0.85 * len(tr_all)); tr, va = tr_all[:cut], tr_all[cut:]
        AT = team_graph(gs, tr_all, tid2i); P = np.zeros(len(te)); nets = []
        for seed in range(SEEDS):
            m1 = fit_stage1(gs, tr, va, seed); Z = embed_matches(gs, tr_all + te, m1); H = HistIndex(gs, Z, tr_all + te)
            net, ym, ys = fit_stage2(gs, tr, va, H, tid2i, AT, seed); p = predict(net, ym, ys, gs, te, H, tid2i, AT); P += p / SEEDS; nets.append((net, ym, ys, H))
            print(f"  [{S}] 시드 {seed}: 1단계 완료 · 2단계 R² {r2(np.array([gs[k]['y'] for k in te]), p):+.4f}", flush=True)
        for k, p in zip(te, P): preds[k] = p
        models_by_S[S] = (nets, AT); print(f"[{S}] HIGFormer R² (경기, {SEEDS}시드 평균) {r2(np.array([gs[k]['y'] for k in te]), P):+.4f} (n={len(te)})", flush=True)
    # 팀-경기 R² (양 관점: 원정 = 부호 반전) — 본문 T1 과 같은 단위
    y = np.array([gs[k]["y"] for k in preds] + [-gs[k]["y"] for k in preds]); p = np.array([preds[k] for k in preds] + [-preds[k] for k in preds])
    per = " · ".join(f"{S}: {r2(np.array([gs[k]['y'] for k in preds if sea[k] == S] + [-gs[k]['y'] for k in preds if sea[k] == S]), np.array([preds[k] for k in preds if sea[k] == S] + [-preds[k] for k in preds if sea[k] == S])):+.4f}" for S in TESTS)
    print(f"\n[HIGFormer {'·'.join(map(str, TESTS))}] 팀-경기 R² {r2(y, p):+.4f} (n={len(y):,})  ({per})", flush=True)
    pd.to_pickle(preds, ROOT / f"outputs/higformer_preds_{''.join(map(str, TESTS))}.pkl")
    if os.environ.get("T2") == "1": t2(gs, models_by_S, tid2i)


def t2(gs, models_by_S, tid2i):
    """교체 탐색 추천 (§4.5 재채점의 탐색판) → pool_eval 덤프"""
    from gap_diagnose import absences
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv"); st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])  # noqa: E712
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date); g = g.sort_values("game_date").reset_index(drop=True); gpos = {int(r.game_id): i for i, r in g.iterrows()}; AB = absences(gpos)
    prev_xi = {}; last = {}
    for r in g.itertuples(index=False):
        for t in (int(r.home_team_id), int(r.away_team_id)):
            prev_xi[(int(r.game_id), t)] = last.get(t)
            if (int(r.game_id), t) in st.index and len(st[(int(r.game_id), t)]) == 11: last[t] = list(st[(int(r.game_id), t)])
    for S, (nets, AT) in models_by_S.items():
        A = pd.read_pickle(ROOT / os.environ.get("PROD", "outputs/gap_soft_fullonball_gk_resid_merged_daysset25_pc20_all_npxg_nozvzn_fixsd.pkl")); A = A[A.season == S].reset_index(drop=True)
        rows = []; shapes = []; cshape = []; recs = []
        def lab(G): return "-".join(str(int(x)) for x in np.asarray(G).sum(1) if x > 0)
        for r in A.itertuples(index=False):
            gid, tid, oid = int(r.gid), int(r.tid), int(r.oid)
            if gid not in gs: continue
            d = gs[gid]; home = d["home"] == tid; opp = prev_xi.get((gid, oid)) or (d["nodes"][11:] if home else d["nodes"][:11])
            coach = [int(p) for p in r.coach_xi]; gks = set(int(p) for p in r.gks); bench = [int(p) for p in r.pool if int(p) not in set(coach)]
            def mk(xi): return (list(xi) + list(opp)) if home else (list(opp) + list(xi))
            def score(xis):
                P = np.concatenate([predict(net, ym, ys, gs, [gid] * len(xis), H, tid2i, AT, lineup=[mk(x) for x in xis]) for net, ym, ys, H in nets[:1]]) if len(xis) else np.array([])
                return P if home else -P
            base = float(score([coach])[0]); cur = list(coach); cur_v = base
            for _ in range(3):
                cands = []
                for u in cur:
                    if u in gks: continue
                    lv = int(np.argmax(r.qmap[u])) if r.qmap.get(u) is not None else 2
                    for p in bench:
                        if p in gks or p in cur: continue
                        q = r.qmap.get(p)
                        if q is None or q[lv] <= 0.05: continue
                        cands.append([p if x == u else x for x in cur])
                if not cands: break
                v = score(cands); j = int(np.argmax(v))
                if v[j] > cur_v + 1e-6: cur, cur_v = cands[j], float(v[j])
                else: break
            plc = float(score([list(r.prev)])[0]) - base if r.prev and len(r.prev) == 11 and set(r.prev) <= set(int(p) for p in r.pool) else np.nan
            rows.append(dict(gid=gid, tid=tid, season=S, y=float(r.y), gg=cur_v - base, hit=len(set(cur) & set(coach)), same=True, plc=plc, absent=AB.get((gid, tid), (np.nan,))[0])); shapes.append(lab(r.G)); cshape.append(lab(r.G))
            recs.append(dict(gid=gid, tid=tid, season=S, rec=[int(x) for x in cur], coach=[int(x) for x in coach], sc={int(p): 0.0 for p in r.pool}))   # 교체 정렬용 추천 XI 덤프
        if os.environ.get("SAVEREC"): pd.to_pickle(recs, os.environ["SAVEREC"].replace("SEASON", str(S)))
        D = pd.DataFrame(rows); out = ROOT / f"outputs/eval_higformer_{S}.pkl"; pd.to_pickle(dict(D=D, nulls=None, shapes=shapes, cshape=cshape), out)
        print(f"[HIGFormer T2 {S}] 사례 {len(D):,} · 적중 {D.hit.mean():.3f} · 교체 0회 {np.mean(D.hit == 11) * 100:.1f}% → {out.name}", flush=True)


if __name__ == "__main__":
    main()
