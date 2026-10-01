"""DraftRec (Lee et al., WWW'22) 이식 — **현재 T3 선발 예측 프레임** (multitask_heads DUMP_E 덤프) 위에서.

원 DraftRec 에서 유지: 드래프트 상태(지금까지의 픽)를 Transformer 로 인코딩, 토큰 = [CLS] + 우리 스쿼드 + 상대 토큰,
타입 임베딩 + 픽 여부 상태 임베딩, policy 헤드(다음 픽) + value 헤드(결과 회귀), greedy 자기회귀 디코딩, GK 정확히 1명.
각색: 픽 순서가 없으므로 감독 XI 의 무작위 순열로 학습; 선수 표현 = T3 프레임의 같은 피처(BASE 32 + 점수·출전분·칸 피처 + 선수 벡터 v_p);
상대 토큰 = 상대 스쿼드 중 직전 경기 선발(st_last=1) — 상대의 당일 XI 는 사전 정보가 아니므로 쓰지 않는다.
평가: 시즌 고정(학습 < S, 검증 = 학습 뒤 15%), 정답 11명인 사례의 적중/11 — Table 4 와 같은 사례 집합.
실행: python -m experiments.draftrec_t3  (FRAME=outputs/t3_frame_daysset26.parquet, SEEDS=3)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
FRAME = ROOT / os.environ.get("FRAME", "outputs/t3_frame_daysset26.parquet")
TESTS = [int(x) for x in os.environ.get("TESTSEASONS", "2025,2026").split(",")]
SEEDS = int(os.environ.get("SEEDS", 3)); EPOCHS = int(os.environ.get("EPOCHS", 60)); D_ = 64; XI = 11
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DraftRecXI(nn.Module):
    def __init__(self, fdim, d=D_, nh=4, nl=2, dp=0.1):
        super().__init__()
        self.proj = nn.Linear(fdim, d); self.type_emb = nn.Embedding(3, d); self.state_emb = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.cls, std=.02)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=nh, dim_feedforward=4 * d, dropout=dp, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=nl); self.policy = nn.Linear(d, 1); self.value = nn.Linear(d, 1)

    def forward(self, xo, xp, picked, pado, padp):
        B, N, _ = xo.shape
        h = torch.cat([self.cls.expand(B, 1, -1) + self.type_emb.weight[0], self.proj(xo) + self.type_emb.weight[1] + self.state_emb(picked.long()), self.proj(xp) + self.type_emb.weight[2]], 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=h.device), pado, padp], 1)
        h = self.enc(h, src_key_padding_mask=kpm)
        return self.policy(h[:, 1:1 + N, :]).squeeze(-1), self.value(h[:, 0, :]).squeeze(-1)


def build_cases(E, C, feats):
    """key → (ids, X_own[n,f], xi(정답), X_opp[m,f], y_outcome, date)"""
    E = E.sort_values(["game_id", "team_id"]); grp = {k: v for k, v in E.groupby(["game_id", "team_id"]).indices.items()}
    X = E[feats].to_numpy(np.float32); pid = E.player_id.to_numpy(int); yy = E.y.to_numpy(int); gk = E.is_gk.to_numpy(bool); prev = np.nan_to_num(E.st_last.to_numpy(float)) > 0.5
    cases = {}
    for r in C.itertuples(index=False):
        k = (int(r.gid), int(r.tid)); ko = (int(r.gid), int(r.oid))
        if k not in grp: continue
        ix = grp[k]; xi = pid[ix][yy[ix] == 1]
        if len(xi) != XI or len(ix) < 13: continue
        io = grp.get(ko, np.array([], int)); io = io[prev[io]] if len(io) else io
        cases[k] = dict(key=k, ids=pid[ix], X=X[ix], gk=gk[ix], xi=set(int(p) for p in xi), Xo=X[io] if len(io) else np.zeros((1, X.shape[1]), np.float32), y=float(r.y), season=int(r.season), date=str(r.date))
    return cases


def pack(cs, mu, sd):
    NMAX = max(len(c["ids"]) for c in cs); MMAX = max(len(c["Xo"]) for c in cs); f = cs[0]["X"].shape[1]; B = len(cs)
    xo = np.zeros((B, NMAX, f), np.float32); xp = np.zeros((B, MMAX, f), np.float32); pado = np.ones((B, NMAX), bool); padp = np.ones((B, MMAX), bool); gk = np.zeros((B, NMAX), bool); yv = np.zeros(B, np.float32)
    for b, c in enumerate(cs):
        n, m = len(c["ids"]), len(c["Xo"]); xo[b, :n] = np.clip((c["X"] - mu) / sd, -10, 10); xp[b, :m] = np.clip((c["Xo"] - mu) / sd, -10, 10); pado[b, :n] = False; padp[b, :m] = False; gk[b, :n] = c["gk"]; yv[b] = c["y"]
    T = lambda a: torch.as_tensor(a, device=DEV)  # noqa: E731
    return T(xo), T(xp), T(pado), T(padp), T(gk), T(yv)


def pos_targets(cs, rng, NMAX):
    """감독 XI 무작위 순열 → (picked 상태, 다음 픽 인덱스); NMAX = 패딩 폭(pack 과 동일)"""
    B = len(cs); picked = np.zeros((B, NMAX), bool); tgt = np.zeros(B, int)
    for b, c in enumerate(cs):
        pos = {int(p): q for q, p in enumerate(c["ids"])}; xi = list(c["xi"]); order = rng.permutation(len(xi)); k = int(rng.integers(0, XI))
        for o in order[:k]: picked[b, pos[xi[o]]] = True
        tgt[b] = pos[xi[order[k]]]
    return torch.as_tensor(picked, device=DEV), torch.as_tensor(tgt, device=DEV)


def train(tr, va, mu, sd, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = DraftRecXI(tr[0]["X"].shape[1]).to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xo, xp, pado, padp, gk, yv = pack(tr, mu, sd); ym, ys = float(yv.mean()), float(yv.std() + 1e-6); yv = (yv - ym) / ys
    vxo, vxp, vpado, vpadp, _, _ = pack(va, mu, sd); ce = nn.CrossEntropyLoss(); best, state, pat = 1e9, None, 0; n = len(tr)
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(n)
        for s in range(0, n, 64):
            idx = perm[s:s + 64]; sub = [tr[i] for i in idx]; picked, tgt = pos_targets(sub, rng, xo.shape[1])
            lg, vl = model(xo[idx], xp[idx], picked, pado[idx], padp[idx]); lg = lg.masked_fill(picked | pado[idx], -1e9)
            loss = ce(lg, tgt) + 0.2 * ((vl - yv[idx]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()                                                        # 검증: 고정 시드 순열의 다음 픽 CE
        with torch.no_grad():
            vr = np.random.default_rng(0); tot = 0.0
            for _ in range(4):
                picked, tgt = pos_targets(va, vr, vxo.shape[1]); lg, _ = model(vxo, vxp, picked, vpado, vpadp); tot += ce(lg.masked_fill(picked | vpado, -1e9), tgt).item()
        if tot < best - 1e-5: best, state, pat = tot, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 8: break
    model.load_state_dict(state); model.eval(); return model


@torch.no_grad()
def decode(model, c, mu, sd):
    xo, xp, pado, padp, gk, _ = pack([c], mu, sd); n = len(c["ids"]); picked = torch.zeros_like(pado)
    for step in range(XI):
        lg, _ = model(xo, xp, picked, pado, padp); m = picked | pado
        if bool((picked & gk).any()): m = m | gk
        elif XI - step == 1: m = m | (~gk)
        picked[0, int(lg.masked_fill(m, -1e9)[0].argmax())] = True
    return [int(c["ids"][q]) for q in range(n) if bool(picked[0, q])]


def main():
    E = pd.read_parquet(FRAME); C = pd.read_parquet(str(FRAME).replace(".parquet", "_cases.parquet"))
    feats = [c for c in E.columns if c not in ("game_id", "team_id", "player_id", "season", "y")]
    FE = os.environ.get("FEATS", "all")                                   # all = 48피처+EventXI 벡터 · base = 로스터 사용 피처 32 만(원 DraftRec 에 가까운 입력) · vec = EventXI 벡터+점수만
    if FE == "base": feats = [c for c in feats if not (c.startswith("v") and c[1:].isdigit()) and c not in ("dl_sc", "our_sc", "hist_min", "cell_need", "cell_n", "cell_ratio", "r_sc", "r_min", "r_st", "r_sc_t", "r_min_t", "r_st_t")]
    if FE == "vec": feats = [c for c in feats if (c.startswith("v") and c[1:].isdigit()) or c in ("dl_sc", "is_gk", "st_last")]
    E[feats] = E[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cases = build_cases(E, C, feats); print(f"프레임 {FRAME.name} · 피처 {len(feats)} · 사례 {len(cases):,}", flush=True)
    allc = sorted(cases.values(), key=lambda c: c["date"]); res = {}
    for S in TESTS:
        tr_all = [c for c in allc if c["season"] < S]; te = [c for c in allc if c["season"] == S]
        if not te: continue
        cut = int(0.85 * len(tr_all)); tr, va = tr_all[:cut], tr_all[cut:]
        Xtr = np.vstack([c["X"] for c in tr]); mu, sd = Xtr.mean(0), Xtr.std(0).clip(1e-6)
        hits = np.zeros(len(te))
        for seed in range(SEEDS):
            model = train(tr, va, mu, sd, seed); h = np.array([len(set(decode(model, c, mu, sd)) & c["xi"]) for c in te], float); hits += h / SEEDS
            print(f"  시즌 {S} 시드 {seed}: 적중 {h.mean():.3f}/11 (n={len(te)})", flush=True)
        res[S] = hits
    allh = np.concatenate([res[S] for S in res])
    print(f"\nDraftRec (T3 프레임, {SEEDS}시드 평균) {allh.mean():.3f}/11  n={len(allh):,}  " + " · ".join(f"{S}: {res[S].mean():.3f}" for S in res), flush=True)
    pd.to_pickle({c["key"]: float(h) for S in res for c, h in zip([c for c in allc if c["season"] == S], res[S])}, ROOT / f"outputs/draftrec_t3_hits{'' if FE == 'all' else '_' + FE}.pkl")


if __name__ == "__main__":
    main()
