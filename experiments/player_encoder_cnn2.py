"""구조 인코더 v2 — v1(player_encoder_cnn.py)의 이상한 부분을 고친 판.

고친 것
  1. 이력 잘림: K=32 → K=96 (릿지 기준선의 전체-이력 감쇠 평균과 정보량을 맞춤)
  2. LayerNorm 제거: 토큰·선수 임베딩의 **크기**(활동량·가치 합)를 보존. 안정화는 0 초기화 머리·표준화 입력으로.
  3. 밴드 풀링 제거: 범주별 (4채널 × 6밴드) 를 그대로 두고 범주 공유 Linear(24→4) → 39×4=156 → 혼합.
     (합성곱은 유지: 범주 공유 Conv1d(4→4,k3) 로 인접 밴드 평활만)
  4. 존 지도 절제: ZONES=1|0
  5. AdamW wd 1e-2, lr 3e-4
판정: 보류 R² Δ vs 릿지(같은 피처 집합), 부트스트랩. RESID=1 이면 릿지 skip 잔차 학습.

실행: ZONES=1 RESID=0 python -m experiments.player_encoder_cnn2
"""
from __future__ import annotations
import os, sys, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp.parquet")
ZONES = os.environ.get("ZONES", "1") == "1"
os.environ["DROPFAM"] = "" if ZONES else "zv,zn"
os.environ["K"] = os.environ.get("K", "96")
import torch, torch.nn as nn, torch.nn.functional as F   # noqa: E402
import torch.utils.checkpoint                                # noqa: E402
from sklearn.linear_model import Ridge              # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from lineup_pipeline import r2w, AL                 # noqa: E402
from lineup_event_panel import HL                   # noqa: E402
from player_encoder import pack, gather, K, DEV     # noqa: E402
from player_encoder_cnn import load_structured, NPC # noqa: E402
os.environ["DROPFAM"] = "" if ZONES else "zv,zn"   # player_encoder_cnn 이 import 시 DROPFAM 을 덮어쓰므로 복원 (릿지 기준선 피처 집합)
from config import VAEP_OUTPUT_DIR                  # noqa: E402

H = int(os.environ.get("H", 16)); RES = os.environ.get("RESID", "0") == "1"


class MatchEncoder2(nn.Module):
    def __init__(self, tidx, zidx, sidx, h=H, drop=float(os.environ.get("DROP", 0.2)), zones=True):
        super().__init__()
        self.register_buffer("tidx", torch.as_tensor(tidx)); self.register_buffer("zidx", torch.as_tensor(zidx)); self.register_buffer("sidx", torch.as_tensor(sidx))
        NC, NB = tidx.shape[1], tidx.shape[2]; self.zones = zones
        self.bandpool = os.environ.get("BANDPOOL", "0") == "1"        # 밴드 축 제거 절제: 6구간을 합쳐 위치 정보를 없앤다 (채널·면은 유지)
        self.conv_x = nn.Conv1d(4, 4, kernel_size=3, padding=1)      # 범주 공유 밴드 평활
        self.percat = nn.Linear(4 * (1 if self.bandpool else NB), 4)  # 범주 공유, 위치 보존 (풀링 없음)
        # 범주 게이트: 62 범주 중 쓸모없는 채널을 모델이 스스로 낮추게 하는 스칼라 게이트 (CATGATE=1, 파라미터 NC 개)
        self.catgate = nn.Parameter(torch.zeros(NC)) if os.environ.get("CATGATE", "0") == "1" else None
        MR = int(os.environ.get("MIXRANK", 0))                         # 저랭크 혼합: NC*4 → r → h (기본 0 = 전결합)
        # 계열별 혼합(FAMMIX=1): 62 범주는 기본 39 + 세분 패스 12 + 리시브 1 + 슛 4 + 수비 6 의 다섯 계열이다.
        # 계열 안에서 먼저 섞고 계열 요약만 합치면, 세분 채널이 기본 채널의 혼합을 흐리지 못한다.
        self.fam = None
        if os.environ.get("FAMMIX", "0") == "1" and NC >= 62:
            bnd = [(0, 39), (39, 51), (51, 52), (52, 56), (56, 62)] + ([(62, NC)] if NC > 62 else []); self.fam = bnd; fw = int(os.environ.get("FAMW", 8))
            self.famlin = nn.ModuleList([nn.Linear((b - a) * 4, fw) for a, b in bnd]); self.mix = nn.Linear(fw * len(bnd), h)
        elif MR: self.mix = nn.Sequential(nn.Linear(NC * 4, MR, bias=False), nn.Linear(MR, h))
        else: self.mix = nn.Linear(NC * 4, h)
        self.noband = os.environ.get("NOBAND", "0") == "1"                                                   # 밴드 텐서 분기 제거(존 지도만)
        if zones:
            cz = int(np.asarray(zidx).shape[0]); hz = 8 if cz <= 2 else int(os.environ.get("ZW", 16)); py, px = [int(x) for x in os.environ.get("ZPOOL", "2,3").split(",")]   # 존 분기 폭·풀링 격자 (튜닝 손잡이)
            self.zshared = os.environ.get("ZSHARED", "0") == "1" and cz > 2                                  # 범주 공유 2-D 필터: (n,16범주,2면,8,12) → 공유 conv → 범주별 4 → 혼합
            if self.zshared:
                self.ncat = cz // 2; self.conv_z = nn.Sequential(nn.Conv2d(2, hz, 3, padding=1), nn.ReLU(), nn.Dropout2d(float(os.environ.get("ZDROP", 0.0))), nn.Conv2d(hz, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((py, px)))
                self.zcat = nn.Linear(8 * py * px, 4); self.zlin = nn.Linear(self.ncat * 4, h)
            else:
                self.conv_z = nn.Sequential(nn.Conv2d(cz, hz, 3, padding=1), nn.ReLU(), nn.Dropout2d(float(os.environ.get("ZDROP", 0.0))), nn.Conv2d(hz, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((py, px)))
                self.zlin = nn.Linear(8 * py * px, h)
        self.slin = nn.Linear(len(sidx), h)
        self.out = nn.Sequential(nn.Dropout(drop), nn.Linear((3 if zones else 2) * h, h))

    def forward(self, X):
        n = X.shape[0]; NC, NB = self.tidx.shape[1], self.tidx.shape[2]
        T = X[:, self.tidx].permute(0, 2, 1, 3).reshape(n * NC, 4, NB)
        if self.bandpool: T = T.sum(-1, keepdim=True); NB = 1         # 밴드 합 — 어디서 했는지를 지운다
        else: T = T + F.relu(self.conv_x(T))                          # 원 신호 + 평활 (잔차)
        pc = F.relu(self.percat(T.reshape(n * NC, 4 * NB))).reshape(n, NC, 4)
        if self.catgate is not None: pc = pc * torch.sigmoid(self.catgate)[None, :, None]
        if self.fam is not None:
            ht = F.relu(self.mix(torch.cat([F.relu(l_(pc[:, a:b].reshape(n, (b - a) * 4))) for l_, (a, b) in zip(self.famlin, self.fam)], 1)))
        else: ht = F.relu(self.mix(pc.reshape(n, NC * 4)))
        parts = [torch.zeros_like(ht) if self.noband else ht, F.relu(self.slin(X[:, self.sidx]))]
        if self.zones:
            Zm = X[:, self.zidx]
            if getattr(self, "zshared", False):
                B_ = Zm.shape[0]; Zc = Zm.view(B_ * self.ncat, 2, Zm.shape[2], Zm.shape[3])                       # 범주 순서: (범주, 면) 인터리브 → 각 범주 2면
                def zbranch(Zc_):                                                                                      # 공유 conv 활성이 n·16범주 만큼 커지므로 (n·K=96 이력) 체크포인트 + 청크로 메모리 절감
                    return torch.cat([F.relu(self.zcat(self.conv_z(c).flatten(1))) for c in Zc_.chunk(4)], 0)
                zc = torch.utils.checkpoint.checkpoint(zbranch, Zc, use_reentrant=False) if (self.training and Zc.requires_grad is not None and os.environ.get("ZCKPT", "1") == "1") else zbranch(Zc)
                parts.append(F.relu(self.zlin(zc.view(B_, -1))))
            else: parts.append(F.relu(self.zlin(self.conv_z(Zm).flatten(1))))
        return self.out(torch.cat(parts, 1))                          # LayerNorm 없음 — 크기 보존


class Net2(nn.Module):
    def __init__(self, tidx, zidx, sidx, d_base, h=H, zones=True):
        super().__init__()
        self.menc = MatchEncoder2(tidx, zidx, sidx, h, zones=zones); self.dt = nn.Linear(1, h)
        self.att = nn.Sequential(nn.Tanh(), nn.Linear(h, 1)); self.head = nn.Linear(h, 1); self.base = nn.Linear(d_base, 1)
        self.lam = math.log(2) / HL
        # SKIP=1: 원공간 감쇠 평균 위의 선형 skip 을 CNN 과 **함께** 학습 (직접판을 end-to-end 잔차 구조로)
        self.skip = None                                              # _ensure_skip 에서 입력 차원을 알고 생성
        self.use_skip = os.environ.get("SKIP", "0") == "1"
        for lin in (self.head, self.base): nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def _ensure_skip(self, d_in, device):
        if self.use_skip and self.skip is None:
            self.skip = nn.Linear(d_in, 1).to(device); nn.init.zeros_(self.skip.weight); nn.init.zeros_(self.skip.bias)

    def player(self, Xs, dt, mask):
        n, k, d = Xs.shape
        tok = self.menc(Xs.reshape(n * k, d)).view(n, k, -1)
        w = torch.exp(-self.lam * dt) * mask.float(); w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
        dec = (tok * w.unsqueeze(-1)).sum(1)                          # 고정 감쇠 평균 (크기 보존)
        sc = self.att(tok + self.dt(torch.log1p(dt).unsqueeze(-1))).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        out = self.head(dec + (tok * sc.unsqueeze(-1)).sum(1)).squeeze(-1)
        if self.use_skip:                                             # 선형 skip: 원공간 감쇠 평균 · w
            out = out + self.skip((Xs * w.unsqueeze(-1)).sum(1)).squeeze(-1)
        return out

    def team(self, Xa, dta, ma, Xb, dtb, mb, base):
        B = base.shape[0]
        return self.player(Xa, dta, ma).view(B, 11).sum(1) - self.player(Xb, dtb, mb).view(B, 11).sum(1) + self.base(base).squeeze(-1)


def fit(Xg, TG, S, tr_i, va_i, seed, epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Ia, Da, Ib, Db, BASE, y = TG
    net = Net2(*S, BASE.shape[1], zones=ZONES).to(DEV); net._ensure_skip(Xg.shape[1], DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=float(os.environ.get("LR", 3e-4)), weight_decay=float(os.environ.get("WD", 1e-2)))
    def pred(ix):
        out = []
        for s in range(0, len(ix), 64):
            b = torch.as_tensor(ix[s:s + 64], device=DEV)
            Xa, dta, ma = gather(Xg, Ia[b].view(-1, K), Da[b].view(-1, K)); Xb, dtb, mb = gather(Xg, Ib[b].view(-1, K), Db[b].view(-1, K))
            out.append(net.team(Xa, dta, ma, Xb, dtb, mb, BASE[b]))
        return torch.cat(out)
    best, best_ep, pat, state, hist = 1e9, 0, 0, None, []
    for ep in range(epochs or 60):
        net.train(); perm = np.random.permutation(tr_i)
        for s in range(0, len(perm), 32):
            b = torch.as_tensor(perm[s:s + 32], device=DEV); opt.zero_grad()
            Xa, dta, ma = gather(Xg, Ia[b].view(-1, K), Da[b].view(-1, K)); Xb, dtb, mb = gather(Xg, Ib[b].view(-1, K), Db[b].view(-1, K))
            loss = ((net.team(Xa, dta, ma, Xb, dtb, mb, BASE[b]) - y[b]) ** 2).mean(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
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
    from nonadditive_gate import build_sides
    g, gpos, X, pid, t_arr, gid_arr, bounds, tidx, zidx, sidx = load_structured()
    R, A, Bm, BASE, IDX = build_sides(); y = R.y.to_numpy(); n = len(R)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(lambda s: [int(x) for x in s])  # noqa: E712
    hm = g.set_index("game_id"); me = np.where(R.home.to_numpy() == 1, hm.home_team_id.reindex(R.game_id), hm.away_team_id.reindex(R.game_id)).astype(int)
    op = np.where(R.home.to_numpy() == 1, hm.away_team_id.reindex(R.game_id), hm.home_team_id.reindex(R.game_id)).astype(int)
    tt = R.t.to_numpy()
    Ia, Da = pack([(p, tt[i]) for i in range(n) for p in st.loc[(int(R.game_id.iloc[i]), int(me[i]))][:11]], bounds, t_arr)
    Ib, Db = pack([(p, tt[i]) for i in range(n) for p in st.loc[(int(R.game_id.iloc[i]), int(op[i]))][:11]], bounds, t_arr)
    cover = float(((Ia >= 0).sum(1) < K).mean())
    Ia, Da, Ib, Db = [torch.as_tensor(z.reshape(n, 11, K), device=DEV) for z in (Ia, Da, Ib, Db)]
    D_ = A - Bm; Ds = (D_ - D_.mean(0)) / D_.std(0).clip(1e-9)
    n_par = sum(p.numel() for p in Net2(tidx, zidx, sidx, BASE.shape[1], zones=ZONES).parameters())
    tag = f"K{K}-H{H}-zones{int(ZONES)}-resid{int(RES)}-s{os.environ.get('SEEDS', 2)}" + ("-skip" if os.environ.get("SKIP", "0") == "1" else "")
    print(f"[{tag}] 사례 {n:,} · 파라미터 {n_par:,} · 이력이 K 안에 다 들어오는 선수-경기 {cover:.1%} · 장치 {DEV}", flush=True)
    preds = {"linear": np.full(n, np.nan), "cnn": np.full(n, np.nan)}
    r2 = lambda p, ix: 1 - ((y[ix] - p) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    for f in range(1, 5):
        tr = np.concatenate(IDX[:f]); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]; te = IDX[f]
        pc = PCA(n_components=NPC, random_state=0).fit(Ds[i1]); Zd = pc.transform(Ds); Zd_sd = Zd[i1].std(0).clip(1e-9); Zd = (Zd - Zd[i1].mean(0)) / Zd_sd
        Xl = np.hstack([BASE, Zd]); ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(Xl[i1], y[i1]).predict(Xl[i2]), np.ones(len(i2))))
        p_i1 = Ridge(alpha=ba).fit(Xl[i1], y[i1]).predict(Xl); p_tr = Ridge(alpha=ba).fit(Xl[tr], y[tr]).predict(Xl)
        preds["linear"][te] = p_tr[te]; lin_val = r2(p_i1[i2], i2)
        g1 = set(int(x) for x in R.game_id.iloc[i1]); rows1 = np.flatnonzero(np.isin(gid_arr, list(g1)))
        mu, sd = X[rows1].mean(0), X[rows1].std(0).clip(1e-6); Xg = torch.as_tensor(((X - mu) / sd).clip(-5, 5).astype(np.float32), device=DEV)
        Bg = torch.as_tensor(BASE.astype(np.float32), device=DEV); S = (tidx, zidx, sidx)
        TG = (Ia, Da, Ib, Db, Bg, torch.as_tensor(((y - p_i1) if RES else y).astype(np.float32), device=DEV))
        _, v, ep, pr, hist = fit(Xg, TG, S, i1, i2, 0)
        with torch.no_grad(): vr2 = r2(pr(i2).cpu().numpy() + (p_i1[i2] if RES else 0), i2)
        TG = (Ia, Da, Ib, Db, Bg, torch.as_tensor(((y - p_tr) if RES else y).astype(np.float32), device=DEV))
        pt = np.zeros(len(te)); NS = int(os.environ.get("SEEDS", 2)); nets = []
        for sd_ in range(NS):
            net, _, _, pr, _ = fit(Xg, TG, S, tr, None, sd_, epochs=ep)
            with torch.no_grad(): pt += pr(te).cpu().numpy() / NS
            nets.append({k: t.detach().cpu() for k, t in net.state_dict().items()})
        if os.environ.get("SAVE", "0") == "1":                       # 배터리 채점용: 폴드별 모델·릿지 원공간 가중치·표준화
            w_raw = (pc.components_.T / Zd_sd) @ Ridge(alpha=ba).fit(Xl[tr], y[tr]).coef_[BASE.shape[1]:]
            torch.save(dict(nets=nets, mu=mu, sd=sd, w_raw=w_raw, Dmu=D_.mean(0), Dsd=D_.std(0).clip(1e-9),
                            te_games=sorted(set(int(x) for x in R.game_id.iloc[te])), resid=RES),
                       ROOT / f"outputs/player_encoder_cnn2_{tag}_fold{f}.pt")
        if RES: pt = pt + p_tr[te]
        preds["cnn"][te] = pt
        curve = " ".join(f"{a:.2f}/{b:.2f}" for a, b in hist[:8]) + " …"
        print(f"폴드 {f}: 릿지 val {lin_val:+.3f}/test {r2(preds['linear'][te], te):+.3f} · v2 ep={ep} val {vr2:+.3f}/test {r2(pt, te):+.3f} · train/val MSE {curve}", flush=True)
    ix0 = np.flatnonzero(np.isfinite(preds["linear"])); gids = R.game_id.to_numpy(); G = {}
    for i in ix0: G.setdefault(int(gids[i]), []).append(i)
    keys = list(G); rng = np.random.default_rng(0)
    def r2i(pv, ix): return 1 - ((y[ix] - pv[ix]) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    ds = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys), replace=True); ix = np.concatenate([G[keys[j]] for j in pick]); ds.append(r2i(preds["cnn"], ix) - r2i(preds["linear"], ix))
    ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"\n[{tag}] 기준(릿지) R² {r2i(preds['linear'], ix0):+.4f} · v2 R² {r2i(preds['cnn'], ix0):+.4f}  Δ={ds.mean():+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ>0)={np.mean(ds > 0):.3f}")
    np.savez(ROOT / f"outputs/player_encoder_cnn2_{tag}.npz", **preds, y=y, game_id=gids)


if __name__ == "__main__":
    main()
