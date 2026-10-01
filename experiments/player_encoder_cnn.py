"""구조 반영 인코더 — 682열을 평탄화하지 않고 피처의 실제 격자 구조에 맞춘 귀납 편향으로 임베딩.

선수-경기 벡터의 실제 구조 (onball_vaep_event.py):
  · tv/to/td/tn : (4 값채널) × (NC 액션 범주 = type_result) × (NB x-구간, 수비→공격)  — 1-D 공간축
  · zv/zn       : (2 채널: 가치·횟수) × 12×8 피치 격자                                  — 2-D 공간 지도
  · gk 21 · dr 6: 스칼라
인코더 (선수-경기 → 토큰)
  · 범주-x 블록: 4채널 1-D 합성곱(커널 3)을 **모든 범주가 공유** → x 풀링(평균·최대) → 범주 혼합 선형
                 ("가치가 피치 어디에 분포하는가"의 필터를 액션 유형 간 공유, 어떤 유형이 중요한지는 혼합에서 학습)
  · 존 블록:     2채널 2-D 합성곱 2층(커널 3) → 적응 평균풀링(2×3) → 선형
  · 스칼라:      선형
  · 결합 → LayerNorm → 토큰 d
선수 임베딩 = 과거 K경기 토큰의 [고정 감쇠 평균 + 학습 어텐션 풀링] → LayerNorm
팀 머리 = Σ우리 s_p − Σ상대 s_p + 기저 (가산)
학습: lr 1e-4, wd 1e-1, 드롭아웃, 매 에폭 검증 체크포인트, 폴드별 최적 에폭으로 시드 2 재적합.
판정: 보류 테스트 R² Δ vs 현행 릿지(부트스트랩). 입력 PCA 없음(사용자 지시).

실행: python -m experiments.player_encoder_cnn   [H=16]
"""
from __future__ import annotations
import os, sys, math, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp.parquet")
os.environ["DROPFAM"] = ""                     # 존 지도(zv/zn)까지 전부 사용 — CNN 입력
import torch, torch.nn as nn, torch.nn.functional as F   # noqa: E402
from sklearn.linear_model import Ridge              # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from lineup_pipeline import r2w, AL                 # noqa: E402
from lineup_event_panel import HL, FEAT_FILE        # noqa: E402
from player_encoder import pack, gather, K, DEV     # noqa: E402
from config import VAEP_OUTPUT_DIR                  # noqa: E402

H = int(os.environ.get("H", 16)); NPC = 20; NZX, NZY = 12, 8


def load_structured():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    import player_encoder as _pe; _pe.DAYIDX = ((g.game_date - pd.Timestamp("2020-01-01")).dt.days).to_numpy(np.float32)   # 경기 순번 → 일수 (DTDAYS 용)
    P = pd.read_parquet(FEAT_FILE); P = P[P.game_id.isin(gpos)].copy(); P["t"] = P.game_id.map(gpos)
    P = P.sort_values(["player_id", "t"]).reset_index(drop=True)
    cols = [c for c in P.columns if c not in ("game_id", "player_id", "t")]
    # 범주-x 블록 열 인덱스: tv_{slot}_{b}
    fam = {k: [c for c in cols if c.startswith(k + "_")] for k in ("tv", "to", "td", "tn")}
    slots, nb = [], 0
    for c in fam["tv"]:
        m = re.match(r"tv_(.+)_(\d+)$", c); slots.append(m.group(1)); nb = max(nb, int(m.group(2)) + 1)
    SL = sorted(set(slots), key=slots.index); NC = len(SL); NB = nb
    ci = {c: i for i, c in enumerate(cols)}
    FULL = os.environ.get("FULLCH", "0") == "1"
    X = P[cols].to_numpy(np.float32)
    if FULL:
        # 확장 채널: 기존 39 유형×결과(v·o·d·n 4면) + 세분 패스 12(방향×길이×결과: v=pv, n=pn) + 리시브 1(v=rx, n=rn)
        #            + 세분 슛(부위×결과: v=sv, n=sn) + 세분 수비(유형×결과: v=dv, n=dn). 없는 면·밴드는 0 열. PCA 없음 — 채널 그대로.
        X = np.hstack([X, np.zeros((len(X), 1), np.float32)]); Z = len(cols); cols = cols + ["_zero"]; ci["_zero"] = Z
        def idx(c): return ci.get(c, Z)
        rows4 = [[[ci[f"{k}_{s}_{b}"] for b in range(NB)] for s in SL] for k in ("tv", "to", "td", "tn")]
        def add(chs, vfam, nfam):
            for ch in chs:
                rows4[0].append([idx(f"{vfam}_{ch}_{b}") if ch else idx(f"{vfam}_{b}") for b in range(NB)])
                rows4[1].append([Z] * NB); rows4[2].append([Z] * NB)
                rows4[3].append([idx(f"{nfam}_{ch}_{b}") if ch else idx(f"{nfam}_{b}") for b in range(NB)])
        if os.environ.get("VPLUS", "0") == "1":      # VERSA+ 오프더볼 기여 (build_versaplus): 역할×부호 4채널
            VP = pd.read_parquet(ROOT / "outputs" / os.environ.get("VPFILE", "versaplus.parquet"))
            SEL = os.environ.get("VPSEL", "all")     # all | mate(동료 몫) | opp(상대 벌점)
            pre = ("bv_", "bn_") if SEL == "all" else (("bv_mate", "bn_mate") if SEL == "mate" else ("bv_opp", "bn_opp"))
            vpc = [c for c in VP.columns if c.startswith(pre) and c not in set(cols)]   # 주 파일에 이미 병합됐으면 건너뛴다
            VPv = VP.set_index(["game_id", "player_id"])[vpc].reindex(
                pd.MultiIndex.from_arrays([P.game_id.to_numpy(), P.player_id.to_numpy()])).to_numpy(np.float32)
            VPv = np.nan_to_num(VPv, nan=0.0)
            if vpc:
                X = np.hstack([X, VPv]); base_vp = len(cols); cols = cols + vpc
                for i, c in enumerate(vpc): ci[c] = base_vp + i
                print(f"VPLUS: VERSA+ 오프더볼 {len(vpc)} 열 추가 (커버 {100*(VPv!=0).any(1).mean():.1f}%)", flush=True)
            else:
                print("VPLUS: 오프더볼 열이 주 피처 파일에 이미 있음 — 중복 추가 안 함", flush=True)
        pch = sorted({re.match(r"pv_(.+)_\d+$", c).group(1) for c in cols if c.startswith("pv_")})
        sch = sorted({re.match(r"sv_(.+)_\d+$", c).group(1) for c in cols if c.startswith("sv_")})
        dch = sorted({re.match(r"dv_(.+)_\d+$", c).group(1) for c in cols if c.startswith("dv_")})
        add(pch, "pv", "pn"); add([""], "rx", "rn"); add(sch, "sv", "sn"); add(dch, "dv", "dn")
        if os.environ.get("VPLUS", "0") == "1":
            _sel = os.environ.get("VPSEL", "all")
            bch = sorted({re.match(r"bv_(.+)_\d+$", c).group(1) for c in cols if c.startswith("bv_")})
            if _sel != "all": bch = [c for c in bch if c.startswith(_sel)]        # all=4 · mate/opp=2
            add(bch, "bv", "bn")
        if os.environ.get("TRK", "0") == "1":        # 트래킹 행 (ssac27 trk_channels): 라벨마다 범주 하나 × 깊이 6구간 × 면 4(속도·고강도·압박 참여·시간)
            TK = pd.read_parquet(os.environ.get("TRKFILE", str(ROOT / "outputs" / "trk_channels.parquet")))
            pre = "trk" + os.environ.get("TRKMODE", "phase") + "_"                      # phase | third | all(페이즈 합침)
            tkc = [c for c in TK.columns if c.startswith(pre)]
            TKd = TK.set_index(["game_id", "player_id"])[tkc].reindex(pd.MultiIndex.from_arrays([P.game_id.to_numpy(), P.player_id.to_numpy()]))
            has = TKd.notna().any(axis=1).to_numpy(np.float32)                         # 트래킹 없는 선수-경기(2021–23 등)는 0 행 + 표시 0
            TKv = np.nan_to_num(TKd.to_numpy(np.float32), nan=0.0)
            ncol = [i for i, c in enumerate(tkc) if c.startswith(pre + "n_")]; TKv[:, ncol] = np.log1p(TKv[:, ncol])
            base_tk = len(cols); X = np.hstack([X, TKv, has[:, None]]); cols = cols + tkc + ["trk_has"]
            for i, c in enumerate(cols[base_tk:]): ci[c] = base_tk + i
            labs = sorted({c[len(pre) + 2:].rsplit("_", 1)[0] for c in tkc})
            assert all(f"{pre}v_{lab}_{NB - 1}" in ci for lab in labs), "트래킹 깊이 구간 수가 이벤트 밴드 수와 달라요"
            for lab in labs:
                for r, face in zip(rows4, ("v", "o", "d", "n")): r.append([ci[f"{pre}{face}_{lab}_{b}"] for b in range(NB)])
            print(f"TRK: 트래킹 {pre[3:-1]} 행 {len(labs)}개 × {NB}구간 × 4면 추가 · 트래킹 있는 선수-경기 {100 * has.mean():.1f}%", flush=True)
        tidx = np.array(rows4); NC = tidx.shape[1]
        sidx = np.array([ci[c] for c in cols if c.startswith(("gk_", "dr_", "bio_", "ps_", "trk_has"))])
        print(f"FULLCH: 패스 {len(pch)} · 리시브 1 · 슛 {len(sch)} · 수비 {len(dch)} 채널 추가 → 범주 {NC}", flush=True)
    else:
        tidx = np.array([[[ci[f"{k}_{s}_{b}"] for b in range(NB)] for s in SL] for k in ("tv", "to", "td", "tn")])   # (4,NC,NB)
        sidx = np.array([ci[c] for c in cols if c.startswith(("gk_", "dr_", "ps_"))])
    zidx = np.array([[ci[f"{k}_{i:03d}"] for i in range(NZX * NZY)] for k in ("zv", "zn")]).reshape(2, NZX, NZY).transpose(0, 2, 1)  # (2,8,12) [y,x]
    if os.environ.get("ZONEMAP", "0") == "1":                        # 범주별 2-D 존 지도 (build_zonemaps): 12 범주 × (n, v) × 8×12 → 24 채널 입력, 집계 zv/zn 대체
        ZM = pd.read_parquet(ROOT / "outputs/zonemaps.parquet"); zcols = [c for c in ZM.columns if c.startswith("z")]
        ZMv = ZM.set_index(["game_id", "player_id"])[zcols].reindex(pd.MultiIndex.from_arrays([P.game_id.to_numpy(), P.player_id.to_numpy()])).to_numpy(np.float32)
        ZMv = np.nan_to_num(ZMv, nan=0.0); ncol = [i for i, c in enumerate(zcols) if "_n_" in c]; ZMv[:, ncol] = np.log1p(ZMv[:, ncol])
        base_z = len(cols); X = np.hstack([X, ZMv]); cols = cols + zcols; ci = {c: i for i, c in enumerate(cols)}
        cats = list(dict.fromkeys(re.match(r"z(.+)_[nv]_\d+$", c).group(1) for c in zcols)); chans = [f"z{c}_{p}" for c in cats for p in ("n", "v")]   # (범주, 면) 인터리브 — ZSHARED 의 view 와 정합
        zidx = np.array([[[ci[f"{k}_{y * NZX + x:03d}"] for x in range(NZX)] for y in range(NZY)] for k in chans])   # (24, 8, 12) [y,x]
        print(f"ZONEMAP: 범주별 존 지도 {len(chans)} 채널 × {NZY}×{NZX} 추가 (열 +{len(zcols):,})", flush=True)
    if os.environ.get("TOKCOV", "0") == "1":
        # 경기 토큰 공변량: 출전 분/90 · 직전 출전 이후 휴식일 log1p · 홈 · (대상까지의 일수는 gather 단계 dt 로 대체) — 스칼라 블록에 추가
        pl_ = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv", usecols=["game_id", "team_id", "player_id", "minutes_played"]).drop_duplicates(["game_id", "player_id"])
        gd_ = g.set_index("game_id"); pl_ = pl_[pl_.game_id.isin(gpos)].copy()
        pl_["date"] = pl_.game_id.map(gd_.game_date); pl_["home"] = (pl_.team_id.to_numpy() == gd_.home_team_id.reindex(pl_.game_id).to_numpy()).astype(np.float32)
        pl_ = pl_.sort_values(["player_id", "date"]); pl_["rest"] = pl_.groupby("player_id").date.diff().dt.days
        pl_["rest"] = np.log1p(pl_.rest.fillna(30.0).clip(0, 60)).astype(np.float32); pl_["mins"] = (pl_.minutes_played.fillna(0.0) / 90.0).astype(np.float32)
        key = pl_.set_index(["game_id", "player_id"])[["mins", "rest", "home"]]
        cov = key.reindex(pd.MultiIndex.from_arrays([P.game_id.to_numpy(), P.player_id.to_numpy()])).to_numpy(np.float32)
        cov = np.nan_to_num(cov, nan=0.0); X = np.hstack([X, cov]); base = len(cols); cols = cols + ["cov_mins", "cov_rest", "cov_home"]
        sidx = np.concatenate([sidx, np.arange(base, base + 3)])
        print(f"TOKCOV: 토큰 공변량 3 추가 (출전분 평균 {cov[:, 0].mean() * 90:.0f}분 · 휴식일 중앙값 {np.expm1(np.median(cov[:, 1])):.0f}일)", flush=True)
    # 횟수 계열은 log1p (긴 꼬리)
    cnt_cols = [ci[c] for c in cols if c.startswith(("tn_", "zn_", "pn_", "rn_", "sn_", "dn_", "bn_"))]; X[:, cnt_cols] = np.log1p(X[:, cnt_cols])
    pid = P.player_id.to_numpy(int); t = P.t.to_numpy(int); gid = P.game_id.to_numpy(int)
    bounds, start = {}, {}
    for i, p in enumerate(pid):
        if p not in start: start[p] = i
    ps = list(start)
    for j, p in enumerate(ps): bounds[p] = (start[p], start[ps[j + 1]] if j + 1 < len(ps) else len(pid))
    print(f"범주 {NC} × x-구간 {NB} · 존 {NZY}×{NZX} · 스칼라 {len(sidx)} · 열 {len(cols)}", flush=True)
    return g, gpos, X, pid, t, gid, bounds, tidx, zidx, sidx


class MatchEncoder(nn.Module):
    """선수-경기 벡터(원공간) → 토큰. 구조별 귀납 편향."""
    def __init__(self, tidx, zidx, sidx, h=H, drop=0.2):
        super().__init__()
        self.register_buffer("tidx", torch.as_tensor(tidx)); self.register_buffer("zidx", torch.as_tensor(zidx)); self.register_buffer("sidx", torch.as_tensor(sidx))
        NC, NB = tidx.shape[1], tidx.shape[2]
        self.conv_x = nn.Conv1d(4, 4, kernel_size=3, padding=1)          # 범주 간 공유 1-D 공간 필터
        self.mix = nn.Linear(NC * 8, h)                                  # 범주 혼합 (평균·최대 풀링 4×2)
        self.conv_z = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 3)))
        self.zlin = nn.Linear(8 * 2 * 3, h); self.slin = nn.Linear(len(sidx), h)
        self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(3 * h, h)); self.norm = nn.LayerNorm(h)

    def forward(self, X):                                                # X [n, d_in]
        n = X.shape[0]
        T = X[:, self.tidx]                                              # [n,4,NC,NB]
        T = T.permute(0, 2, 1, 3).reshape(n * self.tidx.shape[1], 4, self.tidx.shape[2])
        T = F.relu(self.conv_x(T))                                       # [n·NC,4,NB]
        T = torch.cat([T.mean(-1), T.amax(-1)], 1).reshape(n, -1)        # [n, NC·8]
        ht = F.relu(self.mix(T))
        Z = X[:, self.zidx]                                              # [n,2,8,12]
        hz = F.relu(self.zlin(self.conv_z(Z).flatten(1)))
        hs = F.relu(self.slin(X[:, self.sidx]))
        return self.norm(self.out(torch.cat([ht, hz, hs], 1)))


class Net(nn.Module):
    def __init__(self, tidx, zidx, sidx, d_base, h=H, drop=0.2):
        super().__init__()
        self.menc = MatchEncoder(tidx, zidx, sidx, h, drop); self.dt = nn.Linear(1, h)
        self.att = nn.Sequential(nn.Tanh(), nn.Linear(h, 1)); self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, 1); self.base = nn.Linear(d_base, 1); self.lam = math.log(2) / HL
        for lin in (self.head, self.base): nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def player(self, Xs, dt, mask):                                      # Xs [n,K,d_in]
        n, k, d = Xs.shape
        tok = self.menc(Xs.reshape(n * k, d)).view(n, k, -1) + self.dt(torch.log1p(dt).unsqueeze(-1))
        w = torch.exp(-self.lam * dt) * mask.float(); w = w / w.sum(1, keepdim=True).clamp(min=1e-6)
        dec = (tok * w.unsqueeze(-1)).sum(1)
        sc = self.att(tok).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        return self.head(self.norm(dec + (tok * sc.unsqueeze(-1)).sum(1))).squeeze(-1)

    def team(self, Xa, dta, ma, Xb, dtb, mb, base):
        B = base.shape[0]
        return self.player(Xa, dta, ma).view(B, 11).sum(1) - self.player(Xb, dtb, mb).view(B, 11).sum(1) + self.base(base).squeeze(-1)


def fit(Xg, TG, S, tr_i, va_i, seed, epochs=None, wd=1e-1):
    torch.manual_seed(seed); np.random.seed(seed)
    Ia, Da, Ib, Db, BASE, y = TG
    net = Net(*S, BASE.shape[1]).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=wd)
    def pred(ix):
        out = []
        for s in range(0, len(ix), 128):
            b = torch.as_tensor(ix[s:s + 128], device=DEV)
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
    Ia, Da, Ib, Db = [torch.as_tensor(z.reshape(n, 11, K), device=DEV) for z in (Ia, Da, Ib, Db)]
    D_ = A - Bm; Ds = (D_ - D_.mean(0)) / D_.std(0).clip(1e-9)
    n_par = sum(p.numel() for p in Net(tidx, zidx, sidx, BASE.shape[1]).parameters())
    print(f"사례 {n:,} · H={H} · 파라미터 {n_par:,} · 장치 {DEV}", flush=True)
    preds = {"linear": np.full(n, np.nan), "cnn": np.full(n, np.nan)}
    r2 = lambda p, ix: 1 - ((y[ix] - p) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    for f in range(1, 5):
        tr = np.concatenate(IDX[:f]); cut = int(.8 * len(tr)); i1, i2 = tr[:cut], tr[cut:]; te = IDX[f]
        pc = PCA(n_components=NPC, random_state=0).fit(Ds[i1]); Zd = pc.transform(Ds); Zd = (Zd - Zd[i1].mean(0)) / Zd[i1].std(0).clip(1e-9)
        Xl = np.hstack([BASE, Zd]); ba = max(AL, key=lambda a: r2w(y[i2], Ridge(alpha=a).fit(Xl[i1], y[i1]).predict(Xl[i2]), np.ones(len(i2))))
        preds["linear"][te] = Ridge(alpha=ba).fit(Xl[tr], y[tr]).predict(Xl[te]); lin_val = r2(Ridge(alpha=ba).fit(Xl[i1], y[i1]).predict(Xl[i2]), i2)
        # RESID=1: 릿지 예측을 고정 skip 으로 두고 CNN 은 잔차만 학습 (튜닝 단계는 i1 릿지, 재적합 단계는 tr 릿지)
        RES = os.environ.get("RESID", "0") == "1"
        p_i1 = Ridge(alpha=ba).fit(Xl[i1], y[i1]).predict(Xl); p_tr = Ridge(alpha=ba).fit(Xl[tr], y[tr]).predict(Xl)
        g1 = set(int(x) for x in R.game_id.iloc[i1]); rows1 = np.flatnonzero(np.isin(gid_arr, list(g1)))
        mu, sd = X[rows1].mean(0), X[rows1].std(0).clip(1e-6); Xg = torch.as_tensor(((X - mu) / sd).clip(-5, 5).astype(np.float32), device=DEV)
        S = (tidx, zidx, sidx); Bg = torch.as_tensor(BASE.astype(np.float32), device=DEV)
        y_tune = (y - p_i1) if RES else y; y_fit = (y - p_tr) if RES else y
        TG = (Ia, Da, Ib, Db, Bg, torch.as_tensor(y_tune.astype(np.float32), device=DEV))
        _, v, ep, pr, hist = fit(Xg, TG, S, i1, i2, 0)
        with torch.no_grad(): vr2 = r2(pr(i2).cpu().numpy() + (p_i1[i2] if RES else 0), i2)
        TG = (Ia, Da, Ib, Db, Bg, torch.as_tensor(y_fit.astype(np.float32), device=DEV))
        pt = np.zeros(len(te))
        for sd_ in (0, 1):
            net, _, _, pr, _ = fit(Xg, TG, S, tr, None, sd_, epochs=ep)
            with torch.no_grad(): pt += pr(te).cpu().numpy() / 2
        if RES: pt = pt + p_tr[te]
        preds["cnn"][te] = pt
        curve = " ".join(f"{a:.2f}/{b:.2f}" for a, b in hist[:6]) + " …"
        print(f"폴드 {f}: 릿지 val {lin_val:+.3f}/test {r2(preds['linear'][te], te):+.3f} · CNN ep={ep} val {vr2:+.3f}/test {r2(pt, te):+.3f} · train/val MSE {curve}", flush=True)
    ix0 = np.flatnonzero(np.isfinite(preds["linear"])); gids = R.game_id.to_numpy(); G = {}
    for i in ix0: G.setdefault(int(gids[i]), []).append(i)
    keys = list(G); rng = np.random.default_rng(0)
    def r2i(pv, ix): return 1 - ((y[ix] - pv[ix]) ** 2).sum() / ((y[ix] - y[ix].mean()) ** 2).sum()
    ds = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys), replace=True); ix = np.concatenate([G[keys[j]] for j in pick]); ds.append(r2i(preds["cnn"], ix) - r2i(preds["linear"], ix))
    ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"\n기준(릿지) R² {r2i(preds['linear'], ix0):+.4f} · 구조 인코더(H={H}) R² {r2i(preds['cnn'], ix0):+.4f}  Δ={ds.mean():+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ>0)={np.mean(ds > 0):.3f}")
    np.savez(ROOT / f"outputs/player_encoder_cnn_H{H}{'_resid' if os.environ.get('RESID', '0') == '1' else ''}.npz", **preds, y=y, game_id=gids)


if __name__ == "__main__":
    main()
