"""칸 역할 균형 벌점 — 같은 칸(포메이션 셀) 안 선수들의 **실제 수행 역할** 조합을 감독 전형에 맞춘다.

문제(사용자): 중앙 미드 2인 칸을 공격형 둘로, 최전방 2인 칸을 윙어·AM 으로 채워도 현행 벌점(깊이 q·레인·전진)은 무벌점.
해법
  선수 기능 프로필 τ_p (과거-only 감쇠 스냅샷의 tn_ 횟수 열에서): [수비 액션 비율(태클·인터셉트·클리어·파울), 공격 1/3 관여 비율(밴드 4·5), 슛·돌파 비율]
  칸 전형 μ_cell, σ_cell: 감독 라인업(declared_grid)에서 (포메이션, 칸) 별 Σ_{p∈칸} τ_p 의 평균·SD (희소하면 (칸, 인원) 로 대체)
  목적함수 J = Σ sc − β Σ(qpen − lp) − βf Σ max(0, k−최빈단) − λ Σ_칸 ‖(Σ τ − μ)/σ‖²     ← 칸 내 비가산
  최적화: 헝가리안 초기해 → 국소 교환(벤치 교체·칸 간 자리 교환) 탐색, 포메이션 16종 중 J 최대
판정: 격차(J·가산 sc)의 fe_ols 효과, 적중, 동일 모양, 투톱(최전방 2인) 비율, 칸 이탈(모델 vs 감독), 위약(직전 XI), λ 격자
실행: LAMBDAS=0,0.01,0.02 python -m experiments.role_balance_battery
"""
from __future__ import annotations
import os, sys, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp.parquet"); os.environ.setdefault("DROPFAM", "zv,zn")
from lineup_event_panel import _snapshots            # noqa: E402
from asym_penalty import pick_asym, qpen             # noqa: E402
from gap_lane import declared_grid                   # noqa: E402
from gap_constrained import fe_ols                   # noqa: E402
from config import VAEP_OUTPUT_DIR                   # noqa: E402

BETA, BF = 0.05, 0.05
PROD = ROOT / os.environ.get("PROD", "outputs/gap_soft_fullonball_gk_resid_merged_mix25d_pc20_all_npxg_nozvzn_fixsd.pkl")   # PROD 로 점수 pkl 교체 가능
DEF_T = {8, 9, 10, 18}; FIN_T = {7, 11, 12, 13, 21}; ATT_B = {4, 5}


def build():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date); g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    by, T, dim, FC = _snapshots(gpos)
    tn = [(i, int(m.group(1)), int(m.group(3))) for i, c in enumerate(FC) for m in [re.match(r"tn_(\d+)_([0-9x]+)_(\d)$", c)] if m]
    i_all = np.array([i for i, _, _ in tn]); i_def = np.array([i for i, t, b in tn if t in DEF_T]); i_att = np.array([i for i, t, b in tn if b in ATT_B]); i_fin = np.array([i for i, t, b in tn if t in FIN_T])
    def tau(p, tt):
        a = T.get(int(p))
        if a is None: return None
        j = np.searchsorted(a, tt, "right") - 1
        if j < 0: return None
        s = by[int(p)][j][1]; tot = float(s[i_all].sum())
        if tot <= 0: return None
        return np.array([s[i_def].sum() / tot, s[i_att].sum() / tot, s[i_fin].sum() / tot])
    CELL, PER, _ = declared_grid()
    HOLD = int(os.environ.get("HOLDOUT", 0))                          # 보류 시즌: 칸 전형·라이브러리는 그 시즌 이전 감독 라인업으로만
    season_of = {int(r.game_id): int(r.season) for r in g.itertuples(index=False)}
    FIT = {k: G for k, G in CELL.items() if (not HOLD) or season_of.get(k[0], 0) < HOLD}
    if HOLD: print(f"보류 {HOLD}: 칸 전형·라이브러리 적합에 {len(FIT):,}/{len(CELL):,} 라인업 사용", flush=True)
    cn = Counter(tuple(np.asarray(G).ravel()) for G in FIT.values()); LIB = [np.array(k, int).reshape(5, 3) for k, v in cn.items() if v >= 20]
    # 칸 전형: (shape, k, l) → Σ τ 목록
    acc = defaultdict(list); acc_kn = defaultdict(list)
    for (gid, tid), G in FIT.items():
        if gid not in gpos: continue
        tt = gpos[gid]; cells = defaultdict(list)
        for (gg, t2, p), (k, l) in PER.items():
            if gg == gid and t2 == tid:
                v = tau(p, tt)
                if v is not None: cells[(k, l)].append(v)
        shape = tuple(G.ravel())
        for (k, l), vs in cells.items():
            if len(vs) == G[k, l]:
                s_ = np.sum(vs, 0); acc[(shape, k, l)].append(s_); acc_kn[(k, l, len(vs))].append(s_)
    MU = {}
    for key, vs in acc.items():
        if len(vs) >= 20: MU[key] = (np.mean(vs, 0), np.std(vs, 0).clip(0.03))
    MUkn = {key: (np.mean(vs, 0), np.std(vs, 0).clip(0.03)) for key, vs in acc_kn.items() if len(vs) >= 20}
    print(f"칸 전형 (포메이션별) {len(MU)} · 대체 (칸,인원) {len(MUkn)} · 라이브러리 {len(LIB)}", flush=True)
    A = pd.read_pickle(PROD); LP = A.LPk.iloc[0]
    rows = [r for r in A.itertuples(index=False) if int(r.gid) in gpos]

    def cell_dev(G, slot, TAU, lam):
        if lam == 0: return 0.0
        shape = tuple(G.ravel()); sums = defaultdict(lambda: np.zeros(3)); cnt = Counter()
        for p, (k, l) in slot.items():
            v = TAU.get(p)
            if v is not None: sums[(k, l)] += v; cnt[(k, l)] += 1
        d = 0.0
        for (k, l), s_ in sums.items():
            m = MU.get((shape, k, l)) or MUkn.get((k, l, int(G[k, l])))
            if m is None: continue
            z = (s_ - m[0]) / m[1]; d += float((z * z).sum())
        return lam * d
    def pen(r, p, k, l):
        q = r.qmap.get(p); lp = LP[r.lane.get(p, 1)]; qk = float(q[k]) if q is not None else 0.2; mo = int(np.argmax(q)) if q is not None else 2
        return BETA * (qpen(qk) - lp[l]) + BF * max(0, k - mo)
    # 외국인 동시 출전 상한: 공식 국적(player_bio_cur) 기준, 감독 XI 는 2021~26 전 리그에서 ≤4 (5 는 0.2%) → FCAP(기본 4) 초과 인원당 FKAP 벌점
    FCAP = int(os.environ.get("FCAP", 4)); FKAP = float(os.environ.get("FKAP", 1.0))
    try:
        _bio = pd.read_parquet(ROOT / "outputs/player_bio_cur.parquet"); FRN = set(_bio[_bio.foreign == 1].pid.astype(int))
    except Exception:
        FRN = set()
    # 비가산 집합 항 g(XI): set_score SAVEG 덤프(선수 벡터·집합 블록·재척도 k)로 CPU 평가. SETW 로 가중(기본 1 = 학습 단위 그대로, k 로 sc 척도에 맞춤)
    SETG = os.environ.get("SETG", ""); GNET = None
    if SETG:
        import torch; torch.set_num_threads(1)
        from player_encoder_set import SAB, PMA
        _d = pd.read_pickle(SETG); _c = _d["cfg"]; SETW = float(os.environ.get("SETW", 1.0)); NSET = int(os.environ.get("SETSEEDS", 2))
        class _GBlock(torch.nn.Module):
            def __init__(self):
                super().__init__(); h = _c["H"]
                self.sabs = torch.nn.ModuleList([SAB(h, _c["HEADS"], _c["DROP"]) for _ in range(_c["L"])]); self.pma = PMA(h, _c["HEADS"], _c["DROP"])
                self.g = torch.nn.Sequential(torch.nn.Dropout(_c["DROP"]), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))
                self.cellemb = torch.nn.Embedding(17, h)
            def forward(self, X, C=None):
                if C is not None and self.has_cell: X = X + self.cellemb(C)
                for s_ in self.sabs: X = s_(X)
                return self.g(self.pma(X)).squeeze(-1)
        class _GNP:
            """집합 블록의 numpy 전개 (SAB×L → PMA → MLP). 16차원·11토큰이라 torch 호출 오버헤드가 지배적 → numpy 로 ~10배 빠르게."""
            def __init__(self, sd):
                g = lambda k: sd[k].numpy().astype(np.float64); self.h = _c["H"]; self.nh = _c["HEADS"]; self.L = _c["L"]
                self.sab = [dict(Wi=g(f"sabs.{i}.mha.in_proj_weight"), bi=g(f"sabs.{i}.mha.in_proj_bias"), Wo=g(f"sabs.{i}.mha.out_proj.weight"), bo=g(f"sabs.{i}.mha.out_proj.bias"),
                                 W1=g(f"sabs.{i}.ff.0.weight"), b1=g(f"sabs.{i}.ff.0.bias"), W2=g(f"sabs.{i}.ff.3.weight"), b2=g(f"sabs.{i}.ff.3.bias"),
                                 n1w=g(f"sabs.{i}.n1.weight"), n1b=g(f"sabs.{i}.n1.bias"), n2w=g(f"sabs.{i}.n2.weight"), n2b=g(f"sabs.{i}.n2.bias")) for i in range(self.L)]
                self.pma = dict(seed=g("pma.seed")[0, 0], Wi=g("pma.mha.in_proj_weight"), bi=g("pma.mha.in_proj_bias"), Wo=g("pma.mha.out_proj.weight"), bo=g("pma.mha.out_proj.bias"), nw=g("pma.n.weight"), nb=g("pma.n.bias"))
                self.g1w, self.g1b, self.g2w, self.g2b = g("g.1.weight"), g("g.1.bias"), g("g.3.weight"), g("g.3.bias")
                self.cell = g("cellemb.weight") if "cellemb.weight" in sd else None; self.has_cell = self.cell is not None
            @staticmethod
            def _ln(x, w, b): m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True); return (x - m) / np.sqrt(v + 1e-5) * w + b
            def _mha(self, Q, KV, P):
                h, nh = self.h, self.nh; d = h // nh; Wi, bi = P["Wi"], P["bi"]
                q = Q @ Wi[:h].T + bi[:h]; k = KV @ Wi[h:2 * h].T + bi[h:2 * h]; v = KV @ Wi[2 * h:].T + bi[2 * h:]
                out = np.zeros((Q.shape[0], h))
                for j in range(nh):
                    sl = slice(j * d, (j + 1) * d); a = q[:, sl] @ k[:, sl].T / np.sqrt(d); a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True); out[:, sl] = a @ v[:, sl]
                return out @ P["Wo"].T + P["bo"]
            def __call__(self, X, C=None):
                if self.has_cell and C is not None: X = X + self.cell[C]
                for P in self.sab:
                    X = self._ln(X + self._mha(X, X, P), P["n1w"], P["n1b"]); ff = np.maximum(X @ P["W1"].T + P["b1"], 0) @ P["W2"].T + P["b2"]; X = self._ln(X + ff, P["n2w"], P["n2b"])
                S = self.pma["seed"][None]; z = self._ln(S + self._mha(S, X, self.pma), self.pma["nw"], self.pma["nb"])[0]
                return float(np.maximum(z @ self.g1w.T + self.g1b, 0) @ self.g2w.T + self.g2b)
        GNET = {}
        for f_, sds in _d["blocks"].items():
            GNET[f_] = [_GNP(sd_) for sd_ in sds[:NSET]]
        # 검증: torch 블록과 일치하는지
        _t = _GBlock(); _t.has_cell = any(k_.startswith("cellemb") for k_ in sds[0]); _t.load_state_dict(sds[0], strict=False); _t.eval()
        _X = np.random.default_rng(0).standard_normal((11, _c["H"])).astype(np.float32)
        with torch.no_grad(): _ref = float(_t(torch.as_tensor(_X)[None]).item())
        print(f"집합 항 numpy 검증: torch {_ref:+.5f} vs numpy {GNET[f_][0](_X.astype(np.float64)):+.5f}", flush=True)
        VSET, KSET, FOLDG = _d["V"], _d["K"], _d["fold_of"]
        print(f"집합 항 활성: 폴드 {list(GNET)} · 시드 {NSET} · 가중 {SETW}", flush=True)
    GCACHE = {}
    def gterm(r, xi, slot=None):
        if GNET is None: return 0.0
        f_ = FOLDG.get(int(r.gid)); V_ = VSET.get((int(r.gid), int(r.tid)))
        if f_ is None or V_ is None: return 0.0
        ps = [p for p in xi if p in V_]
        if len(ps) < 8: return 0.0
        cells = [((slot[p][0] * 3 + slot[p][1]) if (slot and p in slot) else (15 if p in set(r.gks) else 16)) for p in ps]
        key = (int(r.gid), int(r.tid), frozenset(zip(ps, cells)))
        v = GCACHE.get(key)
        if v is None:
            X = np.stack([V_[p] for p in ps]).astype(np.float64); C = np.array(cells); v = float(np.mean([m_(X, C) for m_ in GNET[f_]]))
            if len(GCACHE) > 200000: GCACHE.clear()
            GCACHE[key] = v
        return SETW * KSET[f_] * v
    PAIRM = pd.read_pickle(ROOT / os.environ["PAIRTERM"]) if os.environ.get("PAIRTERM") else None; PAIRW = float(os.environ.get("PAIRW", 0.0))   # 명시적 쌍 시너지 항: {(gid,tid): {(u,v): 값}}
    if PAIRM is not None: print(f"쌍 항 활성: 사례 {len(PAIRM):,} · 가중 {PAIRW}", flush=True)
    def pairterm(r, xi):
        if PAIRM is None or PAIRW == 0.0: return 0.0
        M_ = PAIRM.get((int(r.gid), int(r.tid)))
        if M_ is None: return 0.0
        return PAIRW * sum(M_.get((u, v), 0.0) for u in xi for v in xi if u != v)
    def J(r, G, xi, slot, TAU, lam, sc=None):
        sc = r.sc if sc is None else sc
        nf = sum(1 for p in xi if p in FRN)
        return sum(sc[p] for p in xi) - sum(pen(r, p, k, l) for p, (k, l) in slot.items()) - cell_dev(G, slot, TAU, lam) - FKAP * max(0, nf - FCAP) + gterm(r, xi, slot) + pairterm(r, xi)
    def solve(r, G, TAU, lam, sc=None):
        sc = r.sc if sc is None else sc
        xi, slot = pick_asym(sc, r.qmap, r.lane, G, list(r.gks), set(r.pool), LP, BETA, BF)
        if not slot: return xi, slot, J(r, G, xi, slot, TAU, lam, sc)
        gk = [p for p in xi if p not in slot]; best = J(r, G, xi, slot, TAU, lam, sc)
        if lam == 0: return xi, slot, best                            # 모양 사전 순위(λ=0)는 국소 탐색 없이 — g 는 TOPK 모양의 본 탐색에서만
        bench = [p for p in sc if p in set(r.pool) and p not in set(xi) and p not in set(r.gks)]
        for _ in range(30):
            improved = False; cur_slot = dict(slot)
            # 벤치 교체
            for p_out, (k, l) in list(cur_slot.items()):
                for b in bench:
                    s2 = dict(cur_slot); del s2[p_out]; s2[b] = (k, l); v = J(r, G, gk + list(s2), s2, TAU, lam, sc)
                    if v > best + 1e-9: best, slot, improved = v, s2, True; bench = [x for x in bench if x != b] + [p_out]; break
                if improved: break
            if improved: continue
            # 칸 간 자리 교환
            items = list(cur_slot.items())
            for a_ in range(len(items)):
                for b_ in range(a_ + 1, len(items)):
                    (pa, ca), (pb, cb) = items[a_], items[b_]
                    if ca == cb: continue
                    s2 = dict(cur_slot); s2[pa], s2[pb] = cb, ca; v = J(r, G, gk + list(s2), s2, TAU, lam, sc)
                    if v > best + 1e-9: best, slot, improved = v, s2, True; break
                if improved: break
            if not improved: break
        return gk + list(slot), slot, best
    return dict(A=A, rows=rows, LIB=LIB, CELL=CELL, FIT=FIT, PER=PER, gpos=gpos, tau=tau, J=J, solve=solve, cell_dev=cell_dev, FRN=FRN)


def main():
    B = build(); A, rows, LIB, CELL, PER, gpos, tau, J, solve, cell_dev = (B[k] for k in ("A", "rows", "LIB", "CELL", "PER", "gpos", "tau", "J", "solve", "cell_dev"))
    lams = [float(x) for x in os.environ.get("LAMBDAS", "0,0.01,0.02").split(",")]
    print(f"{'λ':>6s} {'격차 J':>28s} {'격차 가산sc':>26s} {'적중':>6s} {'동일모양%':>8s} {'투톱%':>6s} {'칸이탈 모델':>10s} {'칸이탈 감독':>10s} {'위약':>24s}", flush=True)
    for lam in lams:
        gj, ga, hit, same, two, dev_m, dev_c, plc = [], [], [], [], [], [], [], []
        for r in rows:
            tt = gpos[int(r.gid)]; TAU = {p: tau(p, tt) for p in r.sc}
            best = (None, None, -1e18, None)
            for G in LIB:
                xi, slot, v = solve(r, G, TAU, lam)
                if v > best[2]: best = (xi, slot, v, G)
            xi, slot, v, G = best
            # 감독 XI 의 J (감독 실제 칸 배정)
            Gc = CELL.get((int(r.gid), int(r.tid))); cslot = {p: PER[(int(r.gid), int(r.tid), p)] for p in r.coach_xi if (int(r.gid), int(r.tid), p) in PER}
            if Gc is None or len(cslot) < 10: continue
            vc = J(r, Gc, r.coach_xi, cslot, TAU, lam)
            gj.append(v - vc); ga.append(sum(r.sc[p] for p in xi) - sum(r.sc[p] for p in r.coach_xi)); hit.append(len(set(xi) & set(r.coach_xi)))
            same.append(np.array_equal(G, Gc)); two.append(int(G[4].sum()) >= 2)
            dev_m.append(cell_dev(G, slot, TAU, 1.0)); dev_c.append(cell_dev(Gc, cslot, TAU, 1.0))
            plc.append((sum(r.sc[p] for p in r.prev) - sum(r.sc[p] for p in r.coach_xi)) if r.prev and set(r.prev) <= set(r.sc) else np.nan)
        D = pd.DataFrame({"gid": [int(r.gid) for r in rows][:len(gj)], "tid": [int(r.tid) for r in rows][:len(gj)], "season": [r.season for r in rows][:len(gj)], "y": [r.y for r in rows][:len(gj)], "gj": gj, "ga": ga, "plc": plc})
        eff = {}
        for c_ in ("gj", "ga", "plc"):
            sd_ = D[c_].std(); b, lo, hi, nn_ = fe_ols(D.dropna(subset=[c_]), c_); eff[c_] = (b * sd_, lo * sd_, hi * sd_)
        f1 = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]{'*' if (t[1] > 0) == (t[2] > 0) else ' '}"
        print(f"{lam:6.3f} {f1(eff['gj']):>28s} {f1(eff['ga']):>26s} {np.mean(hit):6.3f} {np.mean(same) * 100:8.1f} {np.mean(two) * 100:6.1f} {np.mean(dev_m):10.2f} {np.mean(dev_c):10.2f} {f1(eff['plc']):>24s}  n={len(gj)}", flush=True)


if __name__ == "__main__":
    main()
