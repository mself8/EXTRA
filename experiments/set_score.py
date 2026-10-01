"""단일 모델(player_encoder_set, LANE=pca) 점수 → 프로덕션 배터리용 pkl.

선수 점수 = 모델의 가산부 (릿지 동치 선형 레인 β·z_p + CNN 레인 head(v_p)), 폴드 5시드 평균,
폴드별 프로덕션 sc SD 에 재척도 → relb λ=0.25 블렌드 (cnn2_score 와 동일 절차).
집합부 g(XI) 는 여기서 쓰지 않는다(가산 채점) — 집합 목적함수는 set_battery 에서.
실행: TAG=set1-cross0-H16-L1-pca-s5 PROD=<abs pkl> OUT=<out pkl> python -m experiments.set_score
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("ONBALL_FILE", "onball_gk_resid_merged_defresp_ref.parquet")
TAG = os.environ.get("TAG", "set1-cross0-H16-L1-pca-s5")
os.environ["SET"] = "1" if (TAG.startswith("set1") or "-set1-" in TAG) else "0"; os.environ["CROSS"] = "1" if "cross1" in TAG else "0"
os.environ["H"] = TAG.split("-H")[1].split("-")[0]; os.environ["LAYERS"] = TAG.split("-L")[1].split("-")[0]
os.environ["LANE"] = "none" if "-none" in TAG else ("lowrank" if "-lowrank" in TAG else ("raw" if "-raw-" in TAG else "pca"))
os.environ["FULLCH"] = "1" if "-full" in TAG else "0"; os.environ["TOKCOV"] = "1" if "-cov" in TAG else "0"
import re as _re; _m = _re.search(r"-tf(\d+)", TAG); os.environ["MATCHTF"] = _m.group(1) if _m else "0"; os.environ["SLOTEMB"] = "1" if "-slot" in TAG else "0"; os.environ["DTDAYS"] = "1" if "-days" in TAG else "0"; os.environ["ZONEMAP"] = "1" if "-zmap" in TAG else "0"
import torch                                        # noqa: E402
from player_encoder_set import SetNet, load_structured   # noqa: E402
from player_encoder import pack, gather, K, DEV     # noqa: E402
from lineup_event_panel import _snapshots           # noqa: E402
from config import VAEP_OUTPUT_DIR                  # noqa: E402

PROD = ROOT / os.environ.get("PROD", "outputs/gap_soft_fullonball_gk_resid_merged_defresp_ref_pc20_all_npxg_nozvzn_fixsd.pkl")
RELB = ROOT / "outputs/gap_soft_fullonball_gk_resid_merged_relb_pc20_all_npxg_nozvzn_fixsd.pkl"
OUT = ROOT / os.environ.get("OUT", f"outputs/gap_soft_fullonball_gk_resid_merged_{TAG}_pc20_all_npxg_nozvzn_fixsd.pkl")
LAM = float(os.environ.get("BLEND", 0.25))   # relb(릿지) 블렌드 비율; 0 = 순수 딥러닝 점수


def main():
    g, gpos, X, pid, t_arr, gid_arr, bounds, tidx, zidx, sidx = load_structured()
    by, T, dim, FC = _snapshots(gpos)
    P_raw = pd.read_parquet(ROOT / "outputs" / os.environ["ONBALL_FILE"]); P_raw = P_raw[P_raw.game_id.isin(gpos)].copy(); P_raw["t"] = P_raw.game_id.map(gpos)
    P_raw = P_raw.sort_values(["player_id", "t"]).reset_index(drop=True); Xraw = P_raw[FC].to_numpy(np.float32); assert len(Xraw) == len(X)
    Xr = torch.as_tensor(Xraw, device=DEV)
    A = pd.read_pickle(PROD); Rb = pd.read_pickle(RELB); SR = {(int(r.gid), int(r.tid)): r.sc for r in Rb.itertuples(index=False)}
    fold_of, F, nets = {}, {}, {}
    for f in range(1, 5):
        if not (ROOT / f"outputs/player_encoder_set_{TAG}_fold{f}.pt").exists(): continue     # 시즌 폴드는 2개
        d = torch.load(ROOT / f"outputs/player_encoder_set_{TAG}_fold{f}.pt", weights_only=False, map_location="cpu"); F[f] = d
        for gg in d["te_games"]: fold_of[gg] = f
        nets[f] = []
        for sdict in d["nets"]:
            net = SetNet(*d["S"])
            if d.get("pca") is not None: net.set_pca(*d["pca"])
            net = net.to(DEV); net.load_state_dict(sdict); net.eval(); nets[f].append(net)
    new_sc, miss, cur_f, Xg = [], 0, None, None
    VD = {}                                                                      # (gid,tid) → {pid: v_p}  (SAVEG 용)
    for r in A.itertuples(index=False):
        gid, tid = int(r.gid), int(r.tid); f = fold_of.get(gid)
        if f is None: new_sc.append(r.sc); miss += 1; continue
        if f != cur_f:
            d = F[f]; Xg = torch.as_tensor(((X - d["mu"]) / d["sd"]).clip(-5, 5).astype(np.float32), device=DEV); cur_f = f
        tt = gpos[gid]; pids = list(r.sc)
        idx_, dt_ = pack([(p, tt) for p in pids], bounds, t_arr); ib = torch.as_tensor(idx_, device=DEV); db = torch.as_tensor(dt_, device=DEV)
        with torch.no_grad():
            Xs, dts, ms = gather(Xg, ib, db); Rs = gather(Xr, ib, db)[0] if os.environ["LANE"] == "pca" else None
            outs = [n_.player(Xs, dts, ms, Rs) for n_ in nets[f]]
            res = np.mean([o[0].cpu().numpy() for o in outs], 0); vv = np.mean([o[1].cpu().numpy() for o in outs], 0)
        has = (idx_ >= 0).any(1)
        new_sc.append({p: (float(res[i]) if has[i] else r.sc[p]) for i, p in enumerate(pids)})
        if os.environ.get("SAVEG"): VD[(gid, tid)] = {p: vv[i].astype(np.float32) for i, p in enumerate(pids) if has[i]}
    A = A.copy(); A["sc_dl"] = new_sc; KF = {}
    for f in F:
        ix = [i for i, r in enumerate(A.itertuples(index=False)) if fold_of.get(int(r.gid)) == f]
        p_all = np.array([v for i in ix for v in A.sc.iloc[i].values()]); d_all = np.array([v for i in ix for v in A.sc_dl.iloc[i].values()])
        k = p_all.std() / max(d_all.std(), 1e-12); m = p_all.mean() - k * d_all.mean(); KF[f] = float(k)
        for i in ix: A.sc_dl.iloc[i].update({p: k * v + m for p, v in A.sc_dl.iloc[i].items()})
    out = []
    for r in A.itertuples(index=False):
        sb = SR.get((int(r.gid), int(r.tid))); base = r.sc_dl
        out.append(base if sb is None else {p: (1 - LAM) * v + LAM * sb.get(p, v) for p, v in base.items()})
    A["sc"] = out; A.drop(columns=["sc_dl"], inplace=True)
    A["coach_sc"] = [sum(r.sc[p] for p in r.coach_xi) for r in A.itertuples(index=False)]
    A.to_pickle(OUT)
    if os.environ.get("SAVEG"):                                               # 비가산 목적함수용: 집합 블록(sabs·pma·g) 상태 + 선수 벡터 + 재척도 k
        blocks = {f: [{k_: t.detach().cpu() for k_, t in n_.state_dict().items() if k_.startswith(("sabs", "pma", "g.", "side", "cellemb"))} for n_ in nets[f]] for f in F}
        pd.to_pickle(dict(V=VD, K=KF, fold_of=fold_of, blocks=blocks, cfg=dict(H=int(os.environ["H"]), HEADS=4, L=int(os.environ["LAYERS"]), DROP=float(os.environ.get("DROP", 0.4)))), os.environ["SAVEG"])
        print(f"집합 항 덤프 → {os.environ['SAVEG']} (사례 {len(VD):,})", flush=True)
    corr = np.corrcoef([v for r in pd.read_pickle(PROD).itertuples(index=False) for v in r.sc.values()][:50000], [v for r in A.itertuples(index=False) for v in r.sc.values()][:50000])[0, 1]
    print(f"사례 {len(A):,} · 폴드 없음(원본 유지) {miss} · 프로덕션 절대점수와 상관 {corr:.3f} · 저장 {OUT.name}")


if __name__ == "__main__":
    main()
