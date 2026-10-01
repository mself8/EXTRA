"""VERSA+ 오프더볼 기여 — 온볼 액션의 가치를 주변 선수들과 나눈다 (트래킹 불요).

versa 연구(`Experiments/Sports/Football/versa`, `datatools/vaep/versa/offball.py`)의
VERSA+ 를 우리 SPADL·VAEP 위에 구현한다. 이벤트 스트림은 공을 가진 한 명만 평가하는데,
그 값을 **그 자리에 있던 동료**에게 나눠 주고 **막지 못한 상대**에게 물린다.

선수 p 의 존재 지도는 그 구간 T 안 자기 이벤트 위치의 가우시안 커널 합:
    H_p(x, y) = Σ_k K(x, y | x_k, y_k),  K = exp(-(Δx)²/2σ² - (Δy)²/2σ²) / (2πσ²)
액션 a_i (가치 V, 팀 P, 상대 O, 실행자 p_a) 에 대해
    C_p  = +V · H_p / Σ_{P∖{p_a}} H       (동료)
    C_q  = -V · H_q / Σ_{O} H             (상대)
동료 몫의 합은 V, 상대 벌점의 합은 -V 다.

우리 채널 구조에 맞춰 **역할 × 가치 부호 4채널 × 6밴드**로 집계한다.
밴드는 **받는 선수 자신의 공격 방향** x 로 잡는다(상대는 좌표를 뒤집는다).
    bv_matep/maten/oppp/oppn_{b} : 값,   bn_..._{b} : 소프트 횟수(몫의 합)
versa 논문 규약: 재시작 패스(스로인·프리킥·코너·골킥)의 가치는 나누지 않는다
(다만 존재 지도의 근거로는 쓴다). 파라미터는 σ=30m, T=15분(논문 채택값).

출력: outputs/versaplus.parquet  (game_id, player_id, bv_*/bn_*)
실행: [SIGMA=30 WINDOW=15 FLAT=0] python -m experiments.build_versaplus
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR  # noqa: E402
from segment_panel import pid_maps  # noqa: E402

NB = 6
L, W = 105.0, 68.0
SIGMA = float(os.environ.get("SIGMA", 30.0))       # 커널 폭 (versa 논문 채택값)
WINDOW = float(os.environ.get("WINDOW", 15.0))     # 구간 길이(분)
FLAT = os.environ.get("FLAT", "0") == "1"          # 공간 무시 — 관여도만으로 배분 (논문의 대조)
RESTART = {2, 3, 4, 5, 6, 22}                      # throw_in·freekick×2·corner×2·goalkick
CHAN = ["matep", "maten", "oppp", "oppn"]
TAG = os.environ.get("VP_TAG", "")
OUT = ROOT / "outputs" / f"versaplus{TAG}.parquet"


def main():
    V = pd.read_parquet(VAEP_OUTPUT_DIR / "vaep_oof.parquet",
                        columns=["game_id", "action_id", "period_id", "time_seconds",
                                 "team_id", "player_id", "type_id", "vaep_value"])
    S = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet",
                        columns=["game_id", "action_id", "start_x", "start_y"])
    for D in (V, S):
        D["game_id"] = D.game_id.astype(np.int64); D["action_id"] = D.action_id.astype(np.int64)
    A = V.merge(S, on=["game_id", "action_id"], how="inner")
    A = A.dropna(subset=["player_id", "team_id", "start_x", "start_y"])
    A["player_id"] = A.player_id.astype(np.int64); A["team_id"] = A.team_id.astype(np.int64)
    gm = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")[["game_id", "home_team_id"]]
    A = A.merge(gm, on="game_id", how="left")
    # 선수 id 는 파이프라인 브릿지 공간으로 — build_refined 와 같은 규약이라야 병합 키가 맞는다
    gs = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")[["game_id", "season"]]
    g26 = set(gs[gs.season == 2026].game_id.astype(int)); old_map, new_map = pid_maps()
    A["player_id"] = [new_map(p) if gg in g26 else old_map(p) for p, gg in
                      zip(A.player_id.to_numpy(np.int64), A.game_id.to_numpy(np.int64))]
    A["blk"] = np.maximum(A.time_seconds.to_numpy() // (WINDOW * 60.0), 0).astype(np.int64)
    A = A.sort_values(["game_id", "period_id", "blk"], kind="stable").reset_index(drop=True)
    print(f"액션 {len(A):,} · 경기 {A.game_id.nunique():,} · σ={SIGMA:g} T={WINDOW:g}분"
          + (" · FLAT(공간 무시)" if FLAT else ""), flush=True)

    key = A.game_id.to_numpy() * 10 ** 7 + A.player_id.to_numpy()
    uk, kinv = np.unique(key, return_inverse=True)
    NCOL = len(CHAN) * NB
    OV = np.zeros((len(uk), NCOL)); ON = np.zeros((len(uk), NCOL))

    gx = A.start_x.to_numpy(float); gy = A.start_y.to_numpy(float)
    val = A.vaep_value.to_numpy(float); typ = A.type_id.to_numpy()
    tm = A.team_id.to_numpy(); home = A.home_team_id.to_numpy()
    # 밴드: 홈 선수는 +x, 원정 선수는 뒤집은 x (그 선수 자신의 공격 방향)
    bh = np.clip((gx / L * NB).astype(int), 0, NB - 1)
    ba = np.clip(((L - gx) / L * NB).astype(int), 0, NB - 1)
    shareable = (val != 0.0) & ~np.isin(typ, list(RESTART))

    gk = A.game_id.to_numpy() * 100 + A.period_id.to_numpy() * 10 + np.minimum(A.blk.to_numpy(), 9)
    bnd = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1], True])
    inv2s2 = 1.0 / (2.0 * SIGMA * SIGMA)
    nblk = len(bnd) - 1
    for bi in range(nblk):
        s, e = bnd[bi], bnd[bi + 1]
        pl = kinv[s:e]                                    # 전역 (경기,선수) 행
        up, ploc = np.unique(pl, return_inverse=True)     # 블록 내 선수
        npl = len(up)
        if npl < 2: continue
        act = np.flatnonzero(shareable[s:e])
        if not len(act): continue
        ex, ey = gx[s:e], gy[s:e]
        ax, ay = ex[act], ey[act]
        if FLAT:
            K = np.ones((len(act), e - s))
        else:
            K = np.exp(-((ax[:, None] - ex[None, :]) ** 2 + (ay[:, None] - ey[None, :]) ** 2) * inv2s2)
        H = np.zeros((len(act), npl))
        np.add.at(H.T, ploc, K.T)                         # 선수별 커널 합 (n_a, n_p)
        pteam = np.zeros(npl, np.int64); pteam[ploc] = tm[s:e]
        phome = (pteam == home[s])                        # 그 선수가 홈인가
        side0 = pteam == pteam[0]
        tot0 = H @ side0; tot1 = H @ (~side0)
        aloc = ploc[act]; aside = side0[aloc]
        Ha = H[np.arange(len(act)), aloc]
        den_m = np.where(aside, tot0, tot1) - Ha
        den_o = np.where(aside, tot1, tot0)
        opp = side0[None, :] != aside[:, None]            # (n_a, n_p) 상대 여부 — 실행자를 빼기 전에 잡는다
        mate = ~opp.copy()
        mate[np.arange(len(act)), aloc] = False           # 실행자는 자기에게 안 준다
        v = val[s:e][act]
        cm = np.where(mate & (den_m[:, None] > 0), H * v[:, None] / np.maximum(den_m, 1e-12)[:, None], 0.0)
        co = np.where(opp & (den_o[:, None] > 0), -H * v[:, None] / np.maximum(den_o, 1e-12)[:, None], 0.0)
        sm = np.where(mate & (den_m[:, None] > 0), H / np.maximum(den_m, 1e-12)[:, None], 0.0)
        so = np.where(opp & (den_o[:, None] > 0), H / np.maximum(den_o, 1e-12)[:, None], 0.0)
        band = np.where(phome[None, :], bh[s:e][act][:, None], ba[s:e][act][:, None])
        pos = (v > 0)
        for arr_v, arr_n, ch_pos, ch_neg in ((cm, sm, 0, 1), (co, so, 2, 3)):
            ch = np.where(pos, ch_pos, ch_neg)[:, None] * NB + band
            idx = up[None, :] * NCOL + ch                 # 전역 평탄 인덱스
            np.add.at(OV.reshape(-1), idx.ravel(), arr_v.ravel())
            np.add.at(ON.reshape(-1), idx.ravel(), arr_n.ravel())
        if bi % 2000 == 0: print(f"  블록 {bi:,}/{nblk:,}", flush=True)

    cols = [f"{k}_{c}_{b}" for k, M in (("bv", OV), ("bn", ON)) for c in CHAN for b in range(NB)]
    Q = pd.DataFrame(np.hstack([OV, ON]).astype(np.float32), columns=cols)
    Q.insert(0, "player_id", (uk % 10 ** 7).astype(np.int64))
    Q.insert(0, "game_id", (uk // 10 ** 7).astype(np.int64))
    Q.to_parquet(OUT, index=False)
    # 주 피처 파일에도 붙인다 — 릿지 기준선과 추천기 사례 표가 읽는 것은 이 파일이고,
    # 여기 없으면 "같은 입력 위의 선형 모델"이라는 비교가 성립하지 않는다(build_defresp 의 dr_ 과 같은 방식).
    MERGE_INTO = os.environ.get("VP_MERGE", "onball_gk_resid_merged_defresp_ref.parquet")
    if MERGE_INTO:
        tgt = ROOT / "outputs" / MERGE_INTO
        if tgt.exists():
            P = pd.read_parquet(tgt); n0 = P.shape[1]
            P = P[[c for c in P.columns if not c.startswith(("bv_", "bn_"))]]      # 멱등
            M = P.merge(Q, on=["game_id", "player_id"], how="left")
            new = [c for c in Q.columns if c not in ("game_id", "player_id")]
            M[new] = M[new].fillna(0.0)
            M.to_parquet(tgt, index=False)
            print(f"  주 피처 파일 병합: {MERGE_INTO} · 열 {n0} → {M.shape[1]} (+{len(new)})", flush=True)
        else:
            print(f"  병합 건너뜀 (없음): {MERGE_INTO}", flush=True)
    tot = Q[[c for c in Q.columns if c.startswith("bv_")]].to_numpy().sum()
    print(f"\n선수-경기 {len(Q):,} × {len(cols)}열 → {OUT.name}")
    print(f"  합 {tot:+.2f} (동료 +V 와 상대 -V 가 상쇄되므로 0 에 가까워야 정상)")
    for c in CHAN:
        s = Q[[f"bv_{c}_{b}" for b in range(NB)]].to_numpy().sum()
        n = Q[[f"bn_{c}_{b}" for b in range(NB)]].to_numpy().sum()
        print(f"  {c:6s} 값 합 {s:+10.2f} · 소프트 횟수 {n:12,.0f}")
    return Q


if __name__ == "__main__":
    main()
