"""이벤트 유형 세분 피처 — 기존 682열에 **추가 열**로 붙인다 (파이프라인 체인 재실행 없음).

가족 (선수-경기 행, 공격 방향 정규화, 밴드 6 = x 6분할)
  pv_/pn_  패스 세분: 방향(f 전진≥10m · l 측면 · b 후진) × 길이(s <30m · L ≥30m) × 결과(0/1) × 밴드  → VAEP 합 · 횟수   (72 열씩)
  rn_/rx_  리시브: 성공 패스를 받은 위치 밴드별 횟수 · 그 위치의 xT 합 (xt_grid_12x8)                         (6 열씩)
  sv_/sn_  슛 세분: 부위(foot/head/other) × 결과(0/1) × 밴드                                                  (36 열씩)
  dv_/dn_  수비 세분: 태클·인터셉트·클리어·파울 × 결과(0/1) × 밴드                                            (48 열씩)
출력: outputs/onball_gk_resid_merged_defresp_ref.parquet (기존 열 + 위 가족)
실행: python -m experiments.build_refined
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR  # noqa: E402
from segment_panel import pid_maps  # noqa: E402

NB = 6; SRC = ROOT / "outputs" / os.environ.get("ONBALL_FILE", "onball_gk_resid_merged_defresp.parquet")
OUT = Path(str(SRC).replace(".parquet", "_ref.parquet"))
PASS = {0, 1, 3, 4, 5, 6}; SHOT = {11, 12, 13}; DEF = {9: "tk", 10: "ic", 18: "cl", 8: "fl"}


def main():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date); g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}; home = {int(r.game_id): int(r.home_team_id) for r in g.itertuples(index=False)}
    g26 = set(g[g.season == 2026].game_id.astype(int)); old_map, new_map = pid_maps()
    xt = np.load(ROOT / "outputs/xt_grid_12x8.npy").reshape(-1)
    sp = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet", columns=["game_id", "action_id", "period_id", "time_seconds", "team_id", "player_id", "start_x", "start_y", "end_x", "end_y", "type_id", "bodypart_id", "result_id"])
    v = pd.read_parquet(VAEP_OUTPUT_DIR / "vaep_oof.parquet", columns=["game_id", "action_id", "vaep_value"])
    df = sp.merge(v, on=["game_id", "action_id"], how="left"); df["vaep_value"] = df.vaep_value.fillna(0.0)
    df = df[df.game_id.isin(gpos)].dropna(subset=["player_id", "team_id", "start_x", "start_y"]).sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)
    gid = df.game_id.to_numpy(int); tm = df.team_id.to_numpy(int); ty = df.type_id.to_numpy(int); rs = (df.result_id.to_numpy(int) == 1).astype(int); bp = df.bodypart_id.to_numpy(int); va = df.vaep_value.to_numpy(float)
    away = tm != np.array([home[gg] for gg in gid])
    sx = np.where(away, 105 - df.start_x, df.start_x); sy = np.where(away, 68 - df.start_y, df.start_y)
    ex = np.where(away, 105 - df.end_x.fillna(df.start_x), df.end_x.fillna(df.start_x)); ey = np.where(away, 68 - df.end_y.fillna(df.start_y), df.end_y.fillna(df.start_y))
    band = np.clip((sx / 105 * NB).astype(int), 0, NB - 1); eband = np.clip((ex / 105 * NB).astype(int), 0, NB - 1)
    pid = np.array([new_map(int(p)) if gg in g26 else old_map(int(p)) for p, gg in zip(df.player_id.to_numpy(int), gid)])
    dx = ex - sx; dist = np.hypot(ex - sx, ey - sy)
    dirn = np.where(dx >= 10, "f", np.where(dx <= -10, "b", "l")); leng = np.where(dist >= 30, "L", "s")
    body = np.where(bp == 0, "foot", np.where(bp == 1, "head", "oth"))   # SPADL bodypart: 0 foot, 1 head, 2 other
    rows = {}
    key = list(zip(gid, pid))
    def add(k, col, val): rows.setdefault(k, {}); rows[k][col] = rows[k].get(col, 0.0) + val
    is_pass = np.isin(ty, list(PASS)); is_shot = np.isin(ty, list(SHOT)); is_def = np.isin(ty, list(DEF))
    for i in np.flatnonzero(is_pass):
        c = f"{dirn[i]}{leng[i]}_{rs[i]}_{band[i]}"; add(key[i], "pv_" + c, va[i]); add(key[i], "pn_" + c, 1.0)
    # 리시브: 성공 패스 a → 다음 같은 팀 다른 선수 액션 b
    same = (gid[1:] == gid[:-1]) & (tm[1:] == tm[:-1]) & (pid[1:] != pid[:-1]) & is_pass[:-1] & (rs[:-1] == 1)
    for i in np.flatnonzero(same):
        j = i + 1; c = f"{band[j]}"; add(key[j], "rn_" + c, 1.0); add(key[j], "rx_" + c, float(xt[np.clip((sx[j] / 105 * 12).astype(int), 0, 11) * 8 + np.clip((sy[j] / 68 * 8).astype(int), 0, 7)]))
    for i in np.flatnonzero(is_shot):
        c = f"{body[i]}_{rs[i]}_{band[i]}"; add(key[i], "sv_" + c, va[i]); add(key[i], "sn_" + c, 1.0)
    for i in np.flatnonzero(is_def):
        c = f"{DEF[int(ty[i])]}_{rs[i]}_{band[i]}"; add(key[i], "dv_" + c, va[i]); add(key[i], "dn_" + c, 1.0)
    Q = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0); Q.index = pd.MultiIndex.from_tuples(Q.index, names=["game_id", "player_id"]); Q = Q.reset_index()
    P = pd.read_parquet(SRC); n0 = P.shape[1]
    M = P.merge(Q, on=["game_id", "player_id"], how="left"); new = [c for c in Q.columns if c not in ("game_id", "player_id")]; M[new] = M[new].fillna(0.0)
    M.to_parquet(OUT, index=False)
    fam = pd.Series([c[:2] for c in new]).value_counts().to_dict()
    print(f"행 {len(M):,} · 열 {n0} → {M.shape[1]} (+{len(new)}) · 가족 {fam} · 미매칭 행 {int(M[new].abs().sum(1).eq(0).sum()):,} → {OUT.name}")


if __name__ == "__main__":
    main()
