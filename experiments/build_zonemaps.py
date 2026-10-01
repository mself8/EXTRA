"""선수-경기별 **범주별 2-D 존 지도** — 원 데이터(액션 좌표·유형·결과·부위·VAEP)를 최대한 보존한 딥러닝 입력.

행 = (game_id, player_id 정규) · 열 = 범주 12 × 면 2 (횟수 n, VAEP 값 v) × 존 8(y)×12(x) = 2,304
범주: pass_ok · pass_fail · cross_ok · cross_fail · setpiece(스로인·프리킥·코너·골킥) · dribble(테이크온·드리블) · shot_foot · shot_head ·
      def_win(태클 성공·인터셉트) · def_fail(태클 실패·파울) · clear(클리어·배드터치) · keeper
좌표 = 액션 시작점(SPADL, 공격 방향 좌→우 정규화), VAEP = act_vaep_cache 의 offensive+defensive 값.
선수 ID = spadl 원시 → pid_merge(2021~25) / pid_bridge_2026→pid_merge_2026(2026) 사슬로 정규화(onball 파케이와 동일 공간).
출력: outputs/zonemaps.parquet  (game_id, player_id, z{cat}_{n|v}_{000..095})
실행: python -m experiments.build_zonemaps
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

NZX, NZY = 12, 8; NZ = NZX * NZY; L, W = 105.0, 68.0
CATS = ["pass_ok", "pass_fail", "cross_ok", "cross_fail", "setpiece", "dribble", "shot_foot", "shot_head", "def_win", "def_fail", "clear", "keeper", "pass_to", "cross_to", "recv_pass", "recv_cross"]
# pass_to/cross_to: 성공 패스·크로스의 **도착점**(주는 위치맵) · recv_pass/recv_cross: 같은 팀 다음 액션 선수의 **받는 위치맵**(패스 도착점, 받는 선수에게 귀속)


def category(ty, rs, bp):
    if ty == 0: return 0 if rs == 1 else 1
    if ty == 1: return 2 if rs == 1 else 3
    if ty in (2, 3, 4, 5, 6, 22): return 4
    if ty in (7, 21): return 5
    if ty in (11, 12, 13): return 7 if bp == 1 else 6
    if ty in (9, 10): return 8 if (ty == 10 or rs == 1) else 9
    if ty == 8: return 9
    if ty in (18, 19): return 10
    if ty in (14, 15, 16, 17): return 11
    return -1


def main():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); G26 = set(g[g.season >= 2026].game_id.astype(int))
    M = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_merge.json")).items()}; M26 = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_merge_2026.json")).items()}; B26 = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_bridge_2026.json")).items()}
    S = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet", columns=["game_id", "action_id", "period_id", "time_seconds", "team_id", "player_id", "type_id", "result_id", "bodypart_id", "start_x", "start_y", "end_x", "end_y"]).dropna(subset=["player_id"]).sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)
    V = pd.read_parquet(ROOT / "outputs/act_vaep_cache.parquet", columns=["game_id", "action_id", "offensive_value", "defensive_value"])
    S = S.merge(V, on=["game_id", "action_id"], how="left"); S["v"] = (S.offensive_value.fillna(0) + S.defensive_value.fillna(0)).astype(np.float32)
    gid = S.game_id.to_numpy(int); raw = S.player_id.to_numpy(int)
    pid = np.array([(M26.get(B26.get(p, p), B26.get(p, p)) if gg in G26 else M.get(p, p)) for p, gg in zip(raw, gid)], dtype=np.int64)
    cat = np.array([category(int(t), int(r), int(b)) for t, r, b in zip(S.type_id, S.result_id, S.bodypart_id)]); keep = cat >= 0
    zx = np.clip((S.start_x.to_numpy() / L * NZX).astype(int), 0, NZX - 1); zy = np.clip((S.start_y.to_numpy() / W * NZY).astype(int), 0, NZY - 1); zi = zy * NZX + zx
    # 주는 위치맵(도착점, 패서 귀속)과 받는 위치맵(도착점, 다음 액션 선수 귀속): 성공 패스(0)/크로스(1) 뒤 같은 경기·같은 팀 다음 액션
    ty = S.type_id.to_numpy(int); rs = S.result_id.to_numpy(int); tm = S.team_id.to_numpy(); ex = S.end_x.to_numpy(); ey = S.end_y.to_numpy(); vv = S.v.to_numpy()
    nxt_same = np.zeros(len(S), bool); nxt_same[:-1] = (gid[1:] == gid[:-1]) & (tm[1:] == tm[:-1]) & (pid[1:] != pid[:-1])
    ez = np.clip((np.nan_to_num(ey) / W * NZY).astype(int), 0, NZY - 1) * NZX + np.clip((np.nan_to_num(ex) / L * NZX).astype(int), 0, NZX - 1)
    ok_pass = (ty == 0) & (rs == 1); ok_cross = (ty == 1) & (rs == 1)
    ext_g, ext_p, ext_c, ext_z, ext_v = [], [], [], [], []
    for mask, c_to, c_rv in ((ok_pass, 12, 14), (ok_cross, 13, 15)):
        i = np.flatnonzero(mask); ext_g.append(gid[i]); ext_p.append(pid[i]); ext_c.append(np.full(len(i), c_to)); ext_z.append(ez[i]); ext_v.append(vv[i])          # 주는 맵
        j = i[nxt_same[i]]; ext_g.append(gid[j]); ext_p.append(pid[j + 1]); ext_c.append(np.full(len(j), c_rv)); ext_z.append(ez[j]); ext_v.append(vv[j])      # 받는 맵(다음 선수)
    G_ = np.concatenate([gid[keep]] + ext_g); P_ = np.concatenate([pid[keep]] + ext_p); C_ = np.concatenate([cat[keep]] + ext_c); Z_ = np.concatenate([zi[keep]] + ext_z); V_ = np.concatenate([S.v.to_numpy()[keep]] + ext_v)
    key = pd.MultiIndex.from_arrays([G_, P_]); codes, uniq = pd.factorize(key); n = len(uniq)
    Zn = np.zeros((n, len(CATS) * NZ), np.float32); Zv = np.zeros((n, len(CATS) * NZ), np.float32)
    col = C_ * NZ + Z_
    np.add.at(Zn, (codes, col), 1.0); np.add.at(Zv, (codes, col), V_)
    names = [f"z{c}_n_{i:03d}" for c in CATS for i in range(NZ)] + [f"z{c}_v_{i:03d}" for c in CATS for i in range(NZ)]
    out = pd.DataFrame(np.hstack([Zn, Zv]), columns=names); out.insert(0, "player_id", [int(k[1]) for k in uniq]); out.insert(0, "game_id", [int(k[0]) for k in uniq])
    out.to_parquet(ROOT / "outputs/zonemaps.parquet", index=False)
    P = pd.read_parquet(ROOT / "outputs/onball_gk_resid_merged_defresp_ref.parquet", columns=["game_id", "player_id"])
    cov = P.merge(out[["game_id", "player_id"]].assign(has=1), on=["game_id", "player_id"], how="left").has.fillna(0).mean()
    print(f"존 지도 {len(out):,} 선수-경기 · 열 {len(names):,} · onball 행 커버리지 {cov:.3f} · 액션 사용 {keep.sum():,}/{len(keep):,}", flush=True)


if __name__ == "__main__":
    main()
