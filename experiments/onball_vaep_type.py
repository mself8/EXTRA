"""선수별 온볼 기여 — VAEP 기반, **전체 트래킹 경기**를 덮는다.

왜 다시 만드는가
  기존 온볼 피처는 SoccerMap EPV 를 un_xpass 액션에 돌린 것이라 565경기에 갇혔고,
  타입도 4종(→8종)뿐이었다. VAEP 는 SPADL 2,540경기 전부에 있고 액션 타입이 21종이라
  **태클·인터셉트·클리어런스·키퍼 액션 같은 수비 온볼**까지 들어온다.
  선수 ID 는 경기단위 헝가리안 배정(pid_assign + pid_assign_api, 1,039경기)으로 잇는다.

분해
  액션 타입 21종 × 성공/실패  → 관측이 충분한 범주만 유지
  각 범주 × 전후 6구간 : (VAEP 합, offensive, defensive, 횟수)
  존 12×8            : (VAEP 합, 횟수)
출력: outputs/onball_vaep_player.parquet  (game_id, trk_pid, ...)
"""
import numpy as np, pandas as pd
NB, NZX, NZY = 6, 12, 8
NZ = NZX * NZY
L, W = 105.0, 68.0
MINCNT = 3000                      # 이보다 드문 (타입,성공) 조합은 합쳐서 잡음을 줄인다

A = pd.concat([pd.read_parquet("outputs/pid_assign.parquet"),
               pd.read_parquet("outputs/pid_assign_api.parquet")], ignore_index=True)
A = A[A.share >= 0.5]
BR = {(int(r.game_id), int(r.assigned_pid)): int(r.trk_pid) for r in A.itertuples()}
GL = A.groupby(["assigned_pid", "trk_pid"]).votes.sum().reset_index() \
      .sort_values("votes", ascending=False).drop_duplicates("assigned_pid")
GLB = dict(zip(GL.assigned_pid.astype(int), GL.trk_pid.astype(int)))
print(f"브릿지: 경기단위 {len(BR):,} | 전역 {len(GLB):,}")

V = pd.read_parquet("vaep/output/vaep_oof.parquet",
                    columns=["game_id", "period_id", "time_seconds", "team_id", "player_id",
                             "type_id", "result_id", "offensive_value", "defensive_value", "vaep_value"])
S = pd.read_parquet("vaep/output/spadl_all.parquet",
                    columns=["game_id", "action_id", "start_x", "start_y"])
V["game_id"] = V.game_id.astype(int)
V = V.dropna(subset=["player_id"])
V["player_id"] = V.player_id.astype(int)
V = V.reset_index(drop=True)
V["start_x"] = S.start_x.to_numpy()[:len(V)]; V["start_y"] = S.start_y.to_numpy()[:len(V)]
gm = pd.read_csv("vaep/output/games.csv")[["game_id", "home_team_id"]]
V = V.merge(gm, on="game_id", how="left")
G = pd.read_parquet("outputs/frame_segments.parquet")
keep_g = set(G.game_id.astype(int))
V = V[V.game_id.isin(keep_g)]
print(f"액션 {len(V):,} | 경기 {V.game_id.nunique():,}")

# 트래킹 pid 로 매핑 — 경기단위 우선, 없으면 전역
tp = np.array([BR.get((g, p), GLB.get(p, -1)) for g, p in zip(V.game_id, V.player_id)])
V = V.assign(trk=tp)
print(f"매핑 성공 {np.mean(tp > 0):.3f}")
V = V[V.trk > 0]

# SPADL 은 홈이 +x. 우리 규약은 '그 팀의 공격 방향 +x' 이므로 원정은 뒤집는다
away = V.team_id.astype(float) != V.home_team_id.astype(float)
x = np.where(away, L - V.start_x, V.start_x)
y = np.where(away, W - V.start_y, V.start_y)
V["xb"] = np.clip((x / L * NB).astype(int), 0, NB - 1)
V["zi"] = np.clip((x / L * NZX).astype(int), 0, NZX - 1) * NZY + \
          np.clip((y / W * NZY).astype(int), 0, NZY - 1)

cnt = V.groupby(["type_id", "result_id"]).size()
big = set(cnt[cnt >= MINCNT].index)
slot = [f"{int(t)}_{int(r)}" if (t, r) in big else f"{int(t)}_x" for t, r in
        zip(V.type_id, V.result_id)]
V["slot"] = slot
SL = sorted(V.slot.unique())
print(f"범주 {len(SL)}종 (관측 {MINCNT}회 미만은 성공/실패 통합)")
si = V.slot.map({s: i for i, s in enumerate(SL)}).to_numpy()
NC = len(SL)
uk, inv = np.unique(V.game_id.to_numpy().astype(np.int64) * 10 ** 7 + V.trk.to_numpy(),
                    return_inverse=True)
n = len(uk)
TV = np.zeros((n, NC * NB)); TO = np.zeros((n, NC * NB)); TD = np.zeros((n, NC * NB)); TN = np.zeros((n, NC * NB))
ZV = np.zeros((n, NZ)); ZN = np.zeros((n, NZ))
f = inv * (NC * NB) + si * NB + V.xb.to_numpy()
np.add.at(TV.reshape(-1), f, V.vaep_value.to_numpy())
np.add.at(TO.reshape(-1), f, V.offensive_value.to_numpy())
np.add.at(TD.reshape(-1), f, V.defensive_value.to_numpy())
np.add.at(TN.reshape(-1), f, 1.)
fz = inv * NZ + V.zi.to_numpy()
np.add.at(ZV.reshape(-1), fz, V.vaep_value.to_numpy())
np.add.at(ZN.reshape(-1), fz, 1.)
cols = [f"{k}_{s}_{b}" for k, M in (("tv", TV), ("to", TO), ("td", TD), ("tn", TN))
        for s in SL for b in range(NB)] + \
       [f"zv_{i:03d}" for i in range(NZ)] + [f"zn_{i:03d}" for i in range(NZ)]
Q = pd.DataFrame(np.hstack([TV, TO, TD, TN, ZV, ZN]).astype(np.float32), columns=cols)
Q.insert(0, "pid", (uk % 10 ** 7).astype(np.int32))
Q.insert(0, "game_id", (uk // 10 ** 7).astype(np.int32))
Q.to_parquet("outputs/onball_vaep_player.parquet", index=False)
print(f"선수-경기 {n:,} × {Q.shape[1]-2}열 | 경기 {Q.game_id.nunique():,} → outputs/onball_vaep_player.parquet")
r = V.groupby("slot").agg(VAEP=("vaep_value", "sum"), 수비=("defensive_value", "sum"), 횟수=("vaep_value", "size"))
r["건당"] = r.VAEP / r.횟수
print("\n범주별 (VAEP 합 기준 상위·하위)")
print(pd.concat([r.nlargest(6, "VAEP"), r.nsmallest(4, "VAEP")]).round(4).to_string())
