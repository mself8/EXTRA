"""선수별 온볼 VAEP 피처 — **이벤트 pid 기준, 전 2,540경기**.

기존 `onball_vaep_type.py` 와 두 가지가 다르다.

  ① 좌표 정렬 버그를 고친다.
     기존 코드는 `V["start_x"] = S.start_x.to_numpy()[:len(V)]` 로 **위치 인덱싱**을
     썼는데, `vaep_oof` 와 `spadl_all` 은 **행 순서가 다르다** (game_id 일치율 0.02%,
     action_id 0.23%). 즉 좌표가 전혀 다른 액션에서 왔고, 결과적으로 밴드(_b)와
     존(zv/zn) 차원이 무작위였다. 여기서는 (game_id, action_id) 로 **병합**한다.

  ② 트래킹 pid 브릿지를 쓰지 않는다.
     기존은 pid_assign 브릿지로 트래킹 pid 에 매핑하느라 782경기로 잘렸다.
     이벤트 player_id 를 그대로 쓰면 VAEP 가 있는 **2,540경기 전부**를 덮는다.
     트래킹 기반 피처(pos2·space)와는 못 섞이지만, 이 표는 라인업 점수 트랙
     (이벤트 전용, 6시즌)을 위한 것이다.

분해는 기존과 동일:
  액션 타입 × 성공/실패 (관측 MINCNT 미만은 통합) × 전후 6밴드 : (VAEP, 공격, 수비, 횟수)
  존 12×8 : (VAEP, 횟수)

출력: outputs/onball_vaep_event.parquet  (game_id, player_id, ...)
실행: python -m experiments.onball_vaep_event
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

NB, NZX, NZY = 6, 12, 8
NZ = NZX * NZY
L, W = 105.0, 68.0
MINCNT = 3000
SRC = os.environ.get("VAEP_SRC", "")          # 예: outputs/vaep_oof_pre10.parquet
TAG = os.environ.get("VAEP_TAG", "")          # 출력 접미사
OUT = ROOT / "outputs" / f"onball_vaep_event{TAG}.parquet"


def main():
    src = (ROOT / SRC) if SRC else (VAEP_OUTPUT_DIR / "vaep_oof.parquet")
    print(f"VAEP 원천: {src}")
    V = pd.read_parquet(src,
                        columns=["game_id", "action_id", "team_id", "player_id",
                                 "type_id", "result_id", "offensive_value",
                                 "defensive_value", "vaep_value"])
    S = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet",
                        columns=["game_id", "action_id", "start_x", "start_y"])
    V["game_id"] = V.game_id.astype(np.int64)
    S["game_id"] = S.game_id.astype(np.int64)
    V["action_id"] = V.action_id.astype(np.int64)
    S["action_id"] = S.action_id.astype(np.int64)
    n0 = len(V)
    V = V.merge(S, on=["game_id", "action_id"], how="inner")      # ← 병합(버그 수정)
    print(f"액션 {len(V):,} / {n0:,} 병합 · 경기 {V.game_id.nunique():,}")
    V = V.dropna(subset=["player_id", "start_x", "start_y"])
    V["player_id"] = V.player_id.astype(np.int64)

    gm = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")[["game_id", "home_team_id"]]
    V = V.merge(gm, on="game_id", how="left")
    # SPADL 은 홈이 +x. 우리 규약은 '그 팀의 공격 방향 +x' 이므로 원정은 뒤집는다.
    away = V.team_id.astype(float) != V.home_team_id.astype(float)
    x = np.where(away, L - V.start_x, V.start_x)
    y = np.where(away, W - V.start_y, V.start_y)
    V["xb"] = np.clip((x / L * NB).astype(int), 0, NB - 1)
    V["zi"] = np.clip((x / L * NZX).astype(int), 0, NZX - 1) * NZY + \
              np.clip((y / W * NZY).astype(int), 0, NZY - 1)

    cnt = V.groupby(["type_id", "result_id"]).size()
    big = set(cnt[cnt >= MINCNT].index)
    V["slot"] = [f"{int(t)}_{int(r)}" if (t, r) in big else f"{int(t)}_x"
                 for t, r in zip(V.type_id, V.result_id)]
    SL = sorted(V.slot.unique())
    print(f"범주 {len(SL)}종 (관측 {MINCNT}회 미만은 성공/실패 통합)")
    si = V.slot.map({s: i for i, s in enumerate(SL)}).to_numpy()
    NC = len(SL)

    uk, inv = np.unique(V.game_id.to_numpy() * 10 ** 7 + V.player_id.to_numpy(),
                        return_inverse=True)
    n = len(uk)
    TV = np.zeros((n, NC * NB)); TO = np.zeros((n, NC * NB))
    TD = np.zeros((n, NC * NB)); TN = np.zeros((n, NC * NB))
    ZV = np.zeros((n, NZ)); ZN = np.zeros((n, NZ))
    f = inv * (NC * NB) + si * NB + V.xb.to_numpy()
    np.add.at(TV.reshape(-1), f, V.vaep_value.to_numpy())
    np.add.at(TO.reshape(-1), f, V.offensive_value.to_numpy())
    np.add.at(TD.reshape(-1), f, V.defensive_value.to_numpy())
    np.add.at(TN.reshape(-1), f, 1.)
    fz = inv * NZ + V.zi.to_numpy()
    np.add.at(ZV.reshape(-1), fz, V.vaep_value.to_numpy())
    np.add.at(ZN.reshape(-1), fz, 1.)

    cols = [f"{k}_{s}_{b}" for k in ("tv", "to", "td", "tn") for s in SL for b in range(NB)] + \
           [f"zv_{i:03d}" for i in range(NZ)] + [f"zn_{i:03d}" for i in range(NZ)]
    Q = pd.DataFrame(np.hstack([TV, TO, TD, TN, ZV, ZN]).astype(np.float32), columns=cols)
    Q.insert(0, "player_id", (uk % 10 ** 7).astype(np.int64))
    Q.insert(0, "game_id", (uk // 10 ** 7).astype(np.int64))
    Q.to_parquet(OUT, index=False)
    print(f"선수-경기 {n:,} × {Q.shape[1]-2}열 | 경기 {Q.game_id.nunique():,} → {OUT}")
    return Q


if __name__ == "__main__":
    main()
