"""사람 브릿지(2026 명단 pid → 과거 pid) 일괄 적용. 멱등.

적용 대상
  ① vaep_output/players.csv        — 2026 행 player_id 재매핑 (이름은 그대로)
  ② outputs/onball_*.parquet 3종    — 2026 행 (이벤트 브릿지가 이미 명단 공간으로
                                       옮겨놨으므로 명단→과거 한 단계만 더)
  ③ outputs/opp_panel.parquet       — player_level_dynamic 의 색인
  ④ outputs/pid_merge.json 확장     — 원시 lineup.json 판독 경로(gap_lane._cid,
                                       viz_gap_constrained 등)가 자동 정규화되도록

안전장치
  · 브릿지 키가 2021~25 pid 우주와 숫자 충돌하면 그 키는 pid_merge 확장에서 제외
    (전역 조회라 과거 선수를 오염시킨다) — 행 단위(2026 한정) 적용은 유지.
  · 미매칭 2026 pid 가 브릿지 값(과거 pid)과 충돌하면 병합 오염 — 건수만 보고
    (시프트는 원시 판독과 어긋나므로 하지 않는다).
  · 전 대상 백업: outputs/backup_pre_person_bridge/

실행: python -m experiments.apply_person_bridge
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

BK = ROOT / "outputs" / "backup_pre_person_bridge"
PQ = ["onball_vaep_event.parquet", "onball_gk_resid_merged.parquet",
      "onball_gk_resid_merged_p90.parquet", "opp_panel.parquet"]


def main():
    B = {int(k): int(v) for k, v in
         json.load(open(ROOT / "outputs" / "person_bridge_2026.json")).items()}
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g26 = set(g[g.season == 2026].game_id.astype(int))
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    old_pids = set(pl[~pl.game_id.isin(g26)].player_id.astype(int))
    cur26 = set(pl[pl.game_id.isin(g26)].player_id.astype(int))

    done = {k for k in B if k not in cur26}          # 이미 재매핑돼 2026 행에 없는 키
    B = {k: v for k, v in B.items() if k in cur26}   # 증분: 아직 남은 키만 적용
    if not B:
        print(f"적용할 신규 키 없음 (기적용 {len(done)}) — 종료"); return
    if done:
        print(f"증분 적용: 신규 {len(B)} (기적용 {len(done)})")

    key_clash = set(B) & old_pids
    val_clash = (cur26 - set(B)) & set(B.values())
    print(f"브릿지 {len(B)}건 · 키↔과거우주 충돌 {len(key_clash)} (pid_merge 확장 제외) · "
          f"미매칭↔값 충돌 {len(val_clash)} (보고만)")

    BK.mkdir(exist_ok=True)
    # ① players.csv
    src = VAEP_OUTPUT_DIR / "players.csv"
    if not (BK / "players.csv").exists(): shutil.copy2(src, BK / "players.csv")
    m = pl.game_id.isin(g26)
    pl.loc[m, "player_id"] = pl.loc[m, "player_id"].map(lambda x: B.get(int(x), int(x)))
    pl.to_csv(src, index=False)
    print(f"players.csv: 2026 행 {int(m.sum()):,} 중 재매핑 "
          f"{int(pl.loc[m, 'player_id'].isin(set(B.values())).sum()):,}")

    # ②③ parquet
    for fn in PQ:
        p = ROOT / "outputs" / fn
        if not p.exists(): print(f"건너뜀 (없음): {fn}"); continue
        if not (BK / fn).exists(): shutil.copy2(p, BK / fn)
        P = pd.read_parquet(p)
        mm = P.game_id.isin(g26)
        n0 = P.loc[mm, "player_id"].isin(set(B)).sum()
        P.loc[mm, "player_id"] = P.loc[mm, "player_id"].map(lambda x: B.get(int(x), int(x)))
        P.to_parquet(p, index=False)
        print(f"{fn}: 2026 행 {int(mm.sum()):,} · 재매핑 {int(n0):,}")

    # ④ pid_merge 확장
    pmf = ROOT / "outputs" / "pid_merge.json"
    if not (BK / "pid_merge.json").exists(): shutil.copy2(pmf, BK / "pid_merge.json")
    M = {int(k): int(v) for k, v in json.load(open(pmf)).items()}
    add = {k: v for k, v in B.items() if k not in key_clash and M.get(k, v) == v}
    conf = [k for k in B if k in M and M[k] != B[k] and k not in key_clash]
    M.update(add)
    json.dump({str(k): v for k, v in M.items()}, open(pmf, "w"))
    print(f"pid_merge.json: +{len(add)} (기존 충돌 {len(conf)} 제외) → 총 {len(M)}")


if __name__ == "__main__":
    main()
