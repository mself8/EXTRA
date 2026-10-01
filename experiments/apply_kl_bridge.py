"""K리그 공식 등록번호 브릿지(kleague_bridge) 를 파이프라인 파일에 적용한다.

현재 파일의 ID 공간: 2021~25 행 = PMb(backup_pre_person_bridge/pid_merge.json) 적용 공간,
2026 행 = 구 사람 브릿지(B2 = pid_merge − PMb) 적용 공간(미매칭은 원시 명단 pid).
새 맵: kl_map_old.json (원시 2021~25 pid → 정규 원시 pid), kl_map_2026.json (원시 2026 pid → (공간, pid)).
합성해 현재 공간의 맵 F_old, F_26 을 만들고 players.csv·onball 파케이·opp_panel·red_cards·subs 에 적용,
pid_merge_old.json(순수 2021~25 병합) 과 pid_merge.json(+2026 원시→정규) 을 다시 쓴다.
백업: outputs/backup_pre_kl_bridge/  (멱등: 백업이 있으면 백업본에서 다시 적용)
실행: python -m experiments.apply_kl_bridge
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

BK = ROOT / "outputs" / "backup_pre_kl_bridge"
PQ = ["onball_gk_resid_merged_defresp.parquet", "onball_gk_resid_merged_defresp_ref.parquet", "onball_gk_resid_merged_relb.parquet",
      "onball_gk_resid_merged.parquet", "onball_vaep_event.parquet", "opp_panel.parquet", "red_cards.parquet", "subs.parquet", "cards.parquet", "player_level.parquet"]


def _bk(p: Path) -> Path:
    """백업본 경로 (없으면 만들고), 항상 백업본을 원본으로 읽는다 → 멱등."""
    BK.mkdir(exist_ok=True); b = BK / p.name
    if not b.exists(): shutil.copy2(p, b)
    return b


def main():
    PMb = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/backup_pre_person_bridge/pid_merge.json")).items()}
    PM = {int(k): int(v) for k, v in json.load(open(_bk(ROOT / "outputs/pid_merge.json"))).items()}
    B2 = {k: v for k, v in PM.items() if PMb.get(k) != v}                       # 구 사람 브릿지 (2026 원시 → 과거 pid)
    m_old = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/kl_map_old.json")).items()}
    m_26 = {int(k): (v[0], int(v[1])) for k, v in json.load(open(ROOT / "outputs/kl_map_2026.json")).items()}
    cur = lambda p: PMb.get(int(p), int(p))
    # F_old: 현재 공간 → 정규(현재 공간)
    F_old, conflict = {}, 0
    for p, c in m_old.items():
        a, t = cur(p), cur(c)
        if a in F_old and F_old[a] != t: conflict += 1; continue
        F_old[a] = t
    # F_26: 2026 행의 현재 id → 정규
    F_26 = {}
    for r, (ns, p) in m_26.items():
        a = B2.get(r, r); t = F_old.get(cur(p), cur(p)) if ns == "old" else p
        F_26[a] = t
    # 2026 원시 pid 가 2021~25 공간의 다른 사람과 숫자 충돌하면 별도 대역(+10,000,000)으로 옮긴다:
    #  ① 2026 내부 정규 id(n26) 가 old 공간에 있으면, ② 미매칭 2026 원시 pid 가 old 공간 또는 정규 목표와 겹치면
    L = pd.read_parquet(ROOT / "outputs/lineup_players_raw.parquet"); raw26 = set(L[L.season >= 2026].pid.astype(int))
    plc = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv", usecols=["game_id", "player_id"]); gg = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv", usecols=["game_id", "season"])
    oldspace = set(plc[plc.game_id.isin(set(gg[gg.season < 2026].game_id))].player_id.astype(int)) | set(PMb) | set(PMb.values()) | set(F_old.values())
    OFF = 10_000_000; targets = set(F_26.values()); moved = 0
    for r in raw26:
        if r in m_26: 
            ns, p = m_26[r]
            if ns == "n26" and p in oldspace: F_26[B2.get(r, r)] = p + OFF; moved += 1
        else:                                                                  # 새 맵 미매칭: 구 브릿지 연결(a≠r)은 유지, 순수 원시(a==r)만 충돌 시 대역 이동
            a = B2.get(r, r)
            if a == r and (a in oldspace or a in targets): F_26[a] = a + OFF; moved += 1
    print(f"2026 숫자 충돌 회피(+{OFF:,}) {moved}", flush=True)
    new_old = sum(k != v for k, v in F_old.items()); new_26 = sum(k != v for k, v in F_26.items())
    print(f"F_old {len(F_old):,} (비항등 {new_old:,} · PMb 와 충돌 {conflict}) · F_26 {len(F_26):,} (비항등 {new_26:,})", flush=True)

    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv", usecols=["game_id", "season"]); g26 = set(g[g.season == 2026].game_id.astype(int))
    def remap(df: pd.DataFrame, col="player_id") -> tuple[pd.DataFrame, int]:
        # game_id 가 없는 표(player_level 처럼 선수 단위)는 전부 옛 공간으로 본다
        m26 = df.game_id.isin(g26).to_numpy() if "game_id" in df.columns else np.zeros(len(df), bool)
        v = df[col].to_numpy().copy(); ok = pd.notna(v); n = 0
        for i in np.flatnonzero(ok):
            x = int(v[i]); y = F_26.get(x, x) if m26[i] else F_old.get(x, x)
            if y != x: v[i] = y; n += 1
        df = df.copy(); df[col] = v; return df, n

    # players.csv (+ 2026 한글 이름 복원)
    src = VAEP_OUTPUT_DIR / "players.csv"; pl = pd.read_csv(_bk(src)); pl, n = remap(pl)
    dup = pl.duplicated(["game_id", "player_id"]).sum()
    bio = pd.read_parquet(ROOT / "outputs/player_bio.parquet")
    # 바이오를 현재 공간으로: pid_canon(원시) → cur → F_old
    bio["pid"] = bio.pid_canon.map(lambda p: F_old.get(cur(p), cur(p)))
    bio = bio.drop_duplicates("pid"); bio.to_parquet(ROOT / "outputs/player_bio_cur.parquet", index=False)
    ko = dict(zip(bio.pid, bio.ko)); m = pl.game_id.isin(g26) & pl.player_id.isin(ko)
    pl.loc[m, "nickname"] = pl.loc[m, "player_id"].map(ko)
    pl.to_csv(src, index=False)
    print(f"players.csv: 재매핑 {n:,} · 중복(경기,선수) {dup} · 2026 한글 이름 복원 {int(m.sum()):,} 행", flush=True)
    for fn in PQ:
        p = ROOT / "outputs" / fn
        if not p.exists(): print(f"건너뜀 (없음): {fn}"); continue
        P = pd.read_parquet(_bk(p)); P, n = remap(P)
        d = P.duplicated(["game_id", "player_id"]).sum() if "game_id" in P else 0
        if d and fn.startswith("onball"):                                     # 같은 경기에 두 pid 가 한 사람으로 합쳐진 경우: 행 합산
            keys = ["game_id", "player_id"] + [c for c in ("team_id",) if c in P.columns]
            num = P.select_dtypes("number").columns.difference(keys); P = P.groupby(keys, as_index=False)[list(num)].sum()
        P.to_parquet(p, index=False); print(f"{fn}: 재매핑 {n:,} · 중복 {d}", flush=True)
    # pid_merge_old.json / pid_merge.json
    PMo = dict(PMb)
    for p, c in m_old.items(): PMo[p] = F_old.get(cur(p), cur(p))
    PMo = {k: v for k, v in PMo.items() if k != v}
    json.dump({str(k): v for k, v in PMo.items()}, open(ROOT / "outputs/pid_merge_old.json", "w"))
    # 2026 원시 명단 pid → 정규 (별도 파일: 2021~25 공간과 숫자 충돌하므로 평면 맵에 섞지 않는다)
    M26 = {}
    for r in raw26 | set(B2):                                                   # 모든 2026 원시 명단 pid → 정규 (미매칭·대역 이동 포함)
        a = B2.get(r, r); t = F_26.get(a, a)
        if t != r: M26[r] = t
    json.dump({str(k): v for k, v in PMo.items()}, open(ROOT / "outputs/pid_merge.json", "w"))
    json.dump({str(k): v for k, v in M26.items()}, open(ROOT / "outputs/pid_merge_2026.json", "w"))
    print(f"pid_merge_old.json = pid_merge.json {len(PMo):,} (2021~25 공간) · pid_merge_2026.json {len(M26):,}", flush=True)
    # 검증: 2026 선발 중 2021~25 이력 있는 사람 비율
    pl26 = pl[pl.game_id.isin(g26) & (pl.is_starter == True)]; hist = set(pl[~pl.game_id.isin(g26)].player_id)  # noqa: E712
    print(f"2026 선발 행 중 2021~25 이력 보유 {pl26.player_id.isin(hist).mean() * 100:.1f}% · 바이오 보유 {pl26.player_id.isin(set(bio.pid)).mean() * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
