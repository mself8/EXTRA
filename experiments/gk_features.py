"""골키퍼 피처 — 온볼 VAEP 분해가 못 잡는 것들.

왜
  GK 액션 202,073건 중 골키퍼 고유 액션은 0.54% 뿐이고, VAEP 는 선방을 거의
  보상하지 않는다(결과 누수 수정 후에도 세이브 +0.145). 그 결과 라인업 점수의
  GK 선발 판별 AUC 가 0.582 로 필드 선수(0.64~0.72)에 못 미치고,
  송범근(79경기 전부 선발) 이 최근 5경기 0분인 백업에게 밀린다.

무엇을
  ① 기대 대비 선방  gsax = Σ xG(마주한 슛, 페널티 제외) − 실점
  ② 마주한 양       n_faced · xg_faced   (수비 부담 — 좋은 팀 GK 는 적게 마주한다)
  ③ 처리 방식       세이브·캐치·펀치·픽업 횟수와 VAEP
  ④ 스위핑          페널티 박스 밖 수비 액션 수
  ⑤ 배급            패스·골킥 VAEP 와 성공률

슛을 마주한 GK 는 **그 시각에 상대 골문을 지키던** 선수로 귀속한다(교체 반영).

출력: outputs/gk_feat.parquet  (game_id, player_id, gk_*)
실행: python -m experiments.gk_features
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

KEEPER = {14: "save", 15: "claim", 16: "punch", 17: "pickup"}
BOX_X, BOX_Y0, BOX_Y1 = 16.5, (68 - 40.3) / 2, (68 + 40.3) / 2


def gk_stints():
    """(game_id, team_id) -> [(pid, t0, t1)]  그 팀 골문을 지킨 구간(분)."""
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    POS = pl.groupby("player_id").starting_position_name.agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else "?")
    S = pd.read_parquet(ROOT / "outputs" / "subs.parquet")
    S["pos"] = S.player_id.map(POS)
    gsub = S[S.pos == "GK"]
    sub_out = {(int(r.game_id), int(r.team_id), int(r.player_id)): float(r.tmin)
               for r in gsub[gsub.dir == "Off"].itertuples(index=False)}
    sub_in = {(int(r.game_id), int(r.team_id), int(r.player_id)): float(r.tmin)
              for r in gsub[gsub.dir == "On"].itertuples(index=False)}
    out = defaultdict(list)
    gk = pl[(pl.player_id.map(POS) == "GK") & (pl.minutes_played.fillna(0) > 0)]
    for r in gk.itertuples(index=False):
        k = (int(r.game_id), int(r.team_id), int(r.player_id))
        t0 = sub_in.get(k, 0.0)
        t1 = sub_out.get(k, 130.0)
        if r.is_starter != 1 and k not in sub_in:
            t0 = max(0.0, 130.0 - float(r.minutes_played or 0))
        out[(int(r.game_id), int(r.team_id))].append((int(r.player_id), t0, t1))
    return out


def main():
    sh = pd.read_parquet(ROOT / "outputs" / "shot_xg.parquet")
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    opp = {}
    for r in g.itertuples(index=False):
        opp[(int(r.game_id), int(r.home_team_id))] = int(r.away_team_id)
        opp[(int(r.game_id), int(r.away_team_id))] = int(r.home_team_id)
    ST = gk_stints()
    rows = defaultdict(lambda: defaultdict(float))
    miss = 0
    for r in sh.itertuples(index=False):
        gid, tid = int(r.game_id), int(r.team_id)
        d = opp.get((gid, tid))
        if d is None: miss += 1; continue
        cand = ST.get((gid, d), [])
        who = None
        for pid, t0, t1 in cand:
            if t0 <= float(r.minute) < t1: who = pid; break
        if who is None and cand: who = cand[0][0]
        if who is None: miss += 1; continue
        k = (gid, who)
        pen = bool(r.pen)
        rows[k]["gk_faced"] += 1.0
        rows[k]["gk_xg_faced"] += float(r.xg)
        rows[k]["gk_ga"] += float(r.goal)
        if not pen:
            rows[k]["gk_faced_op"] += 1.0
            rows[k]["gk_xg_op"] += float(r.xg)
            rows[k]["gk_ga_op"] += float(r.goal)
    print(f"슛 {len(sh):,} 귀속 · 미귀속 {miss:,}")

    V = pd.read_parquet(ROOT / "outputs" / "vaep_src_pre10.parquet")
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    POS = pl.groupby("player_id").starting_position_name.agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else "?")
    V["pos"] = V.player_id.map(POS)
    K = V[V.pos == "GK"]
    for t_, nm in KEEPER.items():
        s = K[K.type_id == t_].groupby(["game_id", "player_id"]).agg(
            n=("vaep_value", "size"), v=("vaep_value", "sum"))
        for (gid, pid), rr in s.iterrows():
            rows[(int(gid), int(pid))][f"gk_{nm}_n"] += float(rr.n)
            rows[(int(gid), int(pid))][f"gk_{nm}_v"] += float(rr.v)
    # 배급 · 스위핑
    sp = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet",
                         columns=["game_id", "action_id", "player_id", "type_id",
                                  "result_id", "start_x", "start_y"])
    sp = sp[sp.player_id.isin(set(K.player_id))]
    sp = sp.merge(V[["game_id", "action_id", "vaep_value"]], on=["game_id", "action_id"], how="left")
    dis = sp[sp.type_id.isin([0, 1, 22])]
    a = dis.groupby(["game_id", "player_id"]).agg(n=("type_id", "size"),
                                                  ok=("result_id", lambda s: float((s == 1).sum())),
                                                  v=("vaep_value", "sum"))
    for (gid, pid), rr in a.iterrows():
        k = (int(gid), int(pid))
        rows[k]["gk_dist_n"] += float(rr.n); rows[k]["gk_dist_ok"] += float(rr.ok)
        rows[k]["gk_dist_v"] += float(rr.v)
    sw = sp[sp.type_id.isin([9, 10, 18, 7]) & (sp.start_x > BOX_X)]
    b = sw.groupby(["game_id", "player_id"]).size()
    for (gid, pid), n in b.items():
        rows[(int(gid), int(pid))]["gk_sweep_n"] += float(n)

    F = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    F.index = pd.MultiIndex.from_tuples(F.index, names=["game_id", "player_id"])
    F = F.reset_index()
    F["gk_gsax"] = F.get("gk_xg_op", 0) - F.get("gk_ga_op", 0)
    F["gk_gsax_all"] = F.get("gk_xg_faced", 0) - F.get("gk_ga", 0)
    F["gk_save_rate"] = np.where(F.get("gk_faced_op", 0) > 0,
                                 1 - F.get("gk_ga_op", 0) / F.get("gk_faced_op", 1), 0.0)
    F["gk_dist_pct"] = np.where(F.gk_dist_n > 0, F.gk_dist_ok / F.gk_dist_n.clip(lower=1), 0.0)
    F["gk_is"] = 1.0
    out = ROOT / "outputs" / "gk_feat.parquet"
    F.to_parquet(out, index=False)
    print(f"GK 피처 {len(F):,}행 × {F.shape[1]-2}열 · 경기 {F.game_id.nunique():,} · GK {F.player_id.nunique()}")
    print(f"gsax  평균 {F.gk_gsax.mean():+.3f}  SD {F.gk_gsax.std():.3f}  범위 [{F.gk_gsax.min():.2f},{F.gk_gsax.max():.2f}]")
    print("열:", [c for c in F.columns if c.startswith("gk_")])
    print(f"저장 {out}")


if __name__ == "__main__":
    main()
