"""트래킹 → 선수×경기×페이즈 표, 팀×경기×페이즈 표.

입력  $TRACKING_DIR/{K-LEAGUE1,K-LEAGUE2}/{2024,2025}/match/{gid}/tracking.parquet (30 Hz), lineup.json
      경기장 크기: $MATCH_INFO_DIR/{KLEAGUE1,KLEAGUE2}/{season}/match/{gid}/info.json (없으면 105×68)
      EventXI 선수 ID: vaep/output/players.csv ((경기, 팀, 등번호)로 연결)
출력  outputs/player_phase_match.parquet · outputs/team_phase_match.parquet · outputs/match_status.csv

페이즈 (팀 관점, 10 Hz)
  transition_attack / transition_defense  인플레이 턴오버 뒤 5초 (공이 나갔다 재시작한 뒤는 아님)
  build_up         우리 점유 · 공이 공격 방향 40% 이전
  settled_attack   그 밖의 우리 점유
  pressing         상대 점유 · 공 10 m 안에 우리 선수 2명 이상 · 그중 1명 이상이 3 m/s 이상으로 공 쪽으로 이동
  settled_defense  그 밖의 상대 점유
대조군 third: 점유(IN/OUT) × 공 위치 3분할 — 페이즈가 공 위치 이상의 정보를 주는지 보려는 것
실행  python tracking/phase_tables.py [--limit N] [--workers 40]
"""
from __future__ import annotations
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

import os
ROOT = Path(__file__).resolve().parent.parent   # repository root
RAW = Path(os.environ.get("TRACKING_DIR", ROOT / "data/tracking"))     # Bepro tracking parquet (not distributed)
LEGACY = Path(os.environ.get("MATCH_INFO_DIR", ROOT / "data/match_info"))  # info.json with pitch size (not distributed)
EX = ROOT / "vaep/output"
OUT = Path(__file__).resolve().parent.parent / "outputs"
LG = {"K-LEAGUE1": "KLEAGUE1", "K-LEAGUE2": "KLEAGUE2"}
STEP = 3                      # 30 Hz → 10 Hz
TRANS_MS, BUILD_X, PR_R, PR_N, CLOSE_V = 5000, 0.4, 10.0, 2, 3.0
HI_V, SPR_V, MIN_HALF_MIN = 5.5, 7.0, 40.0


def pitch(lg, season, gid):
    f = LEGACY / LG[lg] / str(season) / "match" / str(gid) / "info.json"
    try:
        v = json.load(open(f))["result"]["venue"]; return float(v["ground_width"]), float(v["ground_height"]), True
    except Exception:
        return 105.0, 68.0, False


def last_turnover(bs, poss, tm):
    """프레임마다 가장 최근 인플레이 턴오버 시각. 공이 나간 뒤(ballout) 상대가 재시작하면 턴오버가 아니다."""
    out = np.full(len(bs), -1e18); last_p, dead, last_to = None, False, -1e18
    for i in range(len(bs)):
        s = bs[i]
        if s == "ballout": dead = True
        elif s == "home" or s == "away":
            if last_p is not None and poss[i] != last_p and not dead: last_to = tm[i]
            last_p, dead = poss[i], False
        out[i] = last_to
    return out


def one(task, frames_only=False):
    """frames_only=True 면 팀×프레임 페이즈 표(구간 분석용)만 돌려준다."""
    lg, season, gid = task; t0 = time.time()
    try:
        d = RAW / lg / str(season) / "match" / str(gid)
        t = pq.read_table(d / "tracking.parquet", columns=["period_order", "frame_index", "match_time", "ball_state", "player_id", "x", "y"]).to_pandas()
        t = t[(t.frame_index % STEP) == 0]
        mins = t.groupby("period_order").match_time.agg(lambda s: (s.max() - s.min()) / 60000)
        if len(mins) < 2 or (mins < MIN_HALF_MIN).any():
            return None, None, dict(game_id=gid, league=lg, season=season, status="incomplete", halves=str(mins.round(1).to_dict())), None
        L, W, venue_ok = pitch(lg, season, gid)
        lu = json.load(open(d / "lineup.json"))["data"][0]
        home, away = int(lu["home_team_id"]), int(lu["away_team_id"])
        meta = {int(p["player_id"]): (int(p["team_id"]), p["position_name"], bool(p["is_starting"]), int(p["shirt_number"])) for p in lu["lineup"]}
        t["x"] = t.x / 1e4; t["y"] = t.y / 1e4
        fr = t.drop_duplicates(["period_order", "frame_index"])[["period_order", "frame_index", "match_time", "ball_state"]].sort_values(["period_order", "match_time"]).reset_index(drop=True)
        fr = fr.merge(t[t.player_id == 0][["period_order", "frame_index", "x", "y"]].rename(columns={"x": "bx", "y": "by"}), on=["period_order", "frame_index"], how="left")
        bs = fr.ball_state.astype(str).to_numpy()
        fr["poss"] = np.where(bs == "home", home, np.where(bs == "away", away, -1))
        fr["last_to"] = np.concatenate([last_turnover(bs[g.index], fr.poss.to_numpy()[g.index], fr.match_time.to_numpy()[g.index]) for _, g in fr.groupby("period_order")])
        P = t[t.player_id != 0].drop(columns="ball_state").copy()
        P["team"] = P.player_id.map(lambda p: meta.get(p, (-1,))[0]); P = P[P.team > 0]
        gk = [p for p, m in meta.items() if m[1] == "GK"]
        g = P[P.player_id.isin(gk)].groupby(["period_order", "team"]).x.mean()
        sign = {k: (1 if v < 0.5 else -1) for k, v in g.items()}                     # 골키퍼가 x<0.5 쪽이면 +x 로 공격
        P = P.sort_values(["player_id", "period_order", "match_time"])
        grp = P.groupby(["player_id", "period_order"])
        dt = grp.match_time.diff() / 1000; ok = (dt > 0) & (dt < 0.2)
        P["vx"] = np.where(ok, grp.x.diff() * L / dt, np.nan); P["vy"] = np.where(ok, grp.y.diff() * W / dt, np.nan)
        P = P.merge(fr, on=["period_order", "frame_index", "match_time"], how="left")
        s = np.array([sign.get((po, tm), np.nan) for po, tm in zip(P.period_order, P.team)])
        P["ax"] = np.where(s > 0, P.x, 1 - P.x); P["ay"] = np.where(s > 0, P.y, 1 - P.y); P["bax"] = np.where(s > 0, P.bx, 1 - P.bx)
        P["dball"] = np.hypot((P.x - P.bx) * L, (P.y - P.by) * W)
        ux, uy = (P.bx - P.x) * L, (P.by - P.y) * W; n = np.hypot(ux, uy); n = np.where(n > 0, n, np.nan)
        P["close"] = (P.vx * ux + P.vy * uy) / n
        known = (P.poss > 0) & P.bx.notna() & np.isfinite(s)
        own = P.poss == P.team
        trans = (P.match_time - P.last_to) < TRANS_MS
        D = P[known & ~own]
        agg = D.assign(near=D.dball < PR_R, cl=(D.dball < PR_R) & (D.close >= CLOSE_V)).groupby(["period_order", "frame_index", "team"]).agg(nn=("near", "sum"), nc=("cl", "sum")).reset_index()
        agg["press"] = (agg.nn >= PR_N) & (agg.nc >= 1)
        P = P.merge(agg[["period_order", "frame_index", "team", "press"]], on=["period_order", "frame_index", "team"], how="left")
        press = P.press.fillna(False).astype(bool)
        P["phase"] = np.select([~known, trans & own, trans & ~own, own & (P.bax < BUILD_X), own, press],
                               ["unknown", "transition_attack", "transition_defense", "build_up", "settled_attack", "pressing"], default="settled_defense")
        third = np.where(P.bax < 1 / 3, "D", np.where(P.bax < 2 / 3, "M", "A"))
        P["third"] = np.where(~known, "unknown", np.where(own, "IN_", "OUT_") + pd.Series(third, index=P.index))
        if frames_only:
            return P.drop_duplicates(["period_order", "frame_index", "team"])[["period_order", "frame_index", "match_time", "team", "phase", "poss", "bax", "ball_state"]].sort_values(["team", "period_order", "match_time"]).assign(game_id=gid)
        v = np.hypot(P.vx, P.vy); P["v"] = v; P["hi"] = (v > HI_V).astype(float); P["spr"] = (v > SPR_V).astype(float)
        P["engage"] = ((P.dball < PR_R) & (P.close >= CLOSE_V)).astype(float)
        # 팀 중심(필드 선수) 대비 위치
        cen = P[~P.player_id.isin(gk)].groupby(["period_order", "frame_index", "team"]).agg(cx=("ax", "mean"), cy=("ay", "mean"), sy=("ay", "std"), xmax=("ax", "max"), xmin=("ax", "min")).reset_index()
        P = P.merge(cen, on=["period_order", "frame_index", "team"], how="left")
        P["rx"] = (P.ax - P.cx) * L; P["ry"] = (P.ay - P.cy) * W; P["dcen"] = np.hypot(P.rx, P.ry)
        field = ~P.player_id.isin(gk)
        prow, trow = [], []
        for tax in ("phase", "third"):
            K = P[P[tax] != "unknown"]
            f = (K.groupby(["player_id", "team", tax]).agg(n=("ax", "size"), ax=("ax", "mean"), ay=("ay", "mean"), rx=("rx", "mean"), ry=("ry", "mean"),
                                                         v=("v", "mean"), hi=("hi", "mean"), spr=("spr", "mean"), dball=("dball", "mean"), engage=("engage", "mean"))
                 .reset_index().rename(columns={tax: "label"}))
            f["ayw"] = (f.ay - 0.5).abs(); f["tax"] = tax; f["minutes"] = f.n / 600; prow.append(f)
            Kf = K[field.loc[K.index]]
            fl = Kf.drop_duplicates(["period_order", "frame_index", "team"])
            tm_ = (fl.assign(width=fl.sy * W, length=(fl.xmax - fl.xmin) * L).groupby(["team", tax])
                   .agg(frames=("frame_index", "size"), line=("cx", "mean"), width=("width", "mean"), length=("length", "mean")).reset_index().rename(columns={tax: "label"}))
            cp = Kf.groupby(["team", tax]).dcen.mean().rename("compact").reset_index().rename(columns={tax: "label"})
            tm_ = tm_.merge(cp, on=["team", "label"])
            tm_["share"] = tm_.frames / tm_.groupby("team").frames.transform("sum"); tm_["tax"] = tax; trow.append(tm_)
        pf = pd.concat(prow, ignore_index=True); tf = pd.concat(trow, ignore_index=True)
        # 피치 깊이 6구간(자기 골문 → 상대 골문, EventXI 밴드와 같은 방향)별 표: EventXI 입력 텐서의 트래킹 행용
        brow = []
        for tax in ("phase", "third", "all"):
            K = P[P["phase"] != "unknown"] if tax == "all" else P[P[tax] != "unknown"]
            K = K.assign(band=np.clip(np.floor(K.ax * 6), 0, 5).astype(int), label=("all" if tax == "all" else K[tax]))
            b = K.groupby(["player_id", "team", "label", "band"]).agg(n=("ax", "size"), v=("v", "mean"), hi=("hi", "mean"), engage=("engage", "mean")).reset_index()
            b["tax"] = tax; brow.append(b)
        bf = pd.concat(brow, ignore_index=True); bf["minutes"] = bf.n / 600
        bf["api_pid"] = bf.player_id.astype(int); bf["shirt"] = bf.api_pid.map(lambda p: meta[p][3]); bf["pos"] = bf.api_pid.map(lambda p: meta[p][1])
        bf["game_id"] = gid; bf = bf.drop(columns="player_id")
        pf["api_pid"] = pf.player_id.astype(int); pf["shirt"] = pf.api_pid.map(lambda p: meta[p][3]); pf["starter"] = pf.api_pid.map(lambda p: meta[p][2]); pf["pos"] = pf.api_pid.map(lambda p: meta[p][1])
        for df in (pf, tf): df["game_id"] = gid; df["league"] = lg; df["season"] = season
        tf["home"] = (tf.team == home).astype(int)
        return pf.drop(columns="player_id"), tf, dict(game_id=gid, league=lg, season=season, status="ok", L=L, W=W, venue_ok=venue_ok,
                                                      known_share=round(float(known.mean()), 3), sec=round(time.time() - t0, 1)), bf
    except Exception as e:
        return None, None, dict(game_id=gid, league=lg, season=season, status=f"error: {type(e).__name__}: {e}"[:200]), None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=40); a = ap.parse_args()
    games = pd.read_csv(EX / "games.csv"); ex_ids = set(games.game_id.astype(int))
    tasks = [(lg, s, int(p.name)) for lg in LG for s in (2024, 2025) for p in sorted((RAW / lg / str(s) / "match").glob("*"))
             if p.name.isdigit() and int(p.name) in ex_ids and (p / "tracking.parquet").exists() and (p / "tracking.parquet").stat().st_size > 0]
    if a.limit: tasks = tasks[:a.limit]
    print(f"경기 {len(tasks)} · 작업자 {a.workers}", flush=True); t0 = time.time()
    with Pool(a.workers) as pool: res = pool.map(one, tasks, chunksize=2)
    st = pd.DataFrame([r[2] for r in res]); ok = [r for r in res if r[0] is not None]
    pf = pd.concat([r[0] for r in ok], ignore_index=True); tf = pd.concat([r[1] for r in ok], ignore_index=True)
    # EventXI 선수 ID: (경기, 팀, 등번호)
    pl = pd.read_csv(EX / "players.csv", usecols=["game_id", "team_id", "player_id", "jersey_number"])
    br = pl.rename(columns={"team_id": "team", "jersey_number": "shirt", "player_id": "ex_pid"}).drop_duplicates(["game_id", "team", "shirt"])
    pf = pf.merge(br, on=["game_id", "team", "shirt"], how="left")
    bf = pd.concat([r[3] for r in ok], ignore_index=True).merge(br, on=["game_id", "team", "shirt"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    bf.to_parquet(OUT / "player_phase_band.parquet", index=False); pf.to_parquet(OUT / "player_phase_match.parquet", index=False); tf.to_parquet(OUT / "team_phase_match.parquet", index=False); st.to_csv(OUT / "match_status.csv", index=False)
    print(f"완료 {time.time() - t0:.0f}s · 상태 {st.status.str.split(':').str[0].value_counts().to_dict()} · 선수행 {len(pf):,} (EventXI ID 연결 {pf.ex_pid.notna().mean():.1%}) · 팀행 {len(tf):,}", flush=True)
    ph = tf[tf.tax == "phase"].groupby("label").share.median().sort_values(ascending=False)
    print("페이즈 비중(팀-경기 중앙값):", ph.round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
