r"""2026 시즌 어댑터 — Bepro data-api 응답으로 loader 입력 파일을 만든다.

왜 필요한가
  raw/{리그}/2026/match/{gid}/ 에는 event_data.json 만 있다. loader 는 경기당
  info.json · lineup.json · player_stats.json · sequence_data.json 과 시즌당
  team.json · player/{team_id}.json 을 요구한다.

원천
  outputs/lineups_2026.json   meta/lineups  — 선발 여부·포지션·**포메이션 좌표**·출전시간
  outputs/matches_2026.json   meta/matches  — 팀·일시·**스코어**·라운드·경기장
  근사값 없이 전부 원천에서 온다.

좌표 축 변환 (중요)
  API 는 x=전후·y=좌우, 기존 lineup.json 은 y=전후·x=좌우이고 좌우 방향이 반대다.
      x_기존 = 1 - y_API      y_기존 = x_API
  검증: GK 0.080/0.081, CB 0.245/0.248, CM 0.580/0.582, CF 0.915/0.916 일치.

원본을 건드리지 않는다
  raw-data 는 다른 프로젝트와 공유되는 심볼릭 링크다. 로컬 트리를 새로 만들어
  2021~2025 는 링크로 잇고 2026 만 실물로 채운다.

실행: python -m experiments.mk2026 --build
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

TB = Path(__file__).resolve().parent.parent                     # 저장소 루트
SRC = Path(os.environ.get("RAW_OLD", TB / "raw-data"))           # 2021~2025 원본 트리
API_LU = TB / "outputs" / "lineups_2026.json"
API_MT = TB / "outputs" / "matches_2026.json"
DST = TB / "raw-data-2026"
LEAGUES = [("K-LEAGUE1", "KLEAGUE1", 587, 90001),
           ("K-LEAGUE2", "KLEAGUE2", 588, 90002)]
YEARS_OLD = ["2021", "2022", "2023", "2024", "2025"]


def build():
    lu = {int(k): v for k, v in json.load(open(API_LU, encoding="utf-8")).items()}
    mt = {int(k): v for k, v in json.load(open(API_MT, encoding="utf-8")).items()}
    print(f"API 라인업 {len(lu)}경기 | 경기메타 {len(mt)}경기")
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    lg = json.load(open(SRC / "league.json"))
    keep = {cid for _, _, cid, _ in LEAGUES}
    lg["result"] = [r for r in lg["result"] if r["id"] in keep]
    lg["count"] = len(lg["result"])
    for row in lg["result"]:
        for _, _, cid, sid26 in LEAGUES:
            if row["id"] == cid and sid26 not in row["season_ids"]:
                row["season_ids"] = list(row["season_ids"]) + [sid26]
    json.dump(lg, open(DST / "league.json", "w"), ensure_ascii=False)

    for src_name, dst_name, cid, sid26 in LEAGUES:
        root = DST / dst_name
        root.mkdir()
        sj = json.load(open(SRC / src_name / "season.json"))
        rows = sj["result"] if isinstance(sj, dict) and "result" in sj else sj
        if not any(r["id"] == sid26 for r in rows):
            rows.append({"id": sid26, "name": "2026"})
        json.dump({"result": rows}, open(root / "season.json", "w"), ensure_ascii=False)
        for y in YEARS_OLD:
            s = SRC / src_name / y
            if s.exists():
                (root / y).symlink_to(s)
        mdir = SRC / src_name / "2026" / "match"
        y26 = root / "2026" / "match"
        y26.mkdir(parents=True)
        # 이 리그의 2026 경기 = 이벤트가 있고 API 응답도 있는 것
        gids = sorted(g for g in lu
                      if (mdir / str(g) / "event_data.json").exists() and g in mt)
        tinfo: dict[int, dict] = {}
        proster: dict[int, dict[int, dict]] = {}
        for gid in gids:
            M, LU = mt[gid], lu[gid]["lineup"]
            for side in ("home_team", "away_team"):
                t = M[side]
                tinfo.setdefault(int(t["id"]), {
                    "id": int(t["id"]), "name": str(t["name"]),
                    "name_en": str(t.get("name_en") or t["name"]),
                    "iso_country_code": t.get("iso_country_code", "KR"),
                    "player_ids": []})
            rnd = M.get("round") or {}
            info = {"result": {
                "id": gid,
                "start_time": M["match_start_time"],
                "round": {"id": int(rnd.get("id") or 0),
                          "name": rnd.get("name") or f"Round {rnd.get('id') or 0}"},
                "season": {"id": sid26, "name": "2026", "season_group_name": "2026",
                           "league_id": cid, "start_year": 2026, "end_year": 2026},
                "home_team": {"id": int(M["home_team"]["id"]),
                              "name": str(M["home_team"]["name"]),
                              "name_en": str(M["home_team"].get("name_en")
                                             or M["home_team"]["name"]),
                              "iso_country_code": "KR"},
                "away_team": {"id": int(M["away_team"]["id"]),
                              "name": str(M["away_team"]["name"]),
                              "name_en": str(M["away_team"].get("name_en")
                                             or M["away_team"]["name"]),
                              "iso_country_code": "KR"},
                "full_time": 90, "half_time_duration": 45,
                "extra_full_time": int(M.get("match_extra_time") or 0),
                "detail_match_result": {
                    "home_team_score": int(M["score"]["home_team"]),
                    "away_team_score": int(M["score"]["away_team"])},
                "extra_match_result": None,
                "venue": {"id": 0, "display_name": str(M.get("location_name") or ""),
                          "ground_width": 105.0, "ground_height": 68.0},
                "is_analysis_finished": bool(M.get("is_analysis_finished", True)),
                "live_analysing": False, "external_ids": []}}
            d = y26 / str(gid)
            d.mkdir()
            (d / "event_data.json").symlink_to(mdir / str(gid) / "event_data.json")
            json.dump(info, open(d / "info.json", "w"), ensure_ascii=False)
            lineup, by_team = [], {}
            for k, p in enumerate(LU, start=1):
                pid, tid = int(p["player_id"]), int(p["team_id"])
                try:
                    bn = int(p.get("shirt_number") or 0)
                except (TypeError, ValueError):
                    bn = 0
                pos = p.get("position")
                # API x=전후/y=좌우 → 기존 x=좌우/y=전후, 좌우 반전
                conv = ({"x": 1.0 - float(pos["y"]), "y": float(pos["x"])}
                        if pos else None)
                nm = (p.get("player_full_name") or "").strip()
                lineup.append({"id": k, "back_number": bn, "player_id": pid,
                               "is_starting_lineup": bool(p.get("is_starting")),
                               "position": conv,
                               "position_name": p.get("position_name"),
                               "player_name": nm, "player_last_name": "",
                               "team_id": tid})
                by_team.setdefault(tid, []).append(
                    {"player_id": pid,
                     "stats": {"play_time": int(p.get("playing_time") or 0),
                               "rating": 0.0}})
                proster.setdefault(tid, {})[pid] = {
                    "id": pid, "root_player_id": pid, "player_name": nm,
                    "player_last_name": "", "player_name_en": nm,
                    "player_last_name_en": "", "back_number": str(bn),
                    "player_role": None, "main_position": p.get("position_name"),
                    "team_id": tid, "external_ids": []}
            json.dump({"result": lineup}, open(d / "lineup.json", "w"),
                      ensure_ascii=False)
            json.dump({"result": [{"team_id": t, "players": v}
                                  for t, v in by_team.items()]},
                      open(d / "player_stats.json", "w"), ensure_ascii=False)
            # sequence_data.json — SPADL 변환에 인자로만 넘어가고 쓰이지 않는다
            ev = json.load(open(mdir / str(gid) / "event_data.json",
                                encoding="utf-8"))["result"]
            seqs, cur = [], None
            for e in sorted(ev, key=lambda z: (z.get("event_period") or "",
                                               float(z.get("event_time") or 0))):
                per, tid_ = e.get("event_period"), e.get("team_id")
                t_ = float(e.get("event_time") or 0)
                if cur is None or cur["team_id"] != tid_ or cur["event_period"] != per:
                    cur = {"start_time": t_, "end_time": t_, "team_id": tid_,
                           "event_period": per, "event_ids": []}
                    seqs.append(cur)
                cur["end_time"] = t_
                if e.get("id") is not None:
                    cur["event_ids"].append(e["id"])
            json.dump({"result": [q for q in seqs
                                  if q["team_id"] is not None and q["event_ids"]]},
                      open(d / "sequence_data.json", "w"), ensure_ascii=False)
            # formation.json — API 는 킥오프 배치만 준다(경기 중 변경 스냅샷 없음).
            # 에피소드 빌더는 시점별 스냅샷을 기대하지만 t=0 하나로 채운다.
            fm = {}
            for p_ in LU:
                if not p_.get("is_starting") or not p_.get("position"):
                    continue
                t_ = int(p_["team_id"])
                fm.setdefault(t_, []).append(
                    {"player_id": int(p_["player_id"]),
                     "position": {"x": 1.0 - float(p_["position"]["y"]),
                                  "y": float(p_["position"]["x"])}})
            json.dump({"result": [{"team_id": t_, "event_period": "FIRST_HALF",
                                   "changed_time": 0, "formation": v}
                                  for t_, v in fm.items()]},
                      open(d / "formation.json", "w"), ensure_ascii=False)
        for tid, r in proster.items():
            if tid in tinfo:
                tinfo[tid]["player_ids"] = sorted(r)
        y26root = root / "2026"
        json.dump({"result": list(tinfo.values())},
                  open(y26root / "team.json", "w"), ensure_ascii=False)
        (y26root / "player").mkdir(exist_ok=True)
        for tid, r in proster.items():
            json.dump({"result": list(r.values())},
                      open(y26root / "player" / f"{tid}.json", "w"), ensure_ascii=False)
        npos = sum(1 for g in gids for p in lu[g]["lineup"]
                   if p.get("is_starting") and p.get("position"))
        nst = sum(1 for g in gids for p in lu[g]["lineup"] if p.get("is_starting"))
        print(f"  {dst_name} 2026: {len(gids)}경기 | 팀 {len(tinfo)} | "
              f"선수 {sum(len(v) for v in proster.values())} | "
              f"선발 좌표 {npos}/{nst}")
    print(f"\n생성 완료 → {DST}")
    for _, dst_name, _, _ in LEAGUES:
        cnt = {p.name: len(list((p / "match").iterdir()))
               for p in sorted((DST / dst_name).iterdir()) if p.is_dir()}
        print(f"  {dst_name}: {cnt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    if ap.parse_args().build:
        build()


if __name__ == "__main__":
    main()
