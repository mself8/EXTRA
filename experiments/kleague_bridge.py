"""K리그 공식 등록번호(kl_id) 기준 선수 브릿지 — 이름 추측이 아니라 공식 명부로 사람을 잇는다.

입력  outputs/lineup_players_raw.parquet   (season, gid, team, pid, bn, ko)  ← raw lineup.json (2026 은 ko 자리에 영문명)
      outputs/kleague_list.parquet          (kl_id, team_code, ko, bn)        ← 팀별 현역+역대 목록
      outputs/kleague_players.parquet       (kl_id, ko, en, team_ko, pos, bn, nat, height, weight, birth)
      outputs/kleague_player_seasons.parquet(kl_id, season, team_ko)          ← 시즌별 소속 표
매칭 2021~25: (시즌, 팀, 한글명) 이 시즌표(kl_id,시즌,팀)+프로필 한글명과 유일 일치 → T1
             실패 시 (팀 목록 소속 + 한글명 유일) → T2, (한글명 전역 유일) → T3
      2026 : 영문명 정규화(소문자·영문자만·토큰 정렬) 로 (2026 시즌표 팀 + 영문명) → T1, (팀 목록 + 영문명) → T2, (영문명 전역 유일) → T3
출력  outputs/kl_bridge.parquet  (season, team, pid, kl_id, tier)
      outputs/kl_person_map.json {pid → canonical pid}: 같은 kl_id 를 가진 pid 들을 최초(가장 이른 시즌) pid 로 통일
      outputs/player_bio.parquet (pid_canon, kl_id, ko, en, birth, nat, height, weight, foreign)
실행: python -m experiments.kleague_bridge
"""
from __future__ import annotations
import json, re, sys, unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BEPRO2KL = {316: "K09", 328: "K02", 2353: "K35", 2354: "K01", 4220: "K29", 4639: "K03", 4640: "K05", 4641: "K04", 4643: "K21", 4644: "K17",
            4645: "K07", 4646: "K18", 4647: "K20", 4648: "K22", 4649: "K06", 4650: "K34", 4651: "K08", 4652: "K26", 4654: "K27", 4655: "K31",
            4656: "K32", 4657: "K10", 5890: "K39", 5893: "K37", 5895: "K40", 5896: "K36", 5902: "K38", 6007: "K41", 57730: "K42"}


def norm_en(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(t for t in re.split(r"[^a-z]+", s) if t))


_FOLD = [("woo", "u"), ("oo", "u"), ("eo", "u"), ("eu", "u"), ("ee", "i"), ("yeo", "yu"), ("ae", "e"), ("hyeon", "hyun"), ("hyeo", "hyu"),
         ("wook", "uk"), ("woong", "ung"), ("joon", "jun"), ("young", "yung"), ("sung", "sung"), ("seong", "sung"), ("kyung", "kyung"), ("gyeong", "kyung"), ("kyoung", "kyung"),
         ("ck", "k"), ("ph", "f"), ("gw", "kw"), ("b", "p"), ("d", "t"), ("g", "k"), ("r", "l"), ("j", "ch"), ("z", "s")]


def fold_en(s: str) -> str:
    """로마자 표기 변형을 접는다 (모음·자음 변형 → 대표형), 문자만 남김."""
    s = re.sub(r"[^a-z]", "", norm_en(s).replace(" ", ""))
    for a, b in _FOLD: s = s.replace(a, b)
    return s


def _ed(a: str, b: str) -> int:
    if a == b: return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_ko(s: str) -> str:
    return re.sub(r"[\s·\-]", "", str(s or ""))


def main():
    L = pd.read_parquet(ROOT / "outputs/lineup_players_raw.parquet")
    KL = pd.read_parquet(ROOT / "outputs/kleague_list.parquet")
    P = pd.read_parquet(ROOT / "outputs/kleague_players.parquet")
    Q = pd.read_parquet(ROOT / "outputs/kleague_player_seasons.parquet")
    P["kon"] = P.ko.map(norm_ko); P["enn"] = P.en.map(norm_en)
    # 팀 코드 ↔ 시즌표 팀 한글 약칭: 팀별 목록의 선수들이 시즌표에서 쓰는 팀명의 최빈값 (2021 이후 행만)
    code2ko = {}
    for code, grp in KL.groupby("team_code"):
        q = Q[Q.kl_id.isin(grp.kl_id) & (Q.season >= 2021)]
        cnt = q.team_ko.value_counts()
        if len(cnt): code2ko[code] = cnt.index[0]
    print("팀 코드→시즌표 팀명:", code2ko, flush=True)
    ko2code = {v: k for k, v in code2ko.items()}
    Q["code"] = Q.team_ko.map(ko2code)
    # 인덱스: (season, code, kon) → kl_ids ; (code, kon) → kl_ids ; kon → kl_ids ; 영문 동일
    kon_of = dict(zip(P.kl_id, P.kon)); enn_of = dict(zip(P.kl_id, P.enn))
    idx_s_ko, idx_s_en = defaultdict(set), defaultdict(set)
    for r in Q.itertuples(index=False):
        if r.code: idx_s_ko[(r.season, r.code, kon_of.get(r.kl_id, ""))].add(r.kl_id); idx_s_en[(r.season, r.code, enn_of.get(r.kl_id, ""))].add(r.kl_id)
    idx_t_ko, idx_t_en = defaultdict(set), defaultdict(set)
    for r in KL.itertuples(index=False):
        idx_t_ko[(r.team_code, kon_of.get(r.kl_id, norm_ko(r.ko)))].add(r.kl_id); idx_t_en[(r.team_code, enn_of.get(r.kl_id, ""))].add(r.kl_id)
    idx_ko, idx_en = defaultdict(set), defaultdict(set)
    for k, v in kon_of.items(): idx_ko[v].add(k)
    for k, v in enn_of.items():
        if v: idx_en[v].add(k)

    KLt = {code: list(grp.itertuples(index=False)) for code, grp in KL.groupby("team_code")}
    out = []
    for r in L.drop_duplicates(["season", "team", "pid", "ko"]).itertuples(index=False):
        code = BEPRO2KL.get(int(r.team)); key = norm_ko(r.ko) if r.season < 2026 else norm_en(r.ko)
        I_s, I_t, I_g = (idx_s_ko, idx_t_ko, idx_ko) if r.season < 2026 else (idx_s_en, idx_t_en, idx_en)
        c1 = I_s.get((int(r.season), code, key), set()); c2 = I_t.get((code, key), set()); c3 = I_g.get(key, set())
        kid, tier = None, None
        if len(c1) == 1: kid, tier = next(iter(c1)), "T1"
        elif len(c2) == 1: kid, tier = next(iter(c2)), "T2"
        elif len(c3) == 1: kid, tier = next(iter(c3)), "T3"
        elif r.season >= 2026 and code:
            cand = set().union(*[v for (ss, cc, _), v in idx_s_en.items() if ss == int(r.season) and cc == code]) if True else set()
            fq = fold_en(r.ko); tq = set(norm_en(r.ko).split()); scored = []
            for k in cand:
                fe = fold_en(enn_of.get(k, "")); te = set(enn_of.get(k, "").split())
                if not fe: continue
                d = _ed(fq, fe); sub = bool(tq) and (tq <= te or te <= tq) and len(tq & te) >= 1
                bn_ok = any((kk.kl_id == k and kk.bn == r.bn) for kk in KLt.get(code, []))
                shared = tq & te; strong_sub = sub and (len(shared) >= 2 or any(len(t) >= 6 for t in shared))
                close = d <= max(1, int(0.25 * max(len(fq), len(fe))))
                if d == 0 or (bn_ok and (close or sub)) or strong_sub: scored.append((d - (2 if bn_ok else 0), k, bn_ok))
            scored.sort()
            if len(scored) == 1 or (len(scored) > 1 and (scored[0][0] < scored[1][0])): kid, tier = scored[0][1], "T4"
            elif len(scored) > 1: tier = "AMB_F"
            else: tier = "NONE"
        elif len(c1) > 1: tier = "AMB_S"
        elif len(c2) > 1: tier = "AMB_T"
        elif len(c3) > 1: tier = "AMB_G"
        else: tier = "NONE"
        out.append(dict(season=int(r.season), team=int(r.team), pid=int(r.pid), name=r.ko, bn=r.bn, kl_id=kid, tier=tier))
    B = pd.DataFrame(out)
    # pid 단위 일관성: 한 pid 가 둘 이상의 kl_id 로 가면 다수결
    B["ns"] = np.where(B.season >= 2026, "n26", "old"); B["key"] = list(zip(B.ns, B.pid))      # 2026 원시 pid 는 2021~25 와 숫자 충돌 → 공간 분리
    g = B.dropna(subset=["kl_id"]).groupby("key").kl_id.agg(lambda s: s.value_counts().index[0]); conf = B.dropna(subset=["kl_id"]).groupby("key").kl_id.nunique()
    print(f"행 {len(B):,} · 층 {B.tier.value_counts().to_dict()} · (공간,pid) 충돌(kl_id>1) {int((conf > 1).sum())}", flush=True)
    for s in sorted(B.season.unique()):
        x = B[B.season == s]; print(f"  {s}: 연결 {x.kl_id.notna().mean() * 100:5.1f}%  ({x.tier.value_counts().to_dict()})", flush=True)
    B["kl_id"] = B.key.map(g)
    # 같은 kl_id 로 묶인 (시즌공간, pid) 들이 같은 경기에 동시 출현하면 동일인일 수 없다 → 충돌 그룹은 T1 다수 pid 만 남기고 나머지 연결 해제
    gof = L.assign(ns=np.where(L.season >= 2026, "n26", "old")).groupby(["ns", "pid"]).gid.agg(set).to_dict()
    tier_n = B[B.tier == "T1"].groupby("key").size().to_dict(); rows_n = B.groupby("key").size().to_dict()
    dropped = 0
    for kid, grp in B.dropna(subset=["kl_id"]).groupby("kl_id"):
        keys = list(dict.fromkeys(grp.key))
        if len(keys) < 2: continue
        clash = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:] if gof.get(a, set()) & gof.get(b, set())]
        if not clash: continue
        keep = max(keys, key=lambda k: (tier_n.get(k, 0), rows_n.get(k, 0)))
        bad = {k for pr in clash for k in pr if k != keep}
        B.loc[B.key.isin(bad), ["kl_id", "tier"]] = [np.nan, "CLASH"]; dropped += len(bad)
    print(f"동시출현 충돌로 연결 해제한 (공간,pid) {dropped}", flush=True)
    B.drop(columns=["key"]).to_parquet(ROOT / "outputs/kl_bridge.parquet", index=False)
    # 사람 통일 — 정규 id = 그 사람의 가장 이른 2021~25 pid (없으면 2026 pid)
    lk = B.dropna(subset=["kl_id"]).sort_values(["season"]); canon = {}
    for r in lk.itertuples(index=False):
        if r.kl_id not in canon: canon[r.kl_id] = (r.ns, int(r.pid))
    map_old = {int(r.pid): canon[r.kl_id][1] for r in lk[lk.ns == "old"].itertuples(index=False)}
    map_26 = {int(r.pid): canon[r.kl_id] for r in lk[lk.ns == "n26"].itertuples(index=False)}
    json.dump({str(k): v for k, v in map_old.items()}, open(ROOT / "outputs/kl_map_old.json", "w"))
    json.dump({str(k): [ns, v] for k, (ns, v) in map_26.items()}, open(ROOT / "outputs/kl_map_2026.json", "w"))
    n_hist = sum(ns == "old" for ns, _ in map_26.values())
    print(f"맵 old: pid {len(map_old):,} → 사람 {len(set(map_old.values())):,} (비항등 {sum(k != v for k, v in map_old.items()):,}) · 맵 2026: {len(map_26):,} 중 2021~25 이력 연결 {n_hist:,}", flush=True)
    canon_pid = {k: v[1] for k, v in canon.items()}
    pmap = {int(p): canon_pid[k] for p, k in zip(lk.pid, lk.kl_id) if True}   # (참고용; 공간 미구분)
    json.dump({str(k): v for k, v in pmap.items()}, open(ROOT / "outputs/kl_person_map.json", "w"))
    canon = canon_pid
    # 바이오
    bio = P[P.kl_id.isin(canon)].copy(); bio["pid_canon"] = bio.kl_id.map(canon)
    bio["birth"] = pd.to_datetime(bio.birth, format="%Y/%m/%d", errors="coerce"); bio["foreign"] = (bio.nat.fillna("") != "한국").astype(int)
    bio["height"] = pd.to_numeric(bio.height, errors="coerce"); bio["weight"] = pd.to_numeric(bio.weight, errors="coerce")
    bio[["pid_canon", "kl_id", "ko", "en", "birth", "nat", "height", "weight", "foreign"]].to_parquet(ROOT / "outputs/player_bio.parquet", index=False)
    print(f"바이오 {len(bio):,} · 생년월일 {bio.birth.notna().mean() * 100:.1f}% · 외국인 {bio.foreign.mean() * 100:.1f}% · 국적 상위 {bio.nat.value_counts().head(6).to_dict()}", flush=True)
    # 기존 사람 브릿지와 대조
    try:
        old = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/person_bridge_2026.json")).items()}
        both = [(k, v) for k, v in old.items() if k in pmap and v in pmap]
        agree = sum(pmap[k] == pmap[v] for k, v in both)
        print(f"기존 사람 브릿지 {len(old)} 중 양쪽 연결 {len(both)} · 일치 {agree} ({agree / max(len(both), 1) * 100:.1f}%)", flush=True)
        new26 = B[(B.season == 2026) & B.kl_id.notna()]; prior = B[(B.season < 2026) & B.kl_id.notna()].kl_id.unique()
        print(f"2026 연결 {new26.pid.nunique()} 명 중 2021~25 이력 있는 사람 {new26[new26.kl_id.isin(prior)].pid.nunique()} (기존 브릿지 572)", flush=True)
    except Exception as e:
        print("대조 생략:", e)


if __name__ == "__main__":
    main()
