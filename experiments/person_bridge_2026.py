"""2026 명단 pid → 2021~25 pid **사람 단위** 브릿지.

배경 (2026-09-03)
  2026 덤프는 선수 ID 공간이 과거와 단절(출전자 897명 중 pid 연속 1명)이고
  이름도 영문뿐이다. 그래서 ① 베테랑도 2026 시즌 초 이력 0 으로 리셋되고
  ② 시각화 이름이 영문으로 나온다. 지문 브릿지(이벤트↔명단)는 2026 내부용 —
  시즌 경계를 넘는 사람 연결은 이 스크립트가 처음 만든다.

방법
  과거: 한글 nickname(100%)을 자모 분해→로마자→접기(fold)로 정규화.
  2026: 영문 player_name 을 같은 접기로 정규화. 로마자 표기 변형(choi/coe,
  lee/i, park/bak …)은 성씨 변형표로 한글 성씨 공간에 합류시킨다.
  키 = (한글 성씨, 접힌 이름). 단계:
    T1 키 유일 일치            T2 키 일치 + 팀 연속으로 유일화
    T3 같은 성씨 + 이름 편집거리 ≤1 유일 + (팀 연속 또는 등번호 일치)
  검증: 같은 팀 잔류자의 등번호 일치율 · GK↔GK 일관성 · 최근 활동 시즌 분포.

출력: outputs/person_bridge_2026.json  {2026 명단 pid: 과거 pid}
실행: python -m experiments.person_bridge_2026

⚠ apply_person_bridge 적용 **후** 재실행하면 이미 재매핑된 pid 를 "2026 신원"으로
  읽어 항등 매핑(k==v)이 섞인다 — 무해하지만(적용 시 no-op), 순수한 브릿지가
  필요하면 outputs/backup_pre_person_bridge/players.csv 를 기준으로 돌릴 것.
  시즌 진행분 재수집 시 순서: 지문 브릿지 → (재수집 데이터에) 사람 브릿지 재빌드
  → apply_person_bridge (증분 적용 지원).
"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

INI = "g gg n d dd l m b bb s ss  j jj c k t p h".split(" ")
INI = ["g","gg","n","d","dd","l","m","b","bb","s","ss","","j","jj","c","k","t","p","h"]
MED = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo",
       "u","wo","we","wi","yu","eu","ui","i"]
FIN = ["","g","gg","gs","n","nj","nh","d","l","lg","lm","lb","ls","lt","lp","lh",
       "m","b","bs","s","ss","ng","j","c","k","t","p","h"]

SUR = {  # 로마자 성씨 변형 → 한글 성씨
 "kim":"김","gim":"김","ghim":"김","lee":"이","yi":"이","rhee":"이","ri":"이","i":"이","e":"이",
 "park":"박","pak":"박","bak":"박","bahk":"박","choi":"최","choe":"최","coe":"최",
 "jung":"정","jeong":"정","chung":"정","cheong":"정","kang":"강","gang":"강",
 "cho":"조","jo":"조","joh":"조","yoon":"윤","yun":"윤","youn":"윤",
 "jang":"장","chang":"장","lim":"임","im":"임","rim":"임","leem":"임","han":"한",
 "oh":"오","o":"오","seo":"서","suh":"서","shin":"신","sin":"신","sihn":"신",
 "kwon":"권","gwon":"권","kweon":"권","hwang":"황","whang":"황","ahn":"안","an":"안",
 "song":"송","yoo":"유","yu":"유","you":"유","ryu":"류","ryoo":"류","rieu":"류",
 "hong":"홍","jeon":"전","jun":"전","chun":"전","moon":"문","mun":"문",
 "baek":"백","paek":"백","back":"백","baik":"백","heo":"허","hur":"허","huh":"허",
 "nam":"남","noh":"노","no":"노","roh":"노","ha":"하","kwak":"곽","gwak":"곽",
 "sung":"성","seong":"성","cha":"차","joo":"주","ju":"주","chu":"주","woo":"우","wu":"우",
 "koo":"구","ku":"구","goo":"구","gu":"구","min":"민","bae":"배","pae":"배",
 "byun":"변","byeon":"변","pyun":"변","yang":"양","ryang":"양","do":"도","doh":"도",
 "ko":"고","go":"고","koh":"고","seok":"석","suk":"석","son":"손","sohn":"손",
 "yeo":"여","yuh":"여","yem":"염","yeom":"염","yum":"염","chae":"채","cae":"채",
 "won":"원","wone":"원","ryoo2":"류","na":"나","ra":"나","ma":"마","gil":"길","kil":"길",
 "ji":"지","chi":"지","jin":"진","chin":"진","tak":"탁","tark":"탁","pyo":"표",
 "hyun":"현","hyeon":"현","kook":"국","guk":"국","gook":"국","um":"엄","eom":"엄","uhm":"엄",
 "kwak2":"곽","cheon":"천","chean":"천","bang":"방","pang":"방","hahn":"한",
 "wang":"왕","ok":"옥","ock":"옥","in":"인","ihn":"인","tae":"태","kai":"개",
 "myung":"명","myeong":"명","sim":"심","shim":"심","gong":"공","kong":"공",
 "kye":"계","gye":"계","pi":"피","yong":"용","maeng":"맹","bu":"부","boo":"부",
 "so":"소","soh":"소","seon":"선","sun":"선","sul":"설","seol":"설",
 "namkoong":"남궁","namgung":"남궁","hwangbo":"황보","jegal":"제갈","sunwoo":"선우",
}


def roman(h):
    """한글 → 로마자 (RR 유사, 접기 전 형태)."""
    out = []
    for ch in h:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            b = o - 0xAC00
            out.append(INI[b // 588] + MED[(b % 588) // 28] + FIN[b % 28])
        elif ch.isalpha():
            out.append(ch.lower())
    return "".join(out)


def fold(s):
    """표기 변형 접기 — 양쪽(한글 로마자화·2026 영문)에 동일 적용."""
    s = re.sub(r"[^a-z]", "", str(s).lower())
    for a, b in (("ch","c"),("sh","s"),("ck","g"),("kh","k"),("th","t"),("ph","p"),
                 ("young","yeong"),("oung","eong"),("hyun","hyeon"),("kyung","gyeong"),
                 ("sung","seong"),("jung","jeong"),("yung","yeong"),("bum","beom"),
                 ("suk","seog"),("hoon","hun"),("joon","jun"),("woo","u"),("oo","u"),
                 ("ee","i"),("oi","oe"),("uh","eo"),("ah","a"),("hh","h")):
        s = s.replace(a, b)
    for a, b in (("k","g"),("t","d"),("p","b"),("r","l")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)          # 겹자 축약
    return s


def lev1(a, b):
    """편집거리 ≤1 인가."""
    if a == b: return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb: a, b, la, lb = b, a, lb, la
    i = j = d = 0
    while i < la and j < lb:
        if a[i] == b[j]: i += 1; j += 1
        else:
            d += 1; j += 1
            if d > 1: return False
    return True


def new_key(name):
    """2026 영문명 → (한글 성씨, 접힌 이름) 후보들."""
    toks = [t for t in re.sub(r"[^A-Za-z ]", " ", str(name)).split() if t]
    outs = []
    for i, t in enumerate(toks):
        h = SUR.get(t.lower())
        if h:
            rest = "".join(toks[:i] + toks[i+1:])
            outs.append((h, fold(rest)))
    if not outs and toks:                     # 성씨 미인식 (외국인 등)
        outs.append((None, fold("".join(toks))))
    return outs


def main():
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")[["game_id", "season"]]
    x = pl.merge(g, on="game_id")
    old = x[x.season < 2026]; new = x[x.season == 2026]

    O = {}
    for pid, gg in old.groupby("player_id"):
        nk = gg.nickname.dropna()
        nick = str(nk.mode().iloc[0]) if len(nk) else ""
        if not re.search(r"[가-힣]", nick): continue
        sur2 = nick[:2] if nick[:2] in ("남궁","황보","제갈","선우","독고") else None
        sur, giv = (sur2, nick[2:]) if sur2 else (nick[0], nick[1:])
        last = gg[gg.season == gg.season.max()]
        O[int(pid)] = dict(sur=sur, giv=fold(roman(giv)), nick=nick,
                           teams=set(gg.team_id), lastteams=set(last.team_id),
                           lastsea=int(gg.season.max()),
                           jer=set(gg.jersey_number.dropna().astype(int)),
                           gk=(gg.starting_position_name == "GK").any())
    by_key = defaultdict(list); by_sur = defaultdict(list)
    for pid, d in O.items():
        by_key[(d["sur"], d["giv"])].append(pid); by_sur[d["sur"]].append(pid)

    N = {}
    for pid, gg in new.groupby("player_id"):
        nm = str(gg.player_name.mode().iloc[0]) if len(gg.player_name.mode()) else ""
        N[int(pid)] = dict(name=nm, keys=new_key(nm), teams=set(gg.team_id),
                           jer=set(gg.jersey_number.dropna().astype(int)),
                           gk=(gg.starting_position_name == "GK").any())

    B, tier = {}, {}
    used = set()
    def take(np_, op_, t):
        if op_ in used: return
        B[np_] = op_; used.add(op_); tier[np_] = t

    for np_, d in N.items():
        cands = []
        for k in d["keys"]:
            if k[0] is None: continue
            cands += by_key.get(k, [])
        cands = [c for c in dict.fromkeys(cands) if c not in used]
        if len(cands) == 1: take(np_, cands[0], "T1"); continue
        if len(cands) > 1:
            tm = [c for c in cands if O[c]["lastteams"] & d["teams"]]
            if len(tm) == 1: take(np_, tm[0], "T2"); continue
    for np_, d in N.items():                  # T3 퍼지
        if np_ in B: continue
        surs = {k[0] for k in d["keys"] if k[0]}
        givs = {k[1] for k in d["keys"]}
        cands = []
        for s in surs:
            for c in by_sur.get(s, []):
                if c in used: continue
                if any(lev1(O[c]["giv"], gv) for gv in givs):
                    cands.append(c)
        cands = list(dict.fromkeys(cands))
        sup = [c for c in cands if (O[c]["lastteams"] & d["teams"]) or (O[c]["jer"] & d["jer"])]
        if len(sup) == 1: take(np_, sup[0], "T3")

    # T4 외국인: 과거 닉네임(한글 음역) 전체 로마자 vs 2026 이름 전체/토큰,
    # 편집거리 ≤1 · 팀 연속 또는 등번호 지지 · 유일.
    # 음역은 자음군·어말에 ㅡ(eu)를 끼워 넣으므로(토마스→tomaseu) eu 제거 변형도 시도.
    full_old = {pid: {fold(roman(d["nick"])),
                      fold(roman(d["nick"])).replace("eu", "")} for pid, d in O.items()}
    for np_, d in N.items():
        if np_ in B: continue
        toks = [fold(t) for t in re.sub(r"[^A-Za-z ]", " ", d["name"]).split() if t]
        toks.append(fold(d["name"]))
        cands = [c for c, fos in full_old.items() if c not in used
                 and any(lev1(fo, t) for fo in fos for t in toks if len(t) >= 4)]
        sup = [c for c in cands if (O[c]["lastteams"] & d["teams"]) or (O[c]["jer"] & d["jer"])]
        if len(sup) == 1: take(np_, sup[0], "T4")

    # 안전장치: GK 플래그 불일치 + (팀 연속·등번호 지지 없음) → 기각
    drop = [n for n, o in B.items()
            if O[o]["gk"] != N[n]["gk"]
            and not ((O[o]["lastteams"] & N[n]["teams"]) and (O[o]["jer"] & N[n]["jer"]))]
    for n in drop:
        used.discard(B[n]); del B[n]; del tier[n]
    if drop: print(f"GK 불일치·무지지 기각: {len(drop)}건")

    from collections import Counter
    tc = Counter(tier.values())
    print(f"2026 선수 {len(N)}명 → 매칭 {len(B)} ({100*len(B)/len(N):.1f}%)  "
          f"T1 {tc['T1']} · T2 {tc['T2']} · T3 {tc['T3']}")
    # 검증
    same = [(n, o) for n, o in B.items() if O[o]["lastteams"] & N[n]["teams"]]
    jer_ok = np.mean([bool(O[o]["jer"] & N[n]["jer"]) for n, o in same]) if same else 0
    gk_ok = np.mean([O[o]["gk"] == N[n]["gk"] for n, o in B.items()])
    ls = Counter(O[o]["lastsea"] for o in B.values())
    print(f"검증 — 같은 팀 잔류 {len(same)}명 등번호 일치 {100*jer_ok:.1f}% · "
          f"GK 일관성 {100*gk_ok:.1f}%")
    print(f"과거 최근 활동 시즌 분포: {dict(sorted(ls.items()))}")
    ex = [(N[n]['name'].strip(), O[o]['nick'], tier[n]) for n, o in list(B.items())[:8]]
    for e in ex: print("  예:", e)
    out = ROOT / "outputs" / "person_bridge_2026.json"
    json.dump({str(k): int(v) for k, v in B.items()}, open(out, "w"))
    print(f"저장 {out}")


if __name__ == "__main__":
    main()
