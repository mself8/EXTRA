"""K리그 공식 사이트 선수 명부·프로필 수집 (공식 등록번호 = playerId).

1단계 목록: /player.do?page=P&type=all&teamId=Kxx  (현역+역대, 30구단) → playerId·한글명·배번·포지션
2단계 상세: /record/playerDetail.do?playerId=ID → 한글명·영문명·소속구단·포지션·배번·국적·키·몸무게·생년월일
             + 시즌별 표 (시즌, 팀, 대회별 출장·득점·도움)
출력: outputs/kleague_players.parquet (프로필) · outputs/kleague_player_seasons.parquet (시즌-팀 행)
캐시: outputs/kleague_html/{id}.html (재실행 시 재사용, KL_CACHE 로 변경)
실행: python -m experiments.kleague_scrape [--list-only]
"""
from __future__ import annotations
import os, re, sys, time, html as htmlmod
from pathlib import Path
import requests, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("KL_CACHE", ROOT / "outputs" / "kleague_html"))
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://www.kleague.com"
TEAMS = "K09 K27 K21 K20 K22 K35 K36 K41 K17 K10 K06 K26 K31 K08 K02 K29 K32 K42 K01 K18 K07 K05 K04 K38 K34 K37 K40 K03 K39".split()
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0 (research crawler)"
SLEEP = float(os.environ.get("SLEEP", 0.25))


def get(url: str, tries: int = 4) -> str:
    for k in range(tries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200: time.sleep(SLEEP); return r.text
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    return ""


def parse_list(h: str, team: str) -> list[dict]:
    out = []
    for m in re.finditer(r'onPlayerClicked\((\d+)\)(.*?)</div>\s*</div>\s*</div>', h, re.S):
        pid, body = int(m.group(1)), m.group(2)
        nm = re.search(r'<span class="name">\s*([^<]+?)\s*(?:<span[^>]*>([^<]*)</span>)?\s*</span>', body)
        no = re.search(r'No\.\s*([0-9]+)', body)
        out.append(dict(kl_id=pid, team_code=team, ko=nm.group(1).strip() if nm else None,
                        pos=(nm.group(2).strip() if nm and nm.group(2) else None), bn=int(no.group(1)) if no else None))
    return out


def scrape_list() -> pd.DataFrame:
    f = ROOT / "outputs/kleague_list.parquet"
    if f.exists(): return pd.read_parquet(f)
    rows = []
    for t in TEAMS:
        prev = None
        for p in range(1, 80):                                   # 페이지는 1부터 (0 은 404)
            h = get(f"{BASE}/player.do?page={p}&type=all&teamId={t}")
            r = parse_list(h, t); ids = tuple(x["kl_id"] for x in r)
            if not r or ids == prev: break                       # 빈 페이지 또는 마지막 페이지 반복
            rows += r; prev = ids
        print(f"{t}: 누적 {len(rows):,}", flush=True)
    L = pd.DataFrame(rows).drop_duplicates(["kl_id", "team_code"]); L.to_parquet(f, index=False); return L


def parse_detail(h: str, pid: int) -> tuple[dict, list[dict]]:
    h2 = re.sub(r"\s+", " ", h)
    prof = dict(kl_id=pid)
    m = re.search(r'<table class="style2 center">(.*?)</table>', h2)
    if m:
        for th, td in re.findall(r"<th>([^<]*)</th> <td>([^<]*)</td>", m.group(1)):
            if th.strip(): prof[th.strip()] = htmlmod.unescape(td.strip())
    seas = []
    i = h2.find("시즌별")
    if i > 0:
        j = h2.find("<tbody>", i); k = h2.find("</tbody>", j)
        for tr in re.findall(r"<tr>(.*?)</tr>", h2[j:k]):
            td = [htmlmod.unescape(x.strip()) for x in re.findall(r"<td[^>]*>([^<]*)</td>", tr)]
            if len(td) >= 2 and td[0].isdigit(): seas.append(dict(kl_id=pid, season=int(td[0]), team_ko=td[1], cells=td[2:]))
    return prof, seas


def scrape_details(ids: list[int]) -> None:
    profs, seas = [], []
    for n, pid in enumerate(ids, 1):
        c = CACHE / f"{pid}.html"
        if c.exists(): h = c.read_text(encoding="utf-8")
        else:
            h = get(f"{BASE}/record/playerDetail.do?playerId={pid}")
            if h: c.write_text(h, encoding="utf-8")
        if not h: continue
        p, s = parse_detail(h, pid); profs.append(p); seas += s
        if n % 500 == 0: print(f"상세 {n:,}/{len(ids):,}", flush=True)
    P = pd.DataFrame(profs).rename(columns={"이름": "ko", "영문명": "en", "소속구단": "team_ko", "포지션": "pos", "배번": "bn", "국적": "nat", "키": "height", "몸무게": "weight", "생년월일": "birth"})
    P.to_parquet(ROOT / "outputs/kleague_players.parquet", index=False)
    Q = pd.DataFrame(seas); Q["cells"] = Q.cells.map(list); Q.to_parquet(ROOT / "outputs/kleague_player_seasons.parquet", index=False)
    print(f"프로필 {len(P):,} · 시즌행 {len(Q):,} · 생년월일 보유 {P.birth.notna().mean() * 100:.1f}% · 영문명 {P.en.fillna('').str.len().gt(0).mean() * 100:.1f}%", flush=True)


if __name__ == "__main__":
    L = scrape_list(); print(f"목록 {len(L):,} 행 · 고유 선수 {L.kl_id.nunique():,}", flush=True)
    if "--list-only" not in sys.argv: scrape_details(sorted(L.kl_id.unique()))
