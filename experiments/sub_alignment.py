"""교체 정렬 지표 — 추천이 맞다면 선발이 추천과 달라도 경기 중 교체는 추천 방향으로 움직여야 한다.

케이스 (경기, 팀): R 추천 XI · C 감독 XI · P 당일 명단 중 점수 있는 선수 · B = P∖C 벤치 · ON 교체 투입 · OFF 교체 아웃
  M1 투입 정렬  벤치 선수 i 가 투입될 확률이 i∈R 이면 더 높은가: on_i = α_{case×pos} + β1·1[i∈R]   (케이스×포지션군 고정효과, 경기 클러스터 부트스트랩)
  M2 아웃 정렬  선발 선수 i 가 빠질 확률이 i∉R 이면 더 높은가: off_i = α_{case×pos} + β2·1[i∉R]   (GK 제외)
  M3 순이동     Δ = |R∩XI_final| − |R∩C| (XI_final = (C∪ON)∖OFF) 를, 같은 수의 투입/아웃을 벤치·선발에서 무작위로 뽑은 귀무(200회)와 비교 → 평균·z
  M4 점수 방향  교체 시각별 mean s(ON_t) − mean s(OFF_t) 를 같은 귀무와 비교 (점수 있는 모델만)
추천 원천: outputs/rec_<name>_<season>.pkl (role_balance_parallel SAVEREC) · 위약 prev = eval 덤프의 직전 XI
실행: python -m experiments.sub_alignment [names...]   (기본 eventxi vaepsum mins ridge prev)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); from config import VAEP_OUTPUT_DIR  # noqa
SEASONS = [int(x) for x in os.environ.get("SEASONS", "2024,2025").split(",")]; NPERM = int(os.environ.get("NPERM", 200)); NB = int(os.environ.get("NBOOT", 500))
POSG = {"GK": "GK", "CB": "DF", "LB": "DF", "RB": "DF", "LWB": "DF", "RWB": "DF", "CM": "MF", "LM": "MF", "RM": "MF", "CDM": "MF", "CAM": "MF", "DM": "MF", "AM": "MF", "LW": "FW", "RW": "FW", "CF": "FW", "ST": "FW", "SS": "FW"}


def load_subs():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); G26 = set(g[g.season >= 2026].game_id.astype(int))
    M = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_merge.json")).items()}; M26 = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_merge_2026.json")).items()}; B26 = {int(k): int(v) for k, v in json.load(open(ROOT / "outputs/pid_bridge_2026.json")).items()}
    def cid(p, gid):
        p = int(p)
        if gid in G26: q = B26.get(p, p); return M26.get(q, q)
        return M.get(p, p)
    S = pd.read_parquet(ROOT / "outputs/subs.parquet"); S["pid"] = [cid(p, int(gg)) for p, gg in zip(S.player_id, S.game_id)]
    S["on"] = S.dir.astype(str).str.lower().str.startswith("on")
    return S


def load_cases(name):
    """→ list of dict(gid, tid, season, R, C, P, sc|None, gk)"""
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    # 점수 사전(sc)에는 명단 밖 직전 선발(결장자)도 들어 있다(gap_soft 가 위약용으로 넣는다) — 그대로 P 로 쓰면 결장자가 벤치로 세진다.
    # 그래서 P 를 당일 명단으로 제한한다. 명단 = 그 경기·팀의 players.csv 전원 (gap_soft 의 pool 과 같은 정의)
    SQ = pl.groupby(["game_id", "team_id"]).player_id.apply(lambda s: {int(x) for x in s}).to_dict()
    pl = pl[pl.starting_position_name.notna()]
    # 포지션군은 선수의 **최빈 포지션**(전 출전 기록)으로 — 당일 기록은 교체 투입된 벤치 선수에게만 있어 M1 의 결과를 그대로 누설한다
    POS = pl.groupby("player_id").starting_position_name.agg(lambda x: x.value_counts().index[0]).to_dict()
    out = []
    for s in SEASONS:
        if name == "prev":
            f_ = ROOT / f"outputs/eval_daysset{os.environ.get('PROTO', '25')}_add_{s}.pkl"; D = pd.read_pickle(f_)["D"]   # 위약: 규약별 eval 덤프의 직전 XI (PROTO=25|26)
            for r in D.itertuples(index=False):
                if not r.prev: continue
                out.append(dict(gid=int(r.gid), tid=int(r.tid), season=s, R=set(int(p) for p in r.prev), C=set(int(p) for p in r.coach_xi), P=set(int(p) for p in r.sc) & SQ.get((int(r.gid), int(r.tid)), set()), sc=None))
        else:
            for d in pd.read_pickle(ROOT / f"outputs/rec_{name}_{s}.pkl"):
                out.append(dict(gid=int(d["gid"]), tid=int(d["tid"]), season=s, R=set(int(p) for p in d["rec"]), C=set(int(p) for p in d["coach"]), P=set(int(p) for p in d["sc"]) & SQ.get((int(d["gid"]), int(d["tid"])), set()), sc={int(k): float(v) for k, v in d["sc"].items()}))
    for c in out: c["pos"] = {p: POSG.get(POS.get(p, ""), "UNK") for p in c["P"]}
    return out


def fe_beta(rows, ycol, xcol, gcol):
    """y ~ α_g + β x : 그룹 내 차분 OLS β"""
    d = rows.copy(); d["yd"] = d[ycol] - d.groupby(gcol)[ycol].transform("mean"); d["xd"] = d[xcol] - d.groupby(gcol)[xcol].transform("mean")
    den = (d.xd ** 2).sum(); return float((d.xd * d.yd).sum() / den) if den > 0 else np.nan


def boot_ci(rows, ycol, xcol, gcol, rng):
    gids = rows.gid.unique(); by = {g: rows[rows.gid == g] for g in gids}; vals = []
    for _ in range(NB):
        pick = rng.choice(gids, len(gids), replace=True); vals.append(fe_beta(pd.concat([by[g] for g in pick], ignore_index=True), ycol, xcol, gcol))
    return np.nanpercentile(vals, [2.5, 97.5])


def evaluate(name, S, rng):
    cases = load_cases(name); sub = {k: v for k, v in S.groupby(["game_id", "team_id"])}
    m1, m2, m3, m4 = [], [], [], []
    for i, c in enumerate(cases):
        s = sub.get((c["gid"], c["tid"]))
        ON = set(int(p) for p in s.pid[s.on]) & c["P"] if s is not None else set(); OFF = set(int(p) for p in s.pid[~s.on]) & c["C"] if s is not None else set()
        B = c["P"] - c["C"]; R, C = c["R"], c["C"]; gk = {p for p in c["P"] if c["pos"][p] == "GK"}
        for p in B: m1.append(dict(case=i, gid=c["gid"], grp=f"{i}_{c['pos'][p]}", on=float(p in ON), inR=float(p in R), season=c["season"]))
        for p in C - gk: m2.append(dict(case=i, gid=c["gid"], grp=f"{i}_{c['pos'][p]}", off=float(p in OFF), notR=float(p not in R), season=c["season"]))
        if ON or OFF:
            fin = (C | ON) - OFF; obs = len(R & fin) - len(R & C)
            Bl, Cl = sorted(B), sorted(C - gk); nulls = []
            for _ in range(NPERM):
                on_ = set(rng.choice(Bl, min(len(ON), len(Bl)), replace=False)) if Bl else set(); off_ = set(rng.choice(Cl, min(len(OFF), len(Cl)), replace=False)) if Cl else set()
                nulls.append(len(R & ((C | on_) - off_)) - len(R & C))
            m3.append(dict(gid=c["gid"], tid=c["tid"], season=c["season"], obs=obs, null=float(np.mean(nulls)), nsub=len(ON), ndis=len(R - C)))
            if c["sc"] is not None and ON and OFF:
                sc = c["sc"]; obs4 = np.mean([sc[p] for p in ON]) - np.mean([sc[p] for p in OFF])
                n4 = [np.mean([sc[p] for p in rng.choice(Bl, len(ON), replace=False)]) - np.mean([sc[p] for p in rng.choice(Cl, len(OFF), replace=False)]) for _ in range(NPERM)] if len(Bl) >= len(ON) and len(Cl) >= len(OFF) else []
                if n4: m4.append(dict(gid=c["gid"], season=c["season"], obs=obs4, null=float(np.mean(n4)), nsd=float(np.std(n4))))
    m1, m2, m3 = pd.DataFrame(m1), pd.DataFrame(m2), pd.DataFrame(m3); m4 = pd.DataFrame(m4)
    b1 = fe_beta(m1, "on", "inR", "grp"); c1 = boot_ci(m1, "on", "inR", "grp", rng); b2 = fe_beta(m2, "off", "notR", "grp"); c2 = boot_ci(m2, "off", "notR", "grp", rng)
    base_on = m1.on.mean(); base_off = m2.off.mean()
    def cz(df):
        d = df.obs - df.null; g = (df.assign(d=d)).groupby("gid")["d"].mean(); vals = [np.mean(rng.choice(g.to_numpy(), len(g), replace=True)) for _ in range(NB)]
        return d.mean(), np.percentile(vals, [2.5, 97.5]), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    d3, ci3, z3 = cz(m3); m3n = ((m3.obs - m3.null) / m3.ndis.clip(lower=1)).mean()
    line = f"{name:8s} 케이스 {len(cases):,} · M1 투입 β {b1:+.3f} [{c1[0]:+.3f},{c1[1]:+.3f}] (기저 {base_on:.3f}) · M2 아웃 β {b2:+.3f} [{c2[0]:+.3f},{c2[1]:+.3f}] (기저 {base_off:.3f}) · M3 순이동 {d3:+.3f} [{ci3[0]:+.3f},{ci3[1]:+.3f}] z={z3:+.2f} · 불일치당 {m3n:+.3f} (교체 {m3.nsub.mean():.2f}/케이스, |R−C| {m3.ndis.mean():.2f})"
    if len(m4): d4, ci4, z4 = cz(m4); line += f" · M4 점수 방향 {d4:+.4f} [{ci4[0]:+.4f},{ci4[1]:+.4f}] z={z4:+.2f}"
    print(line, flush=True)
    for s in SEASONS:
        a, b, c3 = m1[m1.season == s], m2[m2.season == s], m3[m3.season == s]
        if len(c3): d3s, ci3s, z3s = cz(c3); print(f"   {s}: M1 β {fe_beta(a, 'on', 'inR', 'grp'):+.3f} · M2 β {fe_beta(b, 'off', 'notR', 'grp'):+.3f} · M3 {d3s:+.3f} [{ci3s[0]:+.3f},{ci3s[1]:+.3f}] z={z3s:+.2f}", flush=True)
    return dict(name=name, m1=b1, m1ci=c1, m2=b2, m2ci=c2, m3=d3, m3ci=ci3, m3z=z3, m3n=m3n, T1=m1, T2=m2, T3=m3)


def paired(a, b, rng):
    """같은 경기 재표집으로 a−b 차이의 CI: M1 β, M2 β, M3 순이동, M3n"""
    gids = np.array(sorted(set(a["T3"].gid) & set(b["T3"].gid))); out = {}
    def sub(T, pick): by = T.groupby("gid").indices; return T.iloc[np.concatenate([by[g] for g in pick if g in by])]
    d1, d2, d3, d3n = [], [], [], []
    for _ in range(NB):
        pick = rng.choice(gids, len(gids), replace=True)
        d1.append(fe_beta(sub(a["T1"], pick), "on", "inR", "grp") - fe_beta(sub(b["T1"], pick), "on", "inR", "grp"))
        d2.append(fe_beta(sub(a["T2"], pick), "off", "notR", "grp") - fe_beta(sub(b["T2"], pick), "off", "notR", "grp"))
        A3, B3 = sub(a["T3"], pick), sub(b["T3"], pick); d3.append((A3.obs - A3.null).mean() - (B3.obs - B3.null).mean())
        d3n.append(((A3.obs - A3.null) / A3.ndis.clip(lower=1)).mean() - ((B3.obs - B3.null) / B3.ndis.clip(lower=1)).mean())
    for k, v in (("M1", d1), ("M2", d2), ("M3", d3), ("M3n", d3n)):
        lo, hi = np.nanpercentile(v, [2.5, 97.5]); out[k] = (float(np.nanmean(v)), lo, hi)
    print(f"  짝 {a['name']} − {b['name']}: " + " · ".join(f"{k} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if (lo > 0) == (hi > 0) else ''}" for k, (m, lo, hi) in out.items()), flush=True)
    return out


def main():
    names = sys.argv[1:] or ["eventxi", "vaepsum", "mins", "ridge", "prev"]; S = load_subs(); rng = np.random.default_rng(0)
    print(f"교체 행 {len(S):,} · 시즌 {SEASONS} · 귀무 {NPERM}회 · 부트스트랩 {NB}", flush=True)
    res = [evaluate(n, S, rng) for n in names]
    for r in res[1:]: paired(res[0], r, rng)
    pd.to_pickle([{k: v for k, v in r.items() if not k.startswith("T")} for r in res], ROOT / "outputs/sub_alignment.pkl")


if __name__ == "__main__":
    main()
