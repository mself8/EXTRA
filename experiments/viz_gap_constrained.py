"""제약 추천 XI 시각화 — 모델이 **단(depth level)을 직접 출력**하므로 배치가 정당하다.

왼쪽: 감독 실제 XI (그 경기 공시 좌표)
가운데: 제약 추천 XI — 세로는 모델의 단, 가로는 그 단의 감독 슬롯에 통산 폭으로 배정
오른쪽: 격차 오분위 + 위약 비교

실행: python -m experiments.viz_gap_constrained
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR, RAW_DATA_DIR  # noqa: E402
from fig_style import apply_style, CB  # noqa: E402
from gap_constrained import LEVELS, GK_MAX_LVL, fe_ols  # noqa: E402
apply_style()
matplotlib.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans", "serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
L, W = 105.0, 68.0

# 병합 모드 감지: 현재 players.csv 에 비정규 ID 가 없으면 병합본이다 → 원본 좌표를 정규 ID 로 매핑
def _pidmap():
    import json
    f = ROOT / "outputs" / "pid_merge.json"
    if not f.exists(): return {}
    M = {int(k): int(v) for k, v in json.load(open(f)).items()}
    try:
        cur = set(pd.read_csv(VAEP_OUTPUT_DIR / "players.csv", usecols=["player_id"]).player_id.astype(int))
    except Exception:
        return {}
    noncanon = {k for k, v in M.items() if k != v}
    return M if not (noncanon & cur) else {}


PIDMAP = _pidmap()
from gap_lane import _cid as _cid_lane   # noqa: E402  시즌 인지 맵 (2021~25 / 2026 원시 pid 공간 분리)
_cid = lambda p, gid=None: _cid_lane(p, gid)


def pitch(ax):
    ax.add_patch(Rectangle((0, 0), L, W, fc="#f2f6f1", ec="#9aa79a", lw=1.0, zorder=0))
    ax.plot([L / 2, L / 2], [0, W], c="#9aa79a", lw=0.8, zorder=1)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, fc="none", ec="#9aa79a", lw=0.8, zorder=1))
    for x0 in (0, L - 16.5):
        ax.add_patch(Rectangle((x0, (W - 40.3) / 2), 16.5, 40.3, fc="none", ec="#9aa79a", lw=0.8, zorder=1))
    ax.set_xlim(-2, L + 2); ax.set_ylim(-2, W + 2); ax.set_aspect("equal"); ax.axis("off")


def match_xy(gid):
    """그 경기 공시 좌표 {pid: (x_m, y_m)} 와 팀별 (단 -> [폭...]) 슬롯."""
    xy, slots = {}, {}
    for f in RAW_DATA_DIR.glob(f"*/*/match/{gid}/lineup.json"):
        for e in json.load(open(f))["result"]:
            q = e.get("position") or {}
            if not (e.get("is_starting_lineup") and "x" in q and "y" in q): continue
            p, tid = _cid(e["player_id"], gid), int(e.get("team_id", -1))
            yv, xv = float(q["y"]), float(q["x"])
            xy[p] = (yv * L, (1.0 - xv) * W)
            if yv > GK_MAX_LVL:
                k = int(np.argmin(np.abs(LEVELS - yv)))
                slots.setdefault(tid, {}).setdefault(k, []).append((1.0 - xv) * W)
    return xy, slots


def career_width():
    """선수별 통산 공시 가로 중앙값 (단 안 슬롯 배정용)."""
    rows = []
    for f in RAW_DATA_DIR.glob("*/*/match/*/lineup.json"):
        try: d = json.load(open(f))["result"]
        except Exception: continue
        _g = int(f.parent.name)
        for e in d:
            q = e.get("position") or {}
            if e.get("is_starting_lineup") and "x" in q:
                rows.append((_cid(e["player_id"], _g), float(q["x"])))
    D = pd.DataFrame(rows, columns=["pid", "x"]).groupby("pid").x.median()
    return {int(k): (1.0 - float(v)) * W for k, v in D.items()}


def layout_rec(rec, lv, gk, slots_t, CW):
    """추천 XI 배치: 세로=모델의 단, 가로=그 단의 감독 슬롯에 통산 폭으로 헝가리안 배정."""
    out = {}
    out[gk] = (0.08 * L, W / 2)
    by_lvl = {}
    for p in rec:
        if p == gk: continue
        by_lvl.setdefault(int(lv.get(p, 2)), []).append(p)
    from scipy.optimize import linear_sum_assignment
    for k, ps in by_lvl.items():
        av = sorted(slots_t.get(k, []))
        if len(av) < len(ps):                        # 감독 슬롯이 모자라면 균등 슬롯으로 보충
            step = W / (len(ps) + 1)
            av = av + [step * (i + 1) for i in range(len(ps) - len(av))]
            av = sorted(av)
        # 통산 폭과 슬롯 폭의 거리 최소 1:1 배정 (같은 쪽 선수 둘이 겹치지 않게)
        cost = np.array([[abs(CW.get(p, W / 2) - a) for a in av] for p in ps], float)
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            out[ps[i]] = (float(LEVELS[k]) * L, float(av[j]))
    # 같은 자리 겹침 최소 분리
    P = {p: np.array(v, float) for p, v in out.items()}
    ks = list(P)
    for _ in range(80):
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                d = P[b] - P[a]; n = np.linalg.norm(d)
                if n < 8.5:
                    u = d / n if n > 1e-6 else np.array([0.0, 1.0])
                    P[a] -= u * (8.5 - n) / 2; P[b] += u * (8.5 - n) / 2
        for p in ks:
            P[p][0] = np.clip(P[p][0], 5, L - 5); P[p][1] = np.clip(P[p][1], 5, W - 5)
    return {p: tuple(v) for p, v in P.items()}


def main():
    C = pd.read_pickle(ROOT / "outputs" / "gap_constrained_full.pkl")
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    NAME = {}
    for pid, gg in pl.groupby("player_id"):
        nk = gg["nickname"].dropna()
        NAME[int(pid)] = str(nk.mode().iloc[0]) if len(nk) and nk.mode().iloc[0] else str(gg.player_name.mode().iloc[0])
    TEAM = {}
    try:
        tt = pd.read_csv(VAEP_OUTPUT_DIR / "teams.csv")
        TEAM = {int(r.team_id): str(r.team_name) for r in tt.itertuples(index=False)}
    except Exception:
        pass
    CW = career_width()
    c = C.sort_values("gap_a").iloc[-1]
    gid = int(c["gid"]); tid = int(c["tid"])
    XY, SL = match_xy(gid)
    gk = [p for p in c["rec_a"] if p in c["gks"]]
    gk = gk[0] if gk else c["rec_a"][0]
    POSR = layout_rec(list(c["rec_a"]), c["lv"], gk, SL.get(tid, {}), CW)
    both = set(c["coach_xi"]) & set(c["rec_a"])

    fig = plt.figure(figsize=(12.2, 4.2))
    gsp = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.16)
    for k, (key, ttl, src) in enumerate([("coach_xi", "감독의 실제 XI", XY),
                                         ("rec_a", "모델 추천 XI (모양 고정)", POSR)]):
        ax = fig.add_subplot(gsp[0, k]); pitch(ax)
        for p in c[key]:
            x, yy = src.get(int(p), (L / 2, W / 2))
            uniq = p not in both
            col = CB["gray"] if not uniq else (CB["vermillion"] if k == 0 else CB["blue"])
            ax.scatter(x, yy, s=200, c=col, ec="white", lw=1.0, zorder=3, alpha=.95)
            t = ax.text(x, yy - 4.6, NAME.get(int(p), "?"), ha="center", va="top",
                        fontsize=6.8, color="#222222", zorder=5)
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])
        ax.set_title(f"{ttl}   (고유 {11-len(both)}명)", fontsize=9)
    tn = TEAM.get(tid, tid); on = TEAM.get(int(c["oid"]), c["oid"])
    fig.text(0.34, 1.03, f"{c['date']}  {tn} vs {on}   ·   모양 {'-'.join(str(v) for v in c['cA'] if v)}"
                         f"   ·   점수차 {c['gap_a']:+.3f}  ·  실제 xG 차 {c['y']:+.2f}",
             ha="center", fontsize=9)

    ax = fig.add_subplot(gsp[0, 2])
    for col, lab, cc, mk in [("gap_a", "제약 추천 (모양 고정)", CB["blue"], "o"),
                             ("gap_placebo", "위약: 직전 경기 XI", CB["gray"], "s")]:
        d = C.dropna(subset=[col])
        q = pd.qcut(d[col], 5, labels=False)
        m = d.groupby(q).agg(g=(col, "mean"), y=("y", "mean"), n=("y", "size"),
                             se=("y", lambda v: v.std() / np.sqrt(len(v))))
        ax.errorbar(m.g, m.y, yerr=1.96 * m.se, marker=mk, ms=5, lw=1.6, color=cc,
                    ecolor=cc, elinewidth=0.9, capsize=2.5, alpha=.9, label=lab)
    ax.axhline(0, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("모델이 주장하는 이득 (추천 XI 점수 − 감독 XI 점수)", fontsize=8)
    ax.set_ylabel("실제 xG 차", fontsize=8)
    ax.set_title("주장한 손실이 클수록 결과가 나쁘다", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    out = ROOT / "experiments" / "fig_gap_constrained.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"저장 {out}")
    print(f"사례: {c['date']} {tn}  모양 {c['cA']}  적중 {c['hit_a']}/11")
    for p in c["rec_a"]:
        mark = "공통" if p in both else "추천만"
        print(f"  단{c['lv'].get(p, '-')}  {NAME.get(int(p), '?'):8s} {c['sc'][p]:+.4f}  {mark}")


if __name__ == "__main__":
    main()
