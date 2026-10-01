"""감독 XI 와 모델 추천 XI 를 **나란히** 그린다 — 화살표·짝짓기 없음.

왼쪽 피치 = 감독이 실제로 세운 XI (공시 좌표)
오른쪽 피치 = 모델 추천 XI (모델이 배정한 단·레인 칸)
색: 회색 = 양쪽 다 · 빨강 = 감독만(제외됨) · 파랑 = 모델만(추가됨)

실행: PKL=... BETA=... python -m experiments.viz_side [사례수] [top|strat] [all|clean]
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn")); sys.path.insert(0, str(ROOT / "experiments"))
from config import VAEP_OUTPUT_DIR  # noqa: E402
from gap_constrained import LEVELS  # noqa: E402
from gap_soft import pick_soft, QFLOOR  # noqa: E402
from asym_penalty import pick_asym, qpen  # noqa: E402
from gap_lane import declared_grid  # noqa: E402
from gap_diagnose import absences  # noqa: E402
from viz_gap_constrained import match_xy, career_width, L, W  # noqa: E402

PKL = os.environ.get("PKL",   # 기본 = 프로덕션(블렌드 λ=0.25, blend_score 로 재생성)
    "outputs/gap_soft_fullonball_gk_resid_merged_mix25d_pc20_all_npxg_nozvzn_fixsd.pkl")
BETA = float(os.environ.get("BETA", 0.05))
NCASE = int(os.environ.get("NCASE", 2))       # 한 줄에 그릴 사례 수
FORMOPT = os.environ.get("FORMOPT", "") == "1"   # 포메이션도 함께 최적화
BF = float(os.environ.get("BF", 0.05))           # 전진 이동 추가 벌점
CRIT = os.environ.get("CRIT", "pen")             # 포메이션 비교: pen(벌점 포함 목적값) | sum
FS = 7.4 if NCASE == 1 else 4.5              # 이름 글자 크기
LANE_MID = {0: 0.875 * W, 1: 0.5 * W, 2: 0.125 * W}
GRAY, RED, BLUE = "#9aa3a0", "#c2453c", "#1f6fb2"


def pitch(ax, title=None):
    ax.add_patch(Rectangle((0, 0), L, W, fc="#f4f7f3", ec="#a8b3a8", lw=0.7, zorder=0))
    ax.plot([L / 2, L / 2], [0, W], c="#a8b3a8", lw=0.6, zorder=1)
    for x0 in (0, L - 16.5):
        ax.add_patch(Rectangle((x0, (W - 40.3) / 2), 16.5, 40.3, fc="none",
                               ec="#a8b3a8", lw=0.6, zorder=1))
    ax.set_xlim(-3, L + 3); ax.set_ylim(-4, W + 4); ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title, fontsize=6.2, pad=2.0, color="#333333")


def model_xy(slot, gk, CW):
    """모델 슬롯 → 좌표. 칸 안 여러 명은 통산 폭 순서로 벌린다."""
    out = {gk: np.array([0.08 * L, W / 2])}
    byc = defaultdict(list)
    for p, (k, l) in slot.items():
        if p != gk: byc[(k, l)].append(p)
    for (k, l), ps in byc.items():
        ps = sorted(ps, key=lambda p: -CW.get(p, W / 2))     # 폭 큰(왼쪽) 선수가 위로
        SEP = 12.0 if FS > 6 else 9.0
        span = SEP * (len(ps) - 1)
        for i, p in enumerate(ps):
            out[p] = np.array([float(LEVELS[k]) * L,
                               float(LANE_MID[l]) + span / 2 - SEP * i])
    return spread({p: tuple(v) for p, v in out.items()})


def spread(XY, dx=None, dy=None, iters=80):
    """라벨 겹침 밀어내기(비등방): 이름 라벨이 점 아래 가로로 길므로 가로 |Δx|<dx 이면서 세로 |Δy|<dy 인 쌍을
    겹침이 덜한 축으로 벌린다. 감독 패널(실측 좌표)·모델 패널 공통."""
    dx = dx or (19.0 if FS > 6 else 18.0); dy = dy or (9.0 if FS > 6 else 7.5)
    out = {p: np.array(v, float) for p, v in XY.items()}; ks = list(out)
    for _ in range(iters):
        moved = False
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                d = out[b] - out[a]; ax_, ay_ = abs(d[0]), abs(d[1])
                if ax_ < dx and ay_ < dy:
                    if (dy - ay_) / dy <= (dx - ax_) / dx:                # 세로로 벌리는 편이 싸다
                        s_ = (dy - ay_) / 2 + 0.2; sg = 1.0 if d[1] >= 0 else -1.0
                        out[a][1] -= sg * s_; out[b][1] += sg * s_
                    else:
                        s_ = (dx - ax_) / 2 + 0.2; sg = 1.0 if d[0] >= 0 else -1.0
                        out[a][0] -= sg * s_; out[b][0] += sg * s_
                    moved = True
        for p in ks:
            out[p][0] = np.clip(out[p][0], 4, L - 4); out[p][1] = np.clip(out[p][1], 5, W - 5)
        if not moved: break
    return {p: tuple(v) for p, v in out.items()}


def draw(ax, XY, ids, both, NAME, mine_col, sc=None):
    """sc 를 주면 교체된 선수(회색 아님)에 점수를 병기 — 박빙/확신 스왑 구분용."""
    for p in ids:
        if p not in XY: continue
        x, y = XY[p]
        c = GRAY if p in both else mine_col
        ax.scatter(x, y, s=(150 if FS > 6 else 86), c=c, ec="white", lw=0.7, zorder=3, alpha=.95)
        lbl = NAME.get(int(p), "?")
        if sc is not None and p not in both and p in sc:
            lbl += f"\n{sc[p]:+.3f}"          # 교체 선수만 점수 병기 (박빙/확신 구분) — 둘째 줄로 내려 라벨 폭을 줄인다
        t = ax.text(x, y - (4.2 if FS > 6 else 3.2), lbl, ha="center", va="top",
                    fontsize=FS, zorder=5, linespacing=0.95, color="#444444" if p in both else "#111111")
        t.set_path_effects([pe.withStroke(linewidth=1.7, foreground="white")])


def main():
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    MODE = sys.argv[2] if len(sys.argv) > 2 else "top"
    SUB = sys.argv[3] if len(sys.argv) > 3 else "clean"
    C = pd.read_pickle(ROOT / PKL)
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv"); g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    # FILTER_SEASONS="2025,2026" · FILTER_COMP="KLEAGUE1" 로 사례 풀 제한
    fs_ = os.environ.get("FILTER_SEASONS", ""); fc_ = os.environ.get("FILTER_COMP", "")
    ft_ = os.environ.get("FILTER_TID", "")          # 특정 팀만 (팀당 2개 제한 해제)
    if fs_ or fc_ or ft_:
        SEA = dict(zip(g.game_id, g.season)); CMP = dict(zip(g.game_id, g.competition_name))
        ok = [((not fs_) or str(SEA.get(int(x))) in fs_.split(",")) and
              ((not fc_) or CMP.get(int(x)) == fc_) for x in C.gid]
        if ft_:
            ok = [o and int(t) == int(ft_) for o, t in zip(ok, C.tid)]
        C = C[np.array(ok)].reset_index(drop=True)
        print(f"필터: 시즌 {fs_ or '전체'} · {fc_ or '전 리그'} · 팀 {ft_ or '전체'} → {len(C):,}사례")
    if SUB == "clean":
        AB = absences(gpos)
        C["absent"] = [AB.get((int(r.gid), int(r.tid)), (99, 0))[0]
                       for r in C.itertuples(index=False)]
        C = C[C.absent == 0].reset_index(drop=True)
    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    NAME = {}
    for pid, gg in pl.groupby("player_id"):
        nk = gg["nickname"].dropna()
        NAME[int(pid)] = (str(nk.mode().iloc[0]) if len(nk) and nk.mode().iloc[0]
                          else str(gg.player_name.mode().iloc[0]))
    TEAM = {}
    try:
        tt = pd.read_csv(VAEP_OUTPUT_DIR / "teams.csv")
        TEAM = {int(r.team_id): str(r.team_name) for r in tt.itertuples(index=False)}
    except Exception:
        pass
    CW = career_width(); LP = C.LPk.iloc[0]
    LIB = None
    if FORMOPT:
        from collections import Counter
        CELL, _, _ = declared_grid()
        cn = Counter(tuple(np.asarray(G).ravel()) for G in CELL.values())
        LIB = [np.array(k, int).reshape(5, 3) for k, v in cn.items() if v >= 20]
        print(f"포메이션 라이브러리 {len(LIB)}종 · 동시 최적화")

    def choose(c):
        """(포메이션, XI) 결합 최적."""
        def one(F):
            if BF > 0:
                R_, sl = pick_asym(c["sc"], c["qmap"], c["lane"], F,
                                   list(c["gks"]), set(c["pool"]), LP, BETA, BF)
                return R_, sl
            R_, sl, _ = pick_soft(c["sc"], c["qmap"], c["lane"], F,
                                  list(c["gks"]), set(c["pool"]), LP, BETA)
            return R_, sl
        if not FORMOPT:
            R_, sl = one(c["G"]); return R_, sl, c["G"]
        best = None; bs = -1e18
        for F in LIB:
            R_, sl = one(F)
            s = sum(c["sc"][p] for p in R_)
            if CRIT == "pen":
                for p, (k, l) in sl.items():
                    q = c["qmap"].get(p); lp = LP[c["lane"].get(p, 1)]
                    qk = float(q[k]) if q is not None else 0.2
                    mo = int(np.argmax(q)) if q is not None else 2
                    s -= BETA * (qpen(qk) - lp[l]) + BF * max(0, k - mo)
            if s > bs: bs, best = s, (R_, sl, F)
        return best
    # 사례 정렬용 격차도 **현재 픽커**로 재계산한다 (pkl 의 gap_s 는 빌드 시점 설정)
    print("현재 픽커로 전체 사례 재선택 중 ...", flush=True)
    RS = [choose(C.iloc[i]) for i in range(len(C))]
    C = C.assign(rec_v=[r[0] for r in RS], slot_v=[r[1] for r in RS],
                 form_v=[r[2] for r in RS],
                 gap=[sum(C.sc.iloc[i][p] for p in RS[i][0]) - C.coach_sc.iloc[i]
                      for i in range(len(C))])
    C = C.sort_values("gap").reset_index(drop=True)
    idx = (np.arange(len(C) - 1, -1, -1) if MODE == "top"
           else np.linspace(0, len(C) - 1, n_want * 4).astype(int))
    picks, seen = [], defaultdict(int)
    # PINS="gid:tid,gid:tid" — 특정 사례를 맨 앞에 고정
    for pin in [p for p in os.environ.get("PINS", "").split(",") if ":" in p]:
        pg, pt = (int(x) for x in pin.split(":"))
        m = C[(C.gid == pg) & (C.tid == pt)]
        if len(m): picks.append(m.iloc[0]); seen[pt] += 1
    npin = len(picks)
    for i in idx:
        if len(picks) >= n_want: break
        t = int(C.tid.iloc[i])
        cap = 10**9 if os.environ.get("FILTER_TID") else 2
        if seen[t] >= cap or any(int(p["gid"]) == int(C.gid.iloc[i]) and
                                 int(p["tid"]) == t for p in picks): continue
        picks.append(C.iloc[i]); seen[t] += 1
    picks = picks[:npin] + sorted(picks[npin:], key=lambda c: -c["gap"])
    print(f"사례 풀 {len(C):,} · 선택 {len(picks)}")

    ncase = NCASE
    nrow = int(np.ceil(len(picks) / ncase))
    pw = 6.2 if NCASE == 1 else 3.5
    fig, axes = plt.subplots(nrow, ncase * 2, figsize=(pw * ncase * 2, pw * 0.72 * nrow))
    axes = np.atleast_2d(axes)
    for a in axes.ravel(): a.axis("off")
    for i, c in enumerate(picks):
        r_, k_ = i // ncase, (i % ncase) * 2
        gid, tid = int(c["gid"]), int(c["tid"])
        XY, _ = match_xy(gid)
        rec, slot = c["rec_v"], c["slot_v"]
        gks = set(c["gks"]); gk = next((p for p in rec if p in gks), rec[0])
        MP = model_xy(slot, gk, CW)
        both = set(c["coach_xi"]) & set(rec)
        a1, a2 = axes[r_, k_], axes[r_, k_ + 1]
        pitch(a1, "감독 실제 XI"); pitch(a2, "모델 추천 XI")
        if FS > 6:
            a1.title.set_fontsize(9); a2.title.set_fontsize(9)
        draw(a1, XY, c["coach_xi"], both, NAME, RED, sc=c["sc"])
        draw(a2, MP, rec, both, NAME, BLUE, sc=c["sc"])
        tn, on = TEAM.get(tid, tid), TEAM.get(int(c["oid"]), c["oid"])
        fc = "-".join(str(x) for x in np.asarray(c["G"]).sum(1) if x)
        fm = "-".join(str(x) for x in np.asarray(
            c["form_v"] if FORMOPT else c["G"]).sum(1) if x)
        a1.text(L, W + 8.5, f"{c['date']}  {tn} vs {on}   |   감독 {fc} → 모델 {fm} · "
                f"격차 {c['gap']:+.3f} · "
                f"교체 {11-len(both)}명 · 실제 npxG 차 {c['y']:+.2f}",
                ha="center", va="bottom", fontsize=(10.5 if FS > 6 else 6.6),
                color="#111111")
    lg = [Line2D([], [], marker="o", ls="", ms=6, mfc=GRAY, mec="white", label="양쪽 다"),
          Line2D([], [], marker="o", ls="", ms=6, mfc=RED, mec="white", label="감독만 (모델이 제외)"),
          Line2D([], [], marker="o", ls="", ms=6, mfc=BLUE, mec="white", label="모델만 (모델이 추가)")]
    fig.legend(handles=lg, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.004))
    fig.suptitle("감독 실제 XI  vs  모델 추천 XI — 왼쪽이 실제, 오른쪽이 추천 "
                 "(공격 방향 →, 위쪽이 왼쪽 측면)", fontsize=11, y=1.016)
    fig.subplots_adjust(hspace=0.45, wspace=0.04, top=0.985)
    bt = (("" if abs(BETA - 0.02) < 1e-9 else f"_b{BETA:g}") + (f"_f{BF:g}" if BF else "")
          + ("_pen" if CRIT == "pen" else ""))
    out = ROOT / "experiments" / (f"fig_side_{SUB}_{MODE}"
          f"{'_big' if NCASE==1 else ''}{bt}{'_fo' if FORMOPT else ''}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"저장 {out}")


if __name__ == "__main__":
    main()
