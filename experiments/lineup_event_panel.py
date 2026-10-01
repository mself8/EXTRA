"""이벤트 전용 라인업 패널 — 트래킹 없이 **2,540경기 · 6시즌** 전부.

왜
  현행 패널(`lineup_pipeline.build`)은 트래킹 구간이 단위라 974경기·구간 1,948행
  (표본외 1,516)에 갇힌다. 이번 세션에서 검정력 부족으로 막힌 것들이 —
  1차차분 귀무 · 블록 구성 식별 실패 · 신경망이 릿지를 못 이김 — 전부 여기서 온다.

  그런데 살아남은 주장(적합 > 합)은 `vaepon` 블록만 쓰고 그건 순수 이벤트 피처다.
  단위를 **팀-경기**로 바꾸고 목표를 이벤트 산출(xG·슛·득점)로 두면 5,080행이 된다.

무엇을 잃는가 (정직하게)
  · 단위가 거칠다 — 선발 중 평균 4.25명이 교체로 빠지므로 XI 와 결과의 대응이 흐리다.
    WINDOW=firstsub(+WFLOOR) 로 y 를 첫 교체 이전 창으로 제한할 수 있으나
    **관문 기각** (experiments/window_label.py, 2026-09-03): U-22 조기 교체로
    창 중앙값 45분이라 per-90 잡음 증가가 오염 감소를 압도 — 하한 0/45분 ΔR²
    -0.016*, 하한 60분에서도 -0.002 (개선 없음). 경기 전체 라벨 유지.
  · 트래킹 목표(apv·tilt·poss)와 피처(pos2·space)가 빠진다. 둘 다 앞선 검정에서
    "식별 안 됨" 이었으므로 손실은 작다.
  · 배치 연구는 이 트랙에서 불가능하다 — 트래킹 트랙으로 남긴다.

피처는 `onball_vaep_event.parquet` (이벤트 pid · 2,540경기 · 좌표 정렬 버그 수정본).
집계는 현행과 동일: 선수별 **과거-only 지수감쇠 스냅샷**의 XI 평균, 우리 − 상대.

출력: build_event() → (R, F, BASE, IDX)
실행: python -m experiments.lineup_event_panel
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR

import os as _os
FEAT_FILE = ROOT / "outputs" / _os.environ.get("ONBALL_FILE", "onball_vaep_event.parquet")
HL = float(_os.environ.get("SNAP_HL", 60.0))   # 지수감쇠 반감기 (경기 순번)
MINPL = 8          # XI 중 이력이 있는 최소 인원


def _snapshots(gpos, hist=1.0, hl=HL):
    """선수-경기 표 → 과거-only 지수감쇠 스냅샷. lineup_pipeline.load_snapshots 와 동일."""
    P = pd.read_parquet(FEAT_FILE)
    # DROPFAM=zv,zn 처럼 열 계열을 통째로 제외 (기본은 전부 사용)
    _dp = tuple(x.strip() + "_" for x in _os.environ.get("DROPFAM", "").split(",") if x.strip())
    FC = [c for c in P.columns if c not in ("game_id", "player_id")
          and not (_dp and c.startswith(_dp))]
    P = P[P.game_id.isin(gpos)].copy()
    P["t"] = P.game_id.map(gpos)
    P = P.sort_values(["player_id", "t"]).reset_index(drop=True)
    X = P[FC].to_numpy(np.float32)
    # PER90=1: 경기당 합 → per-90 비율 (출전 분으로 정규화, 하한 15분) + 노출량 열(ex_min: 출전 분/90) 추가.
    #   근거(입력 감사 2026-09-05): 경기당 합의 감쇠 평균은 교체 출전 경기가 평균을 희석해 교체 비율 상위 20% 점수가 36% 낮음.
    if _os.environ.get("PER90", "0") == "1":
        _pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv").drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"]).minutes_played
        _min = np.array([float(_pl.get((int(a), int(b)), np.nan)) for a, b in zip(P.game_id, P.player_id)])
        _min = np.where(np.isfinite(_min) & (_min > 0), _min, 90.0)
        _scale = 90.0 / np.clip(_min, 15.0, None)
        _rate = [c for c in FC if not c.startswith(("gk_is",))]                  # 비율화 대상: 합 성격 열 전부
        _ri = np.array([FC.index(c) for c in _rate]); X[:, _ri] = X[:, _ri] * _scale[:, None].astype(np.float32)
        X = np.hstack([X, (_min / 90.0)[:, None].astype(np.float32)]); FC = list(FC) + ["ex_min"]
    lam = np.log(2) / hl
    # PER90=2: 합의 비율 — 스냅샷 = (감쇠 액션 합) / (감쇠 출전 분) × 90, 노출량 열 ex_min = 감쇠 평균 출전 분/90.
    #   경기별 비율 평균(PER90=1)은 짧은 출전이 튀고 감독의 기용량 정보를 지운다; 합의 비율은 희석도 폭주도 없다.
    P90B = _os.environ.get("PER90", "0") == "2"
    if P90B:
        _pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv").drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"]).minutes_played
        _mins = np.array([float(_pl.get((int(a_), int(b_)), np.nan)) for a_, b_ in zip(P.game_id, P.player_id)])
        _mins = np.where(np.isfinite(_mins) & (_mins > 0), _mins, 90.0).astype(np.float32)
        FC = list(FC) + ["ex_min"]
    by, cur, a, w, am, last = {}, None, None, 0., 0., None
    for i, (pid_, t_) in enumerate(zip(P.player_id.to_numpy(), P.t.to_numpy())):
        if pid_ != cur:
            cur = pid_; a = np.zeros(X.shape[1], np.float32); w = 0.; am = 0.; last = None
        if last is not None:
            dd = np.exp(-lam * (t_ - last)); a *= dd; w *= dd; am *= dd
        if w >= hist:
            if P90B: by.setdefault(int(pid_), []).append((int(t_), np.append(a / max(am / 90.0, 0.2), np.float32(am / w / 90.0))))
            else: by.setdefault(int(pid_), []).append((int(t_), a / w))
        a = a + X[i]; w += 1.; last = t_
        if P90B: am += float(_mins[i])
    T = {p: np.array([z[0] for z in v]) for p, v in by.items()}
    return by, T, len(FC), FC


def _shots_goals():
    """팀-경기 슛 수 (SPADL 슛 계열) — 목표 계산용."""
    sp = pd.read_parquet(VAEP_OUTPUT_DIR / "spadl_all.parquet",
                         columns=["game_id", "team_id", "type_id"])
    sh = sp[sp.type_id.isin([11, 12, 13])]
    return sh.groupby(["game_id", "team_id"]).size().rename("sh")


def build_event(target="shots", minpl=MINPL):
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values("game_date").reset_index(drop=True)
    gpos = {int(r.game_id): i for i, r in g.iterrows()}
    by, T, dim, FC = _snapshots(gpos)

    pl = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    st = pl[pl.is_starter == True].groupby(["game_id", "team_id"]).player_id.apply(  # noqa: E712
        lambda s: [int(x) for x in s])
    st = st[st.str.len() == 11]
    xg = pd.read_parquet(ROOT / "outputs" / "team_xg.parquet").set_index(
        ["game_id", "team_id"]).xg
    npxg = pd.read_parquet(ROOT / "outputs" / "team_npxg.parquet").set_index(
        ["game_id", "team_id"]).npxg
    sh = _shots_goals()

    # WINDOW=firstsub: y 를 [0, 양팀 첫 교체) 창의 npxg 차 per-90 으로 바꾼다.
    # K리그는 U-22 조기 교체 탓에 창 중앙값이 45분(전반 중 26%·≤30분 15%)이라
    # 창 길이가 크게 다르다 — nlvl(=WLS 가중)·dur 에 t1 을 실어 소비자가 처리.
    WIN = _os.environ.get("WINDOW", "")
    T1, wnp = None, None
    if WIN == "firstsub":
        assert target == "npxg", "WINDOW=firstsub 는 npxg 목표만 지원"
        sub = pd.read_parquet(ROOT / "outputs" / "subs.parquet")
        # WFLOOR: 창 하한(분) — U-22 조기 교체로 창이 반토막나는 것을 막는 절충.
        # 하한 이전의 교체 오염은 감수하고, 하한 이후 첫 교체에서 자른다.
        wfl = float(_os.environ.get("WFLOOR", "0"))
        T1 = sub[sub.dir == "Off"].groupby("game_id").t.min().clip(lower=wfl, upper=90.)
        sx = pd.read_parquet(ROOT / "outputs" / "shot_xg.parquet")
        sx = sx[sx.pen == 0].copy()
        sx["t1"] = sx.game_id.map(T1).fillna(90.)
        wnp = sx[sx.minute < sx.t1].groupby(["game_id", "team_id"]).xg.sum()

    def fx(ids, tt):
        v = np.zeros(dim, np.float32); c = 0
        for p in ids:
            arr = T.get(p)
            if arr is None:
                continue
            j = np.searchsorted(arr, tt, "right") - 1
            if j < 0:
                continue
            v += by[p][j][1]; c += 1
        return v, c

    rows, FT = [], []
    for r in g.itertuples():
        gid = int(r.game_id)
        ha, aw = int(r.home_team_id), int(r.away_team_id)
        if (gid, ha) not in st.index or (gid, aw) not in st.index:
            continue
        tt = gpos[gid]
        va, ca = fx(st.loc[(gid, ha)], tt)
        vb, cb = fx(st.loc[(gid, aw)], tt)
        if min(ca, cb) < minpl:
            continue
        for me, opp, sgn in ((ha, aw, +1.), (aw, ha, -1.)):
            yv = {
                "xg": float(xg.get((gid, me), np.nan)) - float(xg.get((gid, opp), np.nan)),
                "npxg": float(npxg.get((gid, me), np.nan)) - float(npxg.get((gid, opp), np.nan)),
                "shots": float(sh.get((gid, me), 0)) - float(sh.get((gid, opp), 0)),
                "goals": float(r.home_score - r.away_score) * (1 if me == ha else -1),
            }[target]
            nl, du = 1.0, 90.0
            if WIN == "firstsub":
                t1 = float(T1.get(gid, 90.)) if T1 is not None else 90.
                if np.isfinite(yv):        # 사례 집합은 현행(경기 전체 npxg 결측 제외)과 동일하게
                    yv = (float(wnp.get((gid, me), 0.)) - float(wnp.get((gid, opp), 0.))) * 90. / t1
                nl, du = t1 / 90., t1
            rows.append(dict(game_id=gid, t=tt, season=int(r.season), me=me, opp=opp,
                             home=1. if me == ha else 0., y=yv, nlvl=nl, dur=du))
            FT.append((va - vb) if sgn > 0 else (vb - va))
    R = pd.DataFrame(rows)
    ok = np.isfinite(R.y.to_numpy())
    R = R[ok].reset_index(drop=True)
    M = np.vstack([f for f, k in zip(FT, ok) if k]).astype(np.float64)
    keep = M.std(0) > 1e-9
    M = M[:, keep]
    Mraw = M.copy()                       # 표준화 전 — 세밀도 사다리용
    M = (M - M.mean(0)) / M.std(0)
    F = {"vaepon": M, "raw": Mraw}
    SEA = pd.get_dummies(R.season.astype(str)).to_numpy(float)
    BASE = np.hstack([R[["home"]].to_numpy(float), SEA])
    # REDCTRL=1: 수적 열세 분 차(퇴장 후 남은 시간 합, 우리−상대)를 기저 공변량에 추가.
    # 기저 열은 채점 가중치(coef_[B.shape[1]:])에서 제외되므로 라벨 통제로만 작용한다.
    # 근거: 이 통제 하나로 선형 R² 0.10→0.16, 비선형 잔차 증분 소멸 (nonadditive_redcard_check).
    if _os.environ.get("REDCTRL", "0") == "1":
        RC = pd.read_parquet(ROOT / "outputs" / "red_cards.parquet")
        sxr = pd.read_parquet(ROOT / "outputs" / "shot_xg.parquet")
        gend = sxr.groupby("game_id").minute.max().clip(lower=90.0)
        RC["short"] = [max(0.0, float(gend.get(gg, 95.0)) - tt_) for gg, tt_ in zip(RC.game_id, RC.t)]
        short = RC.groupby(["game_id", "team_id"]).short.sum()
        red = np.array([float(short.get((int(a), int(m)), 0.0)) - float(short.get((int(a), int(o)), 0.0))
                        for a, m, o in zip(R.game_id, R.me, R.opp)])
        BASE = np.hstack([BASE, red[:, None]])
    BASE = (BASE - BASE.mean(0)) / np.where(BASE.std(0) == 0, 1, BASE.std(0))
    gord = R.groupby("game_id").t.first().sort_values().index.to_numpy()
    IDX = [np.flatnonzero(R.game_id.isin(set(x)).to_numpy())
           for x in np.array_split(gord, 5)]
    return R, F, BASE, IDX, np.array(FC)[keep]


if __name__ == "__main__":
    for tg in ["shots", "xg"]:
        R, F, BASE, IDX, cols = build_event(tg)
        print(f"[{tg}] 팀-경기 {len(R):,} · 경기 {R.game_id.nunique():,} · "
              f"시즌 {sorted(R.season.unique())} · 피처 {F['vaepon'].shape[1]} · "
              f"폴드 {[len(i) for i in IDX]} · y SD {R.y.std():.3f}")
