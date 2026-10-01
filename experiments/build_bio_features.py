"""선수 바이오 열을 선수-경기 피처 파케이에 붙인다 (K리그 공식 프로필: 생년월일·국적·신장).

bio_age   경기일 기준 나이(년, 30 을 뺀 값 — 중심화)   bio_age2  (나이−30)²/100
bio_frn   외국인(국적≠한국) 1/0                        bio_ht    신장(cm)−180 (/10)
결측(브릿지 미연결)은 열 평균(0) 으로 두고 bio_miss=1.
입력 ONBALL_FILE (기본 onball_gk_resid_merged_defresp_ref.parquet) → 출력 …_refbio.parquet
실행: python -m experiments.build_bio_features
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gnn"))
from config import VAEP_OUTPUT_DIR  # noqa: E402

SRC = ROOT / "outputs" / os.environ.get("ONBALL_FILE", "onball_gk_resid_merged_defresp_ref.parquet")
OUT = Path(str(SRC).replace(".parquet", "bio.parquet"))


def main():
    P = pd.read_parquet(SRC); n0 = P.shape[1]
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv", usecols=["game_id", "game_date"]); g["game_date"] = pd.to_datetime(g.game_date)
    bio = pd.read_parquet(ROOT / "outputs/player_bio_cur.parquet").set_index("pid")
    d = P[["game_id", "player_id"]].merge(g, on="game_id", how="left")
    b = bio.reindex(d.player_id.to_numpy())
    age = ((d.game_date.reset_index(drop=True) - pd.to_datetime(b.birth.to_numpy())).dt.days / 365.25).to_numpy()
    miss = ~np.isfinite(age)
    P["bio_age"] = np.where(miss, 0.0, age - 30.0); P["bio_age2"] = np.where(miss, 0.0, (age - 30.0) ** 2 / 100.0)
    P["bio_frn"] = np.where(miss, 0.0, b.foreign.to_numpy().astype(float))
    ht = b.height.to_numpy().astype(float); P["bio_ht"] = np.where(np.isfinite(ht) & (ht > 100), (ht - 180.0) / 10.0, 0.0)
    P["bio_miss"] = miss.astype(float)
    P.to_parquet(OUT, index=False)
    print(f"행 {len(P):,} · 열 {n0} → {P.shape[1]} · 바이오 결측 {miss.mean() * 100:.1f}% · 나이 평균 {np.nanmean(age):.1f} · 외국인 {P.bio_frn.mean() * 100:.1f}% → {OUT.name}")


if __name__ == "__main__":
    main()
