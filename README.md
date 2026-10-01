# EXTRA — Phase-Aware Tracking for Starting-Eleven Recommendation in Soccer

EXTRA (**E**vent**X**I with **TRA**cking) adds phase-organised optical tracking to
EventXI, an event-based model that recommends a starting eleven and predicts the
non-penalty expected-goal (npxG) difference from the two starting elevens.
This repository holds the code behind the SSAC 2027 abstract.

**No data are included.** The K League event, lineup and tracking data are licensed
from the league's official data provider and cannot be redistributed. The repository
contains no raw, intermediate or final data and no model checkpoints; `.gitignore`
and `.githooks/pre-commit` block them from being committed
(`git config core.hooksPath .githooks`).

## What EXTRA adds to EventXI

1. **Phase labels** (`tracking/phase_tables.py`). From 30 Hz tracking, every 0.1 s of
   live play with a clear team in possession is labelled with one of six phases:
   build-up (own possession, ball in the first 40% of the pitch), settled attack,
   attacking and defensive transition (5 s after an in-play turnover), pressing
   (two or more players within 10 m of the opponent's ball, one closing at over
   3 m/s) and settled defence.
2. **Phase profiles** (`tracking/trk_channels.py`). Per player and match, each phase
   over six pitch-depth bands: time spent, mean speed, high-intensity share
   (> 5.5 m/s) and pressing engagement.
3. **Model input** (`experiments/player_encoder_cnn.py`, `TRK=1 TRKMODE=phase`).
   The six profiles become six new rows of EventXI's player-match tensor; matches
   without tracking enter as zero rows with a missing-tracking indicator.
4. **Training window** (`experiments/player_encoder_set.py`). `TRAINFROM=2024`
   restricts training matches to seasons with tracking and `HISTFROM=2024` restricts
   player histories to the same seasons. Both default to off, which reproduces EventXI.

## Layout

```
experiments/  gnn/  scripts/  vaep/   EventXI (see README_EventXI.md) + the options above
tracking/     phase labels, tracking channels, descriptive statistics, training runs, evaluation
case_study/   player scores, outcome-blind lineup solver, case scan, Figure 1
```

Comments and docstrings are partly in Korean.

## Reproducing the abstract

Run from the repository root. GPU ids in the shell scripts are set with `GPU_A`, `GPU_B`, `GPU_C`. Paths to the licensed inputs are set with
`TRACKING_DIR` (Bepro `tracking.parquet` / `lineup.json` per match) and
`MATCH_INFO_DIR` (`info.json` with pitch size).

```bash
bash scripts/01_features.sh                 # EventXI: events -> SPADL/VAEP -> player-match features
python tracking/phase_tables.py             # phase labels and player/team phase tables
python tracking/trk_channels.py             # phase profiles -> outputs/trk_channels.parquet
python tracking/phase_descriptives.py       # phase shares, reliability, between-phase rank correlations

bash tracking/run_eventxi_arms.sh           # EventXI vs EXTRA trained on 2021-24 (comparison)
python tracking/eval_arms.py
bash tracking/run_eventxi_2024only.sh       # EventXI vs EXTRA, train 2024 / test 2025, 5 seeds, stage 1 -> 2
python tracking/eval_2024only.py            # Table 1 (TRAINFROM=2024 HISTFROM=2024); also reads the 2021-24 runs

python case_study/prepare_lineup_cases.py   # announced squads, past-only positions
python case_study/score_lineup.py event --h24
python case_study/score_lineup.py phase --h24
SUF=_h24 python case_study/scan_coach_cases.py 0 1
SUF=_h24 python case_study/coach_case_figure.py --pin 165309:4640   # Figure 1
```

| Abstract | Output |
|---|---|
| Table 1 (2025 held-out R²: linear 0.144, EventXI 0.162, EXTRA 0.177) | `tracking/eval_2024only.py` |
| Between-phase rank correlations within position | `tracking/phase_descriptives.py` |
| Figure 1 (Jeonbuk vs Gwangju, 23 Feb 2025) | `case_study/coach_case_figure.py` |
