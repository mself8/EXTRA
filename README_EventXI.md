# EventXI — event-only base model

EventXI recommends a starting eleven and a formation for the next match from
event data alone, and tests the recommendation against match outcomes.
This folder holds the EventXI code that EXTRA builds on.

**No data are included.** The event and lineup data are licensed from the
league's official data provider (Bepro) and cannot be redistributed. The
repository contains no raw, intermediate or final data and no model
checkpoints; `.gitignore` and `.githooks/pre-commit` block them from being
committed.

## Layout

```
experiments/   player representation, outcome model, recommender, predictors, evaluation
vaep/          raw events -> SPADL -> VAEP action values
gnn/config.py  paths
scripts/       the order in which the results were produced (01 -> 05)
```

Comments and docstrings are in Korean.

## Pipeline

Run from the repository root. Each step reads the outputs of the previous one
from `outputs/` and `vaep/output/`.

Evaluation is season-held-out and forward-only: the VAEP and xG models for
season *s* are fitted on earlier seasons only (the first season uses
within-season match folds), and `build_xg.py` also writes the non-penalty xG
target `outputs/team_npxg.parquet`.

```bash
bash scripts/01_features.sh   # raw events -> SPADL/VAEP -> player-match features
bash scripts/02_train.sh      # outcome model under both season protocols (one GPU)
bash scripts/03_recommend.sh  # scores -> constrained recommendation -> gap effect, placebo, shuffle null
bash scripts/04_evaluate.sh   # baseline scores, integer-program and HIGFormer baselines,
                              # substitution alignment, recommender ablations
bash scripts/05_predictors.sh # selection (EventXI-Select, DraftRec, per-player classifiers)
                              # and formation prediction (EventXI-Form)
```

| Paper | Script |
|---|---|
| Table 2, outcome R² | `02_train.sh`; baseline scores in `04_evaluate.sh` |
| Table 3, recommendation | `03_recommend.sh`; baselines and substitution alignment in `04_evaluate.sh` |
| Table 4, selection and formation | `05_predictors.sh` |
| Recommender ablations (appendix) | `04_evaluate.sh` |

Auxiliary builders that the scripts do not call:
`build_xt.py` (xT grid read by `build_refined.py`), `vaep_pre.py`,
`build_versaplus.py` (off-ball defensive credit), `build_zonemaps.py`,
`person_bridge_2026.py`, `mk2026.py` (2026 raw-data tree), `manager_table.py`,
`opp_predict_v2.py`, `build_teamstats.py`, `onball_vaep_type.py`.

## Expected raw data

```
raw-data-2026/
├── league.json
├── KLEAGUE1/
│   └── {season}/
│       ├── match.json, team.json, season.json, player/
│       └── match/{game_id}/lineup.json, event_data.json, info.json
└── KLEAGUE2/            same structure
```

## Requirements

Python 3, `torch>=2.0`, `pandas`, `numpy`, `scikit-learn`, `scipy`,
`xgboost`, `matplotlib`, `requests` (see `vaep/requirements.txt`).

## License

Code: MIT. The data are not covered.
