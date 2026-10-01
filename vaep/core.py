"""
VAEP (Valuing Actions by Estimating Probabilities) 파이프라인 — K-League / J-League

[논문] Decroos et al., "Actions Speak Louder than Goals" (KDD 2019)
      https://dl.acm.org/doi/10.1145/3292500.3330758

[개요]
  축구의 모든 액션(패스·슈팅·드리블 등)이 '10초 이내 득점/실점' 확률을
  얼마나 바꿨는지를 XGBoost로 학습해, 선수별 기여도로 환산한다.

[K-League 파이프라인]
  Bepro raw JSON
    → BeproLoader (lib/datatools/loaders/bepro.py)
    → SPADL 변환 (액션 표준 포맷)
    → XGBoost 특징/레이블 계산
    → Leave-One-Season-Out OOF 학습
    → VAEP 값 (offensive / defensive / total)

[J-League 파이프라인]
  StatsBomb flat JSON (Statsbomb_J1_League.json, sb_matches.json)
    → flat → nested 복원 (StatsBomb 이벤트 형식)
    → SPADL 변환 (lib/datatools/representation/spadl/statsbomb.py)
    → K-League 전체 학습 모델로 VAEP 예측 (크로스-리그 적용)
    → VAEP 값
"""

import sys
import os, json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# lib/ 경로를 sys.path에 추가해 내부 라이브러리를 import할 수 있게 한다.
# 이 파일이 어디서 실행되든 lib/을 찾을 수 있도록 __file__ 기준으로 계산한다.
# ---------------------------------------------------------------------------
_LIB = Path(__file__).parent / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

# 로더와 SPADL 변환 함수 (lib/datatools)
from datatools.loaders.bepro import BeproLoader
from datatools.representation.spadl.bepro import convert_to_actions as bepro_convert_to_actions
from datatools.representation.spadl.statsbomb import convert_to_actions as sb_convert_to_actions
import datatools.representation.spadl as spadl
import datatools.vaep.spadl.features as fs   # VAEP 특징 함수 모음
import datatools.vaep.spadl.labels as lab    # VAEP 레이블 함수 모음

# ---------------------------------------------------------------------------
# VAEP 특징 함수 목록 (socceraction 논문 기본 16개)
#
# 각 함수는 "game state" (현재 + 이전 N개 액션 묶음) → 특징 DataFrame을 반환한다.
# pd.concat으로 수평 결합하면 모델 입력 행렬 X가 된다.
#
# 예: fs.startlocation → (x, y) 좌표 2열
#     fs.actiontype_onehot → 액션 종류를 one-hot 벡터로 변환
#     fs.goalscore → 현재 스코어 상황 (2열: 공격팀, 수비팀 득점 수)
# ---------------------------------------------------------------------------
XFNS = [
    fs.actiontype,        # 액션 종류 (정수 코드)
    fs.actiontype_onehot, # 액션 종류 (one-hot 벡터)
    fs.bodypart,          # 신체 부위 (발/머리 등, 정수 코드)
    fs.bodypart_onehot,   # 신체 부위 (one-hot)
    fs.result,            # 액션 성공/실패 (정수 코드)
    fs.result_onehot,     # 액션 결과 (one-hot)
    fs.goalscore,         # 현재 스코어 (공격팀, 수비팀)
    fs.startlocation,     # 시작 좌표 (x, y)
    fs.endlocation,       # 종료 좌표 (x, y)
    fs.movement,          # 이동 벡터 (dx, dy)
    fs.space_delta,       # 공간 점유 변화량 (위협 공간 delta)
    fs.startpolar,        # 시작 위치 극좌표 (골대까지 거리, 각도)
    fs.endpolar,          # 종료 위치 극좌표
    fs.team,              # 같은 팀이 연속 액션인지 여부
    fs.time_delta,        # 이전 액션과의 시간 간격 (초)
    fs.speed,             # 공의 이동 속도 (m/s 추정)
]

# 게임 상태 구성 시 참조할 이전 액션 수.
# 현재 액션 1개 + 직전 3개 = 총 4개 액션의 문맥으로 특징을 만든다.
NB_PREV_ACTIONS = 3


# ===========================================================================
# 데이터 로딩
# ===========================================================================

def build_loader(raw_data_dir: Path) -> BeproLoader:
    """
    Bepro raw JSON 데이터를 읽는 로더 객체를 생성한다.

    Parameters
    ----------
    raw_data_dir : Path
        raw-data/ 폴더 경로.
        내부 구조: {competition}/{season}/match/{game_id}/event_data.json 등

    Returns
    -------
    BeproLoader
        로컬 파일 시스템 기반 데이터 로더.
    """
    return BeproLoader(getter="local", root=raw_data_dir)


def load_all_games(
    loader: BeproLoader,
    competition_names: list[str] = ("KLEAGUE1", "KLEAGUE2"),
) -> pd.DataFrame:
    """
    지정한 리그의 모든 시즌 경기 메타데이터를 로드한다.

    BeproLoader는 competition/season 계층 구조로 데이터를 관리한다.
    1. 전체 대회(competition) 목록 조회
    2. 원하는 대회 이름으로 필터링 (KLEAGUE1, KLEAGUE2)
    3. 각 (competition_id, season_id) 조합의 경기 목록 수집
    4. games DataFrame에 'season' (연도 문자열)과 'competition_name' 컬럼 추가

    Parameters
    ----------
    loader : BeproLoader
    competition_names : list[str]
        불러올 대회 이름 목록. 기본값은 K리그 1부/2부 리그.

    Returns
    -------
    pd.DataFrame
        컬럼: game_id, home_team_id, away_team_id, competition_id,
               season_id, season (연도), competition_name
    """
    # 전체 대회 목록에서 원하는 대회만 추출
    comps = loader.competitions()
    comps = comps[comps["competition_name"].isin(competition_names)]

    # 각 대회×시즌 조합의 경기 목록을 수직 결합
    games = pd.concat(
        [loader.games(r.competition_id, r.season_id) for r in comps.itertuples()],
        ignore_index=True,
    )

    # (competition_id, season_id) → season 연도 문자열 매핑
    # 예: (587, 10) → "2024"
    season_map = comps.set_index(["competition_id", "season_id"])["season_name"].to_dict()
    comp_name_map = comps.set_index("competition_id")["competition_name"].to_dict()

    games["season"] = games.apply(
        lambda r: season_map.get((r.competition_id, r.season_id), "?"), axis=1
    )
    games["competition_name"] = games["competition_id"].map(comp_name_map)
    return games


# ===========================================================================
# SPADL 변환
# ===========================================================================

def convert_games_to_spadl(
    loader: BeproLoader,
    games: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    모든 K-League 경기의 Bepro 이벤트 데이터를 SPADL 형식으로 변환한다.

    SPADL(Soccer Player Action Description Language)은 축구 이벤트 데이터를
    액션 단위로 표준화한 포맷이다. 각 액션은 다음을 포함한다:
      - 액션 종류 (패스, 슈팅, 드리블 등 20종)
      - 시작/종료 좌표 (0~105m × 0~68m 표준 피치)
      - 신체 부위 (발, 머리, 기타)
      - 결과 (성공/실패)
      - 소요 시간

    변환에 실패한 경기(데이터 불완전 등)는 에러 목록에 기록하고,
    반환 시 해당 경기를 games DataFrame에서 제거한다.

    Parameters
    ----------
    loader : BeproLoader
    games : pd.DataFrame
        load_all_games()의 반환값.
    verbose : bool
        진행 상황 출력 여부.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - games: 변환 성공한 경기만 남긴 DataFrame
        - actions_dict: {game_id: SPADL actions DataFrame}
    """
    actions_dict: dict[int, pd.DataFrame] = {}
    errors: list[int] = []

    iterator = tqdm(list(games.itertuples()), desc="SPADL 변환") if verbose else games.itertuples()

    for game in iterator:
        try:
            # Bepro 이벤트 데이터와 시퀀스 데이터 로드
            events = loader.events(game.game_id)
            seqs = loader.sequences(game.game_id)

            # Bepro 이벤트 → SPADL 액션
            # xy_fidelity_version=2: 소수점 좌표 (고정밀)
            # shot_fidelity_version=2: 슈팅 세부 정보 포함
            acts = bepro_convert_to_actions(
                events, seqs,
                home_team_id=game.home_team_id,
                xy_fidelity_version=2,
                shot_fidelity_version=2,
            )
            actions_dict[game.game_id] = acts

        except Exception as e:
            errors.append(game.game_id)
            if verbose:
                print(f"  ✗ game {game.game_id}: {e}")

    if errors and verbose:
        print(f"변환 실패: {len(errors)}경기")

    # 실패한 경기를 제외하고 반환
    return games[~games.game_id.isin(errors)].reset_index(drop=True), actions_dict


def load_players_and_teams(loader: BeproLoader, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    모든 경기의 선수/팀 정보를 로드하고 중복을 제거한다.

    각 경기 JSON에 선수·팀 정보가 포함되어 있어, 경기마다 개별 요청 후
    전체를 수직 결합한다. 팀은 team_id 기준 중복 제거, 선수는 중복 유지
    (같은 선수가 여러 경기에 출전하는 기록 보존).

    Parameters
    ----------
    loader : BeproLoader
    games : pd.DataFrame
        convert_games_to_spadl() 이후 성공한 경기 목록.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - teams_df: 팀 정보 (team_id, team_name 등)
        - players_df: 선수 정보 (player_id, player_name, team_id 등)
    """
    teams, players = [], []
    for game in tqdm(list(games.itertuples()), desc="선수/팀 로드"):
        try:
            teams.append(loader.teams(game.game_id))
            players.append(loader.players(game.game_id))
        except Exception:
            pass  # 일부 경기 데이터 불완전 시 건너뜀

    teams_df = pd.concat(teams).drop_duplicates(subset="team_id").reset_index(drop=True)
    players_df = pd.concat(players).reset_index(drop=True)
    return teams_df, players_df


# ===========================================================================
# VAEP 특징(Feature) / 레이블(Label) 계산
# ===========================================================================

def _add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """
    SPADL 액션의 정수 코드 컬럼을 이름 컬럼으로 보강한다.

    XGBoost의 enable_categorical 옵션을 위해 문자열/카테고리 타입이 필요하다.
    예: type_id=1 → type_name="pass",  bodypart_id=0 → bodypart_name="foot"

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL 형식의 액션 DataFrame.

    Returns
    -------
    pd.DataFrame
        type_name, bodypart_name, result_name 컬럼이 추가된 DataFrame.
    """
    return spadl.add_names(actions)


def compute_features(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """
    한 경기의 SPADL 액션으로부터 VAEP 모델 입력 특징 행렬을 계산한다.

    [처리 흐름]
    1. add_names: 정수 코드 → 이름 변환
    2. gamestates: 현재 액션 + 이전 N개 액션을 묶어 "game state" 구성
       - 행 수는 동일하나, 각 행에 이전 액션 정보가 추가 컬럼으로 붙는다
    3. play_left_to_right: 홈/어웨이 관계없이 공격 방향을 왼→오른쪽으로 통일
       - 좌표 뒤집기로 모델이 방향에 무관하게 학습 가능
    4. XFNS의 각 함수를 호출해 특징 DataFrame들을 수평 결합

    Parameters
    ----------
    actions : pd.DataFrame
        한 경기의 SPADL 액션.
    home_team_id : int
        홈 팀 ID (방향 통일에 사용).

    Returns
    -------
    pd.DataFrame
        행 수 = 경기 내 액션 수, 열 수 = 모든 특징 함수의 출력 열 합계.
    """
    named = _add_names(actions)

    # game state: 현재 액션 + 직전 NB_PREV_ACTIONS개 액션의 문맥 정보
    gs = fs.gamestates(named, nb_prev_actions=NB_PREV_ACTIONS)

    # 공격 방향을 항상 왼→오른으로 표준화 (좌표 대칭 변환)
    gs = fs.play_left_to_right(gs, home_team_id=home_team_id)

    # 각 특징 함수를 열 방향으로 결합하여 최종 특징 행렬 생성
    return pd.concat([fn(gs) for fn in XFNS], axis=1)


def compute_labels(actions: pd.DataFrame) -> pd.DataFrame:
    """
    한 경기의 SPADL 액션으로부터 VAEP 모델 학습 레이블을 계산한다.

    [레이블 정의]
    - scores_by_seconds:  해당 액션 이후 10초 이내 해당 팀이 득점하면 1, 아니면 0
    - concedes_by_seconds: 해당 액션 이후 10초 이내 해당 팀이 실점하면 1, 아니면 0

    이 두 레이블을 XGBoost 이진 분류로 학습하면,
    각 액션에서 '득점 확률'과 '실점 확률'을 예측할 수 있다.

    Parameters
    ----------
    actions : pd.DataFrame
        한 경기의 SPADL 액션 (add_names 전).

    Returns
    -------
    pd.DataFrame
        컬럼: scores_by_seconds, concedes_by_seconds
        값: 0 또는 1 (이진 레이블)
    """
    named = _add_names(actions)
    scores = lab.scores_by_seconds(named)    # 득점 레이블
    concedes = lab.concedes_by_seconds(named)  # 실점 레이블
    return pd.concat([scores, concedes], axis=1)


# ===========================================================================
# OOF (Out-of-Fold) VAEP 학습 및 예측
# ===========================================================================

def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    VAEP 확률 예측용 XGBoost 이진 분류 모델을 학습한다.

    [하이퍼파라미터 선택 근거]
    - n_estimators=100: 과적합 방지를 위해 보수적으로 설정
    - max_depth=4: 중간 복잡도. 너무 깊으면 노이즈 과적합
    - learning_rate=0.1: 표준적인 학습률
    - enable_categorical=True: 특징 행렬의 pandas Categorical 타입을 자동 처리
      (액션 종류 등 명목형 변수를 one-hot 없이 직접 XGBoost가 처리)

    Parameters
    ----------
    X_train : pd.DataFrame
        compute_features()의 출력. 학습 특징 행렬.
    y_train : pd.Series
        compute_labels()의 단일 컬럼. 이진 레이블 (0/1).

    Returns
    -------
    xgb.XGBClassifier
        학습 완료된 모델. predict_proba()로 확률 추출 가능.
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        enable_categorical=True,  # pandas Categorical 컬럼 직접 지원
        random_state=42,
        n_jobs=-1,  # 모든 CPU 코어 사용
    )
    model.fit(X_train, y_train)
    return model


NFOLD_FIRST = 4          # 최초 시즌의 시즌 내부 폴드 수


def _fit_fold(feats, labels, action_rows, train_ids, test_ids, tag, all_preds, fold_metrics):
    """한 폴드: 학습 경기로 득점·실점 모델을 맞추고 테스트 경기의 확률을 낸다."""
    X_train = pd.concat([feats[g] for g in train_ids])
    y_s = pd.concat([labels[g]["scores_by_seconds"] for g in train_ids])
    y_c = pd.concat([labels[g]["concedes_by_seconds"] for g in train_ids])
    X_test = pd.concat([feats[g] for g in test_ids])
    y_s_te = pd.concat([labels[g]["scores_by_seconds"] for g in test_ids])
    y_c_te = pd.concat([labels[g]["concedes_by_seconds"] for g in test_ids])
    print(f"\n[fold {tag}] train={len(train_ids)}경기, test={len(test_ids)}경기", flush=True)
    m_s = _train_xgb(X_train, y_s); m_c = _train_xgb(X_train, y_c)
    p_s = m_s.predict_proba(X_test)[:, 1]; p_c = m_c.predict_proba(X_test)[:, 1]
    auc_s = roc_auc_score(y_s_te, p_s); auc_c = roc_auc_score(y_c_te, p_c)
    print(f"  AUC scores: {auc_s:.4f}, AUC concedes: {auc_c:.4f}", flush=True)
    fold_metrics.append({"season": tag, "auc_scores": auc_s, "auc_concedes": auc_c, "n_test": len(test_ids)})
    ta = pd.concat([action_rows[g] for g in test_ids]).reset_index(drop=True)
    ta["p_scores"] = p_s; ta["p_concedes"] = p_c
    all_preds.append(ta)


def run_oof_vaep(
    games: pd.DataFrame,
    actions_dict: dict,
    output_dir: Path,
    seasons: Optional[list] = None,
) -> pd.DataFrame:
    """
    Leave-One-Season-Out OOF 방식으로 전 시즌 unbiased VAEP 값을 계산한다.

    [OOF (Out-of-Fold)가 필요한 이유]
    일반적으로 같은 시즌 데이터로 학습하고 같은 시즌을 예측하면
    모델이 자신이 학습한 패턴을 그대로 반영해 '낙관적으로 편향'된다.
    OOF는 각 시즌을 차례로 테스트셋으로 설정하고, 나머지 시즌으로만
    학습해 해당 시즌을 예측하므로, 전 시즌에 걸쳐 공정한 평가가 가능하다.

    [각 fold 처리 흐름]
    1. 해당 시즌 = 테스트셋, 나머지 시즌 = 학습셋 분할
    2. 학습셋으로 득점 모델, 실점 모델 각각 XGBoost 학습
    3. 테스트셋에 두 모델 적용 → 액션별 P(score), P(concede) 예측
    4. AUC 기록 (모델 품질 검증)
    5. 예측값 저장 → 다음 단계에서 VAEP 값으로 변환

    [전체 결과]
    모든 fold의 예측을 합치면 전 시즌에 걸친 unbiased VAEP 확보.
    최종적으로 vaep_oof.parquet에 저장.

    Parameters
    ----------
    games : pd.DataFrame
        game_id, season, home_team_id 컬럼 포함.
    actions_dict : dict
        {game_id: SPADL DataFrame} — convert_games_to_spadl() 반환값.
    output_dir : Path
        결과 파일 저장 폴더.
    seasons : list, optional
        fold로 사용할 시즌 목록. None이면 games에서 자동 추출.

    Returns
    -------
    pd.DataFrame
        컬럼: game_id, action_id, player_id, team_id, ...
              p_scores, p_concedes, offensive_value, defensive_value, vaep_value
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if seasons is None:
        seasons = sorted(games["season"].unique())

    # -----------------------------------------------------------------------
    # 전체 특징/레이블 사전 계산 (OOF 루프 전에 한 번만)
    # 각 경기의 특징·레이블을 game_id를 키로 딕셔너리에 저장해두면
    # fold마다 재계산할 필요 없이 pd.concat으로 빠르게 분할할 수 있다.
    # -----------------------------------------------------------------------
    print("Feature/label 계산 중...")
    feats: dict[int, pd.DataFrame] = {}      # {game_id: 특징 DataFrame}
    labels: dict[int, pd.DataFrame] = {}     # {game_id: 레이블 DataFrame}
    action_rows: dict[int, pd.DataFrame] = {} # {game_id: 원시 액션 메타 DataFrame}

    for game in tqdm(list(games.itertuples()), desc="Feature 계산"):
        gid = game.game_id
        if gid not in actions_dict:
            continue
        acts = actions_dict[gid]
        try:
            feats[gid] = compute_features(acts, game.home_team_id)
            labels[gid] = compute_labels(acts)
            # VAEP 값 할당에 필요한 메타 컬럼만 저장 (메모리 절약)
            action_rows[gid] = acts[
                ["game_id", "action_id", "period_id", "time_seconds",
                 "team_id", "player_id", "type_id", "result_id"]
            ].copy()
        except Exception as e:
            print(f"  ✗ feature {gid}: {e}")

    # -----------------------------------------------------------------------
    # OOF 루프: 시즌별로 학습/테스트 분할 반복
    # -----------------------------------------------------------------------
    all_preds: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []

    # FORWARD=1 (기본): 액션 가치도 **그 시즌 이전 데이터로만** 학습한다.
    # 종전에는 Leave-One-Season-Out 이라 2024 의 VAEP 가 2025·2026 을 보고 만들어졌다 —
    # 라벨이 아니라 입력 쪽 누수지만 규약 위반이다. 가장 이른 시즌은 앞선 시즌이 없으므로
    # 그 시즌 안에서만 경기 단위 K-폴드로 낸다(미래는 여전히 안 본다).
    FORWARD = os.environ.get("VAEP_FORWARD", "1") == "1"
    print(f"학습 분할: {'전방 전용 (시즌 s 는 s 이전만)' if FORWARD else 'Leave-One-Season-Out'}", flush=True)

    for season in seasons:
        test_ids = set(games[games["season"] == season]["game_id"]) & set(feats.keys())
        if FORWARD:
            train_ids = set(games[games["season"] < season]["game_id"]) & set(feats.keys())
        else:
            train_ids = set(games[games["season"] != season]["game_id"]) & set(feats.keys())

        if not test_ids:
            print(f"  [skip] season {season}: test=0"); continue

        if FORWARD and not train_ids:
            # 최초 시즌: 시즌 내부 경기 단위 폴드 (앞선 시즌이 없다)
            print(f"\n[fold {season}] 앞선 시즌 없음 → 시즌 내부 {NFOLD_FIRST}폴드", flush=True)
            gl = sorted(test_ids); rng = np.random.default_rng(0); rng.shuffle(gl)
            for k in range(NFOLD_FIRST):
                te = set(gl[k::NFOLD_FIRST]); tr = set(gl) - te
                if not te or not tr: continue
                _fit_fold(feats, labels, action_rows, tr, te, f"{season}-{k}", all_preds, fold_metrics)
            continue
        if not train_ids:
            print(f"  [skip] season {season}: train=0"); continue

        # 학습/테스트 특징·레이블 행렬 결합
        _fit_fold(feats, labels, action_rows, train_ids, test_ids, season, all_preds, fold_metrics)

    # -----------------------------------------------------------------------
    # 모든 fold의 예측을 합쳐 VAEP 값 계산 및 저장
    # -----------------------------------------------------------------------
    print("\nVAEP 값 계산 중...")
    combined = pd.concat(all_preds, ignore_index=True)
    combined = _compute_vaep_values(combined)

    combined.to_parquet(output_dir / "vaep_oof.parquet", index=False)
    with open(output_dir / "vaep_oof_metrics.json", "w") as f:
        metrics = {
            "folds": fold_metrics,
            "mean_auc_scores": np.mean([m["auc_scores"] for m in fold_metrics]),
            "mean_auc_concedes": np.mean([m["auc_concedes"] for m in fold_metrics]),
        }
        json.dump(metrics, f, indent=2)

    print(f"\n✓ 완료: {len(combined):,}개 액션 VAEP 계산됨")
    print(f"  Mean AUC scores:   {metrics['mean_auc_scores']:.4f}")
    print(f"  Mean AUC concedes: {metrics['mean_auc_concedes']:.4f}")
    return combined


_SHOT_TYPES = {11, 12, 13}      # SPADL: shot, shot_penalty, shot_freekick
_RESULT_OWNGOAL = 3
# 데드볼 재개 — 공이 경기장을 벗어났거나 파울로 멈춘 뒤 다시 시작하는 액션.
#   2 throw_in / 3 fk_crossed / 4 fk_short / 5 corner_crossed
#   6 corner_short / 12 shot_penalty / 13 shot_freekick / 22 goalkick
# 이 시점의 게임상태는 직전 인플레이 상태와 연속이 아니다. 그렇다고 값을 0으로
# 두면 '잘 올린 코너'의 가치까지 사라지므로, 기준을 *그 데드볼 상황 자체의
# 기본 가치*로 잡는다: (타입 × 시작 존)별 평균 확률. 이러면
#   · 데드볼을 수행하는 선수는 자기가 만들지 않은 상황(상대 공격이 걷어내진 것)의
#     가치를 받지 않고,
#   · 배급의 질(위험지역으로 잘 올렸는가)만 자기 값이 된다.
_RESTART_TYPES = {2, 3, 4, 5, 6, 12, 13, 22}
_RESTART_ZX, _RESTART_ZY = 6, 3      # 기준선 격자 (가로 6 × 세로 3)


def _restart_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """데드볼 재개 액션의 기준 확률 — (타입 × 시작 존)별 평균.

    직전 인플레이 액션과 무관하게 정해지므로, 상대의 위험한 공격이 데드볼로
    해소된 가치가 재개 수행자에게 잘못 귀속되는 것을 막는다.
    """
    m = df["type_id"].isin(_RESTART_TYPES)
    out = pd.DataFrame(index=df.index, columns=["b_scores", "b_concedes"],
                       dtype=float)
    if not m.any():
        return out
    if "start_x" not in df.columns or "start_y" not in df.columns:
        # 좌표가 없으면 타입만으로 기준선을 잡는다(정밀도는 낮지만 안전)
        key = df.loc[m, "type_id"].astype(int)
        for col, src in (("b_scores", "p_scores"), ("b_concedes", "p_concedes")):
            out.loc[m, col] = df.loc[m, src].groupby(key).transform("mean").values
        return out
    sx = df.loc[m, "start_x"].fillna(105.0 / 2)
    sy = df.loc[m, "start_y"].fillna(68.0 / 2)
    zx = np.clip((sx / 105.0 * _RESTART_ZX).astype(int), 0, _RESTART_ZX - 1)
    zy = np.clip((sy / 68.0 * _RESTART_ZY).astype(int), 0, _RESTART_ZY - 1)
    key = df.loc[m, "type_id"].astype(int) * 100 + zx * 10 + zy
    for col, src in (("b_scores", "p_scores"), ("b_concedes", "p_concedes")):
        out.loc[m, col] = df.loc[m, src].groupby(key).transform("mean").values
    return out


def _compute_vaep_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    예측된 확률(p_scores, p_concedes)을 VAEP 값으로 변환한다.

    [socceraction 논문 공식]
    액션 a에 의한 게임 상태 S_{a-1} → S_a 변화에 대해:

      V_off(a) = P_score(S_a)   − P_score(S_{a-1})
                 ← 득점 확률이 얼마나 높아졌는가

      V_def(a) = P_concede(S_{a-1}) − P_concede(S_a)
                 ← 실점 확률이 얼마나 낮아졌는가

      V(a)     = V_off(a) + V_def(a)
                 ← 공격 기여 + 수비 기여의 합

    예시:
      - 골 직전 슈팅: 득점 확률 0.01→0.60 증가 → V_off = +0.59 (매우 높음)
      - 자진 실수 패스: 실점 확률 0.01→0.15 증가 → V_def = -0.14 (부정적)
      - 평범한 횡패스: 확률 변화 미미 → V ≈ 0

    [구현 주의사항]
    경기(game_id)별로 독립적으로 shift() 처리해야 한다.
    경기 경계를 무시하고 전체 DataFrame을 shift하면,
    경기 첫 액션의 prev가 이전 경기 마지막 액션이 되는 오류 발생.

    Parameters
    ----------
    df : pd.DataFrame
        p_scores, p_concedes 컬럼 포함. 여러 경기가 혼합된 상태.

    Returns
    -------
    pd.DataFrame
        offensive_value, defensive_value, vaep_value 컬럼 추가.
    """
    baseline = _restart_baseline(df)
    parts = []
    for gid, gdf in df.groupby("game_id"):
        base = baseline.loc[gdf.index]
        # 경기 내에서 시간 순서 정렬 (전/후반, 시간 기준)
        _order = gdf.sort_values(["period_id", "time_seconds"]).index
        base = base.loc[_order].reset_index(drop=True)
        gdf = gdf.loc[_order].reset_index(drop=True)

        # 각 액션의 '이전' 확률: shift(1)으로 한 행 아래로 밀기
        # 경기 첫 번째 액션은 이전이 없으므로 0으로 채움 (킥오프 상태 = 확률 0)
        prev_p_scores = gdf["p_scores"].shift(1).fillna(0)
        prev_p_concedes = gdf["p_concedes"].shift(1).fillna(0)

        # [possession 전환 보정 — socceraction 공식]
        # 직전 액션이 다른 팀이면, 이전 팀의 P(scores)=현재 팀의 P(concedes)
        # (같은 게임상태, 반대 관점)이므로 이전 확률을 swap해야 한다.
        # 이를 빠뜨리면 공을 탈취하는 수비액션(태클·인터셉트·클리어)이 항상
        # 소유권 전환점에 놓여 defensive_value가 음수로 잘못 계산된다.
        same_team = (gdf["team_id"] == gdf["team_id"].shift(1)).fillna(False)
        prev_off = prev_p_scores.where(same_team, prev_p_concedes)
        prev_def = prev_p_concedes.where(same_team, prev_p_scores)

        # [게임상태 단절 보정]
        # 골·피리어드 경계는 새 게임상태의 시작이므로 이전 상태를 물려주면 안 된다.
        # 빠뜨리면 골 직후 킥오프에서 prev_def = prev_p_scores ≈ 1.0 이 되어
        # **실점한 팀이 골마다 defensive_value +1.0 을 받는다**. 실측상 전체
        # defensive_value 총합의 87%가 골 직후 액션에서 나왔고, 그 결과 승리팀의
        # offensive_value 를 패배팀의 defensive_value 가 정확히 상쇄해 팀 총 VAEP
        # 차이가 득실차와 무상관(r=+0.07)이 되었다.
        is_goal = (gdf["type_id"].isin(_SHOT_TYPES) & (gdf["result_id"] == 1)) \
            | (gdf["result_id"] == _RESULT_OWNGOAL)
        # 골·피리어드 경계: 완전 단절 → 값 0
        hard = (is_goal.shift(1).fillna(False)
                | (gdf["period_id"] != gdf["period_id"].shift(1)).fillna(True))
        prev_off = prev_off.where(~hard, gdf["p_scores"])
        prev_def = prev_def.where(~hard, gdf["p_concedes"])
        # 데드볼 재개: 직전 인플레이 대신 그 상황의 기본 가치를 기준으로
        rs = gdf["type_id"].isin(_RESTART_TYPES)
        prev_off = prev_off.where(~rs, base["b_scores"])
        prev_def = prev_def.where(~rs, base["b_concedes"])

        gdf["offensive_value"] = gdf["p_scores"] - prev_off
        gdf["defensive_value"] = prev_def - gdf["p_concedes"]
        gdf["vaep_value"] = gdf["offensive_value"] + gdf["defensive_value"]
        parts.append(gdf)

    return pd.concat(parts, ignore_index=True)


# ===========================================================================
# J-League 데이터 로딩 및 SPADL 변환
# ===========================================================================

# StatsBomb flat JSON에서 extra 딕셔너리를 만들 때 처리할 액션 접두사.
# 이 접두사로 시작하는 dot-notation 컬럼들이 extra 딕셔너리에 들어간다.
# 예: "pass.outcome.name" → extra["pass"]["outcome"]["name"]
_SB_ACTION_PREFIXES = {
    "pass", "carry", "dribble", "duel", "clearance", "shot",
    "goalkeeper", "foul_committed", "foul_won", "ball_recovery",
    "ball_receipt", "interception", "substitution", "injury_stoppage",
    "50_50", "miscontrol", "block", "bad_behaviour", "player_off", "tactics",
}

# StatsBomb flat 컬럼 → StatsBombLoader.events() 출력 컬럼으로 이름 변경
_SB_RENAME = {
    "id":                        "event_id",
    "match_id":                  "game_id",
    "period":                    "period_id",
    "type.id":                   "type_id",
    "type.name":                 "type_name",
    "possession_team.id":        "possession_team_id",
    "possession_team.name":      "possession_team_name",
    "play_pattern.id":           "play_pattern_id",
    "play_pattern.name":         "play_pattern_name",
    "team.id":                   "team_id",
    "team.name":                 "team_name",
    "player.id":                 "player_id",
    "player.name":               "player_name",
    "position.id":               "position_id",
    "position.name":             "position_name",
}


def load_jleague_games(raw_data_dir: Path) -> pd.DataFrame:
    """
    J-League 2024 경기 메타데이터를 로드한다.

    StatsBomb J1 League 데이터는 sb_matches.json 한 파일에
    모든 경기 정보가 flat 형태(도트 표기법 컬럼)로 저장되어 있다.
    이 함수는 VAEP 파이프라인에서 사용하는 표준 컬럼명으로 정규화해 반환한다.

    Parameters
    ----------
    raw_data_dir : Path
        raw-data/ 경로. 하위에 J-league1/ 폴더가 있어야 한다.

    Returns
    -------
    pd.DataFrame
        컬럼: game_id, home_team_id, away_team_id, season, competition_name
    """
    path = raw_data_dir / "J-league1" / "sb_matches.json"
    matches = pd.read_json(path)
    matches = matches.rename(columns={
        "match_id":                     "game_id",
        "home_team.home_team_id":       "home_team_id",
        "home_team.home_team_name":     "home_team_name",
        "away_team.away_team_id":       "away_team_id",
        "away_team.away_team_name":     "away_team_name",
        "season.season_name":           "season",
        "competition.competition_name": "competition_name",
    })
    matches["season"] = matches["season"].astype(str)
    return matches[["game_id", "home_team_id", "home_team_name",
                    "away_team_id", "away_team_name", "season", "competition_name"]]


def _build_extra(row: pd.Series) -> dict:
    """
    flat dot-notation 컬럼들로부터 StatsBomb 이벤트의 'extra' 딕셔너리를 복원한다.

    StatsBombLoader.events()는 중첩 JSON을 파싱해 액션별 세부 정보를
    'extra' 컬럼(dict)에 저장한다. 하지만 J-League 데이터는 이미 flat하게
    펼쳐진 형태(e.g. "pass.outcome.name", "shot.body_part.name")이므로,
    dot으로 분리해 다시 중첩 딕셔너리로 복원해야 StatsBomb SPADL 변환 함수가
    올바르게 작동한다.

    예시:
      "pass.outcome.name" = "Complete"
        → extra["pass"]["outcome"]["name"] = "Complete"
      "shot.body_part.id" = 40
        → extra["shot"]["body_part"]["id"] = 40

    Parameters
    ----------
    row : pd.Series
        flat events DataFrame의 한 행.

    Returns
    -------
    dict
        StatsBombLoader.events() 출력의 'extra' 컬럼과 동일한 중첩 구조.
    """
    extra: dict = {}
    for col, val in row.items():
        # dot이 없는 컬럼(type.id, period 등)은 extra에 포함하지 않음
        if "." not in str(col):
            continue
        # _SB_ACTION_PREFIXES에 해당하는 컬럼만 extra로 분류
        prefix = col.split(".")[0]
        if prefix not in _SB_ACTION_PREFIXES:
            continue
        # None / NaN / 빈 리스트 스킵
        # pd.read_json은 JSON null을 컬럼 타입에 따라 None(object) 또는 float(NaN)으로 변환함
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        if isinstance(val, list) and len(val) == 0:
            continue
        # dot 경로를 따라 중첩 딕셔너리 생성
        parts = col.split(".")
        d = extra
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = val
    return extra


def _flat_to_sb_events(flat_events: pd.DataFrame, game_id: int) -> pd.DataFrame:
    """
    flat StatsBomb JSON 행들을 StatsBombLoader.events() 출력 형식으로 변환한다.

    StatsBomb SPADL 변환 함수(sb_convert_to_actions)는 StatsBombLoader가
    반환하는 표준 컬럼 구조를 기대한다. 이 함수는 flat J-League 이벤트를
    해당 형식에 맞게 변환한다:
      1. 컬럼명 재명명 (type.id → type_id 등)
      2. timestamp를 timedelta 형식으로 변환
      3. extra 딕셔너리 복원 (_build_extra 호출)

    Parameters
    ----------
    flat_events : pd.DataFrame
        Statsbomb_J1_League.json 전체 로드 결과.
    game_id : int
        처리할 경기 ID.

    Returns
    -------
    pd.DataFrame
        StatsBombLoader.events()와 동일한 컬럼 구조.
    """
    # 해당 경기만 추출 후 컬럼명 정규화
    e = flat_events[flat_events["match_id"] == game_id].copy()
    e = e.rename(columns=_SB_RENAME)

    # timestamp를 timedelta로 변환 (convert_to_actions 내부에서 요구)
    e["timestamp"] = pd.to_timedelta(e["timestamp"])

    # related_events: NaN이면 빈 리스트로
    e["related_events"] = e["related_events"].apply(
        lambda d: d if isinstance(d, list) else []
    )

    # bool 컬럼 정규화
    e["under_pressure"] = e["under_pressure"].fillna(False).astype(bool)
    if "counterpress" not in e.columns:
        e["counterpress"] = False
    e["counterpress"] = e["counterpress"].fillna(False).astype(bool)

    # location 정제: JSON 파싱 결과 ['null'] 등 비정상 좌표 → None으로 치환
    # 정상 좌표는 [float, float] 형태의 길이 2 리스트여야 한다.
    def _clean_loc(v):
        if isinstance(v, list) and len(v) == 2:
            try:
                float(v[0]); float(v[1])
                return v
            except (TypeError, ValueError):
                pass
        return None
    e["location"] = e["location"].apply(_clean_loc)

    # extra 딕셔너리 복원 (행별로 dot-notation 컬럼 → 중첩 딕셔너리)
    e["extra"] = e.apply(_build_extra, axis=1)

    return e


def load_jleague_events(raw_data_dir: Path) -> pd.DataFrame:
    """
    J-League 2024 전 경기 이벤트를 단일 flat JSON 파일에서 로드한다.

    StatsBomb J1 League 이벤트는 경기별 분리 파일이 아닌,
    모든 경기(380경기, ~125만 행)가 하나의 JSON 파일에 저장되어 있다.
    이 DataFrame은 convert_jleague_to_spadl()에서 game_id로 그룹화해 사용된다.

    Parameters
    ----------
    raw_data_dir : Path

    Returns
    -------
    pd.DataFrame
        ~1,257,772 행, match_id 컬럼으로 경기 식별.
    """
    path = raw_data_dir / "J-league1" / "Statsbomb_J1_League.json"
    print(f"J-League 이벤트 로딩 중: {path}")
    # convert_dates=False: 'timestamp' 컬럼이 "00:00:00.000" 형태(경기 내 시간)이므로
    # pandas가 날짜로 자동 파싱하지 않도록 한다.
    return pd.read_json(path, convert_dates=False)


def convert_jleague_to_spadl(
    games: pd.DataFrame,
    all_events: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    J-League 경기 이벤트를 SPADL 형식으로 변환한다.

    [처리 흐름]
    1. 경기 ID 기준으로 flat events를 그룹화 (속도를 위해 미리 그룹화)
    2. 각 경기마다 _flat_to_sb_events()로 StatsBomb 형식 복원
    3. sb_convert_to_actions() (StatsBomb SPADL 변환) 호출
    4. 변환 실패 경기는 games에서 제거

    K-League의 convert_games_to_spadl()과 동일한 반환 형식을 가지므로,
    이후 compute_features / compute_labels 함수를 그대로 재사용할 수 있다.

    Parameters
    ----------
    games : pd.DataFrame
        load_jleague_games()의 반환값.
    all_events : pd.DataFrame
        load_jleague_events()의 반환값 (전 경기 flat 이벤트).
    verbose : bool

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - games: 변환 성공한 경기만 남긴 DataFrame
        - actions_dict: {game_id: SPADL actions DataFrame}
    """
    # 경기별 flat events를 미리 그룹화 (루프 내 필터링 반복 방지)
    grouped = {gid: grp for gid, grp in all_events.groupby("match_id")}

    # StatsBomb SPADL 변환에 필요한 dummy sequences DataFrame 생성
    # (이 함수는 sequences를 인자로 받지만 내부에서 사용하지 않음)
    def _dummy_seqs(game_id: int) -> pd.DataFrame:
        return pd.DataFrame({
            "game_id":   [game_id, game_id],
            "period_id": [1, 2],
            "team_id":   [float("nan"), float("nan")],
            "start_time": [float("nan"), float("nan")],
            "end_time":   [float("nan"), float("nan")],
            "event_ids":  [[], []],
        })

    actions_dict: dict[int, pd.DataFrame] = {}
    errors: list[int] = []
    iterator = tqdm(list(games.itertuples()), desc="J-League SPADL 변환") if verbose else games.itertuples()

    for game in iterator:
        gid = game.game_id
        if gid not in grouped:
            errors.append(gid)
            continue
        try:
            events = _flat_to_sb_events(grouped[gid], gid)
            acts = sb_convert_to_actions(
                events,
                _dummy_seqs(gid),
                home_team_id=game.home_team_id,
                xy_fidelity_version=2,
                shot_fidelity_version=2,
            )
            actions_dict[gid] = acts
        except Exception as e:
            errors.append(gid)
            if verbose:
                print(f"  ✗ game {gid}: {e}")

    if errors and verbose:
        print(f"변환 실패: {len(errors)}경기")
    return games[~games.game_id.isin(errors)].reset_index(drop=True), actions_dict


def run_jleague_vaep(
    kleague_games: pd.DataFrame,
    kleague_actions_dict: dict,
    jleague_games: pd.DataFrame,
    jleague_actions_dict: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """
    K-League 전체 데이터로 모델을 학습하고 J-League에 VAEP를 적용한다.

    [크로스-리그 VAEP 적용 이유]
    J-League는 단일 시즌(2024)이므로 OOF를 수행할 수 없다.
    대신 K-League 전 시즌(2021-2025)을 학습 데이터로 사용하고,
    J-League를 테스트셋으로 취급해 득점/실점 확률을 예측한다.
    이렇게 하면 K-League 기준의 "액션 가치" 척도로 J-League를 평가할 수 있다.

    [주의] 리그 간 플레이 스타일 차이로 인한 분포 이동(distribution shift)이
    존재할 수 있으나, 리그 수준 비교 목적에서는 허용 가능한 가정이다.

    Parameters
    ----------
    kleague_games : pd.DataFrame
        K-League 경기 메타 (game_id, home_team_id, season 포함).
    kleague_actions_dict : dict
        {game_id: SPADL DataFrame} — K-League 전 시즌.
    jleague_games : pd.DataFrame
        J-League 경기 메타.
    jleague_actions_dict : dict
        {game_id: SPADL DataFrame} — J-League 2024.
    output_dir : Path
        결과 저장 경로.

    Returns
    -------
    pd.DataFrame
        J-League 액션별 VAEP 값 (offensive_value, defensive_value, vaep_value 포함).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # K-League 전 시즌 특징/레이블 계산 (학습 데이터)
    # -----------------------------------------------------------------------
    print("K-League feature/label 계산 중 (학습용)...")
    kl_feats, kl_labels = {}, {}
    for game in tqdm(list(kleague_games.itertuples()), desc="K-League Feature"):
        gid = game.game_id
        if gid not in kleague_actions_dict:
            continue
        acts = kleague_actions_dict[gid]
        try:
            kl_feats[gid] = compute_features(acts, game.home_team_id)
            kl_labels[gid] = compute_labels(acts)
        except Exception as e:
            print(f"  ✗ feature {gid}: {e}")

    X_train = pd.concat(kl_feats.values())
    y_scores_train = pd.concat([v["scores_by_seconds"] for v in kl_labels.values()])
    y_concedes_train = pd.concat([v["concedes_by_seconds"] for v in kl_labels.values()])

    print(f"\nK-League 학습 데이터: {len(X_train):,}개 액션")

    # 득점 모델, 실점 모델 각각 학습
    print("XGBoost 학습 중 (득점 모델)...")
    model_scores = _train_xgb(X_train, y_scores_train)
    print("XGBoost 학습 중 (실점 모델)...")
    model_concedes = _train_xgb(X_train, y_concedes_train)

    # -----------------------------------------------------------------------
    # J-League 특징 계산 및 VAEP 예측
    # -----------------------------------------------------------------------
    print("\nJ-League feature 계산 및 VAEP 예측 중...")
    jl_feats, jl_action_rows = {}, {}
    for game in tqdm(list(jleague_games.itertuples()), desc="J-League Feature"):
        gid = game.game_id
        if gid not in jleague_actions_dict:
            continue
        acts = jleague_actions_dict[gid]
        try:
            jl_feats[gid] = compute_features(acts, game.home_team_id)
            jl_action_rows[gid] = acts[
                ["game_id", "action_id", "period_id", "time_seconds",
                 "team_id", "player_id", "type_id", "result_id"]
            ].copy()
        except Exception as e:
            print(f"  ✗ J-League feature {gid}: {e}")

    X_test = pd.concat(jl_feats.values())
    p_scores = model_scores.predict_proba(X_test)[:, 1]
    p_concedes = model_concedes.predict_proba(X_test)[:, 1]

    # 예측값을 액션 메타정보와 결합
    test_acts = pd.concat(jl_action_rows.values()).reset_index(drop=True)
    test_acts["p_scores"] = p_scores
    test_acts["p_concedes"] = p_concedes

    # VAEP 값 계산 (경기별 shift)
    result = _compute_vaep_values(test_acts)
    result.to_parquet(output_dir / "vaep_jleague.parquet", index=False)

    print(f"\n✓ 완료: {len(result):,}개 J-League 액션 VAEP 계산됨")
    print(f"  VAEP 범위: {result['vaep_value'].min():.4f} ~ {result['vaep_value'].max():.4f}")
    return result
