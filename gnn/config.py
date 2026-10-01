"""GNN 파이프라인 전체에서 사용하는 경로 및 하이퍼파라미터 상수 모음.

이 파일 하나를 수정하면 학습·평가·그래프 빌드 전반에 반영된다.
"""

from pathlib import Path

# ── 디렉토리 경로 ──────────────────────────────────────────────────────────────
# __file__ = <저장소>/gnn/config.py 이므로 .parent.parent = 저장소 루트
GNN_ROOT = Path(__file__).resolve().parent.parent

# K-League 원시 JSON 데이터 (KLEAGUE1/{시즌}/match/{game_id}/ 구조)
# 2026 어댑터 트리 (2021~25 는 심볼릭 링크, 2026 은 실물). 구 raw-data 의 상위집합.
RAW_DATA_DIR = GNN_ROOT / "raw-data-2026"

# run_vaep.py 실행 결과물 위치
#   - vaep_oof.parquet : 전 경기 VAEP 값 (행동별 공격/수비 기여도)
#   - games.csv        : 경기 메타데이터 (game_id, home/away_team_id, 결과 등)
VAEP_OUTPUT_DIR = GNN_ROOT / "vaep" / "output"

# build_dataset.py 가 저장하는 HeteroData .pt 파일들의 폴더
# 48D 피처 그래프는 별도 폴더에 저장 (기존 276D graphs/ 폴더 보존)
GRAPHS_DIR = GNN_ROOT / "outputs" / "graphs_48d"

# 학습 중 best val_loss 달성 시 저장되는 모델 가중치 폴더
CHECKPOINTS_DIR = GNN_ROOT / "outputs" / "checkpoints"

# 평가 결과 JSON 파일 폴더 (AUC, LogLoss, Brier, ECE 등)
METRICS_DIR = GNN_ROOT / "outputs" / "metrics"


# ── K-League 리그 식별자 ───────────────────────────────────────────────────────
# Bepro 데이터에서 사용하는 competition_id 숫자값
K1_COMPETITION_ID = 587   # K리그1 (1부)
K2_COMPETITION_ID = 588   # K리그2 (2부)

# 이 파이프라인이 처리할 리그 목록 (J리그 등 외부 데이터 제외)
VALID_COMPETITION_IDS = {K1_COMPETITION_ID, K2_COMPETITION_ID}

# 사용하는 시즌 목록 (LOSO 5-fold에서 각 시즌이 한 번씩 test가 됨)
SEASONS = [2021, 2022, 2023, 2024, 2025]


# ── 그래프 피처 차원 ───────────────────────────────────────────────────────────
# 피치 구역 수 (zones.py와 동기화). ZONES=30이면 균일 6×5 그리드, 기본 12.
import os as _os
NUM_ZONES = 30 if _os.environ.get("ZONES") == "30" else 12

# SPADL 표준에서 정의된 행동 유형 목록 (vaep/lib/.../spadl/config.py 기준)
# type_id 0~22에 각 행동이 대응된다
ACTION_TYPES = [
    "pass",              # 0: 패스
    "cross",             # 1: 크로스
    "throw_in",          # 2: 스로인
    "freekick_crossed",  # 3: 크로스성 프리킥
    "freekick_short",    # 4: 짧은 프리킥
    "corner_crossed",    # 5: 크로스성 코너킥
    "corner_short",      # 6: 짧은 코너킥
    "take_on",           # 7: 드리블 돌파 시도
    "foul",              # 8: 파울
    "tackle",            # 9: 태클
    "interception",      # 10: 인터셉트
    "shot",              # 11: 슈팅
    "shot_penalty",      # 12: 페널티킥
    "shot_freekick",     # 13: 프리킥 슈팅
    "keeper_save",       # 14: 골키퍼 선방
    "keeper_claim",      # 15: 골키퍼 잡기
    "keeper_punch",      # 16: 골키퍼 펀칭
    "keeper_pick_up",    # 17: 골키퍼 픽업
    "clearance",         # 18: 클리어링
    "bad_touch",         # 19: 볼 컨트롤 실패
    "non_action",        # 20: 비행동 (기록용)
    "dribble",           # 21: 드리블
    "goalkick",          # 22: 골킥
]
NUM_ACTION_TYPES = len(ACTION_TYPES)   # 23

# 노드 피처 차원: 아래 행동 그룹 섹션에서 재정의됨 (48D)

# 엣지 피처 차원: 구역별 12칸 (B-scheme, zones.py 참고)
# B-scheme: 출발 선수의 vaep → 출발 구역에, 도착 선수의 vaep → 도착 구역에 각자 누적
EDGE_DIM = NUM_ZONES   # 12


# ── 행동 그룹 (276D → 48D 피처 압축) ─────────────────────────────────────────
# 23개 action type을 4개 의미 그룹으로 합산 → sparsity 감소
GROUP_MAP: dict[int, list[int]] = {
    0: [0, 1, 2, 3, 4, 5, 6, 22],    # 패스/배급: pass,cross,throw_in,freekick_*,corner_*,goalkick
    1: [11, 12, 13],                   # 슈팅: shot,shot_penalty,shot_freekick
    2: [8, 9, 10, 14, 15, 16, 18],    # 수비: foul,tackle,interception,keeper_*,clearance
    3: [7, 19, 21],                    # 볼운반: take_on,bad_touch,dribble
    # 제거: 17(keeper_pick_up, 0%), 20(non_action, 0%) — 데이터에 없음
}
NUM_GROUPS = 4

# 노드 피처 차원: 행동그룹(4) × 구역(12) = 48D  (기존 276D → 48D)
NODE_DIM = NUM_GROUPS * NUM_ZONES   # 4 × 12 = 48


# ── GNN 하이퍼파라미터 ─────────────────────────────────────────────────────────
HIDDEN_CHANNELS = 64   # 노드 임베딩 차원
NUM_LAYERS = 2         # GNN 컨볼루션 레이어 수
NUM_HEADS = 4          # 어텐션 헤드 수
DROPOUT = 0.3          # 드롭아웃 비율

# 옵티마이저
LR = 1e-3              # Adam 학습률
WEIGHT_DECAY = 1e-4    # L2 정규화 계수

# 학습 제어
MAX_EPOCHS = 100       # 최대 에폭 수
PATIENCE = 15          # val_loss 개선 없을 때 조기종료까지의 에폭 수
BATCH_SIZE = 32        # 배치당 그래프 수 (DataLoader에 전달)
