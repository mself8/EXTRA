"""bepro data to SPADL converter."""
from typing import Any, cast, Optional
import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_block,
    _fix_recovery,
    _add_dribbles_after_receive,
    shift_with_edge_fix,
    # _fix_direction_of_play,
)
from .schema import SPADLSchema

from .config import (
    field_length,
    field_width,
    HEIGHT_POST,
    TOUCH_LINE_LENGTH,
    GOAL_LINE_LENGTH,
    LEFT_POST,
    RIGHT_POST,
    CENTER_POST,
    ORIGINAL_LEFT_POST,
    ORIGINAL_RIGHT_POST,
    Eighteen_YARD,
)


def convert_to_actions(
    events: pd.DataFrame, 
    sequences: pd.DataFrame, 
    home_team_id: int, 
	xy_fidelity_version: Optional[int] = None,
	shot_fidelity_version: Optional[int] = None,
) -> DataFrame[SPADLSchema]:
    """
    Convert K-league events to SPADL actions.
    """

    events = events.sort_values(
        ["period_id", "event_time"], 
        kind="mergesort"
    ).reset_index(drop=True) 
    # 분석에 활용하지 않은 이벤트 제거: 결측치 & 중복값
    events = _clean_events(events, remove_event_types=["Substitution", "Control Under Pressure"])
    events = _sort_sequence(events, sequences) 

    events = _convert_pass_locations(events) # 패스의 끝 위치를 추정하여 보간
    events = _convert_shot_locations(events) # 슛의 끝 위치를 추정하여 보간

    # 홈팀과 원정팀의 플레이 방향을 고정 : 홈팀은 아래에서 위쪽으로, 원정팀은 위에서 아래쪽으로 플레이합니다.
    events = _fix_direction_of_play(events, home_team_id)

    events = _fix_offside(events)
    events = _fix_defensive_line_support(events)    # Convert defensive_line_support to tackle or interception

    # 공격 이벤트와 수비 이벤트가 동시에 발생한 경우 이를 분리합니다.
    events = insert_defensive_actions(events, defensive_action="Interception")
    events = insert_defensive_actions(events, defensive_action="Tackle")

    # K-league데이터셋 형태의 aciton을 SPALD형태로 변환
    events["type_name"] = events["event_types"].apply(_get_type_name)

    # 각 액션에 대해 액션 ID, 터치 부위, 결과, 끝 위치를 정의함
    events[["type_id", "bodypart_id", "result_id"]] = events.apply(_parse_event, axis=1, result_type="expand")

    actions = pd.DataFrame()
    actions["game_id"] = events.game_id.astype(int)
    actions["original_event_id"] = events.event_id.astype(object)
    actions["period_id"] = events.period_id.astype(int)

    # convert milliseconds to seconds
    # First half kick-off: 0(ms), second half kick-off: 2,700,000(ms)
    actions["time_seconds"] = (
        events["event_time"] * 0.001 
        - ((events.period_id > 1) * 45 * 60) # convert 45(minutes) to 45*60(seconds)
        - ((events.period_id > 2) * 45 * 60)
        - ((events.period_id > 3) * 15 * 60)
        - ((events.period_id > 4) * 15 * 60)
    )
     
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    # K-league 경기장 형태를 SPADL형태로 변환 : 68x105 -> 105x68로 변환
    actions["start_x"] = events.y
    actions["start_y"] = GOAL_LINE_LENGTH - events.x
    actions["end_x"] = events.end_y
    actions["end_y"] = GOAL_LINE_LENGTH - events.end_x

    actions["type_id"] = events.type_id.astype(int)
    actions["bodypart_id"] = events.bodypart_id.astype(int)
    actions["result_id"] = events.result_id.astype(int)

    actions = (
        actions[actions.type_id != spadlconfig.actiontypes.index("non_action")]
        .sort_values(["period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )

    actions = _fix_tackle_result(actions) # 기록되지 않은 수비이벤트의 결과를 수정함.
    actions = _fix_dribble(actions)  # dribble의 끝 위치를 조정
    actions = _fix_clearances(actions)

    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions) # TODO: _add_dribbles_after_receive가 더 정확하게 드리블 생성 가능함. 단, baseline을 위해 제외함

    return cast(DataFrame[SPADLSchema], actions)

def _sort_sequence(df_events: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    """
        Bepro 데이터는 이벤트가 발생한 순서대로 정렬되어 있지 않음.
        원인은 모르나, 이벤트 시퀀스 데이터(sequences 데이터프레임)를 활용하여
        각 이벤트가 발생한 순서대로 정렬하는 작업이 필요함.
    """
    def _insert_seq_events(period_group: pd.DataFrame, period_sequence) -> pd.DataFrame:
        period_group = period_group.copy().set_index('event_id') # indexing: faster

        # 시퀀스 데이터에 포함된 모든 event_id(List) 추출
        seq_event_ids = [
            event_id 
            for event_ids in period_sequence["event_ids"] 
            for event_id in event_ids # event_ids: list
            if event_id in period_group.index # clean_events 함수에서 제거된 이벤트는 제외
        ]

        seq_events = period_group.loc[seq_event_ids]
        seq_events = seq_events[
            ~seq_events.index.duplicated(keep="first") # 주의사항: 시퀀스 데이터는 중복값이 존재함. ex) 시퀀스의 시작 or 끝
        ].reset_index(drop=False) # event_id 복원
        
        # 시퀀스 데이터에 포함되지 않는 이벤트 삽입.
        # # idxmax: Return index of "first" occurrence of maximum over requested axis.
        # ex) if time is 6, event_time = [1, 2, 7, 8] -> idxmax=[False, False, True, True] -> insert_time = 6 -> [1, 2, 6, 7, 8]
        not_seq_events = period_group[~period_group.index.isin(seq_events.event_id)].reset_index(drop=False)
        not_seq_events.index = not_seq_events["event_time"].apply(
            lambda time: (seq_events["event_time"] > time).idxmax() 
            if any(seq_events["event_time"] > time) else len(seq_events)
        )

        return pd.concat(
            [not_seq_events, seq_events], 
            ignore_index=False,
            axis=0
        ).sort_index(kind="mergesort").reset_index(drop=True)
    
    df_events = df_events.groupby("period_id").apply(
        lambda group: _insert_seq_events(group, sequences[sequences["period_id"] == group.name])
        )

    return df_events.reset_index(drop=True)

def _clean_events(df_events: pd.DataFrame, remove_event_types) -> pd.DataFrame:
    """
    데이터프레임에서 특정 이벤트 타입을 제거하는 함수.
    remove_event_types (list): 제거할 이벤트 타입 목록

    - 결측치 조건(missing_cond)
    1.24	[Duel]	[Aerial]	...	NaN	NaN	NaN	NaN	NaN	NaN	[{'event_type': 'Duel', 'sub_event_type': 'Aerial...	NaN	NaN	35
    67	85848	94916404	1	4641.0	259769.0	163181	0.193699	0.5342 event_types이 기록되어 있지 않는 경우 -> parsing이 불가능함, 단순히 이전 정보만으로는 예측 불가능
    2. event_types이 기록되어 있는데, team_id & player_id정보가 기록되어 있지 않는 경우 -> 이전 정보로는 불가능하겠지만, 다음 정보로는 가능함.
    
    - 중복 데이터(duplicated_cond)
    1. event_id가 중복되는 경우 -> 제거
    2. event_id는 다른데 그 외 데이터가 중복되는 경우 -> 제거
    """

    # 해당 이벤트만 제거
    # ex) Pass + Control Under Pressure -> Pass 
    df_events["event_types"] = df_events["event_types"].apply(
        lambda event_list: [event for event in event_list if event.get("event_type") not in remove_event_types]
    )

    # 처음부터 빈 리스트이거나 제거하므로써 remove_event_types로 인해 빈 리스트가 된 행 제거
    missing_cond = (
        (df_events['event_types'].apply(len) == 0)
        # | (events["team_id"].isna())   # we do not remove missing team_id to better capture the context
        # | (events["player_id"].isna()) # we do not remove missing player_id to better capture the context
    )
    df_events = df_events[~missing_cond].reset_index(drop=True)

    # keep=fist: 중복된 데이터 중 첫번째 데이터만 남기고 나머지 제거(첫번째 데이터만 False)
    df = df_events.copy()

    # duplicated_cond1: event_id가 중복되는 경우
    duplicated_cond1 = df.duplicated(subset="event_id", keep="first") # 첫번째 데이터만 False -> ~False=True

    # duplicated_cond2: event_id는 다른데 그 외 데이터가 모두 중복되는 경우
    non_event_cols = [col for col in df.columns if col != "event_id"] # event_id 이외 컬럼
    for col in non_event_cols:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) or isinstance(x, dict) else x) # duplicated함수는 list, dict를 지원하지 않음
    duplicated_cond2 = df.duplicated(subset=non_event_cols, keep="first") # 첫번째 데이터만 False -> ~False=True
    
    return df_events[
        ~(duplicated_cond1 | duplicated_cond2)
    ].reset_index(drop=True)

def _parse_event(event : pd.Series) -> tuple[int, int, float, float]:
    # 23 possible values : pass, cross, throw-in, 
    # crossed free kick, short free kick, crossed corner, short corner, 
    # take-on, foul, tackle, interception, 
    # shot, penalty shot, free kick shot, 
    # keeper save, keeper claim, keeper punch, keeper pick-up, 
    # clearance, bad touch, dribble and goal kick.
    events = {
        "pass": _parse_pass_event,
        "cross": _parse_pass_event,
        "throw_in": _parse_pass_event,
        "freekick_crossed": _parse_pass_event,
        "freekick_short": _parse_pass_event,
        "corner_crossed": _parse_pass_event,
        "corner_short": _parse_pass_event,

        "take_on": _parse_take_on_event,

        "foul": _parse_foul_event,

        "tackle" : _parse_tackle_event,

        "interception": _parse_interception_event,

        "shot": _parse_shot_event,
        "shot_penalty": _parse_shot_event,
        "shot_freekick": _parse_shot_event,

        "keeper_save" : _parse_goalkeeper_event,
        "keeper_claim" : _parse_goalkeeper_event,
        "keeper_punch" : _parse_goalkeeper_event,
        "keeper_pick_up" : _parse_goalkeeper_event,
        "Defensive_Line_Support" : _parse_goalkeeper_event,

        "clearance" : _parse_clearance_event,
        "bad_touch" : _parse_bad_touch_event,
        "dribble" : _parse_dribble_event,

        "goalkick" : _parse_pass_event,
    }

    parser = events.get(event["type_name"], _parse_event_as_non_action)
    bodypart, result = parser(event)

    actiontype = spadlconfig.actiontypes.index(event["type_name"])
    bodypart = spadlconfig.bodyparts.index(bodypart)
    result = spadlconfig.results.index(result)
    
    return actiontype, bodypart, result

def _get_type_name(event_types: list) -> str:
    if any(e["event_type"] == "Pass" for e in event_types):
        pass_dict = next(e for e in event_types if e["event_type"] == "Pass")
        if pass_dict.get("cross", False):
            if any(e.get("sub_event_type") == "Freekick" for e in event_types):
                a = "freekick_crossed"
            elif any(e.get("sub_event_type") == "Corner" for e in event_types):
                a = "corner_crossed"
            else:
                a = "cross"
        else:
            if any(e.get("sub_event_type") == "Freekick" for e in event_types):
                a = "freekick_short"
            elif any(e.get("sub_event_type") == "Corner" for e in event_types):
                a = "corner_short"
            elif any(e.get("sub_event_type") == "Throw-In" for e in event_types):
                a = "throw_in"
            elif any(e.get("sub_event_type") == "Goal Kick" for e in event_types):
                a = "goalkick"
            else:
                a = "pass"
    elif any(e["event_type"] == "Shot" for e in event_types):
        if any(e.get("sub_event_type") == "Freekick" for e in event_types):
            a = "shot_freekick"
        elif any(e.get("sub_event_type") == "Penalty Kick" for e in event_types):
            a = "shot_penalty"
        else:
            a = "shot"
    elif any(e["event_type"] == "Take-On" for e in event_types):
        a = "take_on"
    elif any(e["event_type"] == "Step-in" for e in event_types): # API 2025: rename Carry to Step-in
        a = "dribble"
    elif any(e["event_type"] == "Save" for e in event_types): # 골키퍼 액션 : Save, Aerial Clearnce, Defensive Line Support Succeeded
        if any(e.get("sub_event_type") == "Catch" for e in event_types):
            a = "keeper_save"
        elif any(e.get("sub_event_type") == "Parry" for e in event_types):
            a = "keeper_punch"
        else:
            a = "non_action" # Save는 Catch과 Parry만 존재하고 예외경우는 존재하지는 않음
    elif any((e["event_type"] == "Aerial Clearance") & (e.get("outcome") == "Successful") for e in event_types):  
        # 실패한 Aerial Clearance는 공의 방향에 영향을 미치지 않으므로 non_action으로 처리
        a = "keeper_claim"
    elif any(e["event_type"] == "Clearance" for e in event_types):
        a = "clearance"
    elif any(e["event_type"] == "Foul" for e in event_types):
        a = "foul"
    elif any(e["event_type"] in ["Tackle", "Intervention"] for e in event_types):
        a = "tackle"
    elif any(e["event_type"] == "Interception" for e in event_types):
        a = "interception"
    elif any(e["event_type"] == "Error" for e in event_types) or any(e["event_type"] == "Own Goal" for e in event_types): # 자책골도 bad_touch로 정의함
        a = "bad_touch"
    else:
        a = "non_action"
    
    return a

def _fix_direction_of_play(df_events: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    away_idx = (df_events.team_id != home_team_id).values
    for col in ["x", "end_x"]:
        df_events.loc[away_idx, col] = GOAL_LINE_LENGTH - df_events.loc[away_idx, col].values
    for col in ["y", "end_y"]: 
        df_events.loc[away_idx, col] = TOUCH_LINE_LENGTH - df_events.loc[away_idx, col].values

    return df_events

def _fix_defensive_line_support(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert Defensive_Line_Support events to interception"""  
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1)
    cond_defensive_line_support = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Defensive Line Support" for e in x))
    cond_tackle = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Tackle" for e in x))
    same_player = df_events.player_id == df_events_next.player_id

    cond_interception = cond_defensive_line_support & same_player & ~cond_tackle # 수비 액션 이후 공을 소유하는 경우, Interception으로 정의
    cond_tackle = cond_defensive_line_support & (~same_player | cond_tackle)  # 수비 액션 이후 공을 소유하지 않는 경우, Tackle로 정의

    df_events.loc[cond_interception , "event_types"] = df_events.loc[cond_interception , "event_types"].apply(
        lambda event_list: event_list + [{"event_type": "Interception"}]
    )
    df_events.loc[cond_tackle , "event_types"] = df_events.loc[cond_tackle , "event_types"].apply(
        lambda event_list: event_list + [{"event_type": "Tackle", "outcome": next(e.get("outcome") for e in event_list if e["event_type"] == "Defensive Line Support")}]
    )

    return df_events

def _fix_offside(df_events: pd.DataFrame) -> pd.DataFrame:
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1)

    cond_pass = df_events["event_types"].apply(
        lambda x: any(e["event_type"] == "Pass" for e in x)
    )
    cond_set_piece = df_events["event_types"].apply(
        lambda x: any(e.get("sub_event_type") in ["Corner", "Freekick"] for e in x)
    )
    cond_next_offside = df_events_next["event_types"].apply(
        lambda x: any(e.get("event_type") == "Offside" for e in x)
    )

    df_events.loc[cond_pass & cond_next_offside, "event_types"] = df_events.loc[cond_pass & cond_next_offside, "event_types"].apply(
        lambda event_list: [{**e, "outcome": "offside"} if e.get("event_type") == "Pass" else e for e in event_list]
    )
    df_events.loc[cond_set_piece & cond_next_offside, "event_types"] = df_events.loc[cond_set_piece & cond_next_offside, "event_types"].apply(
        lambda event_list: [{**e, "outcome": "offside"} if e.get("sub_event_type") in ["Corner", "Freekick"] else e for e in event_list]
    )

    return df_events

def _fix_dribble(df_actions: pd.DataFrame) -> pd.DataFrame:
    """
        _fix_dribble : Update the end position of dribble events based on their success or failure.
        If the dribble failed, the end position is set to the position of the next event.
        If the dribble succeeded, the end position is set to the position of the next event that is not a tackle.
    """

    df_actions_next = shift_with_edge_fix(df_actions, shift_value=-1)

    failed_tackle = (
        (df_actions_next['type_id'] == spadlconfig.actiontypes.index('tackle')) &
        (df_actions_next['result_id'] == spadlconfig.results.index('fail'))
    )
    failed_defensive = (
        failed_tackle & 
        (df_actions.team_id != df_actions_next.team_id)
    )

    # next_actions: 실패한 태클이 아닌 다음 이벤트의 위치를 드리블의 끝 위치로 보간
    # ex) dribble(A팀) -> tackle(B팀, fail) -> pass(A팀)의 경우, dribble의 끝 위치는 pass의 시작 위치로 정의
    next_actions = df_actions_next.mask(failed_defensive)[["start_x", "start_y"]].bfill()

    cond_dribble = df_actions.type_id == spadlconfig.actiontypes.index("dribble")
    df_actions.loc[cond_dribble, "end_x"] = next_actions.loc[cond_dribble, "start_x"].values
    df_actions.loc[cond_dribble, "end_y"] = next_actions.loc[cond_dribble, "start_y"].values

    return df_actions

def _fix_tackle_result(df_actions: pd.DataFrame) -> pd.DataFrame:
    """
        spadl에서 태클은 소유권 기반으로 결과를 정의한다.
        반면 bepro에서 태클은 소유권 뿐만 아니라 loose ball 상황을 만드는 행위도 성공으로 간주한다.
        따라서 bepro데이터에서 태클의 결과를 spadl형태로 변환
    """
    cond_tackle = df_actions.type_id == spadlconfig.actiontypes.index("tackle")

    # 예외 케이스(소유권기반으로 정의할 수 없음): 득점 이후 태클을 수행하는 액션은 실패한 태클로 간주
    # 발례) shot(A팀) -> tackle(B팀) -> kick_off(B팀)의 경우 소유권 유지과 상관없이 실패한 액션
    df_actions_prev = shift_with_edge_fix(df_actions, shift_value=1)
    tackle_after_goal = (
        (df_actions_prev.team_id != df_actions.team_id) & # 공격팀의 득점(success shot) 이후 수비팀의 태클 
        (df_actions_prev.type_id == spadlconfig.actiontypes.index("shot")) & 
        (df_actions_prev.result_id == spadlconfig.results.index("success"))
    )

    df_actions_next = shift_with_edge_fix(df_actions, shift_value=-1)
    same_team = df_actions.team_id == df_actions_next.team_id

    df_actions.loc[cond_tackle & same_team, "result_id"] = spadlconfig.results.index("success")
    df_actions.loc[cond_tackle & (~same_team | tackle_after_goal), "result_id"] = spadlconfig.results.index("fail")

    return df_actions

# 공격 이벤트와 수비 이벤트가 함께 존재하는지 확인하는 함수
def insert_defensive_actions(df_events: pd.DataFrame, defensive_action : str) -> pd.DataFrame:
    """Insert defensive actions before offensive actions when both occur at the same time."""

    def is_attack_and_defense(event_types : list) -> bool:
        has_attack = any(e["event_type"] in ["Pass", "Shot", "Take-On", "Step-in", "Clearance"] for e in event_types) # 공격 이벤트
        has_defense = any(e["event_type"] == defensive_action for e in event_types)

        return has_attack and has_defense

    cond_attack_and_defense = df_events["event_types"].apply(is_attack_and_defense)
    df_events_defense = df_events[cond_attack_and_defense].copy()

    if not df_events_defense.empty:
        df_events_defense["event_time"] -= 1e-3
        df_events_defense["event_types"] = df_events_defense["event_types"].apply(
            lambda event_list: [event for event in event_list if event.get("event_type") == defensive_action]
        )
        df_events.loc[cond_attack_and_defense, "event_types"] = df_events.loc[cond_attack_and_defense, "event_types"].apply(
            lambda event_list: [event for event in event_list if event.get("event_type") != defensive_action]
        )

        df_events = pd.concat([df_events_defense, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "event_time"], kind="mergesort")
        df_events = df_events.reset_index(drop=True)

    return df_events

def _convert_pass_locations(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert StatsBomb locations to spadl coordinates.
    
    K-리그 경기장 규격 특징:
    1. 경기장 크기: 68m x 105m
    2. 좌표 (0,0)은 왼쪽 하단을, (1,1)은 오른쪽 상단을 의미함
    3. 하프타임과 상관없이 모든 이벤트는 항상 골 라인(y=0)에서 시작함.
    4. x좌표는 68(GOAL_LINE_LENGTH)단위로, y좌표는 105(TOUCH_LINE_LENGTH)단위로 변환.
    """
    def _get_end_location(relative_event: dict) -> tuple[Optional[float], Optional[float]]:
        if isinstance(relative_event, dict):
            return pd.Series([relative_event.get("x"), relative_event.get("y")])
        else:
            return pd.Series([np.nan, np.nan])

    df_events[["end_x", "end_y"]] = df_events["relative_event"].apply(_get_end_location)

    df_events[["x", "end_x"]] = np.clip(df_events[["x", "end_x"]] * GOAL_LINE_LENGTH, 0, GOAL_LINE_LENGTH)
    df_events[["y", "end_y"]] = np.clip(df_events[["y", "end_y"]] * TOUCH_LINE_LENGTH, 0, TOUCH_LINE_LENGTH)

    return df_events

def _convert_shot_locations(df_events: pd.DataFrame) -> pd.DataFrame:
    cond_shot = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Shot" for e in x))
    cond_own_goal = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Own Goal" for e in x))
    cond_blocked = df_events["event_types"].apply(lambda x: any(e.get("outcome") == "Blocked" for e in x))
    cond_low_quality = df_events["event_types"].apply(lambda x: any(e.get("outcome") == "Low Quality Shot" for e in x))
    cond_keeper_rush_out = df_events["event_types"].apply(lambda x: any(e.get("outcome") == "Keeper Rush-Out" for e in x))

    cond_missing_end_loc = df_events["ball_position"].apply(lambda b: not isinstance(b, dict))
    
    # 슛의 끝 위치가 기록된 경우 해당 좌표로 설정
    cond_existing_end_loc = (
        cond_shot & 
        ~cond_missing_end_loc
    )
    df_events.loc[cond_existing_end_loc, "end_x"] = df_events[
        cond_existing_end_loc
    ]["ball_position"].apply(lambda b: ORIGINAL_LEFT_POST + b.get("x") * (ORIGINAL_RIGHT_POST - ORIGINAL_LEFT_POST))
    df_events.loc[cond_existing_end_loc, "end_x"] = np.clip(df_events.loc[cond_existing_end_loc, "end_x"] * GOAL_LINE_LENGTH, 0, GOAL_LINE_LENGTH)
    df_events.loc[cond_existing_end_loc, "end_y"] = TOUCH_LINE_LENGTH # 슛의 높이 정보는 사용하지 않음

    # 자책골의 경우, 우리 팀 진영(y=0)의 중앙 골 포스트로 설정
    df_events.loc[
        cond_own_goal & cond_missing_end_loc, 
        ["end_x", "end_y"]
    ] = CENTER_POST, 0

    # 결측치 처리 방식: Bepro 데이터는 슛의 끝 위치가 기록되지 않은 경우가 많음.
    # 1. Blocked: 블로킹한 액션의 위치를 슛의 끝 위치로 보간
    # 2. Low Quality: 골 포스트에 많이 빗나간 경우 or 골 포스트까지도 도달하지 못한 경우, 상황에 따른 위치를 보간
    # 3. Keeper Rush-Out: 골키퍼가 슛을 막기 위해 나선 경우, 다음 골키퍼 액션의 위치를 슛의 끝 위치로 보간
    # 4. 그 외: Off-Target(99%), On-Target, Goal 등 소수의 결측치는 휴리스틱 기반으로 보간. ex) Off-Target-> Goal Kick, Corner, Deflection, Substitution, Recovery, Parry

    # 1. Blocked 슛의 경우
    # 주의: 블로킹한 액션을 수행한 팀이 같은 팀이냐 상대팀이냐에 따라 위치 기준이 다름. 이유: 두 팀의 공격 방향이 같기 때문임.
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1) 
    blocked_idx_by_teammate = (
        cond_shot & 
        cond_blocked & 
        cond_missing_end_loc & 
        cond_blocked & 
        (df_events["team_id"] == df_events_next["team_id"]) # 팀원의 블로킹(hit) 액션은 같은 공격 방향이므로 대칭하지 않음
    ) 
    blocked_idx_by_opponent = (
        cond_shot & 
        cond_blocked & 
        cond_missing_end_loc & 
        (df_events["team_id"] != df_events_next["team_id"]) # 수비팀의 블로킹 액션은 수비 진영을 기준으로 기록되어 있으므로 대칭 해야함
    ) 

    # 팀원 블로킹 액션의 위치를 슛의 끝 위치로 보간
    df_events.loc[blocked_idx_by_teammate, ["end_x", "end_y"]] = df_events_next.loc[blocked_idx_by_teammate, ["x", "y"]].values

    # 수비팀 블로킹 액션의 위치를 슛의 끝 위치로 보간
    df_events.loc[blocked_idx_by_opponent, "end_x"] = GOAL_LINE_LENGTH - df_events_next.loc[blocked_idx_by_opponent, "x"].values
    df_events.loc[blocked_idx_by_opponent, "end_y"] = TOUCH_LINE_LENGTH - df_events_next.loc[blocked_idx_by_opponent, "y"].values

    # 2. Low Quality 슛의 경우 (복잡함) 
    cond_next_set_piece = df_events_next["event_types"].apply(lambda x: any(e.get("sub_event_type") in ["Goal Kick", "Corner"] for e in x)) # 골킥은 수비 진영을 기준으로 기록되어 있으므로 대칭 해야함
    low_quality_idx_by_set_piece = (
        cond_shot & 
        cond_low_quality & 
        cond_missing_end_loc & 
        cond_next_set_piece # 세트피스가 다음 액션인 경우, 슛이 골 포스트 바깥으로 벗어난 상황이므로 휴리스틱 기반으로 보간
    )
    low_quality_idx_others_by_teammate = (
        cond_shot & 
        cond_low_quality & 
        cond_missing_end_loc &
        ~low_quality_idx_by_set_piece &
        (df_events["team_id"] == df_events_next["team_id"]) # 인플레이가 유지되는 액션의 경우, 팀원의 다음 액션은 같은 공격 방향이므로 대칭하지 않음
    )
    low_quality_idx_others_by_opponent = (
        cond_shot & 
        cond_low_quality & 
        cond_missing_end_loc & 
        ~low_quality_idx_by_set_piece & 
        (df_events["team_id"] != df_events_next["team_id"]) # 인플레이가 유지되는 액션의 경우, 수비팀의 다음 액션은 수비 진영을 기준으로 기록되어 있으므로 대칭 해야함
    )

    cond_out_left = (
        df_events["x"] < (LEFT_POST - Eighteen_YARD) # 왼쪽 측면에서의 슛은 왼쪽 골 포스트 바깥으로 설정
    )
    cond_out_center = (

        (df_events["x"] >= (LEFT_POST - Eighteen_YARD)) # 중앙에서의 슛은 중앙 골 포스트 방향으로 설정
        & (df_events["x"] <= (RIGHT_POST + Eighteen_YARD))
    )
    cond_out_right = (
        df_events["x"] > (RIGHT_POST + Eighteen_YARD) # 오른쪽 측면에서의 슛은 오른쪽 골 포스트 바깥으로 설정
    )
    df_events.loc[low_quality_idx_by_set_piece & cond_out_left, ["end_x", "end_y"]] = LEFT_POST - Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[low_quality_idx_by_set_piece & cond_out_right, ["end_x", "end_y"]] = RIGHT_POST + Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[low_quality_idx_by_set_piece & cond_out_center, ["end_x", "end_y"]] = CENTER_POST, TOUCH_LINE_LENGTH


    df_events.loc[low_quality_idx_others_by_teammate, ["end_x", "end_y"]] = df_events_next.loc[low_quality_idx_others_by_teammate, ["x", "y"]].values

    df_events.loc[low_quality_idx_others_by_opponent, "end_x"] = GOAL_LINE_LENGTH - df_events_next.loc[low_quality_idx_others_by_opponent, "x"].values
    df_events.loc[low_quality_idx_others_by_opponent, "end_y"] = TOUCH_LINE_LENGTH - df_events_next.loc[low_quality_idx_others_by_opponent, "y"].values

    # 3. Keeper Rush-Out 슛의 경우
    # 상대팀의 골키퍼 액션 위치를 슛의 끝 위치로 보간하므로 대칭 해야함
    df_events.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "end_x"] = GOAL_LINE_LENGTH - df_events_next.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "x"].values
    df_events.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "end_y"] = TOUCH_LINE_LENGTH - df_events_next.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "y"].values
    
    # 4. 그 외 결측치의 경우
    cond_other_missing_end_loc = (
        cond_shot &
        (df_events["end_x"].isna() | df_events["end_y"].isna())
    )
    
    df_events.loc[cond_other_missing_end_loc & cond_out_left, ["end_x", "end_y"]] = LEFT_POST - Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[cond_other_missing_end_loc & cond_out_right, ["end_x", "end_y"]] = RIGHT_POST + Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[cond_other_missing_end_loc & cond_out_center, ["end_x", "end_y"]] = CENTER_POST, TOUCH_LINE_LENGTH

    return df_events

def _parse_event_as_non_action(event):
    bodypart = "other"
    result = "fail"
    return bodypart, result

def _parse_pass_event(event):
    cond_aerial = any(
        (e.get("event_type") == "Duel") and 
        (e.get("sub_event_type") == "Aerial") 
        for e in event["event_types"]
    )
    cond_throw_in = any(
        (e.get("event_type") == "Set Piece") and 
        (e.get("sub_event_type") == "Throw-In")
        for e in event["event_types"]
    )
    if cond_aerial:
        bodypart = "head" 
    elif cond_throw_in:
        bodypart = "other"
    else:
        bodypart = "foot"

    pass_outcome =  next(
        (e.get('outcome') for e in event['event_types'] if e.get('event_type') == 'Pass'), 
        None
    )
    if pass_outcome == "Successful":
        result = "success"
    elif pass_outcome == "Unsuccessful":
        result = "fail"  # Offside situations are handled in _fix_offside
    elif pass_outcome == "offside":
        result = "offside"
    else:
        raise ValueError(f"Unexpected outcome value: {pass_outcome}")

    return bodypart, result

def _parse_take_on_event(event):
    bodypart = next(
        (e.get("body_part") for e in event["event_types"] if e.get("event_type") == "Take-On"), 
        None
    )
    if bodypart == "Hands":
        bodypart = "other"
    elif bodypart == "Head":
        bodypart = "head"
    elif bodypart == "Left Foot":
        bodypart = "foot_left"
    elif bodypart == "Right Foot":
        bodypart = "foot_right"
    elif bodypart in ["Lower Body", "Upper Body", "Other"]:
        bodypart = "other"
    else:
        bodypart = "foot"

    take_on_outcome =  next(
        (e.get('outcome') for e in event['event_types'] if e.get('event_type') == 'Take-On'), 
        None
    )
    if take_on_outcome  == "Successful":
        result = "success"
    elif take_on_outcome  == "Unsuccessful":
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {take_on_outcome}")

    return bodypart, result

def _parse_foul_event(event):
    if any(e.get("sub_event_type") in ["Handball Foul", "Foul Throw"] for e in event["event_types"]):
        bodypart = "other" 
    else:
        bodypart = "foot"

    # foul은 여러 결과가 동시에 존재할 수 있음(경고+퇴장 등)
    if any(e.get("outcome") == "Red Card" for e in event["event_types"]):
        result = "red_card"
    elif any(e.get("outcome") == "Yellow Card" for e in event["event_types"]):
        result = "yellow_card"
    else:
        result = "fail"
    
    return bodypart, result

def _parse_tackle_event(event):
    # Even in cases of "Aerial Duel + Tackle", the tackle itself is performed with the foot after the aerial duel, so "foot" is used.
    bodypart = "foot"

    tackle_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Tackle"), 
        None
    )
    defensive_line_support_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Defensive Line Support"), 
        None
    )

    if tackle_outcome:
        if tackle_outcome  == "Successful":
            result = "success"
        elif tackle_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {tackle_outcome}")
    elif defensive_line_support_outcome:
        if defensive_line_support_outcome  == "Successful":
            result = "success"
        elif defensive_line_support_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {defensive_line_support_outcome}")
    elif any(e.get("event_type") == "Intervention" for e in event["event_types"]):
        # Result is set to "success" for interventions with no recorded outcome, to be fixed later in _fix_defense_result.
        result = "success" 
    else:
        raise ValueError(f'Unexpected event_types: {event}')

    return bodypart, result

def _parse_interception_event(event):
    # Even in cases of "Aerial Duel + Interception", the interception itself is performed with the foot after the aerial duel, so "foot" is used.
    bodypart = "foot"

    defensive_line_support_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Defensive Line Support"), 
        None
    )
    if defensive_line_support_outcome:
        if defensive_line_support_outcome  == "Successful":
            result = "success"
        elif defensive_line_support_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {defensive_line_support_outcome}")
    else:
        # Result is set to "success" for interceptions and interventions with no recorded outcome, to be fixed later in _fix_defense_result.
        result = "success"  

    return bodypart, result

def _parse_shot_event(event):
    bodypart = next(
        (e.get("body_part") for e in event["event_types"] if e.get("event_type") == "Shot"), 
        None
    )
    if bodypart == "Hands":
        bodypart = "other"
    elif bodypart == "Head":
        bodypart = "head"
    elif bodypart == "Left Foot":
        bodypart = "foot_left"
    elif bodypart == "Right Foot":
        bodypart = "foot_right"
    elif bodypart in ["Lower Body", "Upper Body", "Other"]:
        bodypart = "other"
    else:
        bodypart = "foot"

    shot_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Shot"), 
        None
    )
    result = "success" if shot_outcome == "Goal" else "fail"

    return bodypart, result

def _parse_goalkeeper_event(event):
    bodypart = next(
        (e.get("body_part") for e in event["event_types"] if e.get("event_type") == "Save"), 
        None
    )
    if bodypart == "Hands":
        bodypart = "other"
    elif bodypart == "Head":
        bodypart = "head"
    elif bodypart == "Left Foot":
        bodypart = "foot_left"
    elif bodypart == "Right Foot":
        bodypart = "foot_right"
    elif bodypart in ["Lower Body", "Upper Body", "Other"]:
        bodypart = "other"
    else:
        bodypart = "other" # Aerial Clearance는 bodypart 정보가 없음

    # Determine the result based on the event type
    if any(e.get("event_type") == "Save" for e in event["event_types"]): # Catch and parry actions are always successful
        result = "success"
    elif any(e.get("event_type") == "Aerial Clearance" for e in event["event_types"]): # Claim actions can be successful or unsuccessful
        aerial_clearance_outcome = next(
            (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Aerial Clearance"), 
            None
        )
        if aerial_clearance_outcome == "Successful": 
            result = "success"
        elif aerial_clearance_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {aerial_clearance_outcome}")
    elif any(e.get("event_type") == "Defensive Line Support" for e in event["event_types"]): # Defensive Line Support can be successful or unsuccessful
        defensive_line_support_outcome = next(
            (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Defensive Line Support"), 
            None
        )
        if defensive_line_support_outcome  == "Successful":
            result = "success"
        elif defensive_line_support_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {defensive_line_support_outcome}")
    else:
        raise ValueError(f'Unexpected event_types: {event}')

    return bodypart, result

def _parse_clearance_event(event):
    cond_aerial = any(
        (e.get("event_type") == "Duel") and 
        (e.get("sub_event_type") == "Aerial") 
        for e in event["event_types"]
    )
    bodypart = "head" if cond_aerial else "foot"
    result = "success"

    return bodypart, result

def _parse_bad_touch_event(event):
    bodypart = "foot"
    result = "owngoal" if any(e.get("event_type") == "Own Goal" for e in event["event_types"]) else "fail"

    return bodypart, result

def _parse_dribble_event(event):
    bodypart = "foot"
    result = "success"

    return bodypart, result