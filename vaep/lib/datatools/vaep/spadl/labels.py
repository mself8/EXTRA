"""Implements the label tranformers of the VAEP framework."""

import pandas as pd  # type: ignore
from pandera.typing import DataFrame

import datatools.representation.spadl.config as spadl
from datatools.representation.spadl.schema import SPADLSchema

N_SECONDS = 10
N_ACTIONS = 10

# n_seconds 내에 세트피스가 있을 때, 세트피스 이후부터 n_seconds를 label 계산
# n_seconds 내에 세트피스가 있는지 확인하고, 세트피스 이후부터 n_seconds를 계산하도록 로직을 추가하겠습니다.
# ex) pass(280) -> out(282) -> Throw-In(290) -> pass(292) -> pass(293) -> goal(294)
# 기존 방식: n_seconds = 10일 때, positive_label은 Throw-In(290)부터 10초 이내에 발생한 goal(294)까지
# 개선 방식: n_seconds = 10일 때, positive_label은 Pass(290-n_seconds) ~ goal(294)까지
set_piece_types = ["freekick_crossed", "freekick_short ", "shot_freekick", "corner_crossed", "corner_short", 
                    "shot_penalty", "throw_in", "goalkick"]  # 예시로 세트피스 이벤트 타입 정의


def scores_by_actions(actions: DataFrame[SPADLSchema], nr_actions: int = N_ACTIONS) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x actions; otherwise False.
    """
    # merging goals, owngoals and team_ids

    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )
    # 자살골 액션 정의 error
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoals = actions["result_id"] == spadl.results.index("owngoal")
    y = pd.concat([goals, owngoals, actions["team_id"]], axis=1)
    y.columns = ["goal", "owngoal", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "goal", "owngoal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c].iloc[len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["goal"]
    for i in range(1, nr_actions):
        gi = y["goal+%d" % i] & (y["team_id+%d" % i] == y["team_id"])
        ogi = y["owngoal+%d" % i] & (y["team_id+%d" % i] != y["team_id"])
        res = res | gi | ogi

    return pd.DataFrame(res, columns=["scores_by_actions"])


def concedes_by_actions(actions: DataFrame[SPADLSchema], nr_actions: int = N_ACTIONS) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action set to
        True if a goal was conceded by the team possessing the ball within the
        next x actions; otherwise False.
    """
    # merging goals,owngoals and team_ids
    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoals = actions["result_id"] == spadl.results.index("owngoal")
    y = pd.concat([goals, owngoals, actions["team_id"]], axis=1)
    y.columns = ["goal", "owngoal", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "goal", "owngoal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c].iloc[len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["owngoal"]
    for i in range(1, nr_actions):
        gi = y["goal+%d" % i] & (y["team_id+%d" % i] != y["team_id"])
        ogi = y["owngoal+%d" % i] & (y["team_id+%d" % i] == y["team_id"])
        res = res | gi | ogi

    return pd.DataFrame(res, columns=["concedes_by_actions"])

def scores_by_seconds(actions: DataFrame[SPADLSchema], n_seconds: int = N_SECONDS) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x seconds.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_seconds : int, default=10  # noqa: DAR103
        Number of seconds after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x seconds; otherwise False.
    """
    # merging goals, owngoals and team_ids
    goal_idx = actions[actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )].index
    # 자살골 액션 정의 error
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoal_idx = actions[actions["result_id"] == spadl.results.index("owngoal")].index

    res = pd.Series([False] * len(actions))
    for idx in goal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]
        
        # n_seconds 내에 세트피스가 있는지 확인(소유권 여부과 상관없음)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        goal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] == team_id) &
            (actions.index <= idx) # Event Stream순서를 조정하기 때문에 시간 정보만으로 label부여를 할 수 없음
        )  
        res = res | goal_cond

    for idx in owngoal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]

        # n_seconds 내에 세트피스가 있는지 확인(소유권 여부과 상관없음)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        owngoal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] != team_id) &
            (actions.index <= idx) # Event Stream순서를 조정하기 때문에 시간 정보만으로 label부여를 할 수 없음
        )
        res = res | owngoal_cond

    return pd.DataFrame(res, columns=["scores_by_seconds"])

def concedes_by_seconds(actions: DataFrame[SPADLSchema], n_seconds: int = N_SECONDS) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x seconds.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_seconds : int, default=10  # noqa: DAR103
        Number of seconds after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x seconds; otherwise False.
    """
    # merging goals, owngoals and team_ids
    goal_idx = actions[actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )].index
    # 자살골 액션 정의 error
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoal_idx = actions[actions["result_id"] == spadl.results.index("owngoal")].index
    
    res = pd.Series([False] * len(actions))
    for idx in goal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]
        
        # n_seconds 내에 세트피스가 있는지 확인(소유권 여부과 상관없음)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        goal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] != team_id) &
            (actions.index <= idx) # Event Stream순서를 조정하기 때문에 시간 정보만으로 label부여를 할 수 없음
        )  

        res = res | goal_cond

    for idx in owngoal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]

        # n_seconds 내에 세트피스가 있는지 확인(소유권 여부과 상관없음)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        owngoal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] == team_id) &
            (actions.index <= idx) # Event Stream순서를 조정하기 때문에 시간 정보만으로 label부여를 할 수 없음
        )
        res = res | owngoal_cond

    return pd.DataFrame(res, columns=["concedes_by_seconds"])

def goal_from_shot(actions: DataFrame[SPADLSchema]) -> pd.DataFrame:
    """Determine whether a goal was scored from the current action.

    This label can be use to train an xG model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.
    """
    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )

    return pd.DataFrame(goals, columns=["goal_from_shot"])
