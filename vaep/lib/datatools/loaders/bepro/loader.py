import os

# Import standard libraries
from typing import cast, Optional
from pathlib import Path

# Import third-party libraries
import pandas as pd
from unidecode import unidecode
from pandera.typing import DataFrame
from unidecode import unidecode

from datatools.loaders.base import (
    EventDataLoader,
    ParseError,
    _expand_minute,
    _localloadjson,
)

from .schema import (
    BeproCompetitionSchema,
    BeproEventSchema,
    BeproGameSchema,
    BeproPlayerSchema,
    BeproTeamSchema,
    BeproPlayerStatsSchema,
    BeproSequenceSchema,
)

class BeproLoader(EventDataLoader):
    def __init__(
            self,
            getter: str = "remote",
            root: Optional[str] = None,
    ):
        if getter == "local":
            if root is None:
                raise ValueError("""The 'root' parameter is required when loading local data.""")
            self._local = True
            self._root = root
        
        # id: name으로 경로정보를 제공
        self.season_info = {}
        self.game_info = {}

        self.PERIOD_DICT = {"FIRST_HALF": 1, "SECOND_HALF": 2, "EXTRA_FIRST_HALF": 3, "EXTRA_SECOND_HALF": 4}

    def get_path(self, info):
        if 'competition_id' in info.keys() and 'season_id' in info.keys():
            competition_name, season_name = self.season_info[(info["competition_id"], info["season_id"])]
        elif 'game_id' in info:
            competition_name, season_name = self.game_info[info["game_id"]]
        else:
            raise ValueError("Please provide valid keys: either 'season_id' and 'competition_id', or 'game_id'.")

        return Path(self._root / competition_name / season_name)
    
    def competitions(self) -> DataFrame[BeproCompetitionSchema]:
        """Return a dataframe with all available competitions and seasons.
            
        file : league.json & season.json

        """
        cols = [
            "season_id",
            "competition_id",
            "competition_name",
            "country_name",
            "season_name",
        ]

        league = _localloadjson(path = Path(self._root / "league.json"))

        competitions = []

        for _, row in enumerate(league["result"]):
            competition_id = row["id"]
            competition_name_en = row["name_en"].replace(" ", "")

            season = _localloadjson(path = Path(self._root / competition_name_en / "season.json"))
            for season_id in row["season_ids"]:
                season_name = next(filter(lambda x : x["id"] == season_id, season["result"]))["name"]
                competitions.append([season_id, competition_id, competition_name_en, row["iso_country_code"], season_name])

                self.season_info[(competition_id, season_id)] = (competition_name_en, season_name) # id: name으로 경로정보를 제공
        obj = BeproCompetitionSchema.validate(pd.DataFrame(competitions, columns=cols))

        return cast(DataFrame[BeproCompetitionSchema], obj)

    def games(self, competition_id: int, season_id: int) -> DataFrame[BeproGameSchema]:
        """Return a dataframe with all available games in a season.

        file : info.json

        Parameters
        ----------
        competition_id : int
            The ID of the competition.
        season_id : int
            The ID of the season.
        """

        cols = [
            "game_id",
            "season_id",
            "competition_id",
            "game_day",       # rename a variable : round -> game_day
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
            "venue",
        ]

        gamesdf = []
        
        competition_name, season_name = self.season_info[(competition_id, season_id)]
        path = Path(self._root / competition_name / season_name)

        game_ids = os.listdir(Path(path / "match"))
        game_ids = list(map(int, game_ids))
        
        for game_id in game_ids:
            self.game_info[game_id] = (competition_name, season_name)
            match_info = _localloadjson(Path(path / "match" / f"{game_id}" / "info.json"))["result"]

            round = match_info["round"]          
            game_day = int(round["name"].split(" ")[-1]) if "Round" in round["name"] else round["id"] # 승강제 데이터셋은 Round가 없음
            game_date = match_info["start_time"]

            home_team_id, away_team_id = match_info["home_team"]["id"], match_info["away_team"]["id"]
            home_score, away_score = match_info["detail_match_result"]["home_team_score"], match_info["detail_match_result"]["away_team_score"]

            venue = match_info["venue"]["display_name"] if isinstance(match_info["venue"], dict) else match_info["venue"]

            gamesdf.append([game_id, season_id, competition_id, game_day, game_date, 
                            home_team_id, away_team_id, home_score, away_score, venue])

        gdf = pd.DataFrame(gamesdf, columns = cols)
        # pandas 2.x + pandera coerce 호환: tz-aware start_time('...+09:00')은
        # datetime64[ns](tz-naive)로 astype이 막히므로 로컬(KST) 벽시계 시간으로 tz 제거.
        gdf["game_date"] = pd.to_datetime(gdf["game_date"]).dt.tz_localize(None)
        obj = BeproGameSchema.validate(gdf)

        return cast(DataFrame[BeproGameSchema], obj)

    def _lineups(self, game_id: int) -> list:
        path = self.get_path({"game_id": game_id})
        lineup = _localloadjson(Path(path / "match" / f"{game_id}" / "lineup.json"))["result"]

        return lineup
    
    def teams(self, game_id: int) -> DataFrame[BeproTeamSchema]:
        """Return a dataframe with both teams that participated in a game.

        file : team.json

        Parameters
        ----------
        game_id : int
            The ID of the game.
        """

        cols = ["team_id", "team_name", "team_name_ko"]
        path = self.get_path({"game_id": game_id})

        # team.json : 시즌별 모든 팀의 정보(team_id, team_name, team_name_en)
        # lineup.json : 경기별 팀의 정보(game_id, team_id)
        team = pd.DataFrame(_localloadjson(Path(path / "team.json"))["result"])
        lineup = pd.DataFrame(self._lineups(game_id))

        obj = pd.merge(lineup, team, left_on = "team_id", right_on = "id", how = "left")
        obj = obj.drop_duplicates(subset="team_id", keep="first")
        obj = obj.rename(columns={"name_en" : "team_name", "name" : "team_name_ko"})

        obj = BeproTeamSchema.validate(obj[cols]) 

        return cast(DataFrame[BeproTeamSchema], obj)

    def players(self, game_id: int) -> DataFrame[BeproPlayerSchema]:
        """Return a dataframe with all players that participated in a game.

        file : lineup.json, player.json, player_stats.json

        Parameters
        ----------
        game_id : int
            The ID of the game.

        """
    
        cols = [
            "game_id",                # parameter
            "team_id",                # lineup
            "player_id",              # lineup
            "player_name",            # player(영어)
            "nickname",               # player(한국어)
            #"birth_date",             # year-month-day
            "jersey_number",          # rename a variable : back_number -> Jersey number
            "is_starter",             # rename a variable : is_starting_lineup -> is_starter
            "starting_position_name", # lineup(position_name) : position_name -> starting_position_name
            #"main_position",          # player(main_position)
            "minutes_played",         # convert milliseconds to seconds : play_time -> minutes_played
            "rating",                 # player_stats(rating)
        ]

        path = self.get_path({"game_id": game_id})
        lineup = pd.DataFrame(self._lineups(game_id))
        lineup["game_id"] = game_id

        # 팀별 player information추출 : 선수full이름, 선수full영문이름, 메인포지션, 생년월일등...
        teamA = _localloadjson(Path(path / "player" / f'{lineup["team_id"].unique()[0]}.json'))["result"]
        teamB = _localloadjson(Path(path / "player" / f'{lineup["team_id"].unique()[1]}.json'))["result"]
        team_df = pd.concat([pd.DataFrame(teamA), pd.DataFrame(teamB)]).reset_index()
        
        lineup = pd.merge(lineup, team_df, left_on = ["team_id", "player_id"], right_on = ["team_id", "id"], how="left", suffixes=("","_drop"))

        # 한국 : 성(Last Name) + 이름(First Name)
        lineup["nickname"] = lineup["player_last_name"] + lineup["player_name"]       

        # 외국 : 이름(First Name) + 성(Last Name)
        lineup["player_name_en"] = lineup.apply(lambda x: unidecode(x["player_name"]) if (pd.isna(x["player_name_en"]) | (x["player_name_en"] == '')) else x["player_name_en"], axis=1)
        lineup["player_last_name_en"] = lineup.apply(lambda x: unidecode(x["player_last_name"]) if (pd.isna(x["player_last_name_en"]) | (x["player_last_name_en"] == '')) else x["player_last_name_en"], axis=1) 
        lineup["player_name"] = lineup["player_name_en"] + ' ' + lineup["player_last_name_en"] 
        
        # 선수별 stat정보 추출 : 선수 총 뛴 시간(추가로 사용하고 싶은 지표있으면 사용)
        player_stats = self.player_stats(game_id = game_id)

        lineup = pd.merge(lineup, player_stats, on = ["team_id", "player_id"], how="left", suffixes=("","_drop"))
        lineup["minutes_played"] = lineup["play_time"].apply(lambda time : 0 if pd.isna(time) else time / 1000 / 60)

        lineup = lineup.rename(columns = {"is_starting_lineup" : "is_starter", "position_name": "starting_position_name"})
        lineup["jersey_number"] = lineup["back_number"].astype("int")
        #lineup["birth_date"] = lineup["birth_date"].fillna("")
        
        obj = BeproPlayerSchema.validate(lineup[cols]) 
        
        return cast(DataFrame[BeproPlayerSchema], obj)

    def player_stats(self, game_id) -> DataFrame:
        path = self.get_path({"game_id": game_id})
        player_stats = _localloadjson(Path(path / "match" / f'{game_id}/player_stats.json'))

        data = []
        cols = ["team_id", "player_id", "play_time", "rating"]

        for team in player_stats["result"]:
            team_id = team['team_id']

            for player in team['players']:
                player_id = player['player_id']

                stats = player['stats']
                stats['team_id'] = team_id
                stats['player_id'] = player_id
                
                data.append(stats)

        obj = BeproPlayerStatsSchema.validate(pd.DataFrame(data, columns = cols)) 

        return cast(DataFrame[BeproPlayerStatsSchema], obj) 
    
    def events(self, game_id: int) -> DataFrame[BeproEventSchema]:
        """Return a dataframe with the event stream of a game.

        Parameters
        ----------
        game_id : int
            The ID of the game.

        """
        cols = [
            "game_id",
            "event_id",
            "period_id",        
            "team_id",
            "player_id",
            "event_time",           # milliseconds
            "x",
            "y",

            "event_types",
            "relative_event",
            "ball_position",
        ]
        
        path = self.get_path({"game_id": game_id})
        events = pd.DataFrame(_localloadjson(Path(path / "match" / f"{game_id}" / "event_data.json"))["result"])
        events = events.rename(columns={"id" : "event_id", "match_id" : "game_id"})
        events["period_id"] = events["event_period"].map(self.PERIOD_DICT)

        obj = BeproEventSchema.validate(events[cols], lazy=True) 
        return cast(DataFrame[BeproEventSchema], obj)

    def sequences(self, game_id: int) -> DataFrame[BeproSequenceSchema]:
        """
            Return a dataframe with the sequences of a game.
        """

        cols = [
            "game_id", 
            "period_id", 
            "team_id", 
            "start_time", 
            "end_time", 
            "event_ids"
        ]
        
        path = self.get_path({"game_id": game_id})
        seqs = pd.DataFrame(_localloadjson(Path(path / "match" / f"{game_id}" / "sequence_data.json"))["result"])
        seqs["game_id"] = game_id
        seqs["period_id"] = seqs["event_period"].map(self.PERIOD_DICT)

        obj = BeproSequenceSchema.validate(seqs[cols])
        return cast(DataFrame[BeproSequenceSchema], obj)
    
    def unpack_json(self, json_object, json_keys) -> pd.DataFrame:
        """주어진 JSON 데이터를 처리하여 각 key를 개별 데이터로 확장합니다.

        Parameters
        ----------
        json_objects : JSON 객체
        json_keys : JSON 객체에서 추출할 key목록

        Returns:
            dict: 확장된 데이터
        """

        if isinstance(json_object, dict):                     # 하나의 이벤트 유형만 존재함
            unpacked_data = {key : None for key in json_keys}

            for key in json_keys:
                unpacked_data[key] = json_object.get(key, None)
        elif isinstance(json_object, list):                   # 여러가지 이벤트 유형이 존재하므로 통합시킴
            unpacked_data = {key: [] for key in json_keys}

            for obj in json_object:
                for key in json_keys:
                    unpacked_data[key].append(obj.get(key, None))
        else:
            unpacked_data = {key: None for key in json_keys}   # 기록되어있지 않은 경우(결측치는 아님)
            

        return pd.Series(unpacked_data)
