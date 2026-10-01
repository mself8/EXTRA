"""Common announced squads, past-only positions, frozen pre-2025 lane prior."""
from pathlib import Path
from collections import defaultdict
import sys, os
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];TB=ROOT   # repository root (EventXI code + outputs)
sys.path[:0]=[str(TB/'gnn'),str(TB/'experiments')]
os.environ['ONBALL_FILE']='onball_gk_resid_merged_defresp_ref.parquet'
from gap_lane import declared_grid
from gap_soft import lane_logp

def main():
    g=pd.read_csv(TB/'vaep/output/games.csv',parse_dates=['game_date']).sort_values(['game_date','game_id']).reset_index(drop=True)
    p=pd.read_csv(TB/'vaep/output/players.csv').drop_duplicates(['game_id','team_id','player_id'])
    CELL,PER,HIST=declared_grid();gpos=dict(zip(g.game_id,range(len(g))));season=dict(zip(g.game_id,g.season))
    poshist=defaultdict(list)
    for (gid,tid,pid),(k,l) in PER.items():poshist[gid].append((pid,k,l))
    squads=p.groupby(['game_id','team_id']);qh={};lh={};pl={};rows=[]
    def lag(pid,date):
        h=qh.get(pid,[])
        if not h:return None,None
        assert max(x[0] for x in h)<date
        weights=.5**(np.arange(len(h)-1,-1,-1)/10)
        q=np.bincount([x[1] for x in h],weights=weights,minlength=5)+.01
        return q/q.sum(),int(np.argmax(np.bincount([x[2] for x in h],weights=weights,minlength=3)))
    # Same-date fixtures are queried before histories update.
    for date,day in g.groupby('game_date',sort=True):
        for r in day.itertuples():
            for tid,oid in [(r.home_team_id,r.away_team_id),(r.away_team_id,r.home_team_id)]:
                key=(r.game_id,tid)
                if key not in squads.groups:continue
                sq=squads.get_group(key);pool=sorted(sq.player_id.astype(int).tolist())
                for pid in pool:
                    q,l=lag(pid,date)
                    if l is not None:pl[(r.game_id,pid)]=l
                if r.season!=2025 or key not in CELL:continue
                coach=sq.loc[sq.is_starter==1,'player_id'].astype(int).tolist()
                gks=sq.loc[sq.starting_position_name=='GK','player_id'].astype(int).tolist()
                if len(coach)!=11 or not gks:continue
                qmap={};lane={}
                for pid in pool:
                    if pid in gks:continue
                    q,l=lag(pid,date)
                    if q is None:continue
                    qmap[pid]=q;lane[pid]=l
                # Exclude players without an observed past position; same for all arms.
                pool=sorted(set(qmap)|set(gks))
                if len(pool)<12 or not set(coach)<=set(pool):continue
                rows.append(dict(gid=int(r.game_id),tid=int(tid),oid=int(oid),season=2025,date=str(date.date()),sc={pid:0. for pid in pool},qmap=qmap,lane=lane,G=CELL[key],gks=tuple(gks),pool=tuple(pool),coach_xi=coach))
        for r in day.itertuples():
            for pid,k,l in poshist.get(r.game_id,[]):qh.setdefault(pid,[]).append((date,k,l))
    fitper={key:v for key,v in PER.items() if season.get(key[0],9999)<2025}
    LP=lane_logp(fitper,pl)
    C=pd.DataFrame(rows);C['LPk']=[LP]*len(C)
    C.to_pickle(ROOT/'outputs/lineup_cases.pkl')
    print(f'Prepared {len(C)} common 2025 squad cases; lane prior fitted on {len(fitper)} pre-2025 starter slots',flush=True)

if __name__=='__main__':main()
