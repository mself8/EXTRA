"""Solve event and phase recommendations for every scored 2025 case and compare with the coach's XI.

Selection uses only lineups (no xG, no substitutions): it is the input to choose an illustrative case.
Run in shards from the repository root: SUF=_h24 python case_study/scan_coach_cases.py <shard> <nshard>  -> outputs/coach_case_scan_h24.part<shard>.json
"""
import os,sys,json
import numpy as np
import pandas as pd
from lineup_figure import ROOT,NSet

ARMS=['event','phase']
SUF=os.environ.get('SUF','_existing')    # _existing = 2021-24 training · _h24 = 2024-only training and history (abstract)

def main():
    shard,nshard=(int(sys.argv[1]),int(sys.argv[2])) if len(sys.argv)>2 else (0,1)
    data={a:pd.read_pickle(ROOT/f'outputs/lineup_scores_{a}{SUF}.pkl') for a in ARMS}
    lookup={a:{(r['gid'],r['tid']):r for r in d['rows']} for a,d in data.items()};models={a:[NSet(sd) for sd in d['states']] for a,d in data.items()}
    C=pd.read_pickle(ROOT/'outputs/lineup_cases.pkl');keys=set.intersection(*(set(q) for q in lookup.values()));C=C[[tuple(k) in keys for k in zip(C.gid,C.tid)]].copy()
    cache={}
    def setterm(r,xi):
        key=(r.arm,r.gid,r.tid,frozenset(xi))
        if key not in cache:
            V=lookup[r.arm][(r.gid,r.tid)]['vectors'];X=np.stack([V[p] for p in sorted(xi)])
            cache[key]=float(np.mean([m(X[:,s]) for s,m in enumerate(models[r.arm])]))
        return cache[key]
    os.environ['PROD']=str(ROOT/'outputs/lineup_cases.pkl')
    import role_balance_battery as rb
    rb.RECOMMENDATION_SET_TERM=setterm
    B=rb.build();out=[]

    def solve_case(rr):
        solved={}
        common_pool=set.intersection(*(set(lookup[a][(rr.gid,rr.tid)]['pool']) for a in ARMS))
        for arm in ARMS:
            scores=lookup[arm][(rr.gid,rr.tid)]['sc'];sc={p:scores[p] for p in sorted(common_pool)}
            r=rr._replace(sc=sc,pool=tuple(sorted(common_pool)),gks=tuple(p for p in rr.gks if p in sc));r=type('Case',(),{**r._asdict(),'arm':arm})()
            TAU={p:B['tau'](p,B['gpos'][r.gid]) for p in sc}
            preliminary=sorted([(B['solve'](r,G,TAU,0,sc)[2],i) for i,G in enumerate(B['LIB'])],reverse=True)[:6]
            options=[(*B['solve'](r,B['LIB'][i],TAU,.01,sc),B['LIB'][i]) for _,i in preliminary]
            xi,slot,v,G=max(options,key=lambda x:x[2]);gk=next(p for p in xi if p in r.gks)
            if any(rb.qpen(r.qmap[p][k])>=1e3 for p,(k,l) in slot.items()) or sum(p in B['FRN'] for p in xi)>4:return None,common_pool
            solved[arm]=dict(xi=[int(p) for p in xi],slot={int(p):[int(k),int(l)] for p,(k,l) in slot.items()},G=G.tolist(),gk=int(gk),objective=float(v),
                             sc={int(p):float(scores[p]) for p in set(xi)|set(int(q) for q in rr.coach_xi) if p in scores})
        return solved,common_pool

    for i_,rr in enumerate(C.sort_values(['date','gid','tid']).itertuples(index=False)):
        if i_%nshard!=shard:continue
        try:solved,pool=solve_case(rr)
        except (KeyError,StopIteration) as e:print({'gid':int(rr.gid),'tid':int(rr.tid),'skip':repr(e)},flush=True);continue
        if solved is None:continue
        coach=set(int(p) for p in rr.coach_xi);E=set(solved['event']['xi']);P=set(solved['phase']['xi'])
        row=dict(gid=int(rr.gid),tid=int(rr.tid),oid=int(rr.oid),date=str(rr.date),coach_xi=sorted(coach),arms=solved,
                 coach_event=11-len(coach&E),coach_phase=11-len(coach&P),event_phase=11-len(E&P),coach_in_pool=all(p in pool for p in coach))
        out.append(row);print({k:row[k] for k in ['gid','tid','date','coach_event','coach_phase','event_phase','coach_in_pool']},flush=True)
        (ROOT/f'outputs/coach_case_scan{SUF}.part{shard}.json').write_text(json.dumps(out,indent=1))

if __name__=='__main__':main()
