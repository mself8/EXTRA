"""Five-seed player contributions and vectors; no test-set rescaling/blending."""
from pathlib import Path
import os,sys,json,argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];TB=ROOT   # repository root (EventXI code + outputs)
sys.path[:0]=[str(TB/'gnn'),str(TB/'experiments')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['event','pooled','phase']);ap.add_argument('--existing',action='store_true');ap.add_argument('--limit',type=int,default=100);ap.add_argument('--h24',action='store_true',help='abstract setting: 2024-only training, history from 2024 on');args=ap.parse_args()
    os.environ.update(SET='1',SETSTAGE='2',H='16',LAYERS='1',LANE='none',FULLCH='1',DTDAYS='1',FAMMIX='1',ONBALL_FILE='onball_gk_resid_merged_defresp_ref.parquet',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if args.arm!='event':os.environ.update(TRK='1',TRKMODE='all' if args.arm=='pooled' else 'phase',TRKFILE=str(TB/'outputs/trk_channels.parquet'))
    import torch
    torch.set_num_threads(2)
    from player_encoder_set import SetNet,load_structured
    from player_encoder import pack,gather,DEV
    if args.h24:
        code={'event':'A','phase':'T'}[args.arm]
        checkpoint=Path(TB/f'outputs/player_encoder_set_ssn2025-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-ssac24h{code}-s5_fold1.pt')
    elif args.existing:
        code={'event':'A','pooled':'P','phase':'T'}[args.arm]
        checkpoint=TB/f'outputs/player_encoder_set_ssn-set1-cross0-H16-L1-none-full-stage2-days-warm-fam5-ssac{code}-s5_fold2.pt'
    else:raise SystemExit('choose --h24 (EXTRA / EventXI as in the abstract) or --existing (2021-24 training)')
    d=torch.load(checkpoint,weights_only=False,map_location='cpu')
    g,gpos,X,pid,t_arr,gid_arr,bounds,tidx,zidx,sidx=load_structured()
    assert np.array_equal(tidx,d['S'][0]) and len(d['nets'])==5
    nets=[]
    for state in d['nets']:
        net=SetNet(*d['S']).to(DEV);net.load_state_dict(state);net.eval();nets.append(net)
    SEA=g.set_index('game_id').season.reindex(gid_arr).to_numpy().astype(int)
    Xg=torch.as_tensor(((X-d['mu'])/d['sd']).clip(-5,5).astype(np.float32),device=DEV)
    C=pd.read_pickle(ROOT/'outputs/lineup_cases.pkl').head(args.limit)
    rows=[]
    for i,r in enumerate(C.itertuples(index=False)):
        tt=gpos[r.gid];pids=list(r.pool);idx,dt=pack([(p,tt) for p in pids],bounds,t_arr)
        if args.h24:idx[(idx>=0)&(SEA[np.clip(idx,0,None)]<2024)]=-1       # same history cut as training (HISTFROM=2024)
        valid=(idx>=0).any(1)
        assert np.all(t_arr[idx[idx>=0]]<tt)
        if not all(valid):
            pids=[p for p,k in zip(pids,valid) if k];idx=idx[valid];dt=dt[valid]
        if len(pids)<12 or not set(r.coach_xi)<=set(pids) or not set(pids)&set(r.gks):continue
        with torch.no_grad():
            xs,dts,ms=gather(Xg,torch.as_tensor(idx,device=DEV),torch.as_tensor(dt,device=DEV))
            results=[net.player(xs,dts,ms) for net in nets]
            sc=np.mean([s.cpu().numpy() for s,v in results],axis=0)
            V=np.stack([v.cpu().numpy() for s,v in results],axis=1)
        rows.append(dict(gid=r.gid,tid=r.tid,sc=dict(zip(pids,sc.astype(float))),vectors=dict(zip(pids,V)),pool=pids))
        if (i+1)%20==0:print(f'{args.arm}: {i+1}/{len(C)} cases',flush=True)
    suffix='_h24' if args.h24 else '_existing' if args.existing else ''
    pd.to_pickle(dict(rows=rows,states=d['nets'],checkpoint=str(checkpoint)),ROOT/f'outputs/lineup_scores_{args.arm}{suffix}.pkl')
    print(f'{args.arm}: saved {len(rows)} cases',flush=True)

if __name__=='__main__':main()
