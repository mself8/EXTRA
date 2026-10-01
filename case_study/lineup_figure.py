"""Outcome-blind illustrative three-arm XI comparison with full set objective."""
from pathlib import Path
from collections import Counter,defaultdict
import sys,os,json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
ROOT=Path(__file__).resolve().parents[1];TB=ROOT   # repository root (EventXI code + outputs)
sys.path[:0]=[str(TB/'gnn'),str(TB/'experiments')]
os.environ.update(HOLDOUT='2025',ONBALL_FILE='onball_gk_resid_merged_defresp_ref.parquet',DROPFAM='zv,zn',FCAP='4',FKAP='1',OMP_NUM_THREADS='2')

class NSet:
    """Exact float64 inference for the trained non-cross set block."""
    def __init__(self,sd):
        g=lambda k:sd[k].numpy().astype(float);self.h=16;self.nh=4
        def attention(prefix):return dict(Wi=g(prefix+'.in_proj_weight'),bi=g(prefix+'.in_proj_bias'),Wo=g(prefix+'.out_proj.weight'),bo=g(prefix+'.out_proj.bias'))
        self.sab=[]
        for i in range(1):
            p=f'sabs.{i}';z=attention(p+'.mha');z.update(W1=g(p+'.ff.0.weight'),b1=g(p+'.ff.0.bias'),W2=g(p+'.ff.3.weight'),b2=g(p+'.ff.3.bias'),n1w=g(p+'.n1.weight'),n1b=g(p+'.n1.bias'),n2w=g(p+'.n2.weight'),n2b=g(p+'.n2.bias'));self.sab.append(z)
        self.pma=attention('pma.mha');self.pma.update(seed=g('pma.seed')[0,0],nw=g('pma.n.weight'),nb=g('pma.n.bias'))
        self.w1,self.b1,self.w2,self.b2=[g(k) for k in ['g.1.weight','g.1.bias','g.3.weight','g.3.bias']]
    @staticmethod
    def ln(x,w,b):return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)*w+b
    def attention(self,Q,KV,p):
        h=self.h;d=h//self.nh;wi=p['Wi'];bi=p['bi'];q=Q@wi[:h].T+bi[:h];k=KV@wi[h:2*h].T+bi[h:2*h];v=KV@wi[2*h:].T+bi[2*h:];out=np.zeros_like(q)
        for j in range(self.nh):
            sl=slice(j*d,(j+1)*d);a=q[:,sl]@k[:,sl].T/np.sqrt(d);a=np.exp(a-a.max(1,keepdims=True));a/=a.sum(1,keepdims=True);out[:,sl]=a@v[:,sl]
        return out@p['Wo'].T+p['bo']
    def __call__(self,X):
        for p in self.sab:
            X=self.ln(X+self.attention(X,X,p),p['n1w'],p['n1b']);X=self.ln(X+np.maximum(X@p['W1'].T+p['b1'],0)@p['W2'].T+p['b2'],p['n2w'],p['n2b'])
        S=self.pma['seed'][None];z=self.ln(S+self.attention(S,X,self.pma),self.pma['nw'],self.pma['nb'])[0]
        return float((np.maximum(z@self.w1.T+self.b1,0)@self.w2.T+self.b2).item())

def names(game_id=None):
    b=pd.read_parquet(TB/'outputs/player_bio_cur.parquet');out={}
    p=pd.read_csv(TB/'vaep/output/players.csv',usecols=['game_id','player_id','player_name'])
    if game_id is not None:p=p[p.game_id==game_id]
    p['player_name']=p.player_name.astype(str).str.strip()
    supplied=p.dropna().groupby('player_id').player_name.agg(lambda x:x.mode().iloc[0]).to_dict()
    families={'김':['kim'],'이':['lee','yi','i'],'박':['park','pak'],'최':['choi','choe'],'정':['jeong','jung','chung'],'강':['kang','gang'],'조':['cho','jo'],'윤':['yoon','yun'],'장':['jang'],'임':['lim','im'],'한':['han'],'오':['oh','o'],'서':['seo','suh'],'신':['shin'],'권':['kwon','gwon'],'황':['hwang'],'안':['ahn','an'],'송':['song'],'류':['ryu','yu'],'홍':['hong'],'전':['jeon','chun'],'고':['ko','go'],'문':['moon','mun'],'양':['yang'],'배':['bae'],'허':['heo','huh'],'백':['baek'],'남':['nam'],'노':['noh','roh'],'하':['ha'],'손':['son'],'심':['sim','shim'],'변':['byeon','byun'],'우':['woo'],'구':['koo','ku'],'유':['yoo','yu','you'],'차':['cha'],'지':['ji'],'설':['seol'],'성':['sung','seong']}
    for r in b.itertuples():
        if not isinstance(r.en,str) or not r.en.strip():continue
        words=r.en.strip().split();label=None
        if r.foreign==0:
            candidates=families.get(str(r.ko).strip()[:1],[])
            for full in [r.en,supplied.get(int(r.pid_canon),''),supplied.get(int(r.pid),'')]:
                tokens=str(full).split();sur=[i for i,w in enumerate(tokens) if w.lower().strip('.-') in candidates]
                if not sur:continue
                i=sur[-1];given=[w for j,w in enumerate(tokens) if j!=i]
                if given:
                    surname=tokens[i].capitalize();label=f'{given[0][0].upper()}. {surname}';break
        else:
            short=supplied.get(int(r.pid_canon),supplied.get(int(r.pid),''))
            if short and len(short.split())==1 and short.isascii():label=short.capitalize()
            elif short and len(short.split())==2 and short.isascii():
                tokens=short.split();label=f'{tokens[0][0].upper()}. {tokens[-1].capitalize()}'
        if label is None:
            sur=[w for w in words if w.isupper() and len(w)>1]
            last=sur[-1] if sur and len(sur)!=len(words) else words[-1]
            given=[w for w in words if w not in sur] if sur and len(sur)!=len(words) else words
            label=words[0].capitalize() if len(words)==1 else f'{given[0][0].upper()}. {last.capitalize()}'
        for key in [r.pid,r.pid_canon]:
            if pd.notna(key):out[int(key)]=label
    return out

def draw_case(case,suffix,horizontal=False):
    team=pd.read_csv(TB/'vaep/output/teams.csv').set_index('team_id').team_name.to_dict();name=names(case['gid'])
    union=set().union(*(set(case['arms'][a]['xi']) for a in case['arms']));counts=Counter(name.get(p,str(p)) for p in union)
    pl=pd.read_csv(TB/'vaep/output/players.csv');jersey=pl[(pl.game_id==case['gid'])&(pl.team_id==case['tid'])].set_index('player_id').jersey_number.to_dict()
    for p in union:
        if counts[name.get(p,str(p))]>1:name[p]=name.get(p,str(p))+f' #{int(jersey[p])}'
    common=set.intersection(*(set(d['xi']) for d in case['arms'].values()));colors={'event':'#1976b3','pooled':'#298c7e','phase':'#c76833'}
    titles={'event':'Events','pooled':'Events + tracking','phase':'Events + phase tracking'}
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.3) if horizontal else (7.1,4.25));gray='#929d99'
    for ax,(arm,d) in zip(axes,case['arms'].items()):
        edge='#a4b0a5'
        if horizontal:
            ax.add_patch(Rectangle((0,0),105,68,fc='#f5f8f4',ec=edge,lw=.8));ax.plot([52.5,52.5],[0,68],c=edge,lw=.7);ax.add_patch(Circle((52.5,34),9.15,fill=False,ec=edge,lw=.7))
            for x in [0,88.5]:ax.add_patch(Rectangle((x,13.85),16.5,40.3,fill=False,ec=edge,lw=.7))
        else:
            ax.add_patch(Rectangle((0,0),68,105,fc='#f5f8f4',ec=edge,lw=.8));ax.plot([0,68],[52.5,52.5],c=edge,lw=.7);ax.add_patch(Circle((34,52.5),9.15,fill=False,ec=edge,lw=.7))
            for y in [0,88.5]:ax.add_patch(Rectangle((13.85,y),40.3,16.5,fill=False,ec=edge,lw=.7))
        xy={d['gk']:(34,7)};bycell=defaultdict(list)
        for p,cell in d['slot'].items():bycell[tuple(cell)].append(int(p))
        levels=[.2,.35,.5,.65,.85]
        for (k,l),ps in bycell.items():
            ps=sorted(ps);x0=[10,34,58][l]
            separation=20 if len(ps)>=3 else 17
            for i,p in enumerate(ps):xy[p]=(x0+(i-(len(ps)-1)/2)*separation,levels[k]*105)
        # Keep roles clear, then gently separate nearby labels within a line.
        for p in xy:
            x,y=xy[p];xy[p]=(float(np.clip(x,5,63)),y)
        if horizontal:xy={p:(y,68-x) for p,(x,y) in xy.items()}
        for p in d['xi']:
            x,y=xy[p];c=gray if p in common else colors[arm];ax.scatter(x,y,s=62,facecolors=c,edgecolors='white',linewidths=.6,zorder=4)
            label_y=y-(13 if horizontal and p==d['gk'] else 3.7)
            text=ax.text(x,label_y,name.get(p,str(p)),ha='center',va='top',fontsize=9 if horizontal else 6.6,color='#424b47' if p in common else '#182521',fontweight='normal' if p in common else 'bold',zorder=5)
            text.set_path_effects([pe.withStroke(linewidth=1.5,foreground='white')])
        grid=np.array(d['G']);counts=grid.sum(1).tolist()
        # Wingbacks and central midfielders are separate model depth rows;
        # combine them when displaying a conventional three-back formation.
        if counts[0]==3 and counts[1]==2 and grid[1,1]==0 and counts[2]>=2 and grid[2,0]+grid[2,2]==0:
            counts=[counts[0],counts[1]+counts[2],counts[3],counts[4]]
        formation='-'.join(str(int(n)) for n in counts if n)
        ax.set_title(titles[arm]+'\n'+formation,fontsize=10.5 if horizontal else 8.4,pad=6);ax.set_xlim((-5,110) if horizontal else (-5,73));ax.set_ylim((-10,73) if horizontal else (-10,107));ax.set_aspect('equal');ax.axis('off')
    fig.suptitle(f"{case['date']}  |  {team[case['tid']]} vs {team[case['oid']]}",fontsize=11.5 if horizontal else 9.5,y=.98)
    handles=[Line2D([],[],marker='o',ls='',mfc=gray,mec='white',label='Selected by all three models'),Line2D([],[],marker='o',ls='',mfc='#394b53',mec='white',label='Selections differ across models')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=9 if horizontal else 7)
    a,b,c=case['pairwise_changes'];arrow='→' if horizontal else '↑';fig.text(.5,.025,f'Player changes: events–tracking {a}; events–phase {b}; tracking–phase {c}.   Attack {arrow}',ha='center',fontsize=9 if horizontal else 7)
    fig.subplots_adjust(wspace=.035,top=.78 if horizontal else .84,bottom=.12,left=.02,right=.98)
    orientation='_horizontal' if horizontal else ''
    (ROOT/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'figures/lineup_comparison{suffix}{orientation}.{ext}',dpi=400,bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--existing',action='store_true');args=ap.parse_args();suffix='_existing' if args.existing else ''
    data={a:pd.read_pickle(ROOT/f'outputs/lineup_scores_{a}{suffix}.pkl') for a in ['event','pooled','phase']}
    lookup={a:{(r['gid'],r['tid']):r for r in d['rows']} for a,d in data.items()};models={a:[NSet(sd) for sd in d['states']] for a,d in data.items()}
    C=pd.read_pickle(ROOT/'outputs/lineup_cases.pkl');keys=set.intersection(*(set(q) for q in lookup.values()));C=C[[tuple(k) in keys for k in zip(C.gid,C.tid)]].copy()
    # Each callback uses its own seed's embedding with that seed's set weights.
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
    B=rb.build();checked=[];chosen=None
    # Chronological search and membership criterion, never observed xG or substitutions.
    for rr in C.sort_values(['date','gid','tid']).itertuples(index=False):
        solved={};sets=[]
        common_pool=set.intersection(*(set(lookup[a][(rr.gid,rr.tid)]['pool']) for a in data))
        for arm in data:
            scores=lookup[arm][(rr.gid,rr.tid)]['sc'];sc={p:scores[p] for p in sorted(common_pool)}
            r=rr._replace(sc=sc,pool=tuple(sorted(common_pool)));r=type('Case',(),{**r._asdict(),'arm':arm})()
            TAU={p:B['tau'](p,B['gpos'][r.gid]) for p in sc}
            preliminary=sorted([(B['solve'](r,G,TAU,0,sc)[2],i) for i,G in enumerate(B['LIB'])],reverse=True)[:6]
            options=[(*B['solve'](r,B['LIB'][i],TAU,.01,sc),B['LIB'][i]) for _,i in preliminary]
            xi,slot,v,G=max(options,key=lambda x:x[2]);gk=next(p for p in xi if p in r.gks)
            assert len(xi)==11 and len(set(xi))==11 and len(slot)==10
            # Reject solutions buying an ineligible position or foreign-cap overflow.
            if any(rb.qpen(r.qmap[p][k])>=1e3 for p,(k,l) in slot.items()) or sum(p in B['FRN'] for p in xi)>4:break
            solved[arm]=dict(xi=[int(p) for p in xi],slot={int(p):[int(k),int(l)] for p,(k,l) in slot.items()},G=G.tolist(),gk=int(gk),objective=float(v));sets.append(set(xi))
        if len(solved)!=3:continue
        changes=[11-len(sets[i]&sets[j]) for i,j in [(0,1),(0,2),(1,2)]];nc=len(set.intersection(*sets))
        checked.append(dict(gid=int(rr.gid),tid=int(rr.tid),pairwise_changes=changes,common_players=nc));print(checked[-1],flush=True)
        if min(changes)>=1 and max(changes)<=4 and nc>=7:
            chosen=dict(gid=int(rr.gid),tid=int(rr.tid),oid=int(rr.oid),date=rr.date,arms=solved,pairwise_changes=changes,common_players=nc);break
    if chosen is None:raise RuntimeError('No qualifying case in scored prefix; increase scoring --limit.')
    chosen.update(selection='Earliest date/game/team among scanned cases with 1–4 changes per pair and at least 7 players shared by all models; no outcome or substitution data used.',checked_cases=checked,model_checkpoints={a:d['checkpoint'] for a,d in data.items()},optimizer=dict(beta=.05,beta_forward=.05,lambda_role=.01,topk_formations=6,seeds=5,set_objective=True,test_rescaling=False),historical_position_profile='Exponentially weighted past starts, half-life 10; lane prior and role templates fit before 2025.')
    (ROOT/f'outputs/lineup_case{suffix}.json').write_text(json.dumps(chosen,indent=2));draw_case(chosen,suffix);draw_case(chosen,suffix,horizontal=True)
    caption=f"Three recommended lineups for one match ({chosen['date']}). Grey players appear in all three selections; coloured players differ. All models use identical squads and constraints. The case was selected without examining match outcomes."
    (ROOT/f'figures/lineup_caption{suffix}.txt').write_text(caption)

if __name__=='__main__':main()
