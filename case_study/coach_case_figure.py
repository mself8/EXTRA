"""Coach's eleven vs EventXI (events) vs EventXI + phase tracking, drawn in the style of the WWW case figure.

Case choice reads only lineups from outputs/coach_case_scan{SUF}.json (or its shard parts) (no xG, no substitutions):
the earliest case where both recommendations change the most coach starters (3-5 changes, readable),
the two recommendations differ, and every coach starter is in the candidate pool.
Substitution minutes and the realised npxG difference are drawn only after the case is fixed.
Run: python coach_case_figure.py [--pin gid:tid]
"""
import os,sys,json,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from lineup_figure import ROOT,TB,names
import viz_side as V

SUF=os.environ.get('SUF','_existing')
L,W=V.L,V.W
GRAY,RED,BLUE='#9aa3a0','#c2453c','#1f6fb2'
FS_NAME,FS_HEAD=7.0,8.0
from matplotlib import font_manager
for _f in ['NotoSans-Regular.ttf','NotoSans-Bold.ttf']:font_manager.fontManager.addfont('/usr/share/fonts/truetype/noto/'+_f)
plt.rcParams['font.family']='Noto Sans'

def rot(x,y):return W-y,x          # attack upward, left flank on the left

def pitch(ax,title):
    ax.add_patch(Rectangle((0,0),W,L,fc='#f5f8f4',ec='#9aa79a',lw=.8,zorder=0))
    ax.plot([0,W],[L/2,L/2],c='#9aa79a',lw=.7,zorder=1)
    for y0 in (0,L-16.5):ax.add_patch(Rectangle(((W-40.3)/2,y0),40.3,16.5,fc='none',ec='#9aa79a',lw=.7,zorder=1))
    ax.add_patch(plt.Circle((W/2,L/2),9.15,fill=False,ec='#9aa79a',lw=.7,zorder=1))
    ax.set_xlim(-7,W+7);ax.set_ylim(-9,L+2);ax.set_aspect('equal');ax.axis('off')
    ax.set_title(title,fontsize=FS_HEAD,pad=3.,color='#222222',fontweight='bold')

def draw(ax,XY,ids,both,NAME,col,sc,mark,mark_lbl,only=()):
    """only: players this model picks but the other model does not (black outline)."""
    mark={q:v for q,v in mark.items() if q not in both}       # substitutions shown only for dropped/added players
    MX,MY=6.,6.                                   # keep nodes and rings inside the pitch
    P={}
    for p in ids:
        if p not in XY:continue
        x,y=rot(*XY[p]);P[p]=(min(max(x,MX),W-MX),min(max(y,MY),L-MY))
    # labels: below by default; alternate above/below for close neighbours in the same line
    def lines(q):
        l=[NAME.get(int(q),'?')];sub=''
        if q not in both and sc is not None and q in sc:sub=f'{sc[q]:+.3f}'
        if q in mark:sub+=('  ' if sub else '')+mark[q]
        return l+([sub] if sub else [])
    wid={q:1.9*max(len(t) for t in lines(q))+1.5 for q in P}       # label box in metres at this size
    hgt={q:3.9*len(lines(q)) for q in P}
    def box(q,pos):
        x,y=P[q];w,h=wid[q],hgt[q]
        if pos=='below':return (x-w/2,x+w/2,y-4.6-h,y-4.6)
        if pos=='above':return (x-w/2,x+w/2,y+4.6,y+4.6+h)
        if pos=='left':return (x-2.5-w,x-2.5+0,y-4.6-h,y-4.6)          # below, extending left
        return (x+2.5,x+2.5+w,y-4.6-h,y-4.6)                          # below, extending right
    def hit(a,b):return a[0]<b[1] and b[0]<a[1] and a[2]<b[3] and b[2]<a[3]
    discs=[(x-3.2,x+3.2,y-3.2,y+3.2) for x,y in P.values()]
    placed=[];where={}
    for q in sorted(P,key=lambda q:(-hgt[q],-wid[q])):                # big labels choose first
        for pos in ('below','above','left','right'):
            bx=box(q,pos);x,y=P[q]
            own=(x-3.2,x+3.2,y-3.2,y+3.2)
            inside=bx[0]>=-6 and bx[1]<=W+6 and bx[2]>=-8 and bx[3]<=L+1
            if inside and not any(hit(bx,o) for o in placed) and not any(hit(bx,d) for d in discs if d!=own):break
        else:pos='below'
        where[q]=pos;placed.append(box(q,pos))
    for p,(x,y) in P.items():
        c=GRAY if p in both else col
        if p in mark:ax.scatter(x,y,s=420,facecolors='none',edgecolors='#7d8784' if p in both else col,lw=1.9,zorder=4)
        ax.scatter(x,y,s=200,c=c,ec='#111111' if p in only else 'white',lw=1.8 if p in only else .9,zorder=5,alpha=.97)
        lbl=NAME.get(int(p),'?');sub=''
        if p not in both and sc is not None and p in sc:sub=f'{sc[p]:+.3f}'
        if p in mark:sub+=('  ' if sub else '')+mark[p]
        if sub:lbl+='\n'+sub
        pos=where[p];above=pos=='above'
        ha={'left':'right','right':'left'}.get(pos,'center');xt=x+(-2.5 if pos=='left' else 2.5 if pos=='right' else 0)
        t=ax.text(xt,y+(4.6 if above else -4.6),lbl,ha=ha,multialignment='center',va='bottom' if above else 'top',fontsize=FS_NAME,zorder=6,linespacing=1.02,
                  color='#333333' if p in both and p not in only else '#111111',fontweight='normal' if p in both and p not in only else 'bold')
        t.set_path_effects([pe.withStroke(linewidth=1.9,foreground='white')])

def pick(rows):
    ok=[r for r in rows if r['coach_in_pool'] and r['event_phase']>=1 and 3<=min(r['coach_event'],r['coach_phase']) and max(r['coach_event'],r['coach_phase'])<=5]
    return max(ok,key=lambda r:(min(r['coach_event'],r['coach_phase'])+max(r['coach_event'],r['coach_phase']),r['event_phase'],-ok.index(r)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pin');ap.add_argument('--out',default=str(ROOT/f'figures/coach_case{SUF}'));a=ap.parse_args()
    merged=ROOT/f'outputs/coach_case_scan{SUF}.json'
    rows=json.loads(merged.read_text()) if merged.exists() else sorted((r for f in sorted((ROOT/'outputs').glob(f'coach_case_scan{SUF}.part*.json')) for r in json.loads(f.read_text())),key=lambda r:(r['date'],r['gid'],r['tid']))
    if a.pin:
        g_,t_=(int(x) for x in a.pin.split(':'));best=next(r for r in rows if r['gid']==g_ and r['tid']==t_)
    else:best=pick(rows)
    gid,tid=best['gid'],best['tid'];coach=[int(p) for p in best['coach_xi']]
    print({k:best[k] for k in ['gid','tid','date','coach_event','coach_phase','event_phase']},flush=True)
    # Outcome-side information below is drawn only, never used for the choice above.
    S=pd.read_parquet(TB/'outputs/subs.parquet');S=S[(S.game_id==gid)&(S.team_id==tid)]
    ON={int(r.player_id):float(r.tmin) for r in S.itertuples(index=False) if r.dir=='On'};OFF={int(r.player_id):float(r.tmin) for r in S.itertuples(index=False) if r.dir=='Off'}
    C=pd.read_pickle(ROOT/'outputs/lineup_cases.pkl');c=C[(C.gid==gid)&(C.tid==tid)].iloc[0]
    g=pd.read_csv(TB/'vaep/output/games.csv');gr=g[g.game_id==gid].iloc[0]
    y=None
    try:
        from nonadditive_gate import build_sides
        Rs=build_sides()[0];home=1 if int(gr.home_team_id)==tid else 0;y=float(Rs[(Rs.game_id==gid)&(Rs.home==home)].y.iloc[0])
    except Exception as e:print('npxG lookup failed',e)
    NAME=names(gid)
    shown=set(coach)|set(best['arms']['event']['xi'])|set(best['arms']['phase']['xi'])
    from collections import Counter
    dup={l for l,n in Counter(NAME.get(p) for p in shown).items() if n>1}
    if dup:                                   # same short label for different players: write the given name out
        bio=pd.read_parquet(TB/'outputs/player_bio_cur.parquet')
        EN={int(k):str(r.en) for r in bio.itertuples() for k in (r.pid,r.pid_canon) if pd.notna(k) and isinstance(r.en,str)}
        for p in shown:
            if NAME.get(p) in dup and p in EN:
                w=EN[p].split();sur=[x for x in w if x.isupper() and len(x)>1]
                if not sur or len(sur)==len(w):sur=[w[-1]]           # all-caps or no caps: the last token is the surname
                giv=[x for x in w if x not in sur]
                NAME[p]=f"{''.join(giv).capitalize() if len(giv)>1 and all(x.isupper() for x in giv) else ' '.join(x.title() for x in giv)} {sur[-1].capitalize()}".strip()
        print('disambiguated',{p:NAME[p] for p in shown if p in EN and ' ' in NAME[p] and '.' not in NAME[p]})
    TEAM={int(r.team_id):str(r.team_name) for r in pd.read_csv(TB/'vaep/output/teams.csv').itertuples(index=False)}
    XY,_=V.match_xy(gid);XY=V.spread({p:XY[p] for p in coach if p in XY},dx=24.,dy=18.,iters=600)
    CW=V.career_width()
    fig,axes=plt.subplots(1,3,figsize=(7.1,4.0))
    pitch(axes[0],"Coach's eleven\n(declared)")
    E,P=best['arms']['event'],best['arms']['phase']
    dropped=(set(coach)-set(E['xi']))|(set(coach)-set(P['xi']))
    # coach panel: red = dropped by at least one model, grey = kept by both
    draw(axes[0],XY,coach,set(coach)-dropped,NAME,RED,None,{p:f"off {OFF[p]:.0f}'" for p in coach if p in OFF},'')
    gaps={}
    for ax,arm,title in [(axes[1],'event','Ours: event data'),(axes[2],'phase','Ours: event + tracking data')]:
        d=best['arms'][arm];sc={int(k):v for k,v in d['sc'].items()}
        MP=V.spread(V.model_xy({int(k):tuple(v) for k,v in d['slot'].items()},d['gk'],CW),dx=24.,dy=18.,iters=600)
        slot={int(k):tuple(v) for k,v in d['slot'].items()}
        for q,(k,l) in slot.items():                          # a lone player in the central lane stays on the centre line
            if l==1 and sum(1 for kl in slot.values() if kl==(k,l))==1 and q in MP:MP[q]=np.array([MP[q][0],W/2])
        both=set(coach)&set(d['xi']);added={p:ON[p] for p in set(d['xi'])-set(coach) if p in ON}
        gaps[arm]=sum(sc[p] for p in d['xi'])-sum(sc[p] for p in coach if p in sc)
        other=best['arms']['phase' if arm=='event' else 'event']['xi']
        pitch(ax,f"{title}\n{11-len(both)} changes from the coach");subs={**{p:f"off {OFF[p]:.0f}'" for p in d['xi'] if p in OFF},**{p:f"on {ON[p]:.0f}'" for p in d['xi'] if p in ON}}
        draw(ax,MP,d['xi'],both,NAME,BLUE,sc,subs,'',only=set(d['xi'])-set(other))
    fc='-'.join(str(x) for x in np.asarray(c.G).sum(1) if x)
    hdr=f"{best['date']}   {TEAM.get(tid,tid)} vs {TEAM.get(int(best['oid']),best['oid'])}   ·   declared {fc}"+(f"   ·   realised npxG difference {y:+.2f}" if y is not None else '')
    fig.text(.5,.975,hdr,ha='center',va='bottom',fontsize=FS_HEAD,color='#111111')
    lg=[Line2D([],[],marker='o',ls='',ms=5,mfc=GRAY,mec='white',label='kept from the coach\'s eleven'),
        Line2D([],[],marker='o',ls='',ms=5,mfc=RED,mec='white',label='dropped by a model'),
        Line2D([],[],marker='o',ls='',ms=5,mfc=BLUE,mec='white',label='added by the model (player score)'),
        Line2D([],[],marker='o',ls='',ms=5,mfc='white',mec='#111111',mew=1.5,label='only in this model'),
        Line2D([],[],marker='o',ls='',ms=8,mfc='none',mec='#555555',label='substituted on/off during the match')]
    fig.legend(handles=lg,loc='lower center',ncol=5,frameon=False,columnspacing=1.0,handletextpad=.3,fontsize=FS_HEAD-.6,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(wspace=.04,left=.005,right=.995,top=.93,bottom=.05)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(f'{a.out}.{ext}',dpi=400,bbox_inches='tight')
    meta=dict(case={k:best[k] for k in ['gid','tid','oid','date','coach_event','coach_phase','event_phase']},score_gap=gaps,on=ON,off=OFF,npxg_diff=y,
              selection='Lineups only: coach_in_pool, event!=phase, 3-5 changes vs coach for both models, most changes first, earliest date; substitutions and npxG drawn after selection.')
    open(f'{a.out}.json','w').write(json.dumps(meta,indent=1,default=str));print('saved',a.out,meta['score_gap'],flush=True)

if __name__=='__main__':main()
