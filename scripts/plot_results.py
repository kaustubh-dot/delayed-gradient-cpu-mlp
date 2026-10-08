"""Publication figures from measured CSVs; bands show observed min/max, not CIs."""
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.2,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
COLORS = ['#272727', '#767676', '#B64342', '#9A4D8E', '#42949E', '#0F4D92', '#8BCF8B', '#C39430']
KEYS = [('sequential', 'NA'), ('synchronous','0')] + [('delayed',str(t)) for t in [0,1,2,4,8,16]]
LABELS = ['Sequential', 'Sync (0,1)'] + [f'Delay {t}' for t in [0,1,2,4,8,16]]

def read(path):
    with (ROOT/path).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def key(r):
    return (r['configuration'], 'NA' if r['configuration']=='sequential' else r['tau_max'])

def save(fig,name):
    for suffix in ['pdf','svg','png']:
        fig.savefig(FIG/(name+'.'+suffix),dpi=300,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def learning(rows,name,fresh=False):
    fig, axes=plt.subplots(1,2,figsize=(7.25,3.25))
    groups=defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    for k,label,color in zip(KEYS,LABELS,COLORS):
        if k not in groups:continue
        for ax,metric in zip(axes,['test_accuracy_percent','test_loss']):
            by_epoch=[[float(r[metric]) for r in groups[k] if int(r['epoch'])==e] for e in range(1,6)]
            vals=np.array([mean(v) for v in by_epoch]);low=[min(v) for v in by_epoch];high=[max(v) for v in by_epoch]
            ax.plot(range(1,6),vals,label=label,color=color,marker='o',markersize=3,lw=1.6,linestyle='--' if k[0]=='synchronous' else '-')
            if any(a!=b for a,b in zip(low,high)):ax.fill_between(range(1,6),low,high,color=color,alpha=.1)
            ax.set_xticks(range(1,6));ax.set_xlabel('Epoch')
    min_acc=min(float(r['test_accuracy_percent']) for r in rows)
    max_loss=max(float(r['test_loss']) for r in rows)
    axes[0].axhline(90,color='#aaaaaa',lw=.8,ls=':');axes[0].set_ylabel('Test accuracy (%)');axes[0].set_ylim(min(78,min_acc-2),95)
    axes[1].set_ylabel('Test cross-entropy');axes[1].set_ylim(.2,max(.65,max_loss*1.07))
    axes[0].set_title('(a) Accuracy');axes[1].set_title('(b) Loss')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.5,-.015))
    fig.tight_layout(pad=1,rect=[0,.16,1,1]);save(fig,name)

def comparison(rows,name):
    groups=defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    keys=[k for k in KEYS if k in groups];labels=[LABELS[KEYS.index(k)] for k in keys];colors=[COLORS[KEYS.index(k)] for k in keys]
    fig,axes=plt.subplots(1,3,figsize=(7.25,2.6))
    for ax,metric,title in zip(axes,['total_training_seconds','hidden_update_acceptance_percent','final_test_accuracy_percent'],['(a) Training time (s)','(b) Hidden acceptance (%)','(c) Final accuracy (%)']):
        for i,k in enumerate(keys):
            values=[float(r[metric]) for r in groups[k]]
            ax.bar(i,mean(values),color=colors[i],edgecolor='black',linewidth=.5)
            if len(values)>1:ax.errorbar(i,mean(values),yerr=[[mean(values)-min(values)],[max(values)-mean(values)]],color='black',capsize=2,lw=.8)
        ax.set_title(title,fontsize=9);ax.set_xticks(range(len(keys)),labels,rotation=65,ha='right',fontsize=7)
        ax.set_ylim(0, max(float(r[metric]) for r in rows)*1.15)
        if metric=='final_test_accuracy_percent':ax.set_ylim(80,95);ax.axhline(90,ls=':',color='#aaaaaa',lw=.8)
    fig.tight_layout(pad=1);save(fig,name)

def targets(old,fresh):
    fig,axes=plt.subplots(1,2,figsize=(7.25,2.9))
    for ax,rows,title in zip(axes,[old,fresh],['(a) Supplied measurements','(b) Fresh reconstruction']):
        labels=[]
        for i,t in enumerate([0,1,2,4,8,16] if rows is fresh else [0,1,2,4,8]):
            subset=[r for r in rows if r['configuration']=='delayed' and r['tau_max']==str(t)]
            vals=[float(r['time_to_90_percent_seconds']) for r in subset if r['time_to_90_percent_seconds'] not in ['NA','']]
            labels.append(str(t))
            if vals:
                ax.scatter([i]*len(vals),vals,color=COLORS[2+i],s=25,marker='o')
                ax.plot([i-.18,i+.18],[mean(vals)]*2,color='black',lw=1)
            else:ax.text(i,.03,'NR',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.set_xticks(range(len(labels)),labels);ax.set_xlabel('Delay bound');ax.set_ylabel('First >=90% checkpoint (s)');ax.set_title(title,fontsize=10);ax.set_ylim(bottom=0)
    fig.tight_layout(pad=1);save(fig,'time_to_target')

def timing(rows,name):
    fig,axes=plt.subplots(1,2,figsize=(7.25,2.6))
    rows=sorted(rows,key=lambda r:r['configuration'],reverse=True)
    for ax,w in zip(axes,[0,1]):
        x=np.arange(len(rows));width=.34
        ax.bar(x-width/2,[float(r[f'w{w}_active_seconds']) for r in rows],width,color='#0F4D92',edgecolor='black',lw=.5,label='Active path')
        ax.bar(x+width/2,[float(r[f'w{w}_pop_seconds']) for r in rows],width,color='#CFCECE',edgecolor='black',lw=.5,hatch='//',label='Input pop')
        ax.set_xticks(x,[r['configuration'].capitalize() for r in rows]);ax.set_ylabel('Seconds');ax.set_title(f'Worker {w}');ax.legend(fontsize=8)
    fig.tight_layout(pad=1);save(fig,name)

def architecture():
    fig,ax=plt.subplots(figsize=(7.25,3.2));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05,rounding_size=.05',edgecolor='#272727',facecolor=color,lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    box(.1,1.15,1.75,1.2,'Coordinator\nwindow I\nepoch drain','#eeeeee')
    box(2.4,1.15,1.45,1.2,'Mixed inbox\nTrain / Grad\ncapacity 2I','#DDF3DE')
    box(4.3,1.15,1.8,1.2,'Worker 0\n784 -> 32\nsole hidden owner\ncache: index / h','#d8e8fa')
    box(7.85,1.15,1.95,1.2,'Worker 1\n32 -> 10\nsole output owner','#F6CFCB')
    ax.annotate('',xy=(2.35,1.75),xytext=(1.9,1.75),arrowprops={'arrowstyle':'->','color':'black'})
    ax.annotate('',xy=(4.25,1.75),xytext=(3.9,1.75),arrowprops={'arrowstyle':'->','color':'black'})
    ax.annotate('',xy=(7.8,2),xytext=(6.15,2),arrowprops={'arrowstyle':'->','color':'#0F4D92'})
    ax.text(6.98,2.18,'Activation queue\n(id, h, origin)',ha='center',fontsize=8)
    ax.annotate('',xy=(3.15,2.43),xytext=(8.8,2.43),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=.23','color':'#B64342'})
    ax.text(6.3,3.36,'Real gradient returns to hidden inbox',ha='center',fontsize=9,color='#B64342')
    ax.annotate('',xy=(1,.95),xytext=(5.2,.95),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.13','color':'#767676'})
    ax.text(3.3,.3,'Result queue: completion after apply or discard',ha='center',fontsize=9)
    fig.tight_layout(pad=.5);save(fig,'architecture')

def staleness():
    fig,ax=plt.subplots(figsize=(3.5,2.1));ax.set_xlim(-.6,4.6);ax.set_ylim(-.5,2.5);ax.axis('off')
    ax.plot([0,4],[1.6,1.6],color='#272727',lw=2)
    for i in range(5):
        ax.scatter(i,1.6,color='#0F4D92',s=35);ax.text(i,1.95,f'v{101+i}',ha='center',fontsize=8)
    ax.text(0,.96,'Forward\norigin = 101',ha='center',fontsize=8)
    ax.text(4,.96,'Return\ncurrent = 105',ha='center',fontsize=8)
    ax.annotate('',xy=(4,.35),xytext=(0,.35),arrowprops={'arrowstyle':'<->','color':'#B64342'})
    ax.text(2,.55,'Age = 4 accepted updates, not elapsed time',ha='center',fontsize=8)
    ax.text(2,-.16,'tau 4: accept; tau 2: discard\nOutput update already applied',ha='center',fontsize=8)
    fig.tight_layout(pad=.5);save(fig,'staleness')

def aggregates(rows):
    groups=defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    stats={}
    for k,v in groups.items():
        rec={'n':len(v)}
        for metric in ['total_training_seconds','hidden_update_acceptance_percent','final_test_accuracy_percent','time_to_90_percent_seconds']:
            vals=[float(r[metric]) for r in v if r[metric] not in ['NA','']]
            rec[metric]={'mean':mean(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None,'n_reached':len(vals)}
        stats['/'.join(k)]=rec
    return stats

def tables(old,fresh):
    for name,rows in [('supplied',old),('fresh',fresh)]:
        stats=aggregates(rows);lines=[]
        for k,label in zip(KEYS,LABELS):
            s=stats.get('/'.join(k))
            if not s:continue
            f=lambda m: s[m]['mean']
            target=f('time_to_90_percent_seconds')
            lines.append(f"{label} & {s['n']} & {f('total_training_seconds'):.3f} & {f('final_test_accuracy_percent'):.2f} & {f('hidden_update_acceptance_percent'):.2f} & "+(f'{target:.3f}' if target is not None else 'NR')+r' \\')
        header = r'\begin{tabular}{lrrrrr}\toprule' + '\n' + r'Setting & $n$ & Time (s) & Acc. (\%) & Accept (\%) & $T_{90}$ (s)\\\midrule' + '\n'
        (ROOT/'report'/f'{name}_table.tex').write_text(header+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n',encoding='utf-8')
    (ROOT/'results/derived/summary_statistics.json').write_text(json.dumps({'supplied':aggregates(old),'fresh':aggregates(fresh)},indent=2),encoding='utf-8')

def main():
    FIG.mkdir(exist_ok=True)
    old=read('results/supplied/epoch_curves.csv');new=read('results/fresh/epoch_curves.csv')
    oc=read('results/supplied/epoch_comparison.csv');nc=read('results/fresh/comparison.csv')
    learning(old,'supplied_learning');learning(new,'fresh_learning',True)
    comparison(oc,'supplied_tradeoff');comparison(nc,'fresh_tradeoff')
    targets(oc,nc);timing(read('results/derived/instrumented_comparison.csv'),'supplied_worker_timing')
    architecture();staleness();tables(oc,nc)
    print('Eight publication figures exported as PDF, SVG, and 300 dpi PNG; tables and descriptive summaries generated')

if __name__=='__main__':main()
