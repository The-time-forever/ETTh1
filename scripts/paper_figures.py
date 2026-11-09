"""Regenerate publication assets from saved metrics, without model training."""
from pathlib import Path
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/paper'
FIG = ROOT / 'paper/figures'
TAB = ROOT / 'paper/tables'
for p in (OUT, FIG, TAB): p.mkdir(parents=True, exist_ok=True)
SOURCES = set()
C = ['#0072B2', '#D55E00', '#009E73', '#8E6AA8', '#777777']
plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.unicode_minus':False, 'pdf.fonttype':42, 'savefig.facecolor':'white'})

def read(name):
    p = ROOT / 'outputs' / name
    SOURCES.add(p)
    return pd.read_csv(p)

def row(folder, model):
    d = read(f'{folder}/test_metrics.csv')
    r = d[d.model == model]
    assert len(r) == 1, (folder, model)
    return r.iloc[0]

def save(fig, name):
    for ext in ('pdf','png'):
        target = OUT / f'{name}.{ext}'
        fig.savefig(target, dpi=180, bbox_inches='tight', pad_inches=.12)
        shutil.copy2(target, FIG / target.name)
    plt.close(fig)

def box(ax, x, y, w, h, label, color='#EDF4FA', size=10):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=0.015',facecolor=color,edgecolor='#667788',lw=.9))
    ax.text(x,y,label,ha='center',va='center',fontsize=size)

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#526271',lw=1.1))

def canvas(w=10,h=4):
    f,a=plt.subplots(figsize=(w,h)); a.set(xlim=(-.035,1.035),ylim=(0,1)); a.axis('off')
    return f,a

def pipeline():
    f,a=canvas(10,4.1)
    xs=[.095,.35,.61,.89]
    labels=['历史油温窗口\n$L=96$','减去局部水平\n$X^c=X-\\ell_t$','训练段标准化\n多输出 Ridge','加回局部水平\n输出 $H$ 步油温']
    for x,s in zip(xs,labels): box(a,x,.62,.18,.21,s)
    for i in range(3): arrow(a,(xs[i]+.105,.62),(xs[i+1]-.105,.62))
    box(a,.35,.20,.23,.19,'最近 24 h 均值 $\\ell_t$', '#FFF1DE')
    arrow(a,(.095,.50),(.095,.20)); arrow(a,(.095,.20),(.22,.20))
    arrow(a,(.35,.31),(.35,.50)); arrow(a,(.48,.20),(.89,.20)); arrow(a,(.89,.20),(.89,.50))
    a.text(.66,.27,'同一输入窗口的水平',ha='center',fontsize=9,color='#526271')
    a.text(.5,.94,'预测阶段：所有统计量均来自已观测历史',ha='center',weight='bold')
    a.text(.5,.005,'训练阶段：目标同减 $\\ell_t$，用训练段拟合尺度；验证集选择正则化强度。',ha='center',fontsize=9)
    save(f,'fig1_pipeline')

def data_split():
    p=ROOT/'ETTh1.csv'; SOURCES.add(p)
    d=pd.read_csv(p,parse_dates=['date'])
    f,axes=plt.subplots(2,1,figsize=(10,5),gridspec_kw={'height_ratios':[2.2,1]},layout='constrained')
    a=axes[0]
    for start,end,color,name in [(0,8640,C[0],'训练'),(8640,11520,C[2],'验证'),(11520,14400,C[1],'测试'),(14400,len(d),C[4],'未使用尾部')]:
        a.axvspan(d.date.iloc[start],d.date.iloc[end-1],color=color,alpha=.09)
        a.text(d.date.iloc[(start+end)//2],47,name,ha='center',color=color)
    a.plot(d.date,d.OT,color='#34495E',lw=.5)
    a.set(ylabel='油温（℃）',ylim=(-7,52),title='(a) 完整序列与固定时间划分')
    a=axes[1]; a.axis('off')
    a.text(.02,.88,'(b) 单个滚动窗口：预测目标完整落在所属数据段内',transform=a.transAxes)
    for x,w,color,s in [(.30,.39,C[0],'历史 $L$ 小时'),(.69,.39,C[1],'未来 $H$ 小时')]:
        a.add_patch(FancyBboxPatch((x-w/2,.26),w,.28,boxstyle='square,pad=0',facecolor=color,alpha=.15,transform=a.transAxes))
        a.text(x,.40,s,ha='center',transform=a.transAxes)
    a.annotate('预测起点 $t$',xy=(.495,.26),xytext=(.495,.02),ha='center',arrowprops={'arrowstyle':'->'},xycoords='axes fraction')
    save(f,'fig0_split')

def architectures():
    f,axes=plt.subplots(2,1,figsize=(10,6.3)); f.subplots_adjust(hspace=.12)
    for a in axes: a.set(xlim=(-.035,1.035),ylim=(0,1)); a.axis('off')
    a=axes[0]; a.text(.01,.94,'(a) PatchTST：片段嵌入、序列编码与直接预测',weight='bold')
    xs=[.10,.36,.63,.90]
    labels=['标准化残差\n$B\\times L$','片段长度 16 / 步长 8\n$B\\times N\\times16$','线性嵌入 + 位置编码\n编码器 $B\\times N\\times d$','展平 + 线性头\n$B\\times H$']
    for x,s in zip(xs,labels): box(a,x,.61,.18,.28,s,size=9)
    for i in range(3): arrow(a,(xs[i]+.105,.61),(xs[i+1]-.105,.61))
    box(a,.55,.16,.75,.22,'编码器：LayerNorm → 4 头自注意力 → 残差相加\nLayerNorm → 前馈网络（GELU）→ 残差相加',size=9)
    arrow(a,(.63,.45),(.63,.29))
    a=axes[1]; a.text(.01,.94,'(b) SegRNN：7 个历史片段与 2 个并行预测片段',weight='bold')
    for x,s in [(.11,'336 h 历史\n7 × 48 h'),(.37,'片段嵌入 + ReLU\n7 × 64'),(.63,'共享 GRU 编码\n最终状态 $h$')]: box(a,x,.64,.19,.25,s,size=9)
    arrow(a,(.22,.64),(.26,.64)); arrow(a,(.48,.64),(.52,.64))
    for y,j in [(.68,1),(.20,2)]: box(a,.88,y,.18,.23,f'输入 $p_{j}$；初态 $h$\n共享 GRU\n线性头 → 48 h',color='#FFF1DE',size=8.5)
    arrow(a,(.74,.64),(.775,.68)); arrow(a,(.63,.50),(.63,.20)); arrow(a,(.63,.20),(.775,.20))
    a.text(.14,.17,'两个输出片段拼接为 96 h；\n反标准化后加回参考水平。',fontsize=9)
    save(f,'fig5_architectures')

def main_results():
    specs=[('ridge_level','Anchored Ridge','锚定 Ridge'),('dlinear','Anchored DLinear (3-seed mean)','锚定 DLinear'),('attn_ablation','Base PatchTST (3-seed mean)','PatchTST（16维）'),('patchtst','Anchored PatchTST (3-seed mean)','PatchTST（64维）'),('mlp','Anchored MLP (3-seed mean)','锚定 MLP'),('gru','Anchored GRU (3-seed mean)','锚定 GRU'),('attn_ablation','No-Attention PatchTST (3-seed mean)','片段混合 MLP'),('lstm','Anchored LSTM (3-seed mean)','锚定 LSTM')]
    vals=[row(d,m) for d,m,_ in specs]
    f,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    a=axes[0]; y=np.arange(len(vals))
    a.errorbar([r.mae for r in vals],y,xerr=[r.get('mae_std',0) if pd.notna(r.get('mae_std',0)) else 0 for r in vals],fmt='o',capsize=3,color=C[0])
    a.axvline(1.8198106968,ls='--',color=C[4],lw=1,label='24 h 均值')
    a.set_yticks(y,[s[2] for s in specs]); a.invert_yaxis()
    a.set(xlim=(1.60,1.84),xlabel='测试 MAE（℃）',title='(a) 三种子指标均值 ± 标准差')
    a.text(1.817,-.45,'24 h 均值',ha='right',fontsize=8,color=C[4])
    a=axes[1]; families=['Ridge','MLP','GRU','LSTM','DLinear','PatchTST']
    plain=[row(d,m) for d,m in [('ridge_level','Plain Ridge'),('mlp','Plain MLP (3-seed mean)'),('gru','Plain GRU (3-seed mean)'),('lstm','Plain LSTM (3-seed mean)'),('dlinear','Plain DLinear (3-seed mean)'),('patchtst','Plain PatchTST (3-seed mean)')]]
    anchored=[vals[i] for i in [0,4,5,7,1,3]]
    x=np.arange(6)
    a.bar(x-.18,[r.bias for r in plain],.36,color=C[4],label='绝对值预测')
    a.bar(x+.18,[r.bias for r in anchored],.36,color=C[0],label='24 h 均值锚定')
    a.axhline(0,color='#333333',lw=.7); a.set_xticks(x,families,rotation=35,ha='right')
    a.set(ylabel='平均预测偏差（℃）',title='(b) 同一结构的水平处理对照')
    a.legend(frameon=False,fontsize=8)
    save(f,'fig2_main')

def horizons():
    d=read('h720/fixed_origin_skill.csv'); step=read('h720/per_step_metrics.csv')
    f,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    a=axes[0]
    for col,label,color in [('ridge_mae','锚定 Ridge',C[0]),('b2b_mae','96 h 均值',C[2])]: a.plot(d.horizon,d[col],'o-',label=label,color=color)
    a.set(xticks=d.horizon,xlabel='预测长度 H（h）',ylabel='测试 MAE（℃）',title='(a) 固定 2,161 个共同预测起点'); a.legend(frameon=False)
    a=axes[1]
    for col,label,color in [('Anchored Ridge_mae','锚定 Ridge',C[0]),('B2b_mae','96 h 均值',C[2]),('B1_mae','最后值',C[4])]: a.plot(step.step,step[col],label=label,color=color,lw=1)
    a.set(xlabel='预测步 h（h）',ylabel='逐步 MAE（℃）',title='(b) 720 h 模型的逐预测步误差'); a.legend(frameon=False,fontsize=8)
    save(f,'fig3_horizon')

def ablations():
    pairs=[('负载输入',['仅油温','七通道'],[row('ridge_ms','S_control (per-variable)'),row('ridge_ms','MS (7 channels)')]),
        ('历史长度',['96 h','336 h'],[row('patchtst_l336','PatchTST small L96 (3-seed mean)'),row('patchtst_l336','PatchTST small L336 (3-seed mean)')]),
        ('片段交互',['自注意力','混合 MLP'],[row('attn_ablation','Base PatchTST (3-seed mean)'),row('attn_ablation','No-Attention PatchTST (3-seed mean)')]),
        ('SegRNN 参考水平',['24 h 均值','最后值'],[row('segrnn','Anchored SegRNN (3-seed mean)'),row('segrnn','Last-value SegRNN (3-seed mean)')])]
    f,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for a,(title,labels,rs),letter in zip(axes.flat,pairs,'abcd'):
        for i,r in enumerate(rs):
            std=r.get('mae_std',0); std=std if pd.notna(std) else 0
            a.errorbar(r.mae,i,xerr=std,fmt='o',capsize=4,color=C[i],ms=6)
            a.annotate(f'{r.mae:.3f}',(r.mae,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
        a.set(yticks=[0,1],yticklabels=labels,ylim=(-.6,1.6),xlim=(1.55,1.94),xlabel='测试 MAE（℃）',title=f'({letter}) {title}')
        a.invert_yaxis(); a.grid(axis='x',alpha=.2)
    save(f,'fig4_ablations')

def tables():
    entries=[('ridge_level','Plain Ridge','Ridge（绝对值）'),('ridge_level','Anchored Ridge','Ridge（锚定）'),('mlp','Plain MLP (3-seed mean)','MLP（绝对值）'),('mlp','Anchored MLP (3-seed mean)','MLP（锚定）'),('gru','Plain GRU (3-seed mean)','GRU（绝对值）'),('gru','Anchored GRU (3-seed mean)','GRU（锚定）'),('lstm','Plain LSTM (3-seed mean)','LSTM（绝对值）'),('lstm','Anchored LSTM (3-seed mean)','LSTM（锚定）'),('dlinear','Plain DLinear (3-seed mean)','DLinear（绝对值）'),('dlinear','Anchored DLinear (3-seed mean)','DLinear（锚定）'),('patchtst','Plain PatchTST (3-seed mean)','PatchTST 64维（绝对值）'),('patchtst','Anchored PatchTST (3-seed mean)','PatchTST 64维（锚定）'),('attn_ablation','Base PatchTST (3-seed mean)','PatchTST 16维（锚定）'),('attn_ablation','No-Attention PatchTST (3-seed mean)','片段混合 MLP（锚定）')]
    lines=[]
    for folder,model,label in entries:
        r=row(folder,model)
        def fmt(k):
            v=r.get(k+'_std',np.nan)
            return f'${r[k]:.3f} \\pm {v:.3f}$' if pd.notna(v) and v>0 else f'{r[k]:.3f}'
        lines.append(f'{label} & {fmt("mse")} & {fmt("mae")} & ${r.bias:+.3f}$ \\\\')
    (TAB/'main_rows.tex').write_text('\n'.join(lines)+'\n\\bottomrule\n',encoding='utf-8')
    lines=[]
    for h,folder in [(96,'ridge_level'),(192,'h192'),(336,'h336'),(720,'h720')]:
        r=row(folder,'Anchored Ridge'); b=row('dlinear' if h==96 else folder,'B2b_local_mean_96')
        lines.append(f'{h} & {2881-h} & {r.mse:.3f} & {r.mae:.3f} & {b.mae:.3f} & {100*(1-r.mae/b.mae):.2f}\\% \\\\')
    (TAB/'horizon_rows.tex').write_text('\n'.join(lines)+'\n\\bottomrule\n',encoding='utf-8')

def solution_forecasts():
    """Reconstruct the selected linear predictor and verify saved test metrics."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    source = ROOT / 'outputs/splits/windows_L96.npz'
    SOURCES.add(source)
    with np.load(source) as windows:
        d = pd.read_csv(ROOT/'ETTh1.csv', parse_dates=['date'])
        ot = d.OT.to_numpy(dtype=float)
        def xy(split):
            starts = windows[f'{split}_96_x_start'].astype(int)
            targets = windows[f'{split}_96_y_start'].astype(int)
            return (np.stack([ot[i:i+96] for i in starts]),
                    np.stack([ot[i:i+96] for i in targets]), targets)
        xtr,ytr,_=xy('train'); xte,yte,starts=xy('test')
    levels_tr=xtr[:,-24:].mean(axis=1)
    levels_te=xte[:,-24:].mean(axis=1)
    grid=read('ridge_level/alpha_selection.csv')
    selected=grid[(grid['variant']=='anchored') & (grid.split=='val')].sort_values('mae').iloc[0]
    scaler=StandardScaler().fit(xtr-levels_tr[:,None])
    model=Ridge(alpha=float(selected.alpha),fit_intercept=True)
    model.fit(scaler.transform(xtr-levels_tr[:,None]),ytr-levels_tr[:,None])
    pred=model.predict(scaler.transform(xte-levels_te[:,None]))+levels_te[:,None]
    mse=float(np.mean((pred-yte)**2)); mae=float(np.mean(np.abs(pred-yte)))
    expected=row('ridge_level','Anchored Ridge')
    np.testing.assert_allclose([mse,mae],[expected.mse,expected.mae],rtol=1e-10)
    # Calendar-fixed examples: these dates were specified before inspecting errors.
    dates=['2017-11-06 00:00:00','2018-01-06 00:00:00']
    f,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
    records=[]
    for a,date,letter in zip(axes,dates,'ab'):
        target=int(np.flatnonzero(d.date==pd.Timestamp(date))[0])
        k=int(np.flatnonzero(starts==target)[0])
        a.plot(np.arange(-95,1),xte[k],color='#555555',lw=1,label='历史观测')
        a.plot(np.arange(1,97),yte[k],color='#222222',lw=1.4,label='未来真实油温')
        a.plot(np.arange(1,97),pred[k],color=C[0],lw=1.5,label='锚定岭回归预测')
        a.plot([1,96],[levels_te[k]]*2,color=C[2],ls='--',lw=1,label='24 h 均值预测')
        a.axvline(0,color=C[4],ls=':',lw=1)
        a.set(xlabel='相对预测起点的时间（h）',ylabel='油温（℃）',
              title=f'({letter}) 首个预测时刻：{date[:10]} 00:00',xlim=(-95,96))
        a.grid(alpha=.15); a.legend(frameon=False,ncol=2,fontsize=8,loc='best')
        records.append({'first_target':date,'mae':float(np.mean(np.abs(pred[k]-yte[k]))),
                        'anchor':float(levels_te[k])})
    save(f,'fig6_solution_forecasts')
    result={'alpha':float(selected.alpha),'verified_test_mse':mse,'verified_test_mae':mae,
            'example_selection':'fixed calendar dates, no error-based selection', 'examples':records}
    (OUT/'solution_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    assert hashlib.sha256((ROOT/'ETTh1.csv').read_bytes()).hexdigest() == 'f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066'
    pipeline(); data_split(); architectures(); main_results(); horizons(); ablations(); tables(); solution_forecasts()
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'Generated seven figures and two tables from {len(SOURCES)} source files; verified selected Ridge predictions.')
