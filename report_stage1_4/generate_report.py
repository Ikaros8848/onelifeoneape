import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent; plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
def read_csv(p): return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))
tap=read_csv('results/tap_sweep.csv'); rob=read_csv('results/robustness.csv'); retr=read_csv('results/retrained_random13_ablation.csv')
fig=OUT/'figures'; fig.mkdir(exist_ok=True)
q=[r for r in tap if r['method'] in ('Ternary QAT baseline','Gated 13, seed 7','Gated 13, seed 42','Gated 14, seed 7','Gated 15, seed 7','Gated 12, seed 7','Gated 11, seed 7','Gated 10, seed 7')]
plt.figure(figsize=(8,5)); plt.plot([int(r['k']) for r in q],[float(r['accuracy'])*100 for r in q],'o-'); plt.xlabel('保留 tap 数量 k'); plt.ylabel('准确率 (%)'); plt.title('Tap 数量与准确率'); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(fig/'tap_sweep_accuracy.png',dpi=180); plt.close()
plt.figure(figsize=(8,5)); plt.plot([int(r['k']) for r in q],[int(float(r['active_ops']))/1e6 for r in q],'o-',label='逻辑 active operations'); plt.xlabel('保留 tap 数量 k'); plt.ylabel('百万次操作'); plt.title('Tap 数量与逻辑计算量'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(fig/'tap_sweep_active_ops.png',dpi=180); plt.close()
plt.figure(figsize=(8,5));
for name in ['QAT','13-tap seed 7','13-tap seed 42']:
    z=[r for r in rob if r['model']==name and r['corruption']=='gaussian']; plt.plot([float(r['severity']) for r in z],[float(r['accuracy'])*100 for r in z],'o-',label=name)
plt.xlabel('Gaussian 噪声 sigma'); plt.ylabel('准确率 (%)'); plt.title('Gaussian 噪声鲁棒性'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(fig/'robustness_gaussian.png',dpi=180); plt.close()
plt.figure(figsize=(9,5));
for name in ['QAT','13-tap seed 7','13-tap seed 42']:
    z=[r for r in rob if r['model']==name and r['corruption'].startswith('translation_')]; agg={}
    for s in ['1','2']: agg[s]=sum(float(r['accuracy']) for r in z if r['severity']==s)/4
    plt.plot([1,2],[agg['1']*100,agg['2']*100],'o-',label=name)
plt.xlabel('平移像素'); plt.ylabel('四方向平均准确率 (%)'); plt.title('平移鲁棒性'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(fig/'robustness_translation.png',dpi=180); plt.close()
plt.figure(figsize=(8,5)); plt.scatter([int(float(r['active_ops']))/1e6 for r in tap],[float(r['accuracy'])*100 for r in tap]);
for r in tap: plt.annotate(r['method'].replace('Gated ','k='), (int(float(r['active_ops']))/1e6,float(r['accuracy'])*100),fontsize=7)
plt.xlabel('逻辑 active operations（百万）'); plt.ylabel('准确率 (%)'); plt.title('准确率-逻辑计算量关系'); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(fig/'pareto_accuracy_ops.png',dpi=180); plt.close()
(OUT/'README.md').write_text('''# SCAMP-aware MNIST 项目实验总报告\n\n- `实验总报告.md`：中文汇总报告。\n- `figures/`：使用 matplotlib 生成的图片。\n- `generate_report.py`：图表生成脚本。\n\n运行：`..\\.venv\\Scripts\\python.exe generate_report.py`\n''',encoding='utf-8')
