"""Add the FP32 Pareto point and generate objective Stage-3 summary artifacts."""
import csv
from pathlib import Path
import torch
from baseline.torch_model import build_model
from hardware_model.cost_model import ScampCostModel
from hardware_model.energy import energy_components
from scripts.stage3_analysis import scatter_png


def main():
    out=Path('results'); fig=Path('figures'); fig.mkdir(exist_ok=True)
    fp=build_model(); fp.load_state_dict(torch.load('artifacts/checkpoints/mnist_scamp_software.pt',map_location='cpu',weights_only=True))
    cs=fp.conv.weight.detach().cpu().sign(); fs=fp.fc.weight.detach().cpu().sign()
    cost=ScampCostModel().analyze_ternary('FP32_dense_reference',cs,fs,parameters=41242)
    e=energy_components(cost,orthogonal_registers=True)['total']
    rows=list(csv.DictReader((out/'tap_sweep.csv').open(encoding='utf-8')))
    points=[{'method':'FP32','accuracy':.9708,'active_ops':1089536,'energy_proxy_b':e}]
    points += [{'method':r['method'],'accuracy':float(r['accuracy']),
                'active_ops':int(r['active_ops']),'energy_proxy_b':float(r['energy_proxy_b'])}
               for r in rows if r['category'] in ('ternary_qat','learned_tap')]
    for filename,key in [('pareto_accuracy_active_ops.csv','active_ops'),('pareto_accuracy_energy.csv','energy_proxy_b')]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['method','accuracy',key]); w.writeheader()
            w.writerows({'method':p['method'],'accuracy':p['accuracy'],key:p[key]} for p in points)
    scatter_png(fig/'accuracy_vs_active_ops.png',points,'active_ops','Logical Active Operations','Accuracy (%)')
    scatter_png(fig/'accuracy_vs_energy_proxy.png',points,'energy_proxy_b','Energy Proxy B (dimensionless)','Accuracy (%)')
    print({'fp32_energy_proxy_b':e,'points':len(points)})

if __name__=='__main__': main()
