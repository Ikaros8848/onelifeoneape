"""Stage-3 tap sweep, fixed-mask ablations, and Pareto outputs."""
import csv, json, random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.evaluate import metrics
from baseline.hardware_aware import ternary_quantize
from hardware_model.cost_model import ScampCostModel
from hardware_model.energy import energy_components
from tests.test_mnist import TestMNIST
from PIL import Image, ImageDraw


def evaluate(state, loader, device):
    state={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in state.items()}
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in loader:
            x=x.to(device)
            z = F.conv2d(F.pad(x, (1,2,1,2)), state['conv_weight'], state['conv_bias'])
            z = F.max_pool2d(F.relu(z), 4, 4).flatten(1)
            z = F.linear(z, state['fc_weight'], state['fc_bias'])
            ys.extend(y.tolist()); ps.extend(z.argmax(1).tolist())
    return metrics(ys, ps)


def load_state(path, latent=False):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if latent:
        conv = ternary_quantize(state['conv_weight'], .05, True)[0]
        fc = ternary_quantize(state['fc_weight'], .05, True)[0]
        state = dict(state, conv_weight=conv.detach(), fc_weight=fc.detach())
    return state


def selected(state):
    active = state['conv_weight'].ne(0).any((0,1))
    return [(i,j) for i in range(4) for j in range(4) if bool(active[i,j])]


def apply_mask(state, offsets):
    out = {k: v.clone() if torch.is_tensor(v) else v for k,v in state.items()}
    keep = torch.zeros((4,4), dtype=out['conv_weight'].dtype)
    for i,j in offsets: keep[i,j] = 1
    out['conv_weight'] *= keep.view(1,1,4,4)
    return out


def scatter_png(path, points, xkey, xlabel, ylabel):
    """Small dependency-free scatter plot for fixed audit figures."""
    width,height=1100,720; left,right,top,bottom=110,35,40,95
    image=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(image)
    xs=[float(p[xkey]) for p in points]; ys=[float(p['accuracy'])*100 for p in points]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    xpad=(xmax-xmin)*.08 or 1; ypad=(ymax-ymin)*.18 or .1
    xmin-=xpad; xmax+=xpad; ymin-=ypad; ymax+=ypad
    def xy(x,y): return (left+(x-xmin)/(xmax-xmin)*(width-left-right),
                         height-bottom-(y-ymin)/(ymax-ymin)*(height-top-bottom))
    draw.line((left,top,left,height-bottom),fill='black',width=2)
    draw.line((left,height-bottom,width-right,height-bottom),fill='black',width=2)
    for i in range(6):
        xv=xmin+(xmax-xmin)*i/5; px,_=xy(xv,ymin)
        draw.line((px,top,px,height-bottom),fill='#e4e4e4',width=1)
        draw.text((px-30,height-bottom+12),f'{xv:.0f}',fill='black')
        yv=ymin+(ymax-ymin)*i/5; _,py=xy(xmin,yv)
        draw.line((left,py,width-right,py),fill='#e4e4e4',width=1)
        draw.text((15,py-8),f'{yv:.2f}',fill='black')
    palette=['#2369bd','#df6b24','#17835c','#b23b55','#7149a6','#758b22']
    for idx,p in enumerate(points):
        px,py=xy(float(p[xkey]),float(p['accuracy'])*100)
        color=palette[idx%len(palette)]; draw.ellipse((px-6,py-6,px+6,py+6),fill=color)
        draw.text((px+8,py-14),p['method'],fill=color)
    draw.text((width//2-100,height-35),xlabel,fill='black')
    draw.text((12,10),ylabel,fill='black')
    image.save(path)


def report_row(name, state, task, seed, config, category):
    cs = state['conv_weight'].sign(); fs = state['fc_weight'].sign()
    offsets = selected(state)
    report = ScampCostModel().analyze_ternary(name, cs, fs, parameters=41242)
    e_b = energy_components(report, orthogonal_registers=True)['total']
    return {
        'method': name, 'category': category, 'k': len(offsets), 'seed': seed,
        'checkpoint': config.get('checkpoint', config.get('weight_checkpoint', '')),
        'training_config': json.dumps(config, sort_keys=True),
        'selected_offsets': json.dumps(offsets), 'accuracy': task['accuracy'],
        'macro_precision': task['macro_precision'], 'macro_recall': task['macro_recall'],
        'macro_f1': task['macro_f1'], 'params': 41242,
        'active_ops': report.active_terms, 'skipped_ops': report.skipped_terms,
        'movement_local': report.local_movement_elements,
        'latency_proxy': report.latency_proxy, 'energy_proxy_b': e_b,
        'inference_config': 'PyTorch eval; CUDA if available else CPU; batch 256; full MNIST firmware preprocessing; dense F.conv2d',
        'software_sparse_kernel': False,
    }


def main():
    out = Path('results'); out.mkdir(exist_ok=True)
    figdir = Path('figures'); figdir.mkdir(exist_ok=True)
    ds = TestMNIST('data'); loader = DataLoader(ds, batch_size=256, shuffle=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]; states={}
    specs = [
        (16, None, 'artifacts/checkpoints/mnist_scamp_qat.pt', True, 'ternary_qat'),
        (16, 7, 'artifacts/checkpoints/mnist_scamp_offset16.deploy.pt', False, 'learned_tap'),
        (15, 7, 'artifacts/checkpoints/mnist_scamp_offset15.deploy.pt', False, 'learned_tap'),
        (14, 7, 'artifacts/checkpoints/mnist_scamp_offset_gated.deploy.pt', False, 'learned_tap'),
        (13, 7, 'artifacts/checkpoints/mnist_scamp_offset13.deploy.pt', False, 'learned_tap'),
        (13, 42, 'artifacts/checkpoints/mnist_scamp_offset13_seed42.deploy.pt', False, 'learned_tap'),
        (12, 7, 'artifacts/checkpoints/mnist_scamp_offset12.deploy.pt', False, 'learned_tap'),
        (11, 7, 'artifacts/checkpoints/mnist_scamp_offset11.deploy.pt', False, 'learned_tap'),
        (10, 7, 'artifacts/checkpoints/mnist_scamp_offset10.deploy.pt', False, 'learned_tap'),
    ]
    for k,seed,path,latent,category in specs:
        s=load_state(path,latent); states[(k,seed)]=s
        task=evaluate(s,loader,device)
        if category=='ternary_qat':
            config={'seed':'unknown_historical_baseline','training_config':'historical checkpoint metadata unavailable',
                    'threshold_ratio':.05,'per_channel_scale':True,'checkpoint':path}
            name='ternary_qat_baseline'
        else:
            config={'seed':seed,'epochs':4,'batch_size':128,'optimizer':'Adam','lr':1e-3,
                    'gate_lr':2e-3,'threshold_ratio':.05,'gate_lambda':.001,'budget_lambda':.01,
                    'init':'mnist_scamp_qat.pt','data_augmentation':False,'checkpoint':path}
            name=f'{k}tap_seed{seed}'
        rows.append(report_row(name,s,task,seed,config,category))

    # Mask ablations use the same seed-7 trained latent weights before the
    # learned mask is baked in. This allows arbitrary subsets to contain any
    # of the 16 offsets while keeping all learned weights fixed.
    latent=torch.load('artifacts/checkpoints/mnist_scamp_offset13.pt',map_location='cpu',weights_only=True)
    base={
        'conv_weight':ternary_quantize(latent['conv_weight'],.05,True)[0].detach(),
        'conv_bias':latent['conv_bias'].detach(),
        'fc_weight':ternary_quantize(latent['fc_weight'],.05,True)[0].detach(),
        'fc_bias':latent['fc_bias'].detach(),
    }
    learned_offsets=selected(states[(13,7)])
    learned_state=apply_mask(base,learned_offsets)
    learned_task=evaluate(learned_state,loader,device)
    rows.append(report_row('learned13_fixed_mask',learned_state,learned_task,7,
        {'weight_checkpoint':'mnist_scamp_offset13.pt','training':'learned static top-k; mask applied at inference'},
        'learned_fixed_mask'))
    all_offsets=[(i,j) for i in range(4) for j in range(4)]
    rng=random.Random(20260930)
    random_masks=[]
    for idx in range(1,6):
        mask=sorted(rng.sample(all_offsets,13))
        random_masks.append(mask)
        state=apply_mask(base,mask); task=evaluate(state,loader,device)
        rows.append(report_row(f'random13_{idx}',state,task,20260930+idx,
                    {'weight_checkpoint':'mnist_scamp_offset13.deploy.pt','mask_seed':20260930+idx,
                     'training':'none; fixed-mask inference ablation'},'random_fixed_mask'))
    manual=[o for o in all_offsets if o not in {(0,0),(0,3),(3,0)}]
    state=apply_mask(base,manual); task=evaluate(state,loader,device)
    rows.append(report_row('manual_corner_drop13',state,task,None,
                {'weight_checkpoint':'mnist_scamp_offset13.deploy.pt','rule':'drop three corners (0,0),(0,3),(3,0)',
                'training':'none; fixed-mask inference ablation'},'manual_fixed_mask'))

    # Same five masks on the original QAT weights isolate direct fixed-mask
    # pruning without learned weight adaptation.
    qat_state=load_state('artifacts/checkpoints/mnist_scamp_qat.pt',latent=True)
    for idx,mask in enumerate(random_masks,1):
        masked=apply_mask(qat_state,mask); task=evaluate(masked,loader,device)
        rows.append(report_row(f'random13_qat_{idx}',masked,task,20260930+idx,
            {'weight_checkpoint':'mnist_scamp_qat.pt','mask_seed':20260930+idx,
             'training':'none; same fixed mask applied to original QAT weights'},
            'random_mask_on_qat'))

    with (out/'tap_sweep.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rand=[r for r in rows if r['category'] in ('random_fixed_mask','learned_fixed_mask','random_mask_on_qat')]
    with (out/'random13_ablation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rand+[r for r in rows if r['method']=='13tap_seed7'])
    with (out/'random13_qat_ablation.csv').open('w',newline='',encoding='utf-8') as f:
        qrows=[r for r in rows if r['category']=='random_mask_on_qat']
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(qrows)

    # Add FP32 point to Pareto data; its operation column is dense scalar MACs.
    fp=load_state('artifacts/checkpoints/mnist_scamp_software.pt')
    # Float baseline accuracy was measured by the Stage-2 fixed benchmark.
    basepoints=[{'method':'FP32','accuracy':.9708,'active_ops':1089536,'energy_proxy_b':None}]
    for r in rows:
        if r['category'] in ('learned_tap','ternary_qat'):
            basepoints.append({'method':r['method'],'accuracy':r['accuracy'],
                               'active_ops':r['active_ops'],'energy_proxy_b':r['energy_proxy_b']})
    for filename,xkey in [('pareto_accuracy_active_ops.csv','active_ops'),('pareto_accuracy_energy.csv','energy_proxy_b')]:
        points=[p for p in basepoints if p[xkey] is not None]
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['method','accuracy',xkey]); w.writeheader()
            w.writerows({key:p[key] for key in ('method','accuracy',xkey)} for p in points)

    active=[p for p in basepoints if p['active_ops'] is not None]
    scatter_png(figdir/'accuracy_vs_active_ops.png',active,'active_ops','Logical Active Operations','Accuracy (%)')
    energy=[p for p in basepoints if p['energy_proxy_b'] is not None]
    scatter_png(figdir/'accuracy_vs_energy_proxy.png',energy,'energy_proxy_b','Energy Proxy B (dimensionless)','Accuracy (%)')
    print(json.dumps({'rows':len(rows),'tap_sweep':'results/tap_sweep.csv',
        'random':'results/random13_ablation.csv','figures':[str(figdir/'accuracy_vs_active_ops.png'),
        str(figdir/'accuracy_vs_energy_proxy.png')]},indent=2))


if __name__=='__main__': main()
