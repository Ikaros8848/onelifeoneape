"""Full-test aggregation and fairness checks for retrained random 13 masks."""
import csv, json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.evaluate import metrics
from hardware_model.cost_model import ScampCostModel
from hardware_model.energy import energy_components
from tests.test_mnist import TestMNIST


def evaluate(state, dataset, device):
    loader=DataLoader(dataset,batch_size=256,shuffle=False)
    s={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in state.items()}
    ys=[]; ps=[]
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device)
            z=F.conv2d(F.pad(x,(1,2,1,2)),s['conv_weight'],s['conv_bias'])
            z=F.max_pool2d(F.relu(z),4,4).flatten(1)
            z=F.linear(z,s['fc_weight'],s['fc_bias'])
            ys.extend(y.tolist()); ps.extend(z.argmax(1).tolist())
    return metrics(ys,ps),ys,ps


def cost_row(name, state, offsets, task, mask_seed, training_seed, checkpoint, samples):
    conv=state['conv_weight']; fc=state['fc_weight']
    report=ScampCostModel().analyze_ternary(name,conv.sign(),fc.sign(),parameters=41242)
    return {
        'method':name,'mask_seed':mask_seed,'training_seed':training_seed,
        'selected_offsets':json.dumps(offsets,separators=(',',':')),
        'accuracy':task['accuracy'],'macro_precision':task['macro_precision'],
        'macro_recall':task['macro_recall'],'macro_f1':task['macro_f1'],
        'params':41242,'active_ops':report.active_terms,
        'skipped_ops':report.skipped_terms,'movement_local':report.local_movement_elements,
        'latency_proxy':report.latency_proxy,
        'energy_proxy_b':energy_components(report,orthogonal_registers=True)['total'],
        'static_schedule_k':len(offsets),'nonzero_weight_tap_count':report.retained_offsets,
        'checkpoint':checkpoint,'test_samples':samples,
        'inference_config':'PyTorch eval; CUDA if available else CPU; batch 256; full MNIST firmware preprocessing; dense F.conv2d',
    }


def main():
    root=Path('results/retrained_random13')
    dataset=TestMNIST('data'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]; matrices={}; masks={}
    common=None
    for mask_seed in (101,202,303,404,505):
        for train_seed in ((7,42) if mask_seed in (101,202) else (7,)):
            folder=root/f'random{mask_seed}_seed{train_seed}'
            exp=json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
            mask_record=json.loads((root/f'random{mask_seed}_seed7'/'mask.json').read_text(encoding='utf-8'))
            offsets=[tuple(x) for x in exp['selected_offsets']]
            if offsets != [tuple(x) for x in mask_record['selected_offsets']]:
                raise RuntimeError(f"mask mismatch for {folder}")
            if len(offsets)!=13 or len(set(offsets))!=13:
                raise RuntimeError(f"mask budget mismatch for {folder}")
            cfg=exp['training_config']
            invariant={k:v for k,v in cfg.items() if k not in ('gate_lr_status',)}
            if common is None: common=invariant
            elif invariant!=common: raise RuntimeError(f"training config mismatch for {folder}")
            if exp['test_samples_per_epoch']!=10000:
                raise RuntimeError(f"test set mismatch for {folder}")
            state=torch.load(exp['deploy_checkpoint'],map_location='cpu',weights_only=True)
            task,ys,ps=evaluate(state,dataset,device)
            row=cost_row(f'random13_mask{mask_seed}_train{train_seed}',state,offsets,
                         task,mask_seed,train_seed,exp['deploy_checkpoint'],len(ys))
            rows.append(row); matrices[row['method']]=task['confusion_matrix']
            masks[str(mask_seed)]=offsets

    # Reuse, do not retrain, the two established learned-mask checkpoints.
    sweep={r['method']:r for r in csv.DictReader(open('results/tap_sweep.csv',encoding='utf-8-sig'))}
    for seed,path,method in [
        (7,'artifacts/checkpoints/mnist_scamp_offset13.deploy.pt','learned13_seed7'),
        (42,'artifacts/checkpoints/mnist_scamp_offset13_seed42.deploy.pt','learned13_seed42')]:
        state=torch.load(path,map_location='cpu',weights_only=True)
        learned_row=sweep[f'13tap_seed{seed}']
        offsets=[tuple(x) for x in json.loads(learned_row['selected_offsets'])]
        task,ys,ps=evaluate(state,dataset,device)
        rows.append(cost_row(method,state,offsets,task,'learned',seed,path,len(ys)))
        matrices[method]=task['confusion_matrix']

    if any(r['test_samples']!=10000 for r in rows) or len(rows)!=9:
        raise RuntimeError('comparison row/sample count validation failed')
    output=Path('results/retrained_random13_ablation.csv')
    with output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (root/'confusion_matrices.json').write_text(json.dumps(matrices,indent=2),encoding='utf-8')
    validation={
        'mask_k_all_13':all(r['static_schedule_k']==13 for r in rows),
        'parameter_counts_all_41242':all(r['params']==41242 for r in rows),
        'test_samples_all_10000':all(r['test_samples']==10000 for r in rows),
        'preprocessing_same':True,
        'training_config_common':common,
        'device':str(device),
        'pc_execution_note':'actual inference uses dense F.conv2d; operation metrics are logical schedule counts',
    }
    (root/'fairness_checks.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps({'csv':str(output),'rows':len(rows),'fairness':validation},indent=2))


if __name__=='__main__':main()
