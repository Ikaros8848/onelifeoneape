"""Fair mask/weight-adaptation contribution table for Stage 3A."""
import csv, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from baseline.evaluate import metrics
from baseline.hardware_aware import ternary_quantize
from hardware_model.cost_model import ScampCostModel
from hardware_model.energy import energy_components
from tests.test_mnist import TestMNIST


def evaluate(state, loader, device):
    s={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in state.items()}
    ys=[]; ps=[]
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device)
            z=F.conv2d(F.pad(x,(1,2,1,2)),s['conv_weight'],s['conv_bias'])
            z=F.max_pool2d(F.relu(z),4,4).flatten(1)
            z=F.linear(z,s['fc_weight'],s['fc_bias'])
            ys.extend(y.tolist()); ps.extend(z.argmax(1).tolist())
    return metrics(ys,ps)


def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=TestMNIST('data'); loader=DataLoader(ds,batch_size=256,shuffle=False)
    qat=torch.load('artifacts/checkpoints/mnist_scamp_qat.pt',map_location='cpu',weights_only=True)
    qconv=ternary_quantize(qat['conv_weight'],.05,True)[0].detach()
    qfc=ternary_quantize(qat['fc_weight'],.05,True)[0].detach()
    qstate={'conv_weight':qconv,'conv_bias':qat['conv_bias'],'fc_weight':qfc,'fc_bias':qat['fc_bias']}
    learned=torch.load('artifacts/checkpoints/mnist_scamp_offset13.deploy.pt',map_location='cpu',weights_only=True)
    offsets=[(i,j) for i in range(4) for j in range(4) if bool(learned['conv_weight'][:,:,i,j].ne(0).any())]
    manual=[(i,j) for i in range(4) for j in range(4) if (i,j) not in {(0,0),(0,3),(3,0)}]
    rows=[]
    for name,mask,interpretation in [
        ('C_learned_mask_on_QAT_weights',offsets,'learned subset applied to QAT weights without offset-gated fine-tuning'),
        ('manual_corner_mask_on_QAT_weights',manual,'drop fixed corners (0,0),(0,3),(3,0) from QAT weights'),
    ]:
        masked={k:v.clone() if torch.is_tensor(v) else v for k,v in qstate.items()}
        keep=torch.zeros(4,4)
        for i,j in mask: keep[i,j]=1
        masked['conv_weight']*=keep.view(1,1,4,4)
        task=evaluate(masked,loader,device)
        c=masked['conv_weight'].sign(); f=masked['fc_weight'].sign()
        cost=ScampCostModel().analyze_ternary(name,c,f,parameters=41242)
        rows.append({'method':name,'accuracy':task['accuracy'],
            'macro_precision':task['macro_precision'],'macro_recall':task['macro_recall'],
            'macro_f1':task['macro_f1'],'params':41242,'selected_offsets':json.dumps(mask),
            'active_ops':cost.active_terms,'skipped_ops':cost.skipped_terms,
            'movement_local':cost.local_movement_elements,'latency_proxy':cost.latency_proxy,
            'energy_proxy_b':energy_components(cost,orthogonal_registers=True)['total'],
            'interpretation':interpretation})
    out=Path('results/gate_contribution.csv')
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
