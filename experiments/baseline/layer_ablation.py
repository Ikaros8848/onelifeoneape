"""Measure progressive post-training ternary ablations with per-channel scale."""
import argparse, json, torch
from torch.utils.data import DataLoader
from tests.test_mnist import TestMNIST
from baseline.torch_model import build_model
from baseline.hardware_aware import HardwareAwareCNN

def score(m, dl, dev):
    m.to(dev).eval(); c=n=0
    with torch.inference_mode():
        for x,y in dl:
            c += int((m(x.to(dev)).argmax(1)==y.to(dev)).sum()); n += len(y)
    return c/n

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='data'); p.add_argument('--max-samples',type=int); p.add_argument('--checkpoint',default='artifacts/checkpoints/mnist_scamp_software.pt'); p.add_argument('--threshold-ratio',type=float,default=.05); a=p.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dl=DataLoader(TestMNIST(a.data_root,a.max_samples),256)
    fp=build_model(); fp.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    base=score(fp,dl,dev); rows=[{'configuration':'FP32','accuracy':base,'accuracy_drop':0.0}]
    for name, conv, fc in [('Conv1 ternary',True,False),('Conv1 + FC ternary',True,True)]:
        m=HardwareAwareCNN(True,a.threshold_ratio,True,ternary_conv=conv,ternary_fc=fc).load_from_fp32(fp)
        acc=score(m,dl,dev); rows.append({'configuration':name,'accuracy':acc,'accuracy_drop':base-acc})
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
