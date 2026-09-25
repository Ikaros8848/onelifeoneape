"""Evaluate FP32, naive ternary and hardware-aware/QAT checkpoints."""
import argparse,json,torch
from torch.utils.data import DataLoader
from tests.test_mnist import TestMNIST
from baseline.torch_model import build_model
from baseline.hardware_aware import HardwareAwareCNN,ternary_quantize

def eval_model(m,dl,dev):
 m.to(dev).eval(); c=n=0
 with torch.inference_mode():
  for x,y in dl:
   p=m(x.to(dev)).argmax(1); c+=int((p==y.to(dev)).sum()); n+=len(y)
 return c/n
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',default='data'); p.add_argument('--max-samples',type=int); p.add_argument('--fp32',default='artifacts/checkpoints/mnist_scamp_software.pt'); p.add_argument('--qat',default='artifacts/checkpoints/mnist_scamp_qat.pt'); p.add_argument('--threshold-ratio',type=float,default=.05); a=p.parse_args()
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); dl=DataLoader(TestMNIST(a.data_root,a.max_samples),256)
 fp=build_model(); fp.load_state_dict(torch.load(a.fp32,map_location='cpu',weights_only=True))
 # Post-training naive ternary (no scale) for a transparent baseline.
 naive=build_model(); naive.load_state_dict(fp.state_dict())
 for w in [naive.conv.weight,naive.fc.weight]: w.data.copy_(w.sign()*(w.abs()>a.threshold_ratio*w.abs().max()))
 qat=HardwareAwareCNN(True,a.threshold_ratio,True); qat.load_state_dict(torch.load(a.qat,map_location='cpu',weights_only=True))
 out={'FP32':eval_model(fp,dl,dev),'naive_ternary':eval_model(naive,dl,dev),'QAT_hardware_ternary':eval_model(qat,dl,dev)}
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()
