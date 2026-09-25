"""Train the SCAMP-aware ternary model with STE QAT."""
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from scripts.baseline.train_mnist import FirmwareMNIST, run_epoch
from baseline.hardware_aware import HardwareAwareCNN

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',default='data'); p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=128); p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--threshold-ratio',type=float,default=.05); p.add_argument('--output',default='artifacts/checkpoints/mnist_scamp_qat.pt')
    p.add_argument('--init',default='artifacts/checkpoints/mnist_scamp_software.pt',help='FP32 checkpoint used to initialize latent weights')
    p.add_argument('--activation-limit',type=float); p.add_argument('--accumulator-limit',type=float)
    p.add_argument('--max-train',type=int); p.add_argument('--max-test',type=int)
    a=p.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=FirmwareMNIST(a.data_root,True); te=FirmwareMNIST(a.data_root,False)
    if a.max_train: tr.base.data,tr.base.targets=tr.base.data[:a.max_train],tr.base.targets[:a.max_train]
    if a.max_test: te.base.data,te.base.targets=te.base.data[:a.max_test],te.base.targets[:a.max_test]
    m=HardwareAwareCNN(True,a.threshold_ratio,True,a.activation_limit,a.accumulator_limit)
    if a.init:
        from baseline.torch_model import build_model
        fp=build_model(); fp.load_state_dict(torch.load(a.init,map_location='cpu',weights_only=True)); m.load_from_fp32(fp)
    m=m.to(device)
    opt=torch.optim.Adam(m.parameters(),lr=a.lr); train=DataLoader(tr,a.batch_size,shuffle=True); test=DataLoader(te,a.batch_size)
    hist=[]
    for e in range(1,a.epochs+1):
        a1=run_epoch(m,train,opt,device)
        with torch.no_grad(): a2=run_epoch(m,test,None,device)
        row={'epoch':e,'train':a1,'test':a2}; hist.append(row); print(json.dumps(row))
    os.makedirs(os.path.dirname(a.output) or '.',exist_ok=True); torch.save(m.state_dict(),a.output)
    _,info=m(torch.zeros(1,1,64,64,device=device),True)
    stats={}
    for name,mask in [('conv',info['conv_mask']),('fc',info['fc_mask'])]:
        # sign statistics are from the trained latent weights at export time.
        w=getattr(m,name+'_weight').detach(); q,sg,al,ma=__import__('baseline.hardware_aware',fromlist=['ternary_quantize']).ternary_quantize(w,a.threshold_ratio,True)
        stats[name]={'plus':float((sg>0).float().mean()),'zero':float((sg==0).float().mean()),'minus':float((sg<0).float().mean()),'alpha_mean':float(al.mean())}
    report={'model':'hardware-aware ternary QAT','device':str(device),'epochs':a.epochs,'threshold_ratio':a.threshold_ratio,'per_channel_scale':True,'activation_limit':a.activation_limit,'accumulator_limit':a.accumulator_limit,'history':hist,'weight_stats':stats,'checkpoint':a.output,'note':'functional SCAMP approximation; not cycle-accurate hardware'}
    with open(os.path.splitext(a.output)[0]+'.json','w',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'checkpoint':a.output,'final_test':hist[-1]['test'],'weight_stats':stats},ensure_ascii=False))
if __name__=='__main__': main()
