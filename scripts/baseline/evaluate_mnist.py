"""Evaluate the ordinary reference graph on MNIST without training."""
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from baseline.torch_model import build_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='data'); p.add_argument('--weights'); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--download',action='store_true'); a=p.parse_args()
    model=build_model()
    if a.weights: model.load_state_dict(torch.load(a.weights,map_location='cpu'))
    ds=datasets.MNIST(a.data_root,train=False,download=a.download,transform=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()]))
    dl=DataLoader(ds,batch_size=a.batch_size); correct=total=0; ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl:
            pred=model(x).argmax(1); correct += int((pred==y).sum()); total += len(y); ys.extend(y.tolist()); ps.extend(pred.tolist())
    from baseline.evaluate import metrics
    r=metrics(ys,ps); r['accuracy']=correct/total; r['parameters']=sum(p.numel() for p in model.parameters()); print(r)
if __name__=='__main__': main()
