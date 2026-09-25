"""Train/evaluate the PC binary FCN on a reproducible synthetic task."""
import argparse, json, torch
from torch.utils.data import DataLoader
from paper_baseline.model import BinaryFCN
from paper_baseline.synthetic import SyntheticBirdsEye
from paper_baseline.realdata import OxfordPetFCN
from paper_baseline.webots_like import WebotsRoadGrass

def iou(pred, target):
    p=pred.bool(); t=target.bool(); return float((p&t).sum()/( (p|t).sum().clamp_min(1)))
def dice(pred, target):
    p=pred.bool(); t=target.bool(); return float(2*(p&t).sum()/(p.sum()+t.sum()).clamp_min(1))
def soft_centroid(z):
    p=torch.softmax(z.flatten(1),dim=1).reshape(z.shape[0],z.shape[2],z.shape[3])
    yy,xx=torch.meshgrid(torch.arange(z.shape[2],device=z.device),torch.arange(z.shape[3],device=z.device),indexing='ij')
    return torch.stack([(p*xx).sum((1,2)),(p*yy).sum((1,2))],1)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3); p.add_argument('--samples',type=int,default=512); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--task',choices=['segmentation','localisation'],default='segmentation'); p.add_argument('--dataset',choices=['synthetic','webots_like','oxford_pet'],default='webots_like'); p.add_argument('--data-root',default='data'); p.add_argument('--download',action='store_true'); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--output',default='artifacts/checkpoints/paper_baseline.pt'); a=p.parse_args(); torch.manual_seed(1)
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    dev=torch.device('cuda' if (a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available())) else 'cpu')
    if a.dataset=='oxford_pet': ds=OxfordPetFCN(a.data_root,split='trainval',download=a.download,limit=a.samples); test_ds=OxfordPetFCN(a.data_root,split='test',download=False,limit=0)
    elif a.dataset=='webots_like':
        n=a.samples if a.samples>0 else 4096; ds=WebotsRoadGrass(n=n,split='train'); test_ds=WebotsRoadGrass(n=max(512,n//5),split='test')
    else: ds=SyntheticBirdsEye(a.samples); test_ds=ds
    dual=(a.dataset=='webots_like' and a.task=='segmentation'); dl=DataLoader(ds,a.batch_size,shuffle=True,pin_memory=(dev.type=='cuda')); m=BinaryFCN(output_channels=2 if dual else 1).to(dev); opt=torch.optim.Adam(m.parameters(),lr=2e-3); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    # Gaussian localisation targets are highly sparse; plain BCE learns the
    # all-zero solution.  MSE preserves the heatmap regression objective and
    # gives the positive peak sufficient gradient on the synthetic task.
    pw=torch.tensor([3.0,3.0],device=dev).view(2,1,1) if (a.dataset=='webots_like' and a.task=='segmentation') else torch.tensor([3.0],device=dev)
    bce=torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(a.epochs):
        m.train(); total=0
        for image,mask,heat,xy in dl:
            image,mask,heat,xy=image.to(dev),mask.to(dev),heat.to(dev),xy.to(dev); opt.zero_grad(); out=m(image)
            if a.task=='segmentation':
                target=torch.cat([mask,1-mask],1) if dual else mask
                prob=torch.sigmoid(out); loss=bce(out,target)+(1-(2*(prob*target).sum((1,2,3))+1)/(prob.sum((1,2,3))+target.sum((1,2,3))+1)).mean()
            else:
                predxy=soft_centroid(out); loss=torch.nn.functional.mse_loss(out,heat)+0.1*torch.nn.functional.mse_loss(predxy/64.0,xy/64.0)
            loss.backward(); opt.step(); total+=loss.item()
        sched.step()
        print(f'epoch {ep+1}/{a.epochs} loss={total/len(dl):.4f}')
    m.eval(); seg=[]; dices=[]; loc=[]
    with torch.no_grad():
        for image,mask,heat,xy in DataLoader(test_ds,batch_size=a.batch_size,pin_memory=(dev.type=='cuda')):
            image,mask,heat,xy=image.to(dev),mask.to(dev),heat.to(dev),xy.to(dev); out=m(image); pred=(out[:,0:1]>0) if dual else (out>0); seg.append(iou(pred,mask)); dices.append(dice(pred,mask)); q=out[:,0:1].flatten(1).argmax(1); py,px=q//64,q%64; loc.extend((((px-xy[:,0])**2+(py-xy[:,1])**2).sqrt()<10).tolist())
    result={'dataset':a.dataset,'train_samples':len(ds),'test_samples':len(test_ds),'epochs':a.epochs,'task':a.task,'iou_mean':sum(seg)/len(seg),'dice_f1_mean':sum(dices)/len(dices),'localisation_accuracy_at_10px':sum(loc)/len(loc),'parameters':sum(x.numel() for x in m.parameters()),'weights_binary_in_forward':True,'device':str(dev),'note':'PC software baseline; not SCAMP hardware'}
    torch.save(m.state_dict(),a.output); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
