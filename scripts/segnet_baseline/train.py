import argparse,json,torch
from torch.utils.data import DataLoader
from segnet_baseline.model import SegNet
from paper_baseline.synthetic import SyntheticBirdsEye

def main():
 p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=3); p.add_argument('--samples',type=int,default=512); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--output',default='artifacts/checkpoints/segnet_synthetic.pt'); a=p.parse_args(); torch.manual_seed(1)
 ds=SyntheticBirdsEye(a.samples); dl=DataLoader(ds,a.batch_size,shuffle=True); m=SegNet(2,1); opt=torch.optim.Adam(m.parameters(),lr=2e-3); lossfn=torch.nn.CrossEntropyLoss()
 for ep in range(a.epochs):
  total=0.; m.train()
  for image,mask,_,_ in dl:
   target=mask[:,0].long(); opt.zero_grad(); loss=lossfn(m(image),target); loss.backward(); opt.step(); total+=loss.item()
  print(f'epoch {ep+1}/{a.epochs} loss={total/len(dl):.4f}')
 m.eval(); inter=union=0
 with torch.no_grad():
  for image,mask,_,_ in DataLoader(ds,a.batch_size):
   pred=m(image).argmax(1); t=mask[:,0].bool(); q=pred.bool(); inter+=int((q&t).sum()); union+=int((q|t).sum())
 r={'model':'SegNet','dataset':'SyntheticBirdsEye','samples':a.samples,'epochs':a.epochs,'road_iou':inter/max(union,1),'parameters':sum(x.numel() for x in m.parameters()),'note':'architecture smoke test; not CamVid or paper original data'}; torch.save(m.state_dict(),a.output); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
