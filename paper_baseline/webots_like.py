"""Deterministic Webots-style bird's-eye road/grass/vehicle dataset.

This is a procedural recreation of the data protocol described by Liu & Lu:
bird's-eye grayscale frames, road/grass coarse masks, and a vehicle Gaussian
localisation target. It is not the authors' original Webots files.
"""
import torch
from torch.utils.data import Dataset

class WebotsRoadGrass(Dataset):
    def __init__(self, n=4096, size=64, seed=123, split='train'):
        self.n=n; self.size=size; self.seed=seed + (0 if split=='train' else 100000)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        g=torch.Generator().manual_seed(self.seed+idx); s=self.size
        yy,xx=torch.meshgrid(torch.arange(s),torch.arange(s),indexing='ij')
        # Perspective road corridor widening toward the bottom of the image.
        vanx=float(torch.randint(20,45,(1,),generator=g)); topw=float(torch.randint(8,18,(1,),generator=g)); botw=float(torch.randint(42,62,(1,),generator=g))
        center=vanx + (yy/(s-1)-.5)*float(torch.randint(-10,11,(1,),generator=g))
        width=topw + (botw-topw)*yy/(s-1)
        road=((xx-center).abs() < width/2).float()
        # textured road and grass, plus mild sensor noise.
        road_tex=torch.randn((s,s),generator=g)*.035; grass_tex=torch.randn((s,s),generator=g)*.06
        # Harder domain: overlapping road/grass intensities, gradients and illumination.
        illum=(0.85 + 0.30*xx/s + 0.18*torch.sin(yy/9))
        image=(0.34 + 0.16*road_tex)*(1-road) + (0.43 + grass_tex)*road
        image=image*illum
        # Tree/shadow blobs create road/grass ambiguity and occlusion.
        for _ in range(5):
            tx=int(torch.randint(0,s,(1,),generator=g)); ty=int(torch.randint(0,s,(1,),generator=g)); rr=int(torch.randint(2,7,(1,),generator=g))
            blob=((xx-tx)**2+(yy-ty)**2 < rr*rr)
            image=image*(1-0.35*blob.float()) + 0.12*blob.float()
        # Add a vehicle blob inside the road corridor.
        cy=int(torch.randint(18,52,(1,),generator=g)); cwidth=float(topw+(botw-topw)*cy/(s-1)); cx=int(torch.clamp(torch.tensor(vanx),8,s-9))
        cx=int(torch.clamp(torch.tensor(cx + int(torch.randn((),generator=g)*max(2,cwidth*.18))),8,s-9))
        veh=((xx-cx).abs()<=2)&((yy-cy).abs()<=3); image=image + veh.float()*0.22
        image=(image + torch.randn((s,s),generator=g)*0.045).clamp(0,1).unsqueeze(0)
        heat=torch.exp(-((xx-cx)**2+(yy-cy)**2)/(2*3.5**2)).float().unsqueeze(0)
        return image, road.unsqueeze(0), heat, torch.tensor([cx,cy],dtype=torch.float32)
