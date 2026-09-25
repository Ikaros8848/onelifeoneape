"""Deterministic road/grass-like coarse segmentation and Gaussian localisation data."""
import torch
from torch.utils.data import Dataset

class SyntheticBirdsEye(Dataset):
    def __init__(self,n=512,size=64,seed=7): self.n=n; self.size=size; self.seed=seed
    def __len__(self): return self.n
    def __getitem__(self,idx):
        g=torch.Generator().manual_seed(self.seed+idx); h=w=self.size; yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
        cx=int(torch.randint(12,w-12,(1,),generator=g)); cy=int(torch.randint(12,h-12,(1,),generator=g)); slope=(torch.rand((),generator=g)-.5)*.8
        boundary=cy+slope*(xx-cx); mask=(yy>boundary).float(); noise=torch.randn((h,w),generator=g)*.08
        image=(.25+.45*mask+noise).clamp(0,1).unsqueeze(0); heat=torch.exp(-((xx-cx)**2+(yy-cy)**2)/(2*4.0**2)).float().unsqueeze(0)
        return image, mask.unsqueeze(0), heat, torch.tensor([cx,cy])
