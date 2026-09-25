"""Oxford-IIIT Pet real-image adapter for the binary FCN baseline."""
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
import numpy as np

class OxfordPetFCN(Dataset):
    def __init__(self, root="data", split="trainval", download=True, size=64, limit=0):
        self.ds = OxfordIIITPet(root=root, split=split, target_types="segmentation", download=download)
        self.size=size; self.limit=limit if limit and limit < len(self.ds) else len(self.ds)
    def __len__(self): return self.limit
    def __getitem__(self, i):
        img, seg = self.ds[i]
        img = TF.resize(img, [self.size,self.size], antialias=True)
        seg = TF.resize(seg, [self.size,self.size], interpolation=TF.InterpolationMode.NEAREST)
        image = TF.rgb_to_grayscale(TF.to_tensor(img))
        s = torch.as_tensor(np.array(seg), dtype=torch.int64)
        mask = (s == 1).float().unsqueeze(0)
        yy,xx=torch.meshgrid(torch.arange(self.size),torch.arange(self.size),indexing='ij')
        ys,xs=torch.where(mask[0] > .5)
        if len(xs): cx,cy=xs.float().mean(),ys.float().mean()
        else: cx,cy=torch.tensor(self.size/2),torch.tensor(self.size/2)
        heat=torch.exp(-((xx-cx)**2+(yy-cy)**2)/(2*5.0**2)).float().unsqueeze(0)
        return image, mask, heat, torch.stack([cx,cy])
