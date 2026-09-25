import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class SegNet(nn.Module):
    """Compact SegNet: VGG-style encoder and max-unpool decoder.

    Pooling indices are retained and reused by MaxUnpool2d, which is the
    defining SegNet design choice. The default width is intentionally small
    for reproducible CPU experiments; set widths=(64,128,256) for a closer
    VGG16-sized variant.
    """
    def __init__(self, num_classes=2, in_channels=1, widths=(16,32,64)):
        super().__init__(); a,b,c=widths
        self.enc1=nn.Sequential(ConvBNReLU(in_channels,a), ConvBNReLU(a,a))
        self.enc2=nn.Sequential(ConvBNReLU(a,b), ConvBNReLU(b,b))
        self.enc3=nn.Sequential(ConvBNReLU(b,c), ConvBNReLU(c,c))
        self.dec3=nn.Sequential(ConvBNReLU(c,b), ConvBNReLU(b,b))
        self.dec2=nn.Sequential(ConvBNReLU(b,a), ConvBNReLU(a,a))
        self.dec1=nn.Sequential(ConvBNReLU(a,a), nn.Conv2d(a,num_classes,1))
        self.pool=nn.MaxPool2d(2,2,return_indices=True); self.unpool=nn.MaxUnpool2d(2,2)
    def forward(self,x):
        sizes=[]; inds=[]
        x=self.enc1(x); sizes.append(x.size()); x,i=self.pool(x); inds.append(i)
        x=self.enc2(x); sizes.append(x.size()); x,i=self.pool(x); inds.append(i)
        x=self.enc3(x); sizes.append(x.size()); x,i=self.pool(x); inds.append(i)
        x=self.unpool(x,inds.pop(),output_size=sizes.pop()); x=self.dec3(x)
        x=self.unpool(x,inds.pop(),output_size=sizes.pop()); x=self.dec2(x)
        x=self.unpool(x,inds.pop(),output_size=sizes.pop()); return self.dec1(x)
