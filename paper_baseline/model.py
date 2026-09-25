"""Binary-weight/binary-activation FCN matching the paper's topology."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold): return torch.where(x >= threshold, torch.ones_like(x), -torch.ones_like(x))
    @staticmethod
    def backward(ctx, grad): return grad, None

def binarize(x, threshold): return SignSTE.apply(x, threshold)

class BinaryConv(nn.Conv2d):
    def forward(self, x):
        wb=binarize(self.weight, torch.zeros((),device=self.weight.device)); return F.conv2d(x,wb,self.bias,self.stride,self.padding,self.dilation,self.groups)

class BinaryFCN(nn.Module):
    """Input 1x64x64 -> binary conv16 -> grouped conv64 -> 1x1 conv64 heatmap."""
    def __init__(self, output_channels=1):
        super().__init__()
        self.conv1=BinaryConv(1,16,4,padding=0,bias=False); self.bn1=nn.BatchNorm2d(16); self.t1=nn.Parameter(torch.zeros(16))
        self.conv2=BinaryConv(16,128,3,padding=1,groups=8,bias=False); self.bn2=nn.BatchNorm2d(128); self.t2=nn.Parameter(torch.zeros(64))
        self.conv3=BinaryConv(64,64,1,bias=False); self.bn3=nn.BatchNorm2d(64)
        self.head=nn.Conv2d(64,output_channels,1,bias=True)
    def forward(self,x, return_features=False):
        x=self.conv1(F.pad(x,(1,2,1,2))); x=self.bn1(x)-self.t1[None,:,None,None]; x=binarize(x,0)
        x=self.conv2(x); x=self.bn2(x)
        # 128 intermediate maps are fused in pairs, as in the 8-group design.
        x=x.reshape(x.shape[0],64,2,x.shape[2],x.shape[3]).sum(2); x=binarize(self.bn2_fused(x) if hasattr(self,'bn2_fused') else x, self.t2[None,:,None,None])
        x=self.conv3(x); x=F.relu(self.bn3(x)); heat=self.head(x)
        return (heat, x) if return_features else heat
