"""Optional PyTorch graph; imports only when PyTorch is installed."""
def build_model():
    import torch.nn as nn
    import torch.nn.functional as F
    class ReferenceCNN(nn.Module):
        def __init__(self):
            super().__init__(); self.conv=nn.Conv2d(1,16,4); self.pool=nn.MaxPool2d(4,4); self.fc=nn.Linear(16*16*16,10)
        def forward(self,x):
            x=F.pad(x,(1,2,1,2)); return self.fc(self.pool(F.relu(self.conv(x))).flatten(1))
    return ReferenceCNN()
