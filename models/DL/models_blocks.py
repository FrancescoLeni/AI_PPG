import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from models.DL.utility_blocks import LayerNorm, DropPath, SelfAttentionModule, LayerScale


class CNeXtStem(nn.Module):
    def __init__(self, c1, c2, k=4, s=4, p=0):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, p)
        self.norm = LayerNorm(c2)

    def forward(self, x):
        return self.norm(self.conv(x))


class CNeXtBlock(nn.Module):  #come paper
    def __init__(self, dim, k=7, p=3, shortcut=True, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        c_ = 4 * dim
        self.act = nn.GELU()
        self.add = shortcut
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=k, padding=p, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, c_, 1, 1, 0) # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Conv1d(c_, dim, 1, 1, 0)
        # layer scale
        self.gamma = LayerScale(layer_scale_init_value, dim) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma:
            x = self.gamma(x) * x

        if self.add:
            x = input + self.drop_path(x)

        return x


class CNeXtDownSample(nn.Module):
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.norm = LayerNorm(c1)
        self.layer = nn.Conv1d(c1, c2, k, s, p)

    def forward(self, x):
        x = self.layer(self.norm(x))
        return x


class CNeXtSAMDownSample(nn.Module):   # ho aggiunto la GELU
    def __init__(self, c1, c2, k, s, p):
        super().__init__()
        self.layer = nn.Conv2d(c1, c2, k, s, p)
        self.SA = cbam.SpatialGate1(c2)
        #self.act = nn.GELU()
    def forward(self,x):
        x = self.layer(x)
        x = self.SA(x)
        #x = self.act(x)
        return x