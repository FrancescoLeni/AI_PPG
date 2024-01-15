import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from models.DL.utility_blocks import LayerNorm, DropPath, SelfAttentionModule, LayerScale, PositionalEncoding


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


class ConvNormAct(nn.Module):
    def __init__(self, c1, c2, k, s, p, act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.norm = nn.BatchNorm1d(c2)
        self.conv = nn.Conv1d(c1, c2, k, s, p)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, c1, k=5, s=1, p=2):
        super().__init__()

        self.m = nn.Sequential(ConvNormAct(c1, c1//2, 1, 1, 0),
                               ConvNormAct(c1//2, c1//2, k, s, p),
                               ConvNormAct(c1//2, c1, 1, 1, 0)
                               )

    def forward(self, x):
        return self.m(x) + x


class ResBlockDP(nn.Module):
    def __init__(self, c1, k=5, s=1, p=2, dp=0.1):
        super().__init__()

        self.dp = nn.Dropout1d(dp)
        self.m = nn.Sequential(self.dp,
                               ConvNormAct(c1, c1//2, 1, 1, 0),
                               ConvNormAct(c1//2, c1//2, k, s, p),
                               ConvNormAct(c1//2, c1, 1, 1, 0)
                               )

    def forward(self, x):
        return self.m(x) + x


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.q = nn.Linear(input_dim, d_model, bias=False)
        self.k = nn.Linear(input_dim, d_model, bias=False)
        self.v = nn.Linear(input_dim, d_model, bias=False)

        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.feedforward = nn.Sequential(nn.Linear(d_model, d_model//2),
                                         nn.GELU(),
                                         nn.Linear(d_model//2, d_model)
                                         )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):

        x = self.norm1(x)

        q = self.positional_encoding(self.q(x))
        k = self.positional_encoding(self.k(x))
        v = self.positional_encoding(self.v(x))

        attn_output, _ = self.mha(q, k, v, need_weights=False)

        x = q + self.dropout(attn_output)  # da capire se ha senso questo shortcut

        x = self.norm2(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)

        return x



