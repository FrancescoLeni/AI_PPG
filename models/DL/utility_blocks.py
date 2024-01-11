import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(nn.Module):
    def __init__(self, layer_scale_init_value, dim):
        super().__init__()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        return x


class SelfAttentionModule(nn.Module):
    def __init__(self, c_in, k=9, s=1, p=4):
        super().__init__()
        self.conv = nn.Conv1d(c_in, 1, 1, 1)
        self.spatial = nn.Sequential(nn.Conv1d(1, 32, k, s, p),
                                     nn.ReLU(),
                                     nn.Conv1d(32, 1, k, s, p)
                                     )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # (bs, C, L)
        x_ = self.conv(x)
        # (bs, 1, L)
        x_ = self.spatial(x_)
        # (bs, 1, L)
        scale = self.act(x_)

        return x * scale

