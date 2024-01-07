import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from models.DL.models_blocks import CNeXtBlock, CNeXtStem, CNeXtDownSample


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1,10,3,1,1)
        self.fc = nn.Linear(10,2)

    def forward(self,x):
        x = self.conv(x)
        return self.fc(x.mean(dim=2).view(x.shape[0], -1)) # x.mean is GAP


class ConvNeXt(nn.Module):
    # ConvNeXt-T: C = (96; 192; 384; 768), B = (3; 3; 9; 3)
    # ConvNeXt-S: C = (96; 192; 384; 768), B = (3; 3; 27; 3)
    # ConvNeXt-B: C = (128; 256; 512; 1024), B = (3; 3; 27; 3)
    # ConvNeXt-L: C = (192; 384; 768; 1536), B = (3; 3; 27; 3)
    # ConvNeXt-XL: C = (256; 512; 1024; 2048), B = (3; 3; 27; 3)
    def __init__(self, num_classes, model_type='T', drop_path=0.):
        super().__init__()

        if model_type == 'T':
            self.B = [3, 3, 9, 3]
            self.C = [96, 192, 384, 768]
        else:
            self.B = [3, 3, 27, 3]

        if model_type == 'S':
            self.C = [96, 192, 384, 768]
        elif model_type == 'B':
            self.C = [128, 385, 768, 1024]
        elif model_type == 'L':
            self.C = [192, 384, 768, 1536]
        elif model_type == 'XL':
            self.C = [256, 512, 1024, 2048]

        self.stem = CNeXtStem(1, self.C[0])

        self.S = nn.ModuleList([nn.Sequential(*(CNeXtBlock(self.C[i], drop_path=drop_path, layer_scale_init_value=0)
                                                for _ in range(self.B[i]))) for i in range(4)])
        self.DownSample = nn.ModuleList([CNeXtDownSample(self.C[i], self.C[i+1], 2, 2, 0) for i in range(3)])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1]//4),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//4, self.C[-1]//16),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//16, self.C[-1]//64),
                                        nn.GELU(),
                                        nn.Linear(self.C[-1]//64, num_classes)
                                        )

    def forward(self, x):

        x = self.stem(x)

        x = self.S[0](x)
        x = self.DownSample[0](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)










