import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from models.DL.models_blocks import CNeXtBlock, CNeXtStem, CNeXtDownSample, ResBlock, ConvNormAct
from models.DL.utility_blocks import SelfAttentionModule


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
    def __init__(self, num_classes, model_type='T', drop_path=0., layer_scale_init_value=1e-6):
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

        self.stem = CNeXtStem(1, self.C[0], k=2, s=2)

        self.S = nn.ModuleList([nn.Sequential(*(CNeXtBlock(self.C[i], drop_path=drop_path, layer_scale_init_value=layer_scale_init_value)
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


class ConvNeXtSAM(nn.Module):
    # ConvNeXt-T: C = (96; 192; 384; 768), B = (3; 3; 9; 3)
    # ConvNeXt-S: C = (96; 192; 384; 768), B = (3; 3; 27; 3)
    # ConvNeXt-B: C = (128; 256; 512; 1024), B = (3; 3; 27; 3)
    # ConvNeXt-L: C = (192; 384; 768; 1536), B = (3; 3; 27; 3)
    # ConvNeXt-XL: C = (256; 512; 1024; 2048), B = (3; 3; 27; 3)
    def __init__(self, num_classes, model_type='T', drop_path=0., layer_scale_init_value=1e-6):
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

        self.stem = CNeXtStem(1, self.C[0], k=2, s=2)

        self.SAM = nn.ModuleList([SelfAttentionModule(self.C[i]) for i in range(4)])

        self.S = nn.ModuleList([nn.Sequential(*(CNeXtBlock(self.C[i], drop_path=drop_path, layer_scale_init_value=layer_scale_init_value)
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
        x = self.SAM[0](x)

        x = self.S[0](x)
        x = self.DownSample[0](x)
        x = self.SAM[1](x)

        x = self.S[1](x)
        x = self.DownSample[1](x)
        x = self.SAM[2](x)

        x = self.S[2](x)
        x = self.DownSample[2](x)
        x = self.SAM[3](x)

        x = self.S[3](x)
        x = x.mean(dim=2).view(x.shape[0], -1)

        return self.classifier(x)


class ResNet1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.C = [128, 385, 768, 1024]
        self.B = [3, 4, 6, 3]

        self.stem = ConvNormAct(1, self.C[0], 9, 2, 4)

        self.DownSample = nn.ModuleList([CNeXtDownSample(self.C[i], self.C[i + 1], 5, 2, 2) for i in range(3)])

        self.S = nn.ModuleList([nn.Sequential(*(ResBlock(self.C[i]) for _ in range(self.B[i]))) for i in range(4)])

        self.classifier = nn.Sequential(nn.Linear(self.C[-1], self.C[-1] // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 4, self.C[-1] // 16),
                                        nn.ReLU(),
                                        nn.Linear(self.C[-1] // 16, num_classes)
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









