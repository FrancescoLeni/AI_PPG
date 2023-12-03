import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchmetrics



class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1,10,3,1,1)
        self.fc = nn.Linear(10,2)

    def forward(self,x):
        x = self.conv(x)


        return self.fc(x.mean(dim=2)) # x.mean is GAP










