
import numpy as np
import random
import torch




class CropsDataset(torch.utils.data.Dataset):
    def __init__(self, data, mode="binary", transform = None, target_transform = None):

        if mode not in ["binary", "all"]:
            raise ValueError(f"Mode {mode} not yet supported, chose between ""binary"" and ""all"" ")
        else:
            self.mode = mode

        self.V = []
        self.S = []
        self.N = []
        for x, y in data:
            if y == "V" :
                self.V.append(x)
            elif y == "S":
                self.S.append(x)
            else:
                self.N.append(x)

        self.half_len = len(self.V)+len(self.S)

        self.build()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx]
        data = torch.from_numpy(data.astype('float32')).permute(1,0)   # Channel-lenght
        target = torch.LongTensor([target])
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

    def build(self):
        self.shuffle_N()
        N_ = self.N[0:self.half_len]
        N = [(x, y) for x, y in zip(N_, list(np.zeros(self.half_len)))] # N labelled as 0

        if self.mode == "all":
            P = [(x, y) for x, y in zip(self.V, list(np.ones(len(self.V))))] + \
                [(x, y) for x, y in zip(self.S, list(np.ones(len(self.S)))*2)] # V labelled as 1, S labelled as 2
        else: # binary
            P = [(x, y) for x, y in zip(self.V+self.S, list(np.ones(self.half_len)))] # V and S labelled as 1

        if hasattr(self, 'data'):
            self.data = N+P
        else:
            setattr(self, 'data', N+P)
        random.shuffle(self.data)

    def shuffle_N(self):
        random.shuffle(self.N)