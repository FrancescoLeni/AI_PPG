
import numpy as np
import random
import torch



# torch dataset builder for crops (to be used for CNN classifier for example)
class CropsDataset(torch.utils.data.Dataset):
    def __init__(self, data, mode="binary", transform=None, target_transform=None, stratify=True):
        """
            :param
                --data: Crops object split for train, val, test respectively (e.g. data=Crops.train)
                --mode: ["binary", "all"] defines whether to work with binary or allclass problem
                --stratify: whether to have equal class distribution in dataset (actually equal only for binary)
                            (set to FALSE during test to exploit all data)
        """
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

        self.transform = transform
        self.target_transform = target_transform

        self.stratify = stratify

        self.build()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx]
        data = torch.from_numpy(data.astype('float32')).permute(1,0)  # C-L for 1Dconvs
        target = torch.LongTensor([target])
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

    def build(self):
        if self.stratify:
            self.shuffle_N()
            N_ = self.N[0:self.half_len]
            N = [(x, y) for x, y in zip(N_, list(np.zeros(self.half_len)))] # N labelled as 0
        else:
            N = [(x, y) for x, y in zip(self.N, list(np.zeros(self.half_len)))]  # N labelled as 0

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

    # randomly shuffles over-numbered N sample for subsampling
    def shuffle_N(self):
        random.shuffle(self.N)



