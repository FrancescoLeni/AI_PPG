
import numpy as np
import random
import pandas as pd
import torch

from utils.DL.collates import padding_x


# torch dataset builder for crops (to be used for CNN classifier for example)
class CropsDataset(torch.utils.data.Dataset):
    def __init__(self, data, mode="binary", transform=None, target_transform=None, stratify=True,
                 normalization='min_max', stats_file='all_amplitude.csv', raw=False):
        """
            :param
                --data: Crops object split for train, val, test respectively (e.g. data=Crops.train)
                --mode: ["binary", "all"] defines whether to work with binary or allclass problem
                --stratify: whether to have equal class distribution in dataset (actually equal only for binary)
                            (set to FALSE during test to exploit all data)
                --normalization: how to normalize data ['min_max', 'RobustScaler', 'Z-score']
                --stats_file: name of .csv file containing amplitude stats about sequences
                --raw: flag for whether you are passing raw signal crops (used to select right stats in norm)
        """
        if mode not in ["binary", "all"]:
            raise ValueError(f"Mode {mode} not yet supported, chose between ""binary"" and ""all"" ")
        else:
            self.mode = mode

        # loading computed stats
        df = pd.read_csv(f'data/{stats_file}')
        self.stats = df.to_dict(orient='records')[0]
        self.raw = raw

        self.V = []
        self.S = []
        self.N = []
        for x, y in data:
            if len(x.shape) == 1:
                x = x[:, np.newaxis]
            if normalization:
                x = self.normalize(x, mode=normalization)
            if y == "V":
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
            # shuffling N before subsampling
            random.shuffle(self.N)
            N_ = self.N[0:self.half_len]
            N = [(x, y) for x, y in zip(N_, list(np.zeros(self.half_len))) if 30 < x.shape[0] < 439]  # N labelled as 0
        else:
            N = [(x, y) for x, y in zip(self.N, list(np.zeros(len(self.N)))) if 30 < x.shape[0] < 439]  # N labelled as 0

        if self.mode == "all":
            P = [(x, y) for x, y in zip(self.V, list(np.ones(len(self.V))))] + \
                [(x, y) for x, y in zip(self.S, list(np.ones(len(self.S))*2)) if 30 < x.shape[0] < 439]  # V labelled as 1, S labelled as 2
        else: # binary
            P = [(x, y) for x, y in zip(self.V+self.S, list(np.ones(self.half_len))) if 30 < x.shape[0] < 439] # V and S labelled as 1

        if hasattr(self, 'data'):
            self.data = N+P
        else:
            setattr(self, 'data', N+P)
        random.shuffle(self.data)

    def normalize(self, x, mode='min_max'):
        if mode == 'min_max':
            return self.min_max(x)
        elif mode == 'RobustScaler':
            return self.RobustScaler(x)
        elif mode == 'Z-score':
            return self.Z_score(x)
        else:
            raise TypeError('Normalization mode not recognised, use one of ["min_max", "RobustScaler", "Z-score"]')

    def min_max(self, x, data_wide=False):
        if not data_wide:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        else:
            if self.raw:
                d_min = self.stats['min_raw']
                d_max = self.stats['max_raw']
            else:
                d_min = self.stats['min_filtered']
                d_max = self.stats['max_filtered']
            x = (x - d_min) / (d_min - d_max)
        return x

    def RobustScaler(self, x):
        if self.raw:
            median = self.stats['median_raw']
            iqr = self.stats['IQR_raw']
        else:
            median = self.stats['median_filtered']
            iqr = self.stats['IQR_filtered']

        return (x - median) / iqr

    def Z_score(self, x):
        if self.raw:
            mean = self.stats['mean_raw']
            std = self.stats['std_raw']
        else:
            mean = self.stats['mean_filtered']
            std = self.stats['std_filtered']

        return (x - mean) / std


class CroppedSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, mini_batch=256, mode="binary", normalization=None, stats_file='all_amplitude.csv', raw=False):
        """
            :param
                --data: List of cropped sequences for train or val or test (e.g. CroppedSeq.Train)
                --mode: ["binary", "all"] defines whether to work with binary or all-class problem
                --mini_batch: numbers of crops to be analyzed as a single sequence (NOTE it is implemented differently from actual batch size)
                --normalization: how to normalize data ['min_max', 'RobustScaler', 'Z-score']
                --stats_file: name of .csv file containing amplitude stats about sequences
                --raw: flag for whether you are passing raw signal crops (used to select right stats in norm)
        """
        if mode not in ["binary", "all"]:
            raise ValueError(f"Mode {mode} not yet supported, chose between ""binary"" and ""all"" ")
        else:
            self.mode = mode

        # loading computed stats
        df = pd.read_csv(f'data/{stats_file}')
        self.stats = df.to_dict(orient='records')[0]
        self.raw = raw

        self.batch = mini_batch

        # list of sequences of tuple (crop, lab)
        self.data = [[(self.normalize(x, mode=normalization), self.map(y)) for x, y in seq] for seq in data]

        self.len = self.get_length()

        self.now_seq = 0  # to keep track of sequences
        self.now_crops = 0  # to keep track of crops in sequence

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.get_mini_batch(idx)

    def map(self, y):
        if y == 'N':
            y = 0
        else:
            if self.mode == 'binary':
                y = 1
            else:
                if y == 'V':
                    y = 1
                else:  # 'S'
                    y = 2
        return y

    def get_mini_batch(self, idx):
        seq = self.data[self.now_seq]
        last_sfigato = 0
        # print(self.now_seq, self.now_crops, len(seq))
        if self.now_crops+self.batch >= len(seq):
            if self.now_crops == len(seq)-1:
                batch_split = seq[self.now_crops]
                last_sfigato = 1
            else:
                batch_split = seq[self.now_crops:-1]
            self.now_seq += 1
            self.now_crops = 0
        else:
            batch_split = seq[self.now_crops:self.now_crops+self.batch]
            self.now_crops += self.batch
        if not last_sfigato:
            batch_list = [(torch.from_numpy(x.astype('float32')).permute(1, 0), torch.LongTensor([y])) for x, y in batch_split]  # permuting to get C-L
        else:
            x, y = batch_split
            batch_list = [(torch.from_numpy(x.astype('float32')).permute(1, 0), torch.LongTensor([y]))]  # permuting to get C-L
        batch_x, batch_y = padding_x(batch_list)

        return batch_x, batch_y

    # skips last element
    def get_mini_batch2(self, idx):

        seq = self.data[self.now_seq]
        # print(self.now_seq, self.now_crops, len(seq))
        if self.now_crops+self.batch > len(seq):
            self.now_seq += 1
            seq = self.data[self.now_seq]
            self.now_crops = 0

        batch_split = seq[self.now_crops:self.now_crops+self.batch]
        self.now_crops += self.batch
        batch_list = [(torch.from_numpy(x.astype('float32')).permute(1, 0), torch.LongTensor([y])) for x, y in batch_split]  # permuting to get C-L

        batch_x, batch_y = padding_x(batch_list)

        return batch_x, batch_y

    def build(self):
        # actually it should be named shuffle, but it's to be consistent with prev dataloader in calls inside train loop
        self.now_seq = 0
        self.now_crops = 0
        #random.shuffle(self.data)

    def get_length(self):
        # tot numbers of iterations is the numbers of crops per sequence divided the mini_batch shape
        l = 0
        for seq in self.data:
            l += np.ceil(len(seq)/self.batch).astype(np.uint8)
            # l += len(seq) // self.batch
        return l

    def normalize(self, x, mode='min_max'):
        # adjusting shape
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        if not mode:  # no norm
            return x
        if mode == 'min_max':
            return self.min_max(x)
        elif mode == 'RobustScaler':
            return self.RobustScaler(x)
        elif mode == 'Z-score':
            return self.Z_score(x)
        else:
            raise TypeError('Normalization mode not recognised, use one of ["min_max", "RobustScaler", "Z-score"]')

    def min_max(self, x, data_wide=False):
        if not data_wide:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        else:
            if self.raw:
                d_min = self.stats['min_raw']
                d_max = self.stats['max_raw']
            else:
                d_min = self.stats['min_filtered']
                d_max = self.stats['max_filtered']
            x = (x - d_min) / (d_min - d_max)
        return x

    def RobustScaler(self, x):
        if self.raw:
            median = self.stats['median_raw']
            iqr = self.stats['IQR_raw']
        else:
            median = self.stats['median_filtered']
            iqr = self.stats['IQR_filtered']

        return (x - median) / iqr

    def Z_score(self, x):
        if self.raw:
            mean = self.stats['mean_raw']
            std = self.stats['std_raw']
        else:
            mean = self.stats['mean_filtered']
            std = self.stats['std_filtered']

        return (x - mean) / std





