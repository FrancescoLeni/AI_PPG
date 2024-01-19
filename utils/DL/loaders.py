
import numpy as np
import random
import pandas as pd
import torch
import json
from matplotlib import pyplot as plt

from utils.DL.collates import padding_x


# torch dataset builder for crops (to be used for CNN classifier for example)
class CropsDataset(torch.utils.data.Dataset):
    def __init__(self, data, mode="binary", stratify=True, normalization='min_max', stats_file='all_amplitude.csv',
                 raw=False, sig_mode='single', bi_head=False):
        """
            :param
                --data: Crops object split for train, val, test respectively (e.g. data=Crops.train)
                --mode: ["binary", "all"] defines whether to work with binary or allclass problem
                --stratify: whether to have equal class distribution in dataset (actually equal only for binary)
                            (set to FALSE during test to exploit all data)
                --normalization: how to normalize data [None, 'min_max', 'RobustScaler', 'Z-score']
                --stats_file: name of .csv file containing amplitude stats about sequences
                --raw: flag for whether you are passing raw signal crops (used to select right stats in norm)
        """

        self.mode = mode

        self.signal_mode = sig_mode
        self.bi_head = bi_head

        # loading computed stats
        df = pd.read_csv(f'data/{stats_file}')
        df2 = pd.read_csv('data/divided_amp_stats.csv')   # stats divided x signal
        df3 = pd.read_csv('data/all_others_amp.csv')     # stats for derivatives
        self.divided_stats, self.stats = self.build_stats(df, df2, df3)

        self.raw = raw

        self.V = []
        self.S = []
        self.N = []

        # populating above
        self.build_N_V_S(data, normalization)

        if mode == 'binary':
            self.half_len = len(self.V)+len(self.S)
        else:
            self.half_len = len(self.S)  # V is the less present, I'm augmenting the number of positive this way
            print(f'S: {len(self.S)}, V: {len(self.V)}, N: {len(self.N)}')


        self.stratify = stratify

        self.data = []
        # populating above
        self.build()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, target = self.data[idx]
        data = torch.from_numpy(data.astype('float32')).permute(1, 0)  # C-L for 1Dconvs
        if not self.bi_head:
            target = torch.LongTensor([target])
        else:
            if target == 2:
                t1 = 1
            else:
                t1 = target.copy()

            target = torch.LongTensor([t1, target])

        return data, target

    def build_stats(self, d1, d2, d3):
        d1 = d1.to_dict(orient='records')[0]
        d2 = d2.set_index('name')
        d3 = d3.to_dict(orient='records')[0]
        d1.update(d3)

        return d2, d1

    def build_N_V_S(self, data, normalization):
        if self.signal_mode == 'single':
            for x, y in data:
                x = self.normalize(x, mode=normalization, data='signal', raw=self.raw)
                if y == "V":
                    self.V.append(x)
                elif y == "S":
                    self.S.append(x)
                else:
                    self.N.append(x)
        elif self.signal_mode == 'derivatives':
            for (x, _, j, a, v), y in data:
                if normalization:
                    x = self.normalize(x, mode=normalization, data='signal')
                    j = self.normalize(j, mode=normalization, data='jpg')
                    a = self.normalize(a, mode=normalization, data='apg')
                    v = self.normalize(v, mode=normalization, data='vpg')

                    x = np.concatenate((x, j, a, v), axis=-1)  # obtaining (crops_len, 4)

                if y == "V":
                    self.V.append(x)
                elif y == "S":
                    self.S.append(x)
                else:
                    self.N.append(x)
        elif self.signal_mode == 'all':
            # assumes to be given the filtered (default) parent to Crops instance
            for (x, r, j, a, v), y in data:
                if normalization:
                    x = self.normalize(x, mode=normalization, data='signal', raw=False)
                    r = self.normalize(r, mode=normalization, data='signal', raw=True)
                    j = self.normalize(j, mode=normalization, data='jpg', raw=self.raw)
                    a = self.normalize(a, mode=normalization, data='apg', raw=self.raw)
                    v = self.normalize(v, mode=normalization, data='vpg', raw=self.raw)

                    x = np.concatenate((x, r, j, a, v), axis=-1)  # obtaining (crops_len, 5)

                if y == "V":
                    self.V.append(x)
                elif y == "S":
                    self.S.append(x)
                else:
                    self.N.append(x)

    def build(self):
        self.data = self.shuffle()
        random.shuffle(self.data)

    def shuffle(self):
        if self.stratify:
            # shuffling N before subsampling
            random.shuffle(self.N)
            N_ = self.N[0:self.half_len]
            N = [(x, y) for x, y in zip(N_, list(np.zeros(self.half_len))) if 30 < x.shape[0] < 439]  # N labelled as 0
        else:
            N = [(x, y) for x, y in zip(self.N, list(np.zeros(len(self.N)))) if 30 < x.shape[0] < 439]  # N labelled as 0

        if self.mode == "binary":
            P = [(x, y) for x, y in zip(self.V + self.S, list(np.ones(self.half_len))) if 30 < x.shape[0] < 439]  # V and S labelled as 1
        else:
            P = [(x, y) for x, y in zip(self.V, list(np.ones(len(self.V)))) if 30 < x.shape[0] < 439] + \
                [(x, y) for x, y in zip(self.S, list(np.ones(len(self.S))*2)) if 30 < x.shape[0] < 439]  # V labelled as 1, S labelled as 2

        return N+P

    def normalize(self, x, mode='min_max', data='signal', raw=False):
        # adjusting shape
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        if not mode:  # no norm
            return x
        if mode == 'min_max':
            return self.min_max(x, data=data, raw=raw)
        elif mode == 'RobustScaler':
            return self.RobustScaler(x, data=data, raw=raw)
        elif mode == 'Z-score':
            return self.Z_score(x, data=data, raw=raw)
        else:
            raise TypeError('Normalization mode not recognised, use one of ["min_max", "RobustScaler", "Z-score"]')

    def min_max(self, x, data_wide=False, data='signal', raw=False):
        if not data_wide:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
        else:
            if data == 'signal':
                if raw:
                    d_min = self.stats['min_raw']
                    d_max = self.stats['max_raw']
                else:
                    d_min = self.stats['min_filtered']
                    d_max = self.stats['max_filtered']
            elif data == 'jpg':
                d_min = self.stats['jpg_min']
                d_max = self.stats['jpg_max']
            elif data == 'apg':
                d_min = self.stats['apg_min']
                d_max = self.stats['apg_max']
            elif data == 'vpg':
                d_min = self.stats['vpg_min']
                d_max = self.stats['vpg_max']
            else:
                raise TypeError('data type not recognised')

            x = (x - d_min) / (d_min - d_max)
        return x

    def RobustScaler(self, x, data='signal', raw=False):
        if data == 'signal':
            if raw:
                median = self.stats['median_raw']
                iqr = self.stats['IQR_raw']
            else:
                median = self.stats['median_filtered']
                iqr = self.stats['IQR_filtered']
        elif data == 'jpg':
            median = self.stats['jpg_median']
            iqr = self.stats['jpg_IQR']
        elif data == 'apg':
            median = self.stats['apg_median']
            iqr = self.stats['apg_IQR']
        elif data == 'vpg':
            median = self.stats['vpg_median']
            iqr = self.stats['vpg_IQR']
        else:
            raise TypeError('data type not recognised')

        return (x - median) / iqr

    def Z_score(self, x, data='signal', raw=False):
        if data == 'signal':
            if raw:
                mean = self.stats['mean_raw']
                std = self.stats['std_raw']
            else:
                mean = self.stats['mean_filtered']
                std = self.stats['std_filtered']
        elif data == 'jpg':
            mean = self.stats['jpg_mean']
            std = self.stats['jpg_std']
        elif data == 'apg':
            mean = self.stats['apg_mean']
            std = self.stats['apg_std']
        elif data == 'vpg':
            mean = self.stats['vpg_mean']
            std = self.stats['vpg_std']
        else:
            raise TypeError('data type not recognised')

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
        df2 = pd.read_csv('data/divided_amp_stats.csv')   # stats divided x signal
        df3 = pd.read_csv('data/all_others_amp.csv')     # stats for derivatives
        self.divided_stats, self.stats = self.build_stats(df, df2, df3)

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
        return self.get_mini_batch2(idx)

    def build_stats(self, d1, d2, d3):
        d1 = d1.to_dict(orient='records')[0]
        d2 = d2.set_index('column_name')
        d3 = d3.to_dict(orient='records')[0]
        d1.update(d3)

        return d2, d1

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
            # l += np.ceil(len(seq)/self.batch).astype(np.uint8)
            l += len(seq) // self.batch
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


class WindowedSeq:
    def __init__(self, data, mode, normalization=None, stats_file='all_amplitude.csv', raw=False, window=1200, stride=600):

        self.raw = raw
        self.window = window
        self.stride = stride
        self.mode = mode
        self.norm_mode = normalization

        # loading computed stats
        df = pd.read_csv(f'data/{stats_file}')
        df2 = pd.read_csv('data/divided_amp_stats.csv')   # stats divided x signal
        df3 = pd.read_csv('data/all_others_amp.csv')     # stats for derivatives
        self.divided_stats, self.stats = self.build_stats(df, df2, df3)

        self.data = data
        self.windows = []
        self.build_windows()  # filling windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]

        data = torch.from_numpy(x.astype('float32')).permute(1, 0)  # to C-L
        target = torch.LongTensor(y)

        return data, target

    def build_stats(self, d1, d2, d3):
        d1 = d1.to_dict(orient='records')[0]
        d2 = d2.set_index('name')
        d3 = d3.to_dict(orient='records')[0]
        d1.update(d3)

        return d2, d1

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

    def build(self):
        pass

    def build_windows(self):
        for data in self.data:
            self.windows += self.get_one_signal_windows(data)

    def get_one_signal_windows(self, data_pack):
        data, peaks, labels, on, apg, vpg, jpg = data_pack
        on = np.array(on)

        pad_len = data.shape[0] % self.window
        if pad_len != 0:
            pad = np.zeros((pad_len, 1))
            data = np.concatenate((pad, data), axis=0)
            peaks += pad_len  # padding in front => I have to shift all the peaks positions
            on += pad_len

        # creating dummy peaks locations to extract more context in the model (surroundings)
        surroundings = [peaks[i]+k for i in range(peaks.shape[0]) for k in range(1, 6, 1) if peaks[i]+k < data.shape[0]]
        surroundings += [peaks[i]-k for i in range(peaks.shape[0]) for k in range(1, 6, 1) if peaks[i]-k >= 0]
        surroundings = sorted(surroundings)

        # creating positional map for labels
        label_map = np.ones(data.shape)*(-1)  # I assume background == -1
        for i, pos in enumerate(peaks):
            label_map[pos] = self.map(labels[i])

        # # adding onset info (coded as 7)
        # for i, pos in enumerate(on):
        #     label_map[pos] = 7

        # adding surroundings (coded as 5)
        for i, pos in enumerate(surroundings):
            label_map[pos] = 5

        windows = []
        for j, i in enumerate(range(0, data.shape[0] - self.window, self.stride)):
            labs = label_map[i:i+self.window]
            now_lab = labs.copy()

            # removing useless surroundings
            if labs[0] == 5:
                # if labs[5] != 0 or labs[5] != 1 or labs[5] != 2:
                #     continue  # totally skipping those windows (too complex tensor handling)
                if labs[5] == -1:  # if it is not part of a peak => eliminate
                    now_lab[0:5] = -1
            #
            # if labs[0] == 0 or labs[-1] == 0 or labs[0] == 1 or labs[-1] == 1 or labs[0] == 2 or labs[-1] == 2:
            #     continue  # too complex handling

            if labs[-1] == 5:
                # if labs[-6] != 0 or labs[-6] != 1 or labs[-6] != 2:
                #     continue  # totally skipping those windows (too complex tensor handling)
                if labs[-6] == -1:  # if it is not part of a peak => eliminate
                    now_lab[-6:] = -1

            # consider the window only if it contains at least one abnormal peak
            if np.isin(1, now_lab) or np.isin(2, now_lab):

                # onss = np.where(now_lab == 7)
                # pkss = np.where((now_lab == 0) | (now_lab == 1) | (now_lab == 2))
                #
                # ons, _ = onss
                # pks, _ = pkss

                # # placing first onset at beginning
                # if pad_len - self.stride * j <= 0:
                #     if ons[0] < pks[0]:
                #         now_lab[ons[0]] = -1
                #     now_lab[0] = 7
                # # placing last onset at last
                # if ons[-1] > pks[-1]:
                #     now_lab[ons[-1]] = -1
                # now_lab[-1] = 7
                #
                # onss = np.where(now_lab == 7)
                # pkss = np.where((now_lab == 0) | (now_lab == 1) | (now_lab == 2))
                # ons, _ = onss
                # pks, _ = pkss

                # if ons.shape[0]-1 != pks.shape[0]:
                #     continue
                #     if ons.shape[0]-1 < pks.shape[0]:
                #         plt.plot(data[i:i+self.window])
                #         plt.plot(now_lab)
                #         plt.show()

                windows.append((self.normalize(data[i:i+self.window]), now_lab))

        return windows

    def normalize(self, x):
        mode = self.norm_mode

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




