
from preprocessing.filtering import Preprocess
from preprocessing.fiducials import FpCollection
from preprocessing import PPG

from preprocessing import Fiducials, Biomarkers
import pyPPG.biomarkers as BM

from sklearn.model_selection import train_test_split
from pathlib import Path
from dotmap import DotMap
import numpy as np
import pandas as pd
import scipy.io
import random
import os
import h5py
import json

class OneSignal:
    def __init__(self, data_name=None):
        """
        :param
            --data_name: filename like 'S001_128.mat' or tuple (data_path,label_path,peaks_path)
        """
        super().__init__()
        parent = 'dataset'
        child = ["data", "label", "peaks"]
        if not isinstance(data_name,tuple):
            self.data_path = os.path.join(parent, child[0], data_name)
            self.label_path = os.path.join(parent, child[1], data_name)
            self.peaks_path = os.path.join(parent, child[2], data_name)
            self.name = data_name
        else:
            self.data_path = data_name[0]
            self.label_path = data_name[1]
            self.peaks_path = data_name[2]
            self.name = data_name[0].split('.')[0][-8:]

        self.raw = scipy.io.loadmat(self.data_path)['ppg']
        self.fs = int(self.data_path.split('.')[0][-3:])
        self.ppg = "call .filter() to calculate"
        self.vpg = "call .filter() to calculate"
        self.jpg = "call .filter() to calculate"
        self.apg = "call .filter() to calculate"
        # self.on = "call .filter() to calculate"
        self.peaks = scipy.io.loadmat(self.peaks_path)['speaks']
        self.labels = scipy.io.loadmat(self.label_path)['labels']

        self.v = np.squeeze(self.raw)
        correction = pd.DataFrame()
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction.loc[0, corr_on] = True
        self.correction = correction

        # Prepare onset alignment with provided peaks (from fiducials)
        self.indx = 0 # peak to crop
        self.indx_max = int(self.peaks.shape[0])

    def filter(self, fL=0.5, fH=4.3, order=4, sm={'ppg':50,'vpg':10,'apg':10,'jpg':10}, data_min=-90, data_max=90):

        print(f'filtering signal {self.name}...')
        # Class which the functions need...
        s = DotMap()
        s.end_sig = -1
        s.v = np.squeeze(self.raw)
        s.fs = self.fs
        s.filtering = True
        s.name = self.name
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction = pd.DataFrame()
        correction.loc[0, corr_on] = True
        s.correct = correction

        # Filters signal through BP filter [fL,fH]
        prep = Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm)
        s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)
        self.ppg = s.ppg
        self.vpg = s.vpg
        self.apg = s.apg
        self.jpg = s.jpg

        self.s = PPG(s)

    def get_fiducials(self):
        # Initialize fiducials package
        fpex = FpCollection(s=self.s)
        # Extract fiducial points (e.g. 'on': peak onsets)
        fiducials = fpex.get_fiducials(s=self.s, ext_peaks=self.peaks)
        # Create a fiducials class
        self.fp = Fiducials(fiducials)

    def get_biomarkers(self):
        # Initialize biomarkers package
        bmex = BM.BmCollection(s=self.s, fp=self.fp)
        # Extract biomarkers
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()

        # tmp_keys=bm_stats.keys()
        # print('Statistics of the biomarkers:')
        # for i in tmp_keys: print(i,'\n',bm_stats[i])

        # Create a biomarkers class
        self.bm = Biomarkers(bm_defs, bm_vals, bm_stats)

    def align_onsets(self):
        # Prepare onset alignment with provided peaks (from fiducials)
        self.get_fiducials()
        onsets, peaks = list(self.fp.get_fp().on).copy(), list(self.peaks.flatten())
        
        # Always include the first onset in the filtered list
        # If no onset detected before first peak
        if onsets[0] >= peaks[0]:
            print('No onset detected before first peak')
            filtered_onsets = [0]
            # Remaining onsets to be considered: all of them
            remaining_onsets = onsets
        else:
            onsets_before_first_peak = [onset for onset in onsets if onset < peaks[0]]
            filtered_onsets = [onsets_before_first_peak[-1]]
            # Remaining onsets to be considered: 
            remaining_onsets = onsets[len(onsets_before_first_peak):]
        
        # Iterate through peaks to determine the closest onset for each peak
        for i in range(len(peaks) - 1):
            # Current and next peaks
            peak = peaks[i]
            next_peak = peaks[i + 1]

            # Find the onsets that lie between the current and next peak
            possible_onsets = [onset for onset in remaining_onsets if (peak < onset) and (onset < next_peak)]

            if not possible_onsets:
                # If no onset is detected between two peaks, add one in the middle
                middle_onset = int((peak + next_peak) / 2)
                filtered_onsets.append(middle_onset)
                # print('onset added at: ', middle_onset)
            else:
                # Find the onset that minimizes the distance to the next peak
                closest_onset = min(possible_onsets, key=lambda x: abs(x - next_peak))
                # Include the closest onset in the filtered list
                filtered_onsets.append(closest_onset)
                # Remove the chosen onset from the remaining onsets
                remaining_onsets = [onset for onset in remaining_onsets if (onset not in possible_onsets) and (onset > next_peak)]
                # if len(possible_onsets) >= 2:
                #     print('multiple onsets between peaks: ', possible_onsets)

        # Include the last onset in the filtered list
        if remaining_onsets:
            # print('remaining onsets: ', remaining_onsets)
            filtered_onsets.append(remaining_onsets[-1])
        else:
            # print('added end of signal')
            filtered_onsets.append(int(len(self.ppg)-1))
        
        self.on = np.array(filtered_onsets, dtype=int)
    
    def crop(self, raw=False):
        crop = self.ppg[self.on[self.indx]:self.on[self.indx+1]]
        lab = self.labels[self.indx]
        if not raw:
            self.indx += 1
            return crop, lab
        else:
            c_raw = self.raw[self.on[self.indx]:self.on[self.indx+1]]
            l_raw = self.labels[self.indx]
            c_j = self.jpg[self.on[self.indx]:self.on[self.indx+1]]
            c_v = self.vpg[self.on[self.indx]:self.on[self.indx+1]]
            c_a = self.apg[self.on[self.indx]:self.on[self.indx+1]]
            self.indx += 1
            return (crop, lab), (c_raw, l_raw), (c_j, c_v, c_a)

    def fixed_crop(self, left=70, right=137, all=True):
        c_raw = []
        c_filtered = []
        c_jpg = []
        c_vpg = []
        c_apg = []

        # padding to allow windowing
        if all:
            raw = np.concatenate((np.zeros((300, 1)), self.raw))
            raw = np.concatenate((raw, np.zeros((300, 1))))
            ppg = np.concatenate((np.zeros((300, 1)), self.ppg[:, np.newaxis]))
            ppg = np.concatenate((ppg, np.zeros((300, 1))))
            jpg = np.concatenate((np.zeros((300, 1)), self.jpg[:, np.newaxis]))
            jpg = np.concatenate((jpg, np.zeros((300, 1))))
            vpg = np.concatenate((np.zeros((300, 1)), self.vpg[:, np.newaxis]))
            vpg = np.concatenate((vpg, np.zeros((300, 1))))
            apg = np.concatenate((np.zeros((300, 1)), self.apg[:, np.newaxis]))
            apg = np.concatenate((apg, np.zeros((300, 1))))

            for i, peak in enumerate(self.peaks):
                peak = peak[0] + 100
                c_raw.append((raw[peak-left:peak+right], self.labels[i]))
                c_filtered.append((ppg[peak-left:peak+right], self.labels[i]))
                c_jpg.append((jpg[peak-left:peak+right], self.labels[i]))
                c_vpg.append((vpg[peak-left:peak+right], self.labels[i]))
                c_apg.append((apg[peak-left:peak+right], self.labels[i]))
            return c_raw, c_filtered, c_jpg, c_vpg, c_apg
        else:
            raw = np.concatenate((np.zeros((300, 1)), self.raw))
            raw = np.concatenate((raw, np.zeros((300, 1))))
            for i, peak in enumerate(self.peaks):
                peak = peak[0] + 100
                c_raw.append((raw[peak-left:peak+right], self.labels[i]))
            return c_raw


class Crops:
    def __init__(self, N="N_crops.h5", V="V_crops.h5", S="S_crops.h5", parent='dataset/crops', seed=36):
        super().__init__()
        self.names_list = [N, V, S]
        self.parent = Path(parent)

        self.N_crops = []
        self.V_crops = []
        self.S_crops = []
        self.N_labels = []
        self.V_labels = []
        self.S_labels = []

        # populating the above
        for names in self.names_list:
            print(f"loading {self.parent/names}...")
            with h5py.File(self.parent / names, 'r') as file:
                name = Path(names).stem
                setattr(self, name, [file[key][:] for key in file.keys() if key != 'labels'])
                setattr(self, f"{name[0]}_labels", list(file['labels'][:].astype('U')))

        self.j = {'N': [], 'V': [], 'S': []}
        self.v = {'N': [], 'V': [], 'S': []}
        self.a = {'N': [], 'V': [], 'S': []}
        #self.load_derivatives()  # populating above dict

        self.raw = {'N': [], 'V': [], 'S': []}
        #self.load_raw()  # populating above

        # compressing everything to handle splitting
        self.N = [(x, r, j, a, v) for x, r, j, a, v in zip(self.N_crops, self.raw['N'], self.j['N'], self.a['N'], self.v['N'])]
        self.V = [(x, r, j, a, v) for x, r, j, a, v in zip(self.V_crops, self.raw['V'], self.j['V'], self.a['V'], self.v['V'])]
        self.S = [(x, r, j, a, v) for x, r, j, a, v in zip(self.S_crops, self.raw['S'], self.j['S'], self.a['S'], self.v['S'])]

        # populated when split
        self.train = None
        self.val = None
        self.test = None

        self.seed = seed

    def load_raw(self, p='dataset/crops_raw'):
        p = Path(p)
        for names in self.names_list:
            print(f"loading {p / names}...")
            with h5py.File(p / names, 'r') as file:
                name = Path(names).stem.split('_')[0]
                self.raw[name] = [file[key][:] for key in file.keys() if key != 'labels']

    def load_derivatives(self, pv='dataset/crops_vpg', pa='dataset/crops_apg', pj='dataset/crops_jpg'):
        pv = Path(pv)
        pa = Path(pa)
        pj = Path(pj)
        for names in self.names_list:
            print(f"loading {pv/names}...")
            with h5py.File(pv / names, 'r') as file:
                name = Path(names).stem.split('_')[0]
                self.v[name] = [file[key][:] for key in file.keys() if key != 'labels']
            print(f"loading {pa/names}...")
            with h5py.File(pa / names, 'r') as file:
                name = Path(names).stem.split('_')[0]
                self.a[name] = [file[key][:] for key in file.keys() if key != 'labels']
            print(f"loading {pj / names}...")
            with h5py.File(pj / names, 'r') as file:
                name = Path(names).stem.split('_')[0]
                self.j[name] = [file[key][:] for key in file.keys() if key != 'labels']

    def split(self, test_size=.15, everything=False):

        # to get ONLY signal crops, either raw or filtered
        if not everything:
            x = self.V_crops + self.S_crops + self.N_crops
            y = self.V_labels + self.S_labels + self.N_labels

        # to get ALSO derivatives
        else:
            x = self.V + self.S + self.N
            y = self.V_labels + self.S_labels + self.N_labels

        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, random_state=self.seed, test_size=test_size,
                                                                    shuffle=True, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, random_state=self.seed,
                                                          test_size=len(x_test),
                                                          shuffle=True, stratify=y_train_val)
        setattr(self, 'train', [(x, y) for x, y in zip(x_train, y_train)])
        setattr(self, 'val', [(x, y) for x, y in zip(x_val, y_val)])
        setattr(self, 'test', [(x, y) for x, y in zip(x_test, y_test)])


class CroppedSeq:
    def __init__(self, parent='dataset/crops_raw/patients', seed=36):
        self.parent = Path(parent)
        self.sequences = {}
        self.train = []
        self.val = []
        self.test = []
        self.build()
        self.split()

    def build(self):
        print(f"loading signals...")
        for names in os.listdir(self.parent):
            with h5py.File(self.parent / names, 'r') as file:
                name = Path(names).stem
                sig_labs = list(file['labels'][:].astype('U'))
                sig_crops = [(file[key][:], lab) for key, lab in zip(file.keys(), sig_labs) if key != 'labels']
                self.sequences[name] = sig_crops

    def split(self):
        """
            numbers of sequences:
                --more_9 26 (23,3,0)
                --less_9 65 (46,8,11)
                --N 12 (3,5,4)
                --tot 103
                --Train 72
                --val 16
                --test 15
        """
        m9 = [Path(sig).stem for sig in os.listdir('dataset/more_9')]
        l9 = [Path(sig).stem for sig in os.listdir('dataset/less_9')]
        N = [Path(sig).stem for sig in os.listdir('dataset/only_N')]

        more_9_train = [m9[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25]]
        more_9_val = [m9[i] for i in [12, 14, 20]]

        less_9_train = [l9[i] for i in [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35,
                                        36, 37, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 63]]
        less_9_val = [l9[i] for i in [33, 38, 39, 12, 13, 20, 28, 61]]
        less_9_test = [l9[i] for i in [2, 8, 14, 47, 48, 19, 21, 54, 60, 62, 64]]

        N_train = [N[i] for i in [0, 11, 5]]
        N_val = [N[i] for i in [1, 2, 8, 7, 6]]
        N_test = [N[i] for i in [3, 4, 9, 10]]

        train = N_train + more_9_train + less_9_train
        val = N_val + more_9_val + less_9_val
        test = N_test + less_9_test

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        # self.train = [self.sequences[key] for key in train]
        # self.val = [self.sequences[key] for key in val]
        # self.test = [self.sequences[key] for key in test]

        self.train = [self.sequences[key] for key in m9]
        self.val = [self.sequences[key] for key in less_9_val]
        self.test = [self.sequences[key] for key in test]


class Sequences:
    def __init__(self, raw=False):

        self.raw = raw

        self.on = self.load_on(data_path='dataset/onsets.json')  # dict 1D array

        self.data = {}
        self.build()  # filling self.data with signals

        self.train = []
        self.val = []
        self.test = []

        self.split()  # filling .train .val .test

    def load_on(self, data_path='dataset/onsets.json'):
        with open('dataset/onsets.json', 'r') as json_file:
            loaded = json.load(json_file)
        return loaded

    def build(self):
        print('loading sequences...')
        for file_name in os.listdir('dataset/data'):
            name = Path(file_name).stem
            one = OneSignal(file_name)
            if self.raw:
                data = one.raw
            else:
                data = np.load(f'dataset/filtered/{name}.npy')

            apg = np.load(f'dataset/apg/{name}.npy')
            vpg = np.load(f'dataset/vpg/{name}.npy')
            jpg = np.load(f'dataset/jpg/{name}.npy')

            self.data[name] = (data, one.peaks, one.labels[:, np.newaxis], self.on[name], apg, vpg, jpg)

    def split(self):
        """
            numbers of sequences:
                    --more_9 26 (23,3,0)
                    --less_9 65 (49,7,9)
                    --N 12 (0,6,6)
                    --tot 103
                    --Train 72
                    --val 16
                    --test 15
        """
        m9 = [Path(sig).stem for sig in os.listdir('dataset/more_9')]
        l9 = [Path(sig).stem for sig in os.listdir('dataset/less_9')]
        N = [Path(sig).stem for sig in os.listdir('dataset/only_N')]

        more_9_train = [m9[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25]]
        more_9_val = [m9[i] for i in [12, 14, 20]]

        less_9_train = [l9[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35,
                                        36, 37, 40, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 61, 63]]
        less_9_val = [l9[i] for i in [33, 38, 39, 12, 13, 20, 28]]
        less_9_test = [l9[i] for i in [8, 14, 47, 48, 21, 54, 60, 62, 64]]

        N_val = [N[i] for i in [0, 1, 2, 8, 7, 6]]
        N_test = [N[i] for i in [3, 4, 5, 9, 10, 11]]

        train = more_9_train + less_9_train
        val = N_val + more_9_val + less_9_val
        test = N_test + less_9_test

        # random.shuffle(train)
        # random.shuffle(val)
        # random.shuffle(test)

        self.train = [self.data[key] for key in train]
        self.val = [self.data[key] for key in val]
        self.test = [self.data[key] for key in test]

        # self.train = [self.data[key] for key in m9]
        # self.val = [self.data[key] for key in less_9_val]
        # self.test = [self.data[key] for key in test]


class MLdf:
    def __init__(self, mode):
        self.train = []
        self.val = []
        self.test = []

        if mode == 'binary':
            self.get_binary()
        elif mode == 'all':
            self.get_multi()

    def get_binary(self):
        df_x_train_b = pd.read_csv('dataset/ML_split/binary/train_features.csv')
        df_y_train_b = pd.read_csv('dataset/ML_split/binary/train_labels.csv')

        df_x_val_b = pd.read_csv('dataset/ML_split/binary/validation_features.csv')
        df_y_val_b = pd.read_csv('dataset/ML_split/binary/validation_labels.csv')

        df_x_test_b = pd.read_csv('dataset/ML_split/binary/test_features.csv')
        df_y_test_b = pd.read_csv('dataset/ML_split/binary/test_labels.csv')

        self.train = [(x.tolist(), y.tolist()[0]) for (_, x), (_, y) in zip(df_x_train_b.iterrows(), df_y_train_b.iterrows())]
        self.val = [(x.tolist(), y.tolist()[0]) for (_, x), (_, y) in zip(df_x_val_b.iterrows(), df_y_val_b.iterrows())]
        self.test = [(x.tolist(), y.tolist()[0]) for (_, x), (_, y) in zip(df_x_test_b.iterrows(), df_y_test_b.iterrows())]

    def get_multi(self):
        df_x_train_b = pd.read_csv('dataset/ML_split/multiclass/train_features.csv')
        df_y_train_b = pd.read_csv('dataset/ML_split/multiclass/train_labels.csv')

        df_x_val_b = pd.read_csv('dataset/ML_split/multiclass/validation_features.csv')
        df_y_val_b = pd.read_csv('dataset/ML_split/multiclass/validation_labels.csv')

        df_x_test_b = pd.read_csv('dataset/ML_split/multiclass/test_features.csv')
        df_y_test_b = pd.read_csv('dataset/ML_split/multiclass/test_labels.csv')

        self.train = [(x.tolist(), y.tolist()[0]) for (_, x), (_, y) in zip(df_x_train_b.iterrows(), df_y_train_b.iterrows())]
        self.val = [(x.tolist(), y.tolist()[0]) for (_, x), (_, y) in zip(df_x_val_b.iterrows(), df_y_val_b.iterrows())]
        self.test = [(x.tolist(), y.tolist()[0]) for (_, x), (_, y) in zip(df_x_test_b.iterrows(), df_y_test_b.iterrows())]

