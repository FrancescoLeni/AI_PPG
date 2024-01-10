from preprocessing.filtering import Preprocess
from preprocessing.fiducials import FpCollection
from preprocessing import PPG

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
from dotmap import DotMap
import numpy as np
import pandas as pd
import scipy.io
import random
import os
import h5py


class OneSignal:
    def __init__(self, data_name = None):
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
        self.on = "call .filter() to calculate"
        self.peaks = scipy.io.loadmat(self.peaks_path)['speaks']
        self.labels = scipy.io.loadmat(self.label_path)['labels']



        self.v = np.squeeze(self.raw)
        correction = pd.DataFrame()
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction.loc[0, corr_on] = True
        self.correction = correction

        self.indx = 0 # peak to crop
        self.indx_max = int(self.peaks.shape[0])

    def filter(self, fL=0.5, fH=4.3, order=4, sm = {'ppg':50,'vpg':10,'apg':10,'jpg':10}, data_min=-90, data_max=90):

        print(f'filtering signal {self.name}...')
        # stupid class that the functions need...
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

        #filters signal through BP filter [fL,fH]
        prep = Preprocess(fL=fL, fH=fH, order=order, sm_wins=sm)
        s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)
        self.ppg = s.ppg
        self.vpg = s.vpg
        self.apg = s.apg
        self.jpg = s.jpg

        # get on_set points
        s = PPG(s)
        fpex = FpCollection(s=s)
        fiducials = fpex.get_fiducials(s=s, ext_peaks= self.peaks)

        #added by SAMUEL
        self.fiducials = fiducials #still need conversion to Fiducials class
        self.s = s


        self.on = list(fiducials['on'][:]) # here i'm wasting a lot more points (and time for calculating them...)

        self.add_onsets()

        self.normalize(data_min, data_max)

    def get_crops(self):

        crops = []
        labs = []
        r_crops = []
        r_labs = []
        while (self.indx < self.indx_max):
            (x, y), (r_x, r_y) = self.crop()
            crops.append(x)
            labs.append(y)
            r_crops.append(r_x)
            r_labs.append(r_y)

        return (crops, labs), (r_crops, r_labs)

    def normalize(self, data_min, data_max):
        new = self.ppg.reshape(-1, 1)
        scaler = MinMaxScaler()

        scaler.data_min_ = data_min
        scaler.data_max_ = data_max

        self.ppg = scaler.fit_transform(new)

    def add_onsets(self):
        k = 0
        o = []
        for i in range(len(self.on) - 1):
            o.append(self.on[i])
            o2 = self.on[i + 1]
            p = self.peaks[k]
            p2 = self.peaks[k + 1]
            p3 = self.peaks[k + 2]
            k += 1
            if p2 < o2:
                o.append(int((p + p2) // 2))
                k += 1
                if p3 < o2:
                    o.append(int((p3 + p2) // 2))
                    k += 1
            if i == (len(self.on) - 2):
                o.append(o2)
        if o[-1] < self.peaks[-3]:
            o.append(int((self.peaks[-4] + self.peaks[-3]) // 2))
        if o[-1] < self.peaks[-2]:
            o.append(int((self.peaks[-3]+self.peaks[-2]) // 2))
        if o[-1] < self.peaks[-1]:
            o.append(int((self.peaks[-2] + self.peaks[-1]) // 2))
        o.append(int(len(self.ppg)-1))

        self.on = np.array(o, dtype=int)


    def crop(self):

        crop = self.ppg[self.on[self.indx]:self.on[self.indx+1]]
        lab = self.labels[self.indx]
        raw_crop = self.raw[self.on[self.indx]:self.on[self.indx+1]]
        self.indx+=1
        return (crop, lab), (raw_crop, lab)



class Crops():
    def __init__(self, N="N_crops.h5", V="V_crops.h5", S="S_crops.h5", parent=Path('dataset/crops'), seed=36):
        super().__init__()
        names_list = [N,V,S]
        for names in names_list:
            print(f"loading {names}...")
            with h5py.File(parent / names, 'r') as file:
                name = Path(names).stem
                setattr(self, name, [file[key][:] for key in file.keys() if key != 'labels'])
                setattr(self, f"{name[0]}_labels", list(file['labels'][:].astype('U')))

        self.seed = seed

    def split(self, test_size=.15):

        x = self.V_crops + self.S_crops + self.N_crops
        y = self.V_labels + self.S_labels + self.N_labels

        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, random_state=self.seed, test_size=test_size,
                                                                    shuffle=True, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, random_state=self.seed, test_size=len(x_test),
                                                                    shuffle=True, stratify=y_train_val)
        setattr(self,'train', [(x, y) for x, y in zip(x_train, y_train)])
        setattr(self,'val', [(x, y) for x, y in zip(x_val, y_val)])
        setattr(self,'test', [(x, y) for x, y in zip(x_test, y_test)])

