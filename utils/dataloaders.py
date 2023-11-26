
from preprocessing.filtering import Preprocess
from preprocessing.fiducials import FpCollection
from preprocessing import PPG

from dotmap import DotMap
import numpy as np
import pandas as pd
import scipy.io
import os
from sklearn.preprocessing import MinMaxScaler


class OneSignal:
    def __init__(self, data_name = None):
        """
        :param data_name: filename like 'S001_128.mat' or tuple (data_path,label_path,peaks_path)

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

    def filter(self, fL=0.8, fH=3.3, order=4, sm = {'ppg':50,'vpg':10,'apg':10,'jpg':10}, data_min= -90, data_max= 90):

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

        self.on = list(fiducials['on'][:]) # here i'm wasting a lot more points (and time for calculating them...)

        self.add_onsets()

        self.normalize(data_min, data_max)

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
        if o[-1] < self.peaks[-1]:
            o.append(int((self.peaks[-2]+self.peaks[-1]) // 2))

        self.on = np.array(o, dtype=int)


