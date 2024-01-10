
from preprocessing.filtering import Preprocess
from preprocessing.fiducials import FpCollection
from preprocessing import PPG

from preprocessing import Fiducials, Biomarkers
import pyPPG.biomarkers as BM

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
    
    def crop(self):
        crop = self.ppg[self.on[self.indx]:self.on[self.indx+1]]
        lab = self.labels[self.indx]
        self.indx+=1
        return (crop, lab)

class Crops():
    def __init__(self, N="N_crops.h5", V="V_crops.h5", S="S_crops.h5", parent=Path('dataset/crops'), seed=36):
        super().__init__()
        names_list = [N, V, S]
        for names in names_list:
            print(f"loading {parent/names}...")
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
        setattr(self, 'train', [(x, y) for x, y in zip(x_train, y_train)])
        setattr(self, 'val', [(x, y) for x, y in zip(x_val, y_val)])
        setattr(self, 'test', [(x, y) for x, y in zip(x_test, y_test)])

