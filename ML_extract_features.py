# Import packages
import os
import numpy as np
import pandas as pd
from scipy.stats import tvar, skew, kurtosis
from scipy.integrate import simps
from tqdm import tqdm

from utils.dataloaders import OneSignal
from utils import random_state
# import random

random_state(36)

class FeatureConstructor:
    def __init__(self, data_name=None):
        """
        :param
            --data_name: filename like 'S001_128.mat' or tuple (data_path,label_path,peaks_path)
        """
        # Initialize class arguments
        self.data_name = data_name

        # Get information for patient with data_name
        self.signal = OneSignal(data_name=self.data_name)
        # Filter the PPG signal
        self.signal.filter(fL = 0.5, fH = 4.3, order = 4)
        # Align onsets to determine crops: always 1 peak between 2 onsets
        self.signal.align_onsets()

        # Set attributes of FeatureExtractor
        self.ppg = self.signal.ppg              # get filtered ppg and derivatives
        self.vpg = self.signal.vpg
        self.apg = self.signal.apg
        self.jpg = self.signal.jpg
        self.fs = self.signal.fs                # sampling frequency                        --> int
        self.peaks = self.signal.peaks.flatten()# peaks array                               --> (number_of_peaks,)
        self.labels = self.signal.labels        # labels                                    --> (number_of_peaks,)
        self.onsets = self.signal.on            # determined by self.signal.align_onsets()  --> (number_of_peaks+1,)


    def generate_crops(self):
        self.crops = []
        while self.signal.indx < self.signal.indx_max:
            # (x, y), (x_r, y_r) = self.signal.crop(raw=True)
            crop, _ = self.signal.crop(raw=False)
            self.crops.append(crop)

    def get_intra_crop_features(self):
        ""
        self.ft_intra_crop_names = ['crop_duration','t_peak','mean','median','std','tvar','skew','kurt',
                                    'auc','peak_amplitude','pulse_width','symmetry']
        self.ft_intra_crop = np.zeros(((self.peaks.shape[0]), len(self.ft_intra_crop_names)))

        # Construct self.crops
        self.generate_crops()

        # Loop over all crops: extract features
        for i, crop in enumerate(self.crops):
            self.ft_intra_crop[i,:] = np.array(
                [crop.shape[0] / self.fs,
                 np.argmax(crop) / self.fs,
                 np.mean(crop),
                 np.median(crop),
                 np.std(crop),
                 tvar(crop), # tune values?
                 skew(crop),
                 kurtosis(crop),
                 simps(np.abs(crop), dx=1/self.fs), # AUC: Simpson's rule for numeral integration
                 np.max(crop)-np.min(crop),
                 self.pulse_width(crop),
                 self.symmetry_index(crop)
                 ])

    def get_inter_crop_features(self):
        ""
        self.ft_inter_crop_names = ['PTP','N_last_X_s']

        self.ft_inter_crop = np.zeros(((self.peaks.shape[0]), len(self.ft_inter_crop_names)))
        
        self.ft_inter_crop[:,0] = self.peak_to_peak_times()
        self.ft_inter_crop[:,1] = self.N_ratio()

    def get_patient_specific_features(self):
        ""
        self.ft_patient_names = ['name']
        self.ft_patient = np.zeros(((self.peaks.shape[0]), len(self.ft_patient_names)))
        
        self.ft_patient[:,0] = self.data_name.split('_')[0][1:]

        

    def construct_dataframe(self, out_file=''):
        # Get features
        self.feature_names = self.ft_intra_crop_names + self.ft_inter_crop_names + self.ft_patient_names
        features = np.concatenate([self.ft_intra_crop, self.ft_inter_crop, self.ft_patient], axis=-1)

        # Create a DataFrame
        data = {'peaks': self.peaks, 'labels': self.labels}
        for i, feature_name in enumerate(self.feature_names):
            data[feature_name] = features[:, i]

        self.df = pd.DataFrame(data)

        # Save DataFrame to .csv file
        current_directory = os.path.dirname(os.path.abspath('__file__'))
        folder_name = 'dataset/ML_features/'
        target_folder = os.path.join(current_directory, folder_name)

        ## Check if the folder exists and create it if not
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        file_path = os.path.join(target_folder, self.data_name.split('.')[0] + '.csv')
        self.df.to_csv(file_path, index=False)

    #---------------------------------------------------
    #   Functions used in get_intra_crop_features()
    #---------------------------------------------------
        
    def pulse_width(self, crop):
        if len(crop) == 0:
            return 0  # or some default value, as appropriate
        
        half_peak = max(crop) / 2
        idx_peak = np.argmax(crop)
        
        # Find indices on both sides of the peak
        # Index of half value of peak before peak
        idx_t1 = self.find_nearest(crop[:idx_peak], half_peak)
        # Index of half value of peak after peak
        idx_t2 = self.find_nearest(crop[idx_peak:], half_peak) + idx_peak

        # Calculate the width
        width = (idx_t2 - idx_t1)/self.fs # [s]

        return width

    #From https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def find_nearest(self, array, value):
        array = np.asarray(array)
        if len(array) == 0:
            idx = 0
        else:
            idx = (np.abs(array - value)).argmin()
        return idx
    
    def symmetry_index(self, crop):
        middle_idx = len(crop) // 2
        
        left_half = crop[:middle_idx]
        right_half = crop[middle_idx:]
        
        mean_left = np.mean(left_half)
        mean_right = np.mean(right_half)

        symmetry_index = mean_right / mean_left
        
        return symmetry_index
    
    #---------------------------------------------------
    #   Functions used in get_inter_crop_features()
    #---------------------------------------------------

    def peak_to_peak_times(self):
        time_between_peaks = np.diff(self.peaks) / self.fs
        mean_PTP = np.mean(time_between_peaks)
        time_between_peaks = mean_PTP
        time_between_peaks = np.insert(time_between_peaks, 0, mean_PTP)

        return time_between_peaks
    
    def A_to_N_ratio(self, time_window=20):
        indices_before = int(time_window*self.fs)

        ratio = np.zeros(len(self.peaks))
        for i, peak_idx in enumerate(self.peaks):
            indices_in_window = np.where((peak_idx - indices_before <= self.peaks) & (self.peaks < peak_idx))[0]

            count_N = np.count_nonzero((self.labels[indices_in_window] ==  'N'))
            count_V = np.count_nonzero((self.labels[indices_in_window] ==  'V'))
            count_S = np.count_nonzero((self.labels[indices_in_window] ==  'S'))
            
            # Handle the case were count_N is 0
            if count_N == 0:
                count_N = 1

            ratio[i] = (count_V+count_S)/count_N

        return ratio
        
    # def drop_empty_crops(self, crops):

    #     out = [crop for crop in crops if len(crop) > 4]
    #     print("Number of crops eliminated: ", len(crops)-len(out))
    #     return out

    # def get_max_freq(self, crop, fs): #NEED FS OF INDIVIDUAL SIGNAL :(
    #     fft = np.fft.fft(crop)
    #     freqs = np.fft.fftfreq(len(crop), d=1/fs)
    #     dom_freq_idx = np.argmax(np.abs(fft))
    #     dom_freq = np.abs(freqs[dom_freq_idx])
    #     return dom_freq

def process_files(directory):
    compute = False

    for file_name in tqdm(os.listdir(directory)):

        if '108' in file_name:
            compute = True
        
        if compute:
            recording = FeatureConstructor(file_name)
            recording.get_intra_crop_features()
            recording.get_inter_crop_features()
            recording.get_patient_specific_features()
            recording.construct_dataframe()

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_name = 'dataset/data/'
    target_folder = os.path.join(current_directory, folder_name)
    
    process_files(target_folder)