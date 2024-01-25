import os
import shutil
from pathlib import Path
import h5py
import numpy as np
import json

import pandas as pd

from utils.dataloaders import OneSignal
from utils import random_state

random_state(36)


# be sure to create a dataset folder with the data

# data will be divided in data, peaks, labels folder respectively

# FILE ALL RENAMED as Snum_frq.mat



dad = 'dataset'
l = ["data", "peaks", "label", "blacklisted_data" ]
for i in l:
    if not os.path.isdir(os.path.join(dad, i)):
        os.mkdir(os.path.join(dad, i))

for files in os.listdir(dad):
    path = os.path.join(dad, files)
    if os.path.isfile(path):
        if "_spk" in files:
            dst = os.path.join(dad,l[1],files[0:8]+'.mat')
            shutil.move(path, dst)
        elif "_ann" in files:
            dst = os.path.join(dad,l[2],files[0:8]+'.mat')
            shutil.move(path,dst)
        else:
            dst = os.path.join(dad,l[0],files[0:8]+'.mat')
            if files[0:8] != "S120_250":  # blacklisted
                shutil.move(path, dst)
            else:
                shutil.move(path, os.path.join(dad, l[3], files[0:8] + '.mat'))

# ----------------------------------------------------------------------------------------------------------------------
# to retrive samples with reasonable number of positives
# ----------------------------------------------------------------------------------------------------------------------

parent = Path('dataset')

if not os.path.isdir(parent / 'more_9'):
    os.mkdir(parent / 'more_9')
if not os.path.isdir(parent / 'less_9'):
    os.mkdir(parent / 'less_9')
if not os.path.isdir(parent / 'only_N'):
    os.mkdir(parent / 'only_N')

for name in os.listdir(parent / 'data'):
    signal = OneSignal(name)
    lab = list(signal.labels)
    N = lab.count('N')
    V = lab.count('V')
    S = lab.count('S')

    if (S+V)/(N+S+V)*100 >= 9.0:
        shutil.copy((parent / 'data' / name), (parent / 'more_9' / name))
    elif (S+V)/(N+S+V)*100 < 9.0 and not (S+V)/(N+S+V)*100 == 0:
        shutil.copy((parent / 'data' / name), (parent / 'less_9' / name))
    else:
        shutil.copy((parent / 'data' / name), (parent / 'only_N' / name))

# ----------------------------------------------------------------------------------------------------------------------
# to save files of cropped peaks divided into classes
# ----------------------------------------------------------------------------------------------------------------------

parent = Path('dataset')

if not os.path.isdir(parent / 'crops_apg'):
    os.mkdir(parent / 'crops_apg')
    os.mkdir(parent / 'crops_apg/patients')
    os.mkdir(parent / 'crops_vpg')
    os.mkdir(parent / 'crops_vpg/patients')
    os.mkdir(parent / 'crops_jpg')
    os.mkdir(parent / 'crops_jpg/patients')

data_path = parent / 'data'

crops_n = []
labs_n = []
crops_v = []
labs_v = []
crops_s = []
labs_s = []

crops_n_j = []
crops_v_j = []
crops_s_j = []

crops_n_a = []
crops_v_a = []
crops_s_a = []

crops_n_v = []
crops_v_v = []
crops_s_v = []

tutte = []

tutte_filt = []

for sample in os.listdir(data_path):

    sig_crops = []
    sig_labs = []

    sig_c_r = []
    sig_l_r = []

    j = []
    v = []
    a = []

    signal = OneSignal(sample)
    signal.filter(fL = 0.5, fH = 4.3, order = 4)
    # Align onsets to determine crops_old: always 1 peak between 2 onsets
    signal.align_onsets()

    while signal.indx < signal.indx_max:

        (x, y), (x_r, y_r), (c_j, c_v, c_a) = signal.crop(raw=True)
        # sig_crops.append(x_r)
        sig_labs.append(y_r)
        #
        # sig_c_r.append(x)
        # sig_l_r.append(y)

        c_j = c_j[:, np.newaxis]
        c_a = c_a[:, np.newaxis]
        c_v = c_v[:, np.newaxis]

        j.append(c_j)
        a.append(c_a)
        v.append(c_v)

        # tutte += list(x_r)
        # tutte_filt += list(x)

        if y_r == 'N':
            crops_n_j.append(c_j)
            crops_n_v.append(c_v)
            crops_n_a.append(c_a)
            labs_n.append(y_r)
        if y_r == 'S':
            crops_s_j.append(c_j)
            crops_s_v.append(c_v)
            crops_s_a.append(c_a)
            labs_s.append(y_r)
        if y_r == 'V':
            crops_v_j.append(c_j)
            crops_v_v.append(c_v)
            crops_v_a.append(c_a)
            labs_v.append(y_r)

    with h5py.File(parent / f'crops_apg/patients/{sample[:-4]}.h5', 'w') as file:
        for i, crop in enumerate(a):
            file.create_dataset(f'crop_{i}', data=crop)
        file.create_dataset('labels', data=np.array(sig_labs, dtype='S'))

    with h5py.File(parent / f'crops_vpg/patients/{sample[:-4]}.h5', 'w') as file:
        for i, crop in enumerate(v):
            file.create_dataset(f'crop_{i}', data=crop)
        file.create_dataset('labels', data=np.array(sig_labs, dtype='S'))

    with h5py.File(parent / f'crops_jpg/patients/{sample[:-4]}.h5', 'w') as file:
        for i, crop in enumerate(j):
            file.create_dataset(f'crop_{i}', data=crop)
        file.create_dataset('labels', data=np.array(sig_labs, dtype='S'))
#
# Q1 = np.percentile(tutte, 25)
# Q3 = np.percentile(tutte, 75)
# IQR = Q3 - Q1
#
# Q1_filt = np.percentile(tutte_filt, 25)
# Q3_filt = np.percentile(tutte_filt, 75)
# IQR_filt = Q3 - Q1
#
# print('RAW SIGNAL')
# print(f'media:{np.mean(tutte)}, std:{np.std(tutte)}, mediana:{np.median(tutte)}, Q1:{Q1}, Q3:{Q3}, IQR:{IQR}')
# print('FILTERED SIGNAL')
# print(f'media:{np.mean(tutte_filt)}, std:{np.std(tutte_filt)}, mediana:{np.median(tutte_filt)}, Q1:{Q1_filt}, Q3:{Q3_filt}, IQR:{IQR_filt}')
#
with h5py.File(parent / 'crops_apg/N_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_n_a):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_n, dtype='S'))

with h5py.File(parent / 'crops_apg/S_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_s_a):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_s, dtype='S'))

with h5py.File(parent / 'crops_apg/V_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_v_a):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_v, dtype='S'))

with h5py.File(parent / 'crops_vpg/N_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_n_v):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_n, dtype='S'))

with h5py.File(parent / 'crops_vpg/S_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_s_v):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_s, dtype='S'))

with h5py.File(parent / 'crops_vpg/V_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_v_v):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_v, dtype='S'))

with h5py.File(parent / 'crops_jpg/N_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_n_j):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_n, dtype='S'))

with h5py.File(parent / 'crops_jpg/S_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_s_j):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_s, dtype='S'))

with h5py.File(parent / 'crops_jpg/V_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_v_j):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_v, dtype='S'))


parent = Path('dataset')

if not os.path.isdir(parent / 'filtered'):
    os.mkdir(parent / 'filtered')
    os.mkdir(parent / 'vpg')
    os.mkdir(parent / 'apg')
    os.mkdir(parent / 'jpg')


data_path = parent / 'data'


onset_dict = {}

stats = {'name': [], 'filt_mean': [], 'filt_std': [], 'filt_median': [], 'filt_IQR': [], 'filt_max': [], 'filt_min': [],
                     'raw_mean': [], 'raw_std': [], 'raw_median': [], 'raw_IQR': [], 'raw_max': [], 'raw_min': [],
                     'apg_mean': [], 'apg_std': [], 'apg_median': [], 'apg_IQR': [], 'apg_max': [], 'apg_min': [],
                     'vpg_mean': [], 'vpg_std': [], 'vpg_median': [], 'vpg_IQR': [], 'vpg_max': [], 'vpg_min': [],
                     'jpg_mean': [], 'jpg_std': [], 'jpg_median': [], 'jpg_IQR': [], 'jpg_max': [], 'jpg_min': [],
         }

tutte_apg = []
tutte_jpg = []
tutte_vpg = []

for sample in os.listdir(data_path):

    name = Path(sample).stem

    signal = OneSignal(sample)
    signal.filter(fL = 0.5, fH = 4.3, order = 4)
    # Align onsets to determine crops_old: always 1 peak between 2 onsets
    signal.align_onsets()

    on = list(signal.on)
    raw = signal.raw
    filt = signal.ppg
    vpg = list(signal.vpg)
    apg = list(signal.apg)
    jpg = list(signal.jpg)

    tutte_vpg += vpg
    tutte_apg += apg
    tutte_jpg += jpg

    on = [int(o) for o in on]

    onset_dict[name] = on

    stats['name'].append(name)
    stats['filt_mean'].append(np.mean(filt))
    stats['filt_std'].append(np.std(filt))
    stats['filt_median'].append(np.median(filt))
    stats['filt_IQR'].append(np.percentile(filt, 75) - np.percentile(filt, 25))
    stats['filt_max'].append(np.max(filt))
    stats['filt_min'].append(np.min(filt))

    stats['raw_mean'].append(np.mean(raw))
    stats['raw_std'].append(np.std(raw))
    stats['raw_median'].append(np.median(raw))
    stats['raw_IQR'].append(np.percentile(raw, 75) - np.percentile(raw, 25))
    stats['raw_max'].append(np.max(raw))
    stats['raw_min'].append(np.min(raw))

    stats['vpg_mean'].append(np.mean(vpg))
    stats['vpg_std'].append(np.std(vpg))
    stats['vpg_median'].append(np.median(vpg))
    stats['vpg_IQR'].append(np.percentile(vpg, 75) - np.percentile(vpg, 25))
    stats['vpg_max'].append(np.max(vpg))
    stats['vpg_min'].append(np.min(vpg))

    stats['apg_mean'].append(np.mean(apg))
    stats['apg_std'].append(np.std(apg))
    stats['apg_median'].append(np.median(apg))
    stats['apg_IQR'].append(np.percentile(apg, 75) - np.percentile(apg, 25))
    stats['apg_max'].append(np.max(apg))
    stats['apg_min'].append(np.min(apg))

    stats['jpg_mean'].append(np.mean(jpg))
    stats['jpg_std'].append(np.std(jpg))
    stats['jpg_median'].append(np.median(jpg))
    stats['jpg_IQR'].append(np.percentile(jpg, 75) - np.percentile(jpg, 25))
    stats['jpg_max'].append(np.max(jpg))
    stats['jpg_min'].append(np.min(jpg))


    # np.save(parent / 'filtered' / f'{name}.npy', filt[:, np.newaxis])
    # np.save(parent / 'apg' / f'{name}.npy', apg[:, np.newaxis])
    # np.save(parent / 'vpg' / f'{name}.npy', vpg[:, np.newaxis])
    # np.save(parent / 'jpg' / f'{name}.npy', jpg[:, np.newaxis])


df_stats = pd.DataFrame(stats)
df_stats.to_csv('data/divided_amp_stats.csv', index=False)

all_ = {'jpg_mean': [np.mean(tutte_jpg)], 'jpg_std': [np.std(tutte_jpg)], 'jpg_median': [np.median(tutte_jpg)],
        'jpg_IQR': [np.percentile(tutte_jpg, 75) - np.percentile(tutte_jpg, 25)], 'jpg_max': [np.max(tutte_jpg)],
        'jpg_min': [np.min(tutte_jpg)], 'vpg_mean': [np.mean(tutte_vpg)], 'vpg_std': [np.std(tutte_vpg)],
        'vpg_median': [np.median(tutte_vpg)], 'vpg_IQR': [np.percentile(tutte_vpg, 75) - np.percentile(tutte_vpg, 25)],
        'vpg_max': [np.max(tutte_vpg)], 'vpg_min': [np.min(tutte_vpg)], 'apg_mean': [np.mean(tutte_apg)],
        'apg_std': [np.std(tutte_apg)], 'apg_median': [np.median(tutte_apg)],
        'apg_IQR': [np.percentile(tutte_apg, 75) - np.percentile(tutte_apg, 25)], 'apg_max': [np.max(tutte_apg)],
        'apg_min': [np.min(tutte_apg)]}

df = pd.DataFrame(all_)
df.to_csv('data/all_others_amp.csv', index=False)

with open(parent / 'onsets.json', 'w') as json_file:
    json.dump(onset_dict, json_file)

