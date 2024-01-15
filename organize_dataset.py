import os
import shutil
from pathlib import Path
import h5py
import numpy as np

from utils.dataloaders import OneSignal
from utils import random_state

random_state(36)


# be sure to create a dataset folder with the data

# data will be divided in data, peaks, labels folder respectively

# FILE ALL RENAMED as Snum_frq.mat

#
#
# dad = 'dataset'
# l = ["data", "peaks", "label", "blacklisted_data" ]
# for i in l:
#     if not os.path.isdir(os.path.join(dad, i)):
#         os.mkdir(os.path.join(dad, i))
#
# for files in os.listdir(dad):
#     path = os.path.join(dad, files)
#     if os.path.isfile(path):
#         if "_spk" in files:
#             dst = os.path.join(dad,l[1],files[0:8]+'.mat')
#             shutil.move(path, dst)
#         elif "_ann" in files:
#             dst = os.path.join(dad,l[2],files[0:8]+'.mat')
#             shutil.move(path,dst)
#         else:
#             dst = os.path.join(dad,l[0],files[0:8]+'.mat')
#             if files[0:8] != "S120_250":  # blacklisted
#                 shutil.move(path, dst)
#             else:
#                 shutil.move(path, os.path.join(dad, l[3], files[0:8] + '.mat'))

# ----------------------------------------------------------------------------------------------------------------------
# to retrive samples with reasonable number of positives
# ----------------------------------------------------------------------------------------------------------------------

# parent = Path('dataset')
#
# if not os.path.isdir(parent / 'more_9'):
#     os.mkdir(parent / 'more_9')
# if not os.path.isdir(parent / 'less_9'):
#     os.mkdir(parent / 'less_9')
# if not os.path.isdir(parent / 'only_N'):
#     os.mkdir(parent / 'only_N')
#
# for name in os.listdir(parent / 'data'):
#     signal = OneSignal(name)
#     lab = list(signal.labels)
#     N = lab.count('N')
#     V = lab.count('V')
#     S = lab.count('S')
#
#     if (S+V)/(N+S+V)*100 >= 9.0:
#         shutil.copy((parent / 'data' / name), (parent / 'more_9' / name))
#     elif (S+V)/(N+S+V)*100 < 9.0 and not (S+V)/(N+S+V)*100 == 0:
#         shutil.copy((parent / 'data' / name), (parent / 'less_9' / name))
#     else:
#         shutil.copy((parent / 'data' / name), (parent / 'only_N' / name))

# ----------------------------------------------------------------------------------------------------------------------
# to save files of cropped peaks divided into classes
# ----------------------------------------------------------------------------------------------------------------------

parent = Path('dataset')

if not os.path.isdir(parent / 'crops_raw'):
    os.mkdir(parent / 'crops_raw')
    os.mkdir(parent / 'crops_raw/patients')

data_path = parent / 'data'

crops_n = []
labs_n = []
crops_v = []
labs_v = []
crops_s = []
labs_s = []

tutte = []

tutte_filt = []

for sample in os.listdir(data_path):

    sig_crops = []
    sig_labs = []

    sig_c_r = []
    sig_l_r = []

    signal = OneSignal(sample)
    signal.filter(fL = 0.5, fH = 4.3, order = 4)
    # Align onsets to determine crops_old: always 1 peak between 2 onsets
    signal.align_onsets()

    while signal.indx < signal.indx_max:

        (x, y), (x_r, y_r) = signal.crop(raw=True)
        sig_crops.append(x_r)
        sig_labs.append(y_r)

        sig_c_r.append(x)
        sig_l_r.append(y)

        tutte += list(x_r)
        tutte_filt += list(x)

        if y_r == 'N':
            crops_n.append(x_r)
            labs_n.append(y_r)
        if y_r == 'S':
            crops_s.append(x_r)
            labs_s.append(y_r)
        if y_r == 'V':
            crops_v.append(x_r)
            labs_v.append(y_r)

    with h5py.File(parent / f'crops_raw/patients/{sample[:-4]}.h5', 'w') as file:
        for i, crop in enumerate(sig_crops):
            file.create_dataset(f'crop_{i}', data=crop)
        file.create_dataset('labels', data=np.array(sig_labs, dtype='S'))

    if not os.path.isdir(parent / f'crops/patients/{sample[:-4]}.h5'):
        with h5py.File(parent / f'crops/patients/{sample[:-4]}.h5', 'w') as file:
            for i, crop in enumerate(sig_c_r):
                file.create_dataset(f'crop_{i}', data=crop)
            file.create_dataset('labels', data=np.array(sig_l_r, dtype='S'))

Q1 = np.percentile(tutte, 25)
Q3 = np.percentile(tutte, 75)
IQR = Q3 - Q1

Q1_filt = np.percentile(tutte_filt, 25)
Q3_filt = np.percentile(tutte_filt, 75)
IQR_filt = Q3 - Q1

print('RAW SIGNAL')
print(f'media:{np.mean(tutte)}, std:{np.std(tutte)}, mediana:{np.median(tutte)}, Q1:{Q1}, Q3:{Q3}, IQR:{IQR}')
print('FILTERED SIGNAL')
print(f'media:{np.mean(tutte_filt)}, std:{np.std(tutte_filt)}, mediana:{np.median(tutte_filt)}, Q1:{Q1_filt}, Q3:{Q3_filt}, IQR:{IQR_filt}')

with h5py.File(parent / 'crops_raw/N_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_n):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_n, dtype='S'))

with h5py.File(parent / 'crops_raw/S_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_s):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_s, dtype='S'))

with h5py.File(parent / 'crops_raw/V_crops.h5', 'w') as file:
    # Save crops_old as separate datasets
    for i, crop in enumerate(crops_v):
        file.create_dataset(f'crop_{i}', data=crop)
    # Save labels as a dataset
    file.create_dataset('labels', data=np.array(labs_v, dtype='S'))

# ----------------------------------------------------------------------------------------------------------------------
# to load .h5
# ----------------------------------------------------------------------------------------------------------------------

# parent = Path('dataset/crops_old')
# for names in ["N_crops.h5","V_crops.h5","S_crops.h5"]:
#
#     with h5py.File(parent / names, 'r') as file:
#         name = Path(names).stem
#         exec(f"{name} = [file[key][:] for key in file.keys() if key != 'labels']")
#         exec(f"{name[0]}_labels = list(file['labels'][:].astype('U'))")
#
# # Access specific elements by index
# N_x = N_crops[-1]
# N_y = N_labels[-1]
# V_x = V_crops[-1]
# V_y = V_labels[-1]
# S_x = S_crops[-1]
# S_y = S_labels[-1]