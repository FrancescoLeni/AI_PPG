import os
import shutil
from pathlib import Path

from utils.dataloaders import OneSignal
from utils import random_state

random_state(36)


# be sure to create a dataset folder with the data

# data will be divided in data, peaks, labels folder respectively

# FILE ALL RENAMED as Snum_frq.mat



dad = 'dataset'
l = ["data", "peaks", "label" ]
for i in l:
    if not os.path.isdir(os.path.join(dad, i)):
        os.mkdir(os.path.join(dad, i))

for files in os.listdir(dad):
    path = os.path.join(dad, files)
    if os.path.isfile(path):
        if "_spk" in files:
            dst = os.path.join(dad,l[1],files[0:8]+'.mat')
            shutil.move(path,dst)
        elif "_ann" in files:
            dst = os.path.join(dad,l[2],files[0:8]+'.mat')
            shutil.move(path,dst)
        else:
            dst = os.path.join(dad,l[0],files[0:8]+'.mat')
            shutil.move(path,dst)

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

