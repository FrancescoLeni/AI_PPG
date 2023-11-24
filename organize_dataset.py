import os
import shutil


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