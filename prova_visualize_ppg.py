from utils.dataloaders_prev import OneSignal

from utils.plots import raw_vs_filtered
from matplotlib import pyplot as plt
from utils import random_state

random_state(36)

signal = OneSignal('S109_250.mat')

signal.filter(fL=0.5, fH=4.3, order=4)
# 445000
raw_vs_filtered(signal.raw, signal.ppg, signal.on, signal.peaks, signal.labels, dt=320050) # dt indicates the displacement of the signal

# crop, lab = signal.crop()
# crop1, lab1 = signal.crop()
# crop2, lab2 = signal.crop()
# crop3, lab3 = signal.crop()
#
# fig, axes = plt.subplots(2, 2)
#
# axes[0,0].plot(range(crop.shape[0]), crop)
# axes[0,0].set_title(lab)
# axes[0,1].plot(range(crop1.shape[0]), crop1)
# axes[0,1].set_title(lab1)
# axes[1,0].plot(range(crop2.shape[0]), crop2)
# axes[1,0].set_title(lab2)
# axes[1,1].plot(range(crop3.shape[0]), crop3)
# axes[1,1].set_title(lab3)
#
#
# plt.tight_layout()
# plt.show()
