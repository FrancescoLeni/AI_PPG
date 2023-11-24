from utils.dataloaders import OneSignal
from utils.plots import raw_vs_filtered


signal = OneSignal('S001_128.mat')

signal.filter(fL = 0.8, fH = 3.3, order = 4)

raw_vs_filtered(signal.raw, signal.ppg, signal.on, dt = 101) # dt indicates the displacement of the signal


