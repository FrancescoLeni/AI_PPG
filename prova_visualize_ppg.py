from utils.dataloaders import OneSignal
from utils.plots import raw_vs_filtered


signal = OneSignal('S022_128.mat')

signal.filter(fL = 0.5, fH = 4.3, order = 4)

raw_vs_filtered(signal.raw, signal.ppg, signal.on, signal.peaks, dt = 0) # dt indicates the displacement of the signal


