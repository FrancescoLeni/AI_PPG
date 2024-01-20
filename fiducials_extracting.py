from preprocessing.fiducials2 import get_apg_fiducials, get_vpg_fiducials, get_jpg_fiducials, get_diastolic_peak, get_dicrotic_notch, \
                                    correct_fiducials

from utils.dataloaders_prev import OneSignal1
from utils.dataloaders import OneSignal
from utils import random_state

random_state()


signal = OneSignal1('S001_128.mat')
sig = OneSignal('S001_128.mat')

signal.filter()
sig.filter()
sig.align_onsets()

print('dict')
print(len(signal.fiducials))

print('prev')
print(len(signal.on))
print(signal.peaks.shape[0])

print('correct')
print(sig.on.shape[0])
print(sig.peaks.shape[0])

# IT is correcting our calculating onsets (like shifting the position to make them better) so the value do not always match

# I'm assuming it is dropping the last onset, simply re-add it or let's



print(signal.fiducials)
print(sig.on)