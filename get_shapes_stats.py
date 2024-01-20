
from utils.dataloaders import Crops, OneSignal
from utils.DL.loaders import CropsDataset

import torch

from utils import random_state

from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np

random_state(36)


# df_dict = {'name': [], 'series_len': [], 'max_shape': [], 'min_shapes': [], 'avg_shape': [], 'std_shape': [], 'indx_max': [],
#            'indx_min': [], 'pos_max': [], 'pos_min': [], 'num_crops': [], 'num_peaks': [], 'num_zero_len': [], 'Q1': [], 'Q3': [], 'IQR': [],
#            'max_x_raw': [], 'min_x_raw': [], 'max_x_filtered': [], 'min_x_filtered': [], 'Q1-1.5IQR': [], 'Q3+1.5IQR': [],
#            'Q1-3IQR': [], 'Q3+3IQR': [], '90thP': [], '10thP': [], '95thP': [], '05thP': [], '01thP': [], '99thP': [], 'below_1.5IQR': [], 'above_1.5IQR': [],
#            'below_3IQR': [], 'above_3IQR': [], 'below_10th': [], 'above_90th': [], 'below_05th': [], 'above_95th': [], 'below_01th': [], 'above_99th': []}
#
# alls = []
# for sample in os.listdir('dataset/data'):
#     shapes = []
#     zeros = []
#     tot = []
#     non_zero = []
#     raws = []
#     cum_len = []
#
#     # if sample == 'S039_128.mat':
#     signal = OneSignal(sample)
#     signal.filter(fL = 0.5, fH = 4.3, order = 4)
#     # Align onsets to determine crops_old: always 1 peak between 2 onsets
#     signal.align_onsets()
#
#     while signal.indx < signal.indx_max:
#
#         (x, y) = signal.crop()
#         alls.append(x.shape[0])
#         shapes.append(x.shape[0])
#         if x.shape[0] == 0:
#             zeros.append(x)
#         else:
#             non_zero.append(x.shape[0])
#
#         tot.append((x, y))
#         if cum_len:
#             cum_len.append(cum_len[-1] + x.shape[0])
#         else:
#             cum_len.append(x.shape[0])
#
#     df_dict['name'].append(sample)
#     df_dict['series_len'].append(signal.ppg.shape[0])
#     df_dict['max_shape'].append(max(non_zero))
#     df_dict['min_shapes'].append(min(non_zero))
#     df_dict['avg_shape'].append(np.mean(non_zero).astype(np.uint8))
#     df_dict['std_shape'].append(np.std(non_zero).astype(np.uint8))
#     df_dict['indx_max'].append(np.argmax(non_zero))
#     df_dict['indx_min'].append(np.argmin(non_zero))
#     df_dict['pos_max'].append(cum_len[np.argmax(non_zero)])
#     df_dict['pos_min'].append((cum_len[np.argmin(non_zero)]))
#     df_dict['num_crops'].append(len(shapes))
#     df_dict['num_peaks'].append(signal.peaks.shape[0])
#     df_dict['num_zero_len'].append(len(zeros))
#     df_dict['Q1'].append(np.percentile(shapes, 25))
#     df_dict['Q3'].append(np.percentile(shapes, 75))
#     df_dict['IQR'].append(np.percentile(shapes, 75) - np.percentile(shapes, 25))
#     df_dict['min_x_raw'].append(np.min(signal.raw))
#     df_dict['max_x_raw'].append((np.max(signal.raw)))
#     df_dict['min_x_filtered'].append(np.min(signal.ppg))
#     df_dict['max_x_filtered'].append((np.max(signal.ppg)))
#     df_dict['Q1-1.5IQR'].append(np.percentile(shapes, 25) - 1.5 * (np.percentile(shapes, 75) - np.percentile(shapes, 25)))
#     df_dict['Q3+1.5IQR'].append(np.percentile(shapes, 75) + 1.5 * (np.percentile(shapes, 75) - np.percentile(shapes, 25)))
#     df_dict['Q1-3IQR'].append(np.percentile(shapes, 25) - 3 * (np.percentile(shapes, 75) - np.percentile(shapes, 25)))
#     df_dict['Q3+3IQR'].append(np.percentile(shapes, 75) + 3 * (np.percentile(shapes, 75) - np.percentile(shapes, 25)))
#     df_dict['90thP'].append(np.percentile(shapes, 90))
#     df_dict['10thP'].append(np.percentile(shapes, 10))
#     df_dict['95thP'].append(np.percentile(shapes, 95))
#     df_dict['05thP'].append(np.percentile(shapes, 5))
#     df_dict['99thP'].append(np.percentile(shapes, 99))
#     df_dict['01thP'].append(np.percentile(shapes, 1))
#     df_dict['below_1.5IQR'].append(len([x for x in shapes if x < np.percentile(shapes, 25) - 1.5 * (np.percentile(shapes, 75) - np.percentile(shapes, 25))]))
#     df_dict['above_1.5IQR'].append(len([x for x in shapes if x > np.percentile(shapes, 75) + 1.5 * (np.percentile(shapes, 75) - np.percentile(shapes, 25))]))
#     df_dict['below_3IQR'].append(len([x for x in shapes if x < np.percentile(shapes, 25) - 3 * (np.percentile(shapes, 75) - np.percentile(shapes, 25))]))
#     df_dict['above_3IQR'].append(len([x for x in shapes if x > np.percentile(shapes, 75) + 3 * (np.percentile(shapes, 75) - np.percentile(shapes, 25))]))
#     df_dict['below_10th'].append(len([x for x in shapes if x < np.percentile(shapes, 10)]))
#     df_dict['above_90th'].append(len([x for x in shapes if x > np.percentile(shapes, 90)]))
#     df_dict['below_05th'].append(len([x for x in shapes if x < np.percentile(shapes, 5)]))
#     df_dict['above_95th'].append(len([x for x in shapes if x > np.percentile(shapes, 95)]))
#     df_dict['below_01th'].append(len([x for x in shapes if x < np.percentile(shapes, 1)]))
#     df_dict['above_99th'].append(len([x for x in shapes if x > np.percentile(shapes, 99)]))
#
# df = pd.DataFrame(df_dict)
# df.to_csv('data/crops_shapes_.csv', index=False)
#
# # IF YOU WANNA PLOT CROPS
#
# # n = 885
# #
# # fig, axs = plt.subplots(2, 1)
# # axs[0].plot(tot[n][0])
# # axs[0].set_title(tot[n][1] + f'_{cum_len[n]}')
# # axs[1].plot(raws[n][0])
# # axs[1].set_title(raws[n][1])
# # plt.show()
#
# Q1 = np.percentile(alls, 25)
# Q3 = np.percentile(alls, 75)
# P05 = np.percentile(alls, 5)
# P10 = np.percentile(alls, 10)
# P90 = np.percentile(alls, 90)
# P95 = np.percentile(alls, 95)
# P99 = np.percentile(alls, 99)
# P01 = np.percentile(alls, 1)
# IQR = Q3 - Q1
#
# df_dict_ = {}
#
# df_dict_['Q1-1.5IQR'] = [Q1 - 1.5 * IQR]
# df_dict_['Q3+1.5IQR'] = [Q3 + 1.5 * IQR]
# df_dict_['Q1-3IQR'] = [Q1 - 3 * IQR]
# df_dict_['Q3+3IQR'] = [Q3 + 3 * IQR]
# df_dict_['90thP'] = [P90]
# df_dict_['10thP'] = [P10]
# df_dict_['95thP'] = [P95]
# df_dict_['05thP'] = [P05]
# df_dict_['99thP'] = [P99]
# df_dict_['01thP'] = [P01]
# df_dict_['below_1.5IQR'] = [len([x for x in alls if x < Q1 - 1.5 * IQR])]
# df_dict_['above_1.5IQR'] = [len([x for x in alls if x > Q3 + 1.5 * IQR])]
# df_dict_['below_3IQR'] = [len([x for x in alls if x < Q1 - 3 * IQR])]
# df_dict_['above_3IQR'] = [len([x for x in alls if x > Q3 + 3 * IQR])]
# df_dict_['below_10th'] = [len([x for x in alls if x < P10])]
# df_dict_['above_90th'] = [len([x for x in alls if x > P90])]
# df_dict_['below_05th'] = [len([x for x in alls if x < P05])]
# df_dict_['above_95th'] = [len([x for x in alls if x > P95])]
# df_dict_['below_01th'] = [len([x for x in alls if x < P01])]
# df_dict_['above_99th'] = [len([x for x in alls if x > P99])]
#
# df_ = pd.DataFrame(df_dict_)
# df_.to_csv('data/all_.csv', index=False)

crops = Crops(parent='dataset/crops_raw')
crops.split(test_size=0.15)
# all = crops.N_crops + crops.S_crops + crops.V_crops
#
# over_3iqr = 0
# over_15iqr = 0
#
# for x in all:
#     if np.max(x) > 8.05 or np.min(x) < -8.05:
#         over_3iqr += 1
#     if np.max(x) > 4.55 or np.min(x) < -4.55:
#         over_15iqr += 1
#
# print(f'over 3iqr: {over_3iqr}, over 1.5iqr: {over_15iqr}, tot: {len(all)}')

dataset = CropsDataset(crops.train, mode='binary', stratify=True, raw=True, normalization='min_max')

x, y = next(iter(dataset))

for i, (x, y) in enumerate(dataset):
    if np.max(x.permute(0,1).numpy()) > 1 or np.min(x.permute(0,1).numpy()) < 0:
        print('aaaa', i)

    # if i == 38:
    #     print(x.shape)
    #     plt.plot(x.squeeze().numpy())
    #     plt.show()
    #
