import numpy as np
from matplotlib import pyplot as plt




def raw_vs_filtered(raw, filtered, onsets, peaks, labels, dt = 0):
    """
    :param
        --raw: raw signal
        --filtered: filtered signal
        --onset: onsets points
        --dt: displacement
    """

    on = np.zeros((len(filtered),1)).squeeze()
    pks = np.zeros((len(filtered),1)).squeeze()
    labs_raw = []
    labs_filt = []
    print(len(filtered),onsets[-1])
    on[onsets]=1
    pks[peaks]=1
    for i in range(len(peaks)):
        labs_raw.append((int(peaks[i]), raw[int(peaks[i])], labels[i]))
        labs_filt.append((int(peaks[i]), filtered[int(peaks[i])], labels[i]))


    fig, axes = plt.subplots(2, 1)
    ax = axes[0]
    ax.plot(range(dt,dt+5000),raw[dt:dt+5000])
    ax.scatter(dt+np.nonzero(on[dt:dt+5000])[0], raw[dt+np.nonzero(on[dt:dt+5000])[0]], color='red', marker='o')
    ax.scatter(dt+np.nonzero(pks[dt:dt+5000])[0], raw[dt+np.nonzero(pks[dt:dt+5000])[0]], color='blue', marker='*')
    for x, y, label in labs_raw[list(peaks).index(dt+np.nonzero(pks[dt:dt+5000])[0][0]):list(peaks).index(dt+np.nonzero(pks[dt:dt+5000])[0][-1])]:
        ax.text(x, y+0.01*abs(y), label, fontsize=8, color='black')
    ax.set_title('RAW')

    axes[1].plot(range(dt,dt+5000),filtered[dt:dt+5000])
    axes[1].scatter(dt+np.nonzero(on[dt:dt+5000])[0], filtered[dt+np.nonzero(on[dt:dt+5000])[0]], color='red', marker='o')
    axes[1].scatter(dt+np.nonzero(pks[dt:dt+5000])[0], filtered[dt+np.nonzero(pks[dt:dt+5000])[0]], color='blue', marker='*')
    for x, y, label in labs_filt[list(peaks).index(dt+np.nonzero(pks[dt:dt+5000])[0][0]):list(peaks).index(dt+np.nonzero(pks[dt:dt+5000])[0][-1])+1]:
        axes[1].text(x, y+0.01*abs(y), label, fontsize=8, color='black')
    axes[1].set_title('Filtered')




    # ax = axes[0]
    # ax.plot(range(onsets[dt+0],onsets[dt+40]),raw[onsets[dt+0]:onsets[dt+40]])
    # axes[0].scatter(onsets[dt+0:dt+41], raw[onsets[dt+0:dt+41]], color='red', marker='o')
    # axes[0].scatter(peaks[dt + 0:dt + 40], raw[peaks[dt + 0:dt + 40]], color='blue', marker='*')
    # ax.set_title('RAW')
    #
    # axes[1].plot(range(onsets[dt+0],onsets[dt+40]), filtered[onsets[dt+0]:onsets[dt+40]])
    # axes[1].scatter(onsets[dt+0:dt+41], filtered[onsets[dt+0:dt+41]], color='red', marker='o')
    # axes[1].scatter(peaks[dt + 0:dt + 40], filtered[peaks[dt + 0:dt + 40]], color='blue', marker='*')
    # axes[1].set_title('Filtered')
    #
    plt.tight_layout()
    plt.show()