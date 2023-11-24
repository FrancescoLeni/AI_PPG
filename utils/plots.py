from matplotlib import pyplot as plt




def raw_vs_filtered(raw, filtered, onsets, dt = 0):
    """
    :param
        --raw: raw signal
        --filtered: filtered signal
        --onset: onsets points
        --dt: displacement
    """
    fig, axes = plt.subplots(2, 1)

    ax = axes[0]
    ax.plot(range(onsets[dt+0],onsets[dt+20]),raw[onsets[dt+0]:onsets[dt+20]])
    axes[0].scatter(onsets[dt+0:dt+21], raw[onsets[dt+0:dt+21]], color='red', marker='o')
    ax.set_title('RAW')

    axes[1].plot(range(onsets[dt+0],onsets[dt+20]), filtered[onsets[dt+0]:onsets[dt+20]])
    axes[1].scatter(onsets[dt+0:dt+21], filtered[onsets[dt+0:dt+21]], color='red', marker='o')
    axes[1].set_title('Filtered')

    plt.tight_layout()
    plt.show()