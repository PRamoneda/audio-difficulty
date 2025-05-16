import os
import random

from utils import load_binary

import matplotlib.pyplot as plt
def main():
    # two cqts spectrograms print both
    #get random file from directory ../videos_download/cqt10
    path10 = f'../videos_download/cqt10/{random.choice(os.listdir("../videos_download/cqt10"))}'
    path5 = path10.replace('cqt10/', 'cqt5/')
    # plt both spectrograms with matplotlib
    cqt10 = load_binary(path10)
    cqt5 = load_binary(path5)
    # Plotting the spectrograms
    # Adjusting the layout to plot the spectrograms one above the other
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    # Spectrogram 1
    cax1 = axs[0].imshow(cqt5, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(cax1, ax=axs[0])
    axs[0].set_title('Spectrogram 1')
    axs[0].set_xlabel('Frequency Bins')
    axs[0].set_ylabel('Time Bins')

    # Spectrogram 2
    cax2 = axs[1].imshow(cqt10, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(cax2, ax=axs[1])
    axs[1].set_title('Spectrogram 2')
    axs[1].set_xlabel('Frequency Bins')
    axs[1].set_ylabel('Time Bins')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


