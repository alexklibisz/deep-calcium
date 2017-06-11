from keras.callbacks import Callback
from math import ceil
from time import sleep, ctime
import keras.backend as K
import logging
import numpy as np

class HistoryPlotCallback(Callback):
    '''Plots all of the metrics in a single figure and saves to the given file name. Plots the same metric's validation and training values on the same subplot for easy comparison and overfit monitoring.'''

    def __init__(self, file_name=None):
        super(Callback, self).__init__()
        self.logs = {}
        self.file_name = file_name

    def on_train_begin(self, logs={}):
        self.logs = {}

    def on_epoch_end(self, epoch, logs={}):
        if self.file_name is not None:
            import matplotlib
            matplotlib.use('agg')
        import matplotlib.pyplot as plt
        if len(self.logs) == 0:
            self.logs = {key: [] for key in logs.keys()}
        for key, val in logs.items():
            self.logs[key].append(val)
        keys = sorted([k for k in self.logs.keys() if not k.startswith('val')])
        nb_metrics = len(keys)
        keys = iter(keys)
        nb_col = 6
        nb_row = int(ceil(nb_metrics * 1.0 / nb_col))
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(min(nb_col * 3, 12), 3 * nb_row))
        for idx, ax in enumerate(fig.axes):
            if idx >= nb_metrics:
                ax.axis('off')
                continue
            key = next(keys)
            ax.set_title(key)
            ax.plot(self.logs[key], label='TR')
            val_key = 'val_%s' % key
            if val_key in self.logs:
                ax.plot(self.logs[val_key], label='VL')
            ax.legend()
        plt.suptitle('Epoch %d: %s' % (epoch, ctime()), y=1.10)
        plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
        if self.file_name is not None:
            plt.savefig(self.file_name)
            plt.close()
        else:
            plt.show()
            plt.close()
