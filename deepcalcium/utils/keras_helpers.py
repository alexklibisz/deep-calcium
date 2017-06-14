from __future__ import division
from keras.callbacks import Callback
from math import ceil
from time import sleep, ctime
import keras.backend as K
import logging
import numpy as np
import pandas as pd


class MetricsPlotCallback(Callback):
    '''Plots all of the metrics in a single figure and saves to the given file name. Can optionally read from a CSV (i.e. using CSVLogger callback to save a CSV then read it.)'''

    def __init__(self, path_png_out, path_csv_in=None):
        super(Callback, self).__init__()
        self.logs = {}
        self.path_png_out = path_png_out
        self.path_csv_in = path_csv_in

    def on_train_begin(self, logs={}):
        self.logs = {}

    def on_epoch_end(self, epoch, logs={}):

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        # Mimic the keras logs structure from a dataframe.
        if self.path_csv_in is not None:
            df = pd.read_csv(self.path_csv_in)
            logs = {key: df[key].values[-1] for key in df.columns}

        # In-memory dict with metric name keys and list vals.
        if len(self.logs) == 0:
            self.logs = {key: [] for key in logs.keys()}

        # Read latest metrics.
        for key, val in logs.items():
            self.logs[key].append(val)

        # Make figure.
        nb_col = 5
        nb_row = int(ceil(len(logs) / nb_col))
        fig, _ = plt.subplots(nb_row, nb_col, figsize=(min(nb_col * 3, 10), 3 * nb_row))
        iterkeys = iter(sorted(logs.keys()))

        for idx, ax in enumerate(fig.axes):
            if idx >= len(logs.keys()):
                ax.axis('off')
                continue
            key = next(iterkeys)
            ax.set_title(key)
            ax.plot(self.logs[key])
            ax.legend()
        plt.suptitle('Epoch %d' % epoch)
        plt.savefig(self.path_png_out, dpi=200)
        plt.close()

        # keys = sorted([k for k in self.logs.keys() if not k.startswith('val')])
        # nb_metrics = len(keys)
        # keys = iter(keys)
        # nb_col = 6
        # nb_row = int(ceil(nb_metrics * 1.0 / nb_col))
        # fig, axs = plt.subplots(nb_row, nb_col, figsize=(min(nb_col * 3, 12), 3 * nb_row))
        # for idx, ax in enumerate(fig.axes):
        #     if idx >= nb_metrics:
        #         ax.axis('off')
        #         continue
        #     key = next(keys)
        #     ax.set_title(key)
        #     ax.plot(self.logs[key], label='TR')
        #     val_key = 'val_%s' % key
        #     if val_key in self.logs:
        #         ax.plot(self.logs[val_key], label='VL')
        #     ax.legend()
        # plt.suptitle('Epoch %d: %s' % (epoch, ctime()), y=1.10)
        # plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
        # if self.file_name is not None:
        #     plt.savefig(self.file_name)
        #     plt.close()
        # else:
        #     plt.show()
        #     plt.close()
