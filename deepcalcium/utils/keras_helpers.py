from __future__ import division
from keras.callbacks import Callback
from math import ceil
from time import sleep, ctime
import keras.backend as K
import logging
import numpy as np
import pandas as pd


def prec(yt, yp):
    """Keras precision metric."""
    yp = K.round(yp)
    return K.sum(yp * yt) / (K.sum(yp) + K.epsilon())


def reca(yt, yp):
    """Keras recall metric."""
    yp = K.round(yp)
    tp = K.sum(yp * yt)
    fn = K.sum(K.clip(yt - yp, 0, 1))
    return K.sum(yp * yt) / (tp + fn + K.epsilon())


def F1(yt, yp):
    """Keras F1 metric."""
    p = prec(yt, yp)
    r = reca(yt, yp)
    return (2 * p * r) / (p + r + K.epsilon())


def jacc(yt, yp):
    """Keras Jaccard coefficient metric."""
    union = K.sum(yt * yp)
    return union / (K.sum(yt) + K.sum(yp) - union)


def load_model_with_new_input_shape(model_path, input_shape, **load_model_args):
    """Given a model_path, configures the model to have a new input shape, then
    loads the model using keras' load_model with the given load_model_args."""

    from keras.models import load_model
    from json import loads, dumps
    from hashlib import md5
    from shutil import copy
    from os import path, remove
    from time import time
    import h5py

    if input_shape[0]:
        input_shape = (None,) + input_shape

    # Make a copy of the model hdf5 file.
    path = '.tmp_%s_%d' % (path.basename(model_path), int(time()))
    copy(model_path, path)

    # Open the copied hdf5 file and modify the input layer's shape.
    h5 = h5py.File(path, 'a')
    config = loads(h5.attrs['model_config'])
    config['config']['layers'][0]['batch_input_shape'] = list(input_shape)
    config['config']['layers'][0]['config']['batch_input_shape'] = list(input_shape)
    h5.attrs['model_config'] = dumps(config)
    h5.close()

    # Load model, delete temporary file, return model.
    model = load_model(path, **load_model_args)
    remove(path)
    return model


class MetricsPlotCallback(Callback):
    """Plots all of the metrics in a single figure and saves to the given file name. Can optionally read from a CSV (i.e. using CSVLogger callback to save a CSV then read it.)"""

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
