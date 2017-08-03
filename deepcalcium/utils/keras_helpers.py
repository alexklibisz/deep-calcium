from __future__ import division
from keras.callbacks import Callback
from math import ceil
import keras.backend as K
import numpy as np
import pandas as pd


def load_model_with_new_input_shape(model_path, input_shape, **load_model_args):
    """Given a model_path, configures the model to have a new input shape, then
    loads the model using keras' load_model with the given load_model_args."""

    from keras.models import load_model
    from json import loads, dumps
    from shutil import copy
    from os import path, remove
    from time import time
    import h5py

    def replace_shape(old_shape):
        old_val = max(old_shape)
        new_val = max(input_shape)
        return [x if x != old_val else new_val for x in old_shape]

    # Make a copy of the model hdf5 file.
    path = '.tmp_%s_%d' % (path.basename(model_path), int(time()))
    copy(model_path, path)

    # Open the copied hdf5 file and modify the input layer's shape.
    h5 = h5py.File(path, 'a')
    config = loads(h5.attrs['model_config'])

    for layer in config['config']['layers']:

        if 'batch_input_shape' in layer['config'] and layer['config']['batch_input_shape']:
            layer['config']['batch_input_shape'] = replace_shape(
                layer['config']['batch_input_shape'])

        if 'output_shape' in layer['config'] and layer['config']['output_shape']:
            layer['config']['output_shape'] = replace_shape(
                layer['config']['output_shape'])

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
        fig, _ = plt.subplots(nb_row, nb_col, figsize=(
            min(nb_col * 3, 10), 3 * nb_row))
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
