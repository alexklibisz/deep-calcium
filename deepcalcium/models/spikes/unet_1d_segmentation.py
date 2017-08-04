from __future__ import division, print_function
from glob import glob
from itertools import cycle
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from math import ceil
from os import path, mkdir
from time import time
import h5py
import keras.backend as K
import logging
import numpy as np
import os

from deepcalcium.utils.runtime import funcname
from deepcalcium.utils.keras_helpers import MetricsPlotCallback
from deepcalcium.models.spikes.utils import F2, prec, reca, maxpool1D, ytspks, ypspks, F2_margin, prec_margin, reca_margin, plot_traces_spikes

rng = np.random


class _SamplePlotCallback(Callback):
    """Keras callback that plots sample predictions during training."""

    def __init__(self, save_path, traces, spikes, nb_plot=30, title='Epoch {epoch:d} loss={loss:.3f}'):

        self.save_path = save_path
        self.traces = traces
        self.spikes = spikes
        self.nb_plot = min(len(traces), nb_plot)
        self.title = title

    def on_epoch_end(self, epoch, logs):

        # Get newest weights, predict, plot.
        spikes_pred = self.model.predict(self.traces[:self.nb_plot])
        plot_traces_spikes(traces=self.traces[:self.nb_plot],
                           spikes_true=self.spikes[:self.nb_plot],
                           spikes_pred=spikes_pred[:self.nb_plot],
                           title=self.title.format(epoch=epoch, **logs),
                           save_path=self.save_path.format(epoch=epoch),
                           dpi=120)


def unet1d(window_shape=(128,), nb_filters_base=32, conv_kernel_init='he_normal', prop_dropout_base=0.15):
    """Builds and returns the UNet architecture using Keras.
    # Arguments
        window_shape: tuple of one integer defining the input/output window shape.
        nb_filters_base: number of convolutional filters used at the first layer. This is doubled
            after every pooling layer, four times until the bottleneck layer, and then it gets
            divided by two four times to the output layer.
        conv_kernel_init: weight initialization for the convolutional kernels. He initialization
            is considered best-practice when using ReLU activations, as is the case in this network.
        prop_dropout_base: proportion of dropout after the first pooling layer. Two-times the
            proportion is used after subsequent pooling layers on the downward pass.
    # Returns
        model: Keras model, not compiled.
    """

    from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, concatenate, BatchNormalization, Lambda, UpSampling1D, Activation
    from keras.models import Model

    drp = prop_dropout_base
    nfb = nb_filters_base
    cki = conv_kernel_init

    # Theano vs. TF setup.
    assert K.backend() == 'tensorflow', 'Theano implementation is incomplete.'

    def up_layer(nb_filters, x):
        return UpSampling1D()(x)

    def conv_layer(nbf, x):
        x = Conv1D(nbf, 5, strides=1, padding='same', kernel_initializer=cki)(x)
        x = BatchNormalization(axis=-1)(x)
        return Activation('relu')(x)

    x = inputs = Input(window_shape)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    dc_0_out = x

    x = MaxPooling1D(2, strides=2)(x)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = Dropout(drp)(x)
    dc_1_out = x

    x = MaxPooling1D(2, strides=2)(x)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = Dropout(drp * 2)(x)
    dc_2_out = x

    x = MaxPooling1D(2, strides=2)(x)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = Dropout(drp * 2)(x)
    dc_3_out = x

    x = MaxPooling1D(2, strides=2)(x)
    x = conv_layer(nfb * 16, x)
    x = conv_layer(nfb * 16, x)
    x = up_layer(nfb * 8, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_3_out], axis=-1)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = up_layer(nfb * 4, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_2_out], axis=-1)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = up_layer(nfb * 2, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_1_out], axis=-1)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = up_layer(nfb, x)
    x = Dropout(drp)(x)

    x = concatenate([x, dc_0_out], axis=-1)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    x = Conv1D(2, 1, activation='softmax')(x)

    x = Lambda(lambda x: x[:, :, -1])(x)
    model = Model(inputs=inputs, outputs=x)

    return model


def weighted_binary_crossentropy(yt, yp, margin=1, weightpos=2., weightneg=1.):
    """Apply different weights to true positives and true negatives with
    binary crossentropy loss. Allow an error margin to prevent penalizing
    for off-by-N errors.

    # Arguments
        yt, yp: Keras ground-truth and predicted batch matrices.
        margin: number of time-steps within which an error is tolerated. e.g.
            margin = 1 would allow off-by-1 errors. This is implemented by
            max-pooling the ground-truth and predicted matrices.
        weightpos: weight multiplier for loss on true-positives.
        weightnet: weight multiplier for loss on true-negatives.

    # Returns
        matrix of loss scalars with shape (batch size x 1).

    """

    L, S = 2 * margin + 1, 1  # pooling length and stride.
    yt, yp = maxpool1D(yt, L, S), maxpool1D(yp, L, S)
    losspos = yt * K.log(yp + 1e-7)
    lossneg = (1 - yt) * K.log(1 - yp + 1e-7)
    return -1 * ((weightpos * losspos) + (weightneg * lossneg))


def _dataset_attrs_func(dspath):
    return


def _dataset_traces_func(dspath):
    fp = h5py.File(dspath)
    traces = fp.get('traces')[...]
    fp.close()
    m = np.mean(traces, axis=1, keepdims=True)
    s = np.std(traces, axis=1, keepdims=True)
    traces = (traces - m) / s
    assert -5 < np.mean(traces) < 5, np.mean(traces)
    assert -5 < np.std(traces) < 5, np.std(traces)
    return traces


def _dataset_spikes_func(dspath):
    fp = h5py.File(dspath)
    spikes = fp.get('spikes')[...]
    fp.close()
    return spikes


class UNet1DSegmentation(object):
    """Trace segmentation wrapper class. In general, this type of model takes a
    calcium trace of length N frames and return a binary segmentation
    of length N frames. e.g. f([0.1, 0.2, ...]) -> [0, 1, ...].

    # Arguments
        cpdir: checkpoint directory for training artifacts and predictions.
        dataset_attrs_func: function f(hdf5 path) -> dataset attributes.
        dataset_traces_func: function f(hdf5 path) -> dataset calcium traces array
            with shape (no. ROIs x no. frames).
        dataset_spikes_func: function f(hdf5 path) -> dataset binary spikes array
            with shape (no. ROIs x no. frames).
        net_builder_func: function that builds and returns the Keras model for
            training and predictions. This allows swapping out the network
            architecture without re-writing or copying all training and prediction
            code.
    """

    def __init__(self, cpdir='%s/.deep-calcium-datasets/tmp' % os.path.expanduser('~'),
                 dataset_attrs_func=_dataset_attrs_func,
                 dataset_traces_func=_dataset_traces_func,
                 dataset_spikes_func=_dataset_spikes_func,
                 net_builder_func=unet1d):

        self.cpdir = cpdir
        self.dataset_attrs_func = dataset_attrs_func
        self.dataset_traces_func = dataset_traces_func
        self.dataset_spikes_func = dataset_spikes_func
        self.net_builder_func = net_builder_func

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, dataset_paths, model_path=None, shape=(4096,), error_margin=1.,
            batch=20, nb_epochs=20, val_type='random_split', prop_trn=0.8,
            prop_val=0.2, nb_folds=5, keras_callbacks=[], optimizer=Adam(0.002)):
        """Constructs model based on parameters and trains with the given data.
        Internally, the function uses a local function to abstract the training
        for both validation types.

        # Arguments
            dataset_paths: list of paths to HDF5 datasets used for training.
            model_path: filesystem path to serialized model that should be
                loaded into the network.
            shape: tuple defining the input length.
            error_margin: number of frames within which a false positive error
                is allowed. e.g. error_margin=1 would allow off-by-1 errors.
            batch: batch size.
            val_type: either 'random_split' or 'cross_validate'.
            prop_trn: proportion of data for training when using random_split.
            prop_val: proportion of data for validation when using random_split.
            nb_folds: number of folds for K-fold cross valdiation using using
                cross_validate.
            keras_callbacks: additional callbacks that should be included.
            nb_epochs: how many epochs. 1 epoch includes 1 sample of every trace.
            optimizer: instantiated keras optimizer.

        # Returns
            history: the keras training history as a dictionary of metrics and
                their values after each epoch.
            model_path: path to the HDF5 file where the best architecture and
                weights were serialized.

        """

        def _fit_single(idxs_trn, idxs_val, model_summary=False):
            """Instantiates model, splits data based on given indices, trains.
            Abstracted in order to enable both random split and cross-validation.

            # Returns
                metrics_trn: dictionary of {name: metric} for training data.
                metrics_val: dictionary of {name: metric} for validation data.
                best_model_path: filesystem path to the best serialized model.
            """

            # Define metrics and loss based on error margin.
            def F2M(yt, yp, margin=error_margin):
                return F2_margin(yt, yp, margin)

            def precM(yt, yp, margin=error_margin):
                return prec_margin(yt, yp, margin)

            def recaM(yt, yp, margin=error_margin):
                return reca_margin(yt, yp, margin)

            def loss(yt, yp, margin=error_margin):
                return weighted_binary_crossentropy(yt, yp, margin)

            metrics = [F2, prec, reca, F2M, precM, recaM, ytspks, ypspks]
            custom_objects = {o.__name__: o for o in metrics + [loss]}

            # Load network from disk.
            if model_path:
                model = load_model_with_new_input_shape(
                    model_path, shape, compile=True, custom_objects=custom_objects)

            # Define, compile network.
            else:
                model = self.net_builder_func(shape)
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                if model_summary:
                    model.summary()

            # Split traces and spikes.
            tr_trn = [traces[i] for i in idxs_trn]
            sp_trn = [spikes[i] for i in idxs_trn]
            tr_val = [traces[i] for i in idxs_val]
            sp_val = [spikes[i] for i in idxs_val]

            # 1 epoch = 1 training sample from every trace.
            steps_trn = int(ceil(len(tr_trn) / batch))

            # Training and validation generators.
            bg = self._batch_gen
            gen_trn = bg(tr_trn, sp_trn, shape, batch, steps_trn)
            gen_val = bg(tr_val, sp_val, shape, len(tr_val) * 2, 1)
            x_val, y_val = next(gen_val)

            # Callbacks.
            cpt, spc = (self.cpdir, int(time())), _SamplePlotCallback
            cb = [
                spc('%s/%d_samples_{epoch:03d}_trn.png' % cpt, *next(gen_trn),
                    title='Epoch {epoch:d} val_F2={val_F2:3f} val_F2M={val_F2M:3f}'),
                spc('%s/%d_samples_{epoch:03d}_val.png' % cpt, x_val, y_val,
                    title='Epoch {epoch:d} val_F2={val_F2:3f} val_F2M={val_F2M:3f}'),
                ModelCheckpoint('%s/%d_model_val_F2M_{val_F2M:3f}_{epoch:03d}.hdf5' % cpt,
                                monitor='val_F2M', mode='max', verbose=1, save_best_only=True),
                CSVLogger('%s/%d_metrics.csv' % cpt),
                MetricsPlotCallback('%s/%d_metrics.png' % cpt)
            ]

            # Train.
            model.fit_generator(gen_trn, steps_per_epoch=steps_trn,
                                epochs=nb_epochs, callbacks=cb,
                                validation_data=(x_val, y_val), verbose=1)

            # Identify best serialized model, assuming newest is best.
            model_path_glob = '%s/%d_model*hdf5' % cpt
            model_paths = sorted(glob(model_path_glob), key=os.path.getmtime)
            best_model_path = model_paths[-1]

            # Training and validation metrics on trained model.
            model.load_weights(best_model_path)
            mt = model.evaluate_generator(gen_trn, steps_trn)
            mt = {n: m for n, m in zip(model.metrics_names, mt)}
            mv = model.evaluate(x_val, y_val)
            mv = {n: m for n, m in zip(model.metrics_names, mv)}

            return mt, mv, best_model_path
            # END OF INTERNAL FUNCTION.

        logger = logging.getLogger(funcname())

        # Error check.
        assert len(shape) == 1
        assert val_type in ['random_split', 'cross_validate']
        assert nb_folds > 1
        assert prop_trn + prop_val == 1.

        # Extract traces and spikes from datasets.
        traces = [t for p in dataset_paths for t in self.dataset_traces_func(p)]
        spikes = [s for p in dataset_paths for s in self.dataset_spikes_func(p)]
        assert len(traces) == len(spikes)

        # Random-split training.
        if val_type == 'random_split':

            idxs = rng.choice(np.arange(len(traces)), len(traces), replace=0)
            idxs_trn = idxs[:int(len(idxs) * prop_trn)]
            idxs_val = idxs[-1 * int(len(idxs) * prop_val):]
            mt, mv, _ = _fit_single(idxs_trn, idxs_val, True)
            for k in sorted(mt.keys()):
                s = (k, mt[k], mv[k])
                logger.info('%-20s trn=%-9.4lf val=%-9.4lf' % s)

        # Cross-validation training.
        elif val_type == 'cross_validate':

            # Randomly-ordered indicies for cross-validation.
            idxs = rng.choice(np.arange(len(traces)), len(traces), replace=0)
            fsz = int(len(idxs) / nb_folds)
            fold_idxs = [idxs[fsz * n:fsz * n + fsz] for n in range(nb_folds)]

            # Train on folds.
            metrics_trn, metrics_val = [], []
            for val_idx in range(nb_folds):

                # Seperate training and validation indexes.
                idxs_trn = [idx for i, fold in enumerate(fold_idxs)
                            if i != val_idx for idx in fold]
                idxs_val = [idx for i, fold in enumerate(fold_idxs)
                            if i == val_idx for idx in fold]
                assert set(idxs_trn).intersection(idxs_val) == set([])

                # Train and report metrics.
                logger.info('\nCross validation fold = %d' % val_idx)
                mt, mv, _ = _fit_single(idxs_trn, idxs_val, val_idx == 0)
                metrics_trn.append(mt)
                metrics_val.append(mv)

                for k in sorted(mt.keys()):
                    s = (k, mt[k], mv[k])
                    logger.info('%-20s trn=%-10.4lf val=%-10.4lf' % s)

            # Aggregate metrics.
            logger.info('\nCross validation summary')
            for k in sorted(metrics_trn[0].keys()):
                vals_trn = [m[k] for m in metrics_trn]
                vals_val = [m[k] for m in metrics_val]
                s = (k, np.mean(vals_trn), np.std(vals_trn),
                     np.mean(vals_val), np.std(vals_val))
                logger.info('%-20s trn=%-9.4lf (%.4lf) val=%-9.4lf (%.4lf)' % s)

    def _batch_gen(self, traces, spikes, shape, batch_size, nb_steps):

        while True:

            idxs = np.arange(len(traces))
            cidxs = cycle(rng.choice(idxs, len(idxs), replace=False))

            for _ in range(nb_steps):

                # Empty batches (traces and spikes).
                tb = np.zeros((batch_size,) + shape, dtype=np.float64)
                sb = np.zeros((batch_size,) + shape, dtype=np.uint8)

                for bidx in range(batch_size):

                    # Dataset and sample indices.
                    idx = next(cidxs)

                    # Pick start and end point around positive spike index.
                    x0 = rng.randint(0, len(spikes[idx]) - shape[0])
                    x1 = x0 + shape[0]

                    # Populate batch.
                    tb[bidx] = traces[idx][x0:x1]
                    sb[bidx] = spikes[idx][x0:x1]

                yield tb, sb

    def predict(self, dataset_paths, model_path, sample_shape=(128,), print_scores=True, save=True):
        pass
