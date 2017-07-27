from __future__ import division, print_function
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from os import path, mkdir
from scipy.misc import imsave
from time import time
import h5py
import keras.backend as K
import logging
import numpy as np
import os

from deepcalcium.utils.runtime import funcname
from deepcalcium.utils.keras_helpers import MetricsPlotCallback, F2, prec, reca

rng = np.random


class _SamplePlotCallback(Callback):

    def __init__(self, model, save_path, traces, spikes, nb_plot=30, title='Epoch {epoch:d} loss={loss:.3f}'):

        self._model = model
        self.save_path = save_path
        self.traces = traces
        self.spikes = spikes
        self.nb_plot = min(len(traces), nb_plot)
        self.title = title

    def on_epoch_end(self, epoch, logs):

        # Get newest weights.
        self._model.set_weights(self.model.get_weights())

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        def scale(x): return x / np.max(x)
        spikes_pred = self._model.predict(self.traces[:self.nb_plot])

        fig, axes = plt.subplots(self.nb_plot, 1, figsize=(8, self.nb_plot))
        for i, ax in enumerate(axes):

            # Compute scaled signal.
            t = scale(self.traces[i])
            ax.plot(t, 'k')

            # True labels as points.
            ax.plot((self.spikes[i] * 3 - 2) * t, 'bo', label='yt', alpha=0.7)

            # Predicted labels as a series.
            ax.plot((spikes_pred[i].round() * 3 - 2) * t, 'rx')

            ax.set_ylim(0., 1.2)
            if i == 0.:
                ax.legend()

        plt.suptitle(self.title.format(epoch=epoch, **logs))
        plt.savefig(self.save_path.format(epoch=epoch), dpi=150)
        plt.close()


class _ValidationMetricsCB(Callback):
    """Keras callback that evaluates validation metrics on full-size predictions
    during training."""

    def __init__(self, model_val, batch_gen, steps):
        self.model_val = model_val
        self.batch_gen = batch_gen
        self.steps = steps

    def on_epoch_end(self, epoch, logs={}):
        self.model_val.set_weights(self.model.get_weights())
        metrics = self.model_val.evaluate_generator(self.batch_gen, self.steps)
        for name, metric in zip(self.model_val.metrics_names, metrics):
            logs['val_%s' % name] = metric


def unet1d(window_shape=(128,), nb_filters_base=32, conv_kernel_init='he_normal', prop_dropout_base=0.2):
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

    def conv_layer(nb_filters, x):
        x = Conv1D(nb_filters, 9, strides=1, padding='same',
                   kernel_initializer=cki)(x)
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


def maxpool1D(x, pool_size, pool_strides):
    """1D pooling along the rows of a 2D array. Requires reshaping to work
    with the keras pool2d function."""
    x = K.expand_dims(K.expand_dims(x, axis=0), axis=-1)
    x = K.pool2d(x, (1, pool_size), (1, pool_strides))
    return x[0, :, :, 0]


def weighted_maxpool_binary_crossentropy(yt, yp, L=3, S=1, wp=2., wn=1.):
    """Applying max-pooling before computing binary crossentropy loss.
    Apply weights wp to ground-truth positives and wn to ground-truth negatives."""
    yt, yp = maxpool1D(yt, L, S), maxpool1D(yp, L, S)
    losspos = yt * K.log(yp + 1e-7)
    lossneg = (1 - yt) * K.log(1 - yp + 1e-7)
    return -1 * ((wp * losspos) + (wn * lossneg))


def F2_maxpool(yt, yp, L=3, S=1):
    return F2(maxpool1D(yt, L, S), maxpool1D(yp, L, S))


def ytspks(yt, yp):
    """On average, how many spikes in yt."""
    return K.sum(yt, axis=1)


def ypspks(yt, yp):
    """On average, how many spikes in yp."""
    return K.sum(K.round(yp), axis=1)


def _dataset_attrs_func(dspath):
    return


def _dataset_traces_func(dspath):
    fp = h5py.File(dspath)
    traces = fp.get('traces')[...]
    fp.close()
    return traces


def _dataset_spikes_func(dspath):
    fp = h5py.File(dspath)
    spikes = fp.get('spikes')[...]
    fp.close()
    return spikes


class TraceSegmentation(object):
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

        cobj = []
        self.custom_objects = {o.__name__: o for o in cobj}

    def fit(self, dataset_paths, model_path=None, proceed=False,
            shape_trn=(128,), shape_val=(4096,),
            batch_trn=32, batch_val=10,
            steps_trn=200, steps_val=20,
            val_type='leave_one_out', val_index=-1, prop_trn=0.8, prop_val=0.2,
            nb_epochs=20, keras_callbacks=[],
            optimizer=Adam(0.002), loss=weighted_maxpool_binary_crossentropy):
        """Constructs model based on parameters and trains with the given data.

        # Arguments
            dataset_paths: list of paths to HDF5 datasets used for training.
            model_path: filesystem path to serialized model that should be
                loaded into the network.
            proceed: whether to continue training where the model left off or
                start over. Only relevant when model_path is given.
            shape_trn: 1D tuple defining the training input length.
            shape_val: 1D tuple defining the validation input length.
            steps_trn: number of updates per training epoch.
            steps_val: number of updates per validation epoch.
            val_type: string defining the train/validation split strategy.
                Either 'leave_one_out' or 'random_split'.
            val_index: if the val_type is 'leave_one_out', this index in the
                dataset_paths list is used for validation.
            prop_trn: if the val_type is 'random_split', this proportion of all
                calcium traces is used for training.
            prop_val: if the val_type is 'random_split', this proportion of all
                calcium traces is used for validation.
            optimizer: instantiated keras optimizer.
            loss: loss function, either string or an actual keras-compatible
                loss function.

        # Returns:
            history: the keras training history as a dictionary of metrics and
                their values after each epoch.
            model_path: path to the HDF5 file where the best architecture and
                weights were serialized.

        """

        # Error check.
        assert len(shape_trn) == 1
        assert len(shape_val) == 1
        assert not (proceed and not model_path)
        assert val_type in {'leave_one_out', 'random_split'}

        # Load network from disk.
        if model_path:
            lmwnis = load_model_with_new_input_shape
            model = lmwnis(model_path, shape_trn, compile=proceed,
                           custom_objects=self.custom_objects)
            model_val = lmwnis(model_path, shape_val, compile=False,
                               custom_objects=self.custom_objects)

        # Define, compile network.
        else:
            model = self.net_builder_func(shape_trn)
            model.summary()
            model_val = self.net_builder_func(shape_val)

        # Recompile network if proceed is false.
        if not proceed:
            m = [F2, prec, reca, F2_maxpool, ytspks, ypspks]
            model.compile(optimizer=optimizer, loss=loss, metrics=m)
            model_val.compile(optimizer=optimizer, loss=loss, metrics=m)

        # Extract traces and spikes from datasets.
        traces = [self.dataset_traces_func(p) for p in dataset_paths]
        spikes = [self.dataset_spikes_func(p) for p in dataset_paths]
        for p, t, s in zip(dataset_paths, traces, spikes):
            assert t.shape == s.shape, "Bad shapes: %s" % p

        # Training/validation split.
        if val_type == 'random_split':
            idxs = [list(range(x.shape[0])) for x in traces]
            idxs_trn = [rng.choice(ix, int(len(ix) * prop_trn), replace=False)
                        for ix in idxs]
            idxs_val = [sorted(list(set(ix) - set(ixt)))
                        for ix, ixt in zip(idxs, idxs_trn)]
            traces_trn = [traces[i][ix, :] for i, ix in enumerate(idxs_trn)]
            spikes_trn = [spikes[i][ix, :] for i, ix in enumerate(idxs_trn)]
            traces_val = [traces[i][ix, :] for i, ix in enumerate(idxs_val)]
            spikes_val = [spikes[i][ix, :] for i, ix in enumerate(idxs_val)]
        elif val_type == 'leave_one_out':
            idxs_trn = [i for i in range(len(traces)) if i != val_index]
            idxs_val = [i for i in range(len(traces)) if i == val_index]
            traces_trn = [traces[i] for i in idxs_trn]
            spikes_trn = [spikes[i] for i in idxs_trn]
            traces_val = [traces[i] for i in idxs_val]
            spikes_val = [spikes[i] for i in idxs_val]

        # Training and validation generators.
        bg = self._batch_gen
        gen_trn = bg(traces_trn, spikes_trn, shape_trn, batch_trn)
        gen_val = bg(traces_val, spikes_val, shape_val, batch_val)

        # Callbacks.
        cpt = (self.cpdir, int(time()))
        cb = [
            _ValidationMetricsCB(model_val, gen_val, steps_val),
            _SamplePlotCallback(model,
                                '%s/%d_samples_{epoch:03d}_trn.png' % cpt, *next(gen_trn),
                                title='Epoch {epoch:d} reca={reca:.3f} prec={prec:.3f}'),
            _SamplePlotCallback(model_val,
                                '%s/%d_samples_{epoch:03d}_val.png' % cpt, *next(gen_val),
                                title='Epoch {epoch:d} val_reca={val_reca:.3f} val_prec={val_prec:.3f}'),
            ModelCheckpoint('%s/%d_model_val_F2_{val_F2:.3f}_{epoch:d}.hdf5' % cpt,
                            monitor='val_F2', mode='max', verbose=1, save_best_only=True),
            CSVLogger('%s/%d_metrics.csv' % cpt),
            MetricsPlotCallback('%s/%d_metrics.png' % cpt),
            EarlyStopping(monitor='val_F2', min_delta=0.001, patience=10,
                          verbose=1, mode='max')
        ]

        # Train.
        trained = model.fit_generator(gen_trn, steps_per_epoch=steps_trn,
                                      epochs=nb_epochs, callbacks=cb, verbose=1)

        # Summary of metrics.
        logger = logging.getLogger(funcname())
        for k in sorted(trained.history.keys(), key=lambda k: k.replace('val_', '')):
            v = trained.history[k]
            logger.info('%-20s %10.4lf (%d) %10.4lf (%d) %10.4lf (%d)' %
                        (k, v[-1], len(v), np.min(v), np.argmin(v), np.max(v), np.argmax(v)))

        # Return history and best model path.
        return trained.history, '%s/model_val_nf_f1_mean.hdf5' % self.cpdir

    def _batch_gen(self, traces, spikes, shape, batch_size):

        # Dataset > ROI > trace mean value.
        trmean = [np.mean(t, axis=1) for t in traces]

        # Dataset > ROI > list of positive (spiked) indices.
        spidxs = [[np.where(_s > 0)[0] if np.sum(_s) else np.arange(len(_s))
                   for _s in s] for s in spikes]

        while True:

            # Empty batches.
            tb = np.zeros((batch_size,) + shape, dtype=np.float16)
            sb = np.zeros((batch_size,) + shape, dtype=np.uint8)

            for bidx in range(batch_size):

                # Sample random indices for dataset, ROI, spike index.
                didx = rng.randint(0, len(traces))
                ridx = rng.randint(0, len(spikes[didx]))
                sidx = rng.choice(spidxs[didx][ridx])

                # Pick start and end point around positive spike index.
                w, = shape
                x0 = rng.randint(sidx + 1 - w, sidx - 1 + w)
                x0 = max(x0, 0)
                x0 = min(x0, len(spikes[didx][ridx]) - w)
                x1 = x0 + w

                # Populate batch.
                tb[bidx] = traces[didx][ridx][x0:x1]
                tb[bidx] /= trmean[didx][ridx]
                sb[bidx] = spikes[didx][ridx][x0:x1]

            yield tb, sb

    def predict(self, dataset_paths, model_path, sample_shape=(128,), print_scores=True, save=True):
        pass
