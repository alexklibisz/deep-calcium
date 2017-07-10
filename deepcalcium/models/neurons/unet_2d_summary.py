# U-Net 2D Summary module.
# The UNet2DS class is a wrapper around the UNet architecture that simplifies
# training and validation on calcium imaging segmentation tasks like Neurofinder.
# The code for building the UNet network can easily be extracted and re-used
# independent of the specific calcium-imaging features.
from __future__ import division, print_function
from itertools import cycle
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam
from keras_contrib.callbacks import DeadReluDetector
from math import ceil
from os import path, mkdir, remove
from scipy.misc import imsave
from skimage import transform
from time import time
from tqdm import tqdm
import json
import keras.backend as K
import logging
import numpy as np
import tensorflow as tf
import sys

from deepcalcium.utils.runtime import funcname
from deepcalcium.datasets.nf import nf_mask_metrics
from deepcalcium.utils.keras_helpers import MetricsPlotCallback
from deepcalcium.utils.visuals import mask_outlines


class ValidationMetricsCB(Callback):

    def __init__(self, model_val, S_summ, M_summ, names, y_coords, cpdir):
        self.model_val = model_val
        self.S_summ = []
        self.M_summ = []
        self.val_coords = []
        self.names = []
        self.cpdir = cpdir

        # Store standard, flipped, rotated summary images and their corresponding
        # validation masks.
        for s, m, name, (y0, y1) in zip(S_summ, M_summ, names, y_coords):

            vm = np.zeros(s.shape, dtype=np.uint8)
            vm[y0:y1, :] = 1

            def append(f):
                self.S_summ.append(f(s))
                self.M_summ.append(f(m))
                self.names.append(name)
                yy, xx = np.where(f(vm) == 1)
                self.val_coords.append([min(yy), max(yy), min(xx), max(xx)])

            append(lambda x: x)
            append(np.fliplr)
            append(np.flipud)
            append(lambda x: np.rot90(x, 1))
            append(lambda x: np.rot90(x, 2))
            append(lambda x: np.rot90(x, 3))

    def on_epoch_end(self, epoch, logs={}):

        logger = logging.getLogger(funcname())
        logger.info('\n')
        tic = time()

        # Save weights from the training model and load them into validation model.
        path_weights = '%s/weights.tmp' % self.cpdir
        self.model.save_weights(path_weights)
        self.model_val.load_weights(path_weights)
        remove(path_weights)

        # Tracking precision, recall, f1 values.
        pp, rr, ff = [], [], []

        # Padding helper.
        _, hw, ww = self.model_val.input_shape
        pad = lambda x: np.pad(x, ((0, hw - x.shape[0]), (0, ww - x.shape[1])), 'reflect')

        for s, m, vc, name in zip(self.S_summ, self.M_summ, self.val_coords, self.names):

            # Coordinates for validation.
            y0, y1, x0, x1 = vc

            # Batch prediction with padding.
            batch = np.zeros((1,) + self.model_val.input_shape[1:])
            batch[0] = pad(s)
            mp = self.model_val.predict(batch)[0, :s.shape[0], :s.shape[1]]

            # Evaluate metrics masks within validation area.
            p, r, i, e, f = nf_mask_metrics(m[y0:y1, x0:x1], mp[y0:y1, x0:x1].round())
            pp.append(p)
            rr.append(r)
            ff.append(f)

            logger.info('%s p=%.3lf r=%.3lf f=%.3lf' % (name, p, r, f))

        # Compute validation score with added epsilon for early epochs.
        eps = 1e-4 * epoch if epoch else 0

        logs['val_nf_f1_mean'] = np.mean(ff) + eps
        logs['val_nf_f1_median'] = np.median(ff) + eps
        logs['val_nf_f1_min'] = np.min(ff) + eps
        logs['val_nf_f1_adj'] = np.mean(ff) * np.min(ff) + eps
        logs['val_nf_prec'] = np.mean(pp)
        logs['val_nf_reca'] = np.mean(rr)

        logger.info('mean precision  = %.3lf' % logs['val_nf_prec'])
        logger.info('mean recall     = %.3lf' % logs['val_nf_reca'])
        logger.info('mean f1         = %.3lf' % logs['val_nf_f1_mean'])
        logger.info('minimum f1      = %.3lf' % logs['val_nf_f1_min'])
        logger.info('median f1       = %.3lf' % logs['val_nf_f1_median'])
        logger.info('adjusted f1     = %.3lf' % logs['val_nf_f1_adj'])
        logger.info('validation time = %.3lf' % (time() - tic))


def unet_builder(window_shape=(128, 128), nb_filters_base=32, conv_kernel_init='he_normal',
                 conv_l2_lambda=0.0, prop_dropout_base=0.25, upsampling_or_transpose='transpose'):
    """Builds and returns the UNet architecture using Keras.

    Arguments:
        window_shape: tuple of two equivalent integers defining the input/output window shape.
        nb_filters_base: number of convolutional filters used at the first layer. This is doubled
            after every pooling layer, four times until the bottleneck layer, and then it gets
            divided by two four times to the output layer.
        conv_kernel_init: weight initialization for the convolutional kernels. He initialization
            is considered best-practice when using ReLU activations, as is the case in this network.
        conv_l2_lambda: lambda term for l2 regularization at each convolutional kernel.
        prop_dropout_base: proportion of dropout after the first pooling layer. Two-times the
            proportion is used after subsequent pooling layers on the downward pass.
        upsampling_or_transpose: whether to use Upsampling2D or Conv2DTranspose layers on the upward
            pass. The original paper used Conv2DTranspose ("Deconvolution").

    """

    from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, BatchNormalization, Lambda, Reshape, UpSampling2D, Activation
    from keras.models import Model
    from keras.regularizers import l2

    drp = prop_dropout_base
    nfb = nb_filters_base
    cki = conv_kernel_init
    cl2 = l2(conv_l2_lambda)

    def up_layer(nb_filters, x):
        if upsampling_or_transpose == 'transpose':
            return Conv2DTranspose(nb_filters, 2, strides=2, activation='relu', kernel_initializer=cki, kernel_regularizer=cl2)(x)
        else:
            return UpSampling2D()(x)

    def conv_layer(nb_filters, x):
        return Conv2D(nb_filters, 3, padding='same', activation='relu', kernel_initializer=cki, kernel_regularizer=cl2)(x)

    x = inputs = Input(window_shape)
    x = Lambda(lambda x: K.expand_dims(x))(x)
    x = BatchNormalization(axis=-1)(x)

    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    dc_0_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = Dropout(drp)(x)
    dc_1_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = Dropout(drp * 2)(x)
    dc_2_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = Dropout(drp * 2)(x)
    dc_3_out = x

    x = MaxPooling2D(2, strides=2)(x)
    x = conv_layer(nfb * 16, x)
    x = conv_layer(nfb * 16, x)
    x = up_layer(nfb * 8, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_3_out], axis=3)
    x = conv_layer(nfb * 8, x)
    x = conv_layer(nfb * 8, x)
    x = up_layer(nfb * 4, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_2_out], axis=3)
    x = conv_layer(nfb * 4, x)
    x = conv_layer(nfb * 4, x)
    x = up_layer(nfb * 2, x)
    x = Dropout(drp * 2)(x)

    x = concatenate([x, dc_1_out], axis=3)
    x = conv_layer(nfb * 2, x)
    x = conv_layer(nfb * 2, x)
    x = up_layer(nfb, x)
    x = Dropout(drp)(x)

    x = concatenate([x, dc_0_out], axis=3)
    x = conv_layer(nfb, x)
    x = conv_layer(nfb, x)
    x = Conv2D(2, 1)(x)
    x = Activation('softmax')(x)
    x = Lambda(lambda x: x[:, :, :, -1])(x)
    # x = Lambda(lambda x: x[:, :, :, -1], output_shape=window_shape)(x)

    return Model(inputs=inputs, outputs=x)


def _summarize_series(ds):
    assert 'series/mean' in ds
    summ = ds.get('series/mean')[...] * 1. / 2**16
    assert np.min(summ) >= 0
    assert np.max(summ) <= 1
    return summ


def _summarize_mask(ds):
    assert 'masks/max' in ds
    return ds.get('masks/max')[...]


class UNet2DSummary(object):

    def __init__(self, cpdir='%s/.deep-calcium-datasets/tmp' % path.expanduser('~'),
                 series_summary_func=_summarize_series,
                 mask_summary_func=_summarize_mask, net_builder=unet_builder):

        self.cpdir = cpdir
        self.net_builder = net_builder
        self.series_summary_func = series_summary_func
        self.mask_summary_func = mask_summary_func

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, datasets, weights_path=None, shape_trn=(96, 96), shape_val=(512, 512), batch_size_trn=32,
            batch_size_val=1, nb_steps_trn=200, nb_epochs=20, prop_trn=0.75, prop_val=0.25, keras_callbacks=[],
            optimizer=Adam(0.002), loss='binary_crossentropy'):
        """Constructs network based on parameters and trains with the given data.

        # Arguments
            datasets: List of HDF5 datasets. Each of these will be passed to self.series_summary_func and 
                self.mask_summary_func to compute its series and mask summaries, so the HDF5 structure 
                should be compatible with those functions.
            weights_path: filesystem path to weights that should be loaded into the network.
            shape_trn: (height, width) shape of the windows cropped for training.
            shape_val: (height, width) shape of the windows used for validation.
            batch_size_trn: Batch size used for training.
            batch_size_val: Batch size used for validation.
            nb_steps_trn: Number of updates per training epoch.
            prop_trn: Proportion of each summary image used to train, cropped from the top of the image.
            prop_val: Proportion of each summary image used to validate, cropped from the bottom of the image.
            keras_callbacks: List of callbacks appended to internal callbacks for training.
            optimizer: Instanitated keras optimizer.
            loss: Loss function, currently either binary_crossentropy or dice_squared from https://arxiv.org/abs/1606.04797.
        """

        assert len(shape_trn) == 2
        assert len(shape_val) == 2
        assert shape_trn[0] == shape_trn[1]
        assert shape_val[0] == shape_val[1]
        assert 0 < prop_trn < 1
        assert 0 < prop_val < 1
        assert loss in {'binary_crossentropy', 'dice_squared'}

        logger = logging.getLogger(funcname())

        # Define, compile neural net.
        model = self.net_builder(shape_trn)
        model_val = self.net_builder(shape_val)
        json.dump(model.to_json(), open('%s/model.json' % self.cpdir, 'w'), indent=2)

        # Metric: True positive proportion.
        def ytpos(yt, yp):
            size = K.sum(K.ones_like(yt))
            return K.sum(yt) / (size + K.epsilon())

        # Metric: Predicted positive proportion.
        def yppos(yt, yp):
            size = K.sum(K.ones_like(yp))
            return K.sum(K.round(yp)) / (size + K.epsilon())

        # Metric: Binary pixel-wise precision.
        def prec(yt, yp):
            yp = K.round(yp)
            tp = K.sum(yt * yp)
            fp = K.sum(K.clip(yp - yt, 0, 1))
            return tp / (tp + fp + K.epsilon())

        # Metric: Binary pixel-wise recall.
        def reca(yt, yp):
            yp = K.round(yp)
            tp = K.sum(yt * yp)
            fn = K.sum(K.clip(yt - yp, 0, 1))
            return tp / (tp + fn + K.epsilon())

        # Metric: Squared dice coefficient from VNet paper.
        def dice_squared(yt, yp):
            nmr = 2 * K.sum(yt * yp)
            dnm = K.sum(yt**2) + K.sum(yp**2) + K.epsilon()
            return (nmr / dnm)

        def dice_squared_loss(yt, yp):
            return 1 - dice_squared(yt, yp)

        if loss == 'dice_squared':
            loss = dice_squared_loss
        else:
            loss = 'binary_crossentropy'

        model.compile(optimizer=optimizer, loss=loss,
                      metrics=[dice_squared, ytpos, yppos, prec, reca])
        model.summary()

        if weights_path is not None:
            model.load_weights(weights_path)
            logger.info('Loaded weights from %s.' % weights_path)

        # Pre-compute summaries once to avoid problems with accessing HDF5.
        S_summ = [self.series_summary_func(ds) for ds in datasets]
        M_summ = [self.mask_summary_func(ds) for ds in datasets]

        # Define generators for training and validation data.
        y_coords_trn = [(0, int(s.shape[0] * prop_trn)) for s in S_summ]
        gen_trn = self.batch_gen_fit(
            S_summ, M_summ, y_coords_trn, batch_size_trn, shape_trn, nb_max_augment=15)

        # Validation setup.
        y_coords_val = [(s.shape[0] - int(s.shape[0] * prop_val), s.shape[0])
                        for s in S_summ]

        names = [ds.attrs['name'] for ds in datasets]

        callbacks = [
            ValidationMetricsCB(model_val, S_summ, M_summ,
                                names, y_coords_val, self.cpdir),
            CSVLogger('%s/metrics.csv' % self.cpdir),
            MetricsPlotCallback('%s/metrics.png' % self.cpdir,
                                '%s/metrics.csv' % self.cpdir),
            ModelCheckpoint('%s/weights_val_nf_f1_mean.hdf5' % self.cpdir, mode='max',
                            monitor='val_nf_f1_mean', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_nf_f1_mean', min_delta=1e-3,
                          patience=10, verbose=1, mode='max'),
            EarlyStopping(monitor='reca', min_delta=1e-3,
                          patience=4, verbose=1, mode='max'),
            EarlyStopping(monitor='prec', min_delta=1e-3,
                          patience=4, verbose=1, mode='max'),
            DeadReluDetector(next(gen_trn)[0], True)

        ] + keras_callbacks

        trained = model.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                                      callbacks=callbacks, verbose=1, max_q_size=100)

        return trained.history

    def batch_gen_fit(self, S_summ, M_summ, y_coords, batch_size, window_shape, nb_max_augment=0):
        """Builds and yields batches of image windows and corresponding mask windows for training.
        Includes random data augmentation.

        Arguments:
            S_summ: list of series summary images stored as individual 2D numpy arrays. They can be
                different sizes, which is why it's not all just one 3D numpy array.
            M_summ: list of mask summary images corresponding to the series summary images. E.g.
                M_summ[i] is the 2D mask corresponding to the 2D summary image at S_summ[i].
            y_coords: list of tuples defining the min and max y-coordinate (rows) that should be 
                sampled for generating batches. Order corresponds to the S_summ and M_summ arrays.
                E.g. if y_coords[i] = (0, 300), then the generator will only sample S_summ[i] and M_summ[i]
                from within that range of rows. This creates a separation such that one instance of the
                generator can be used for training and another instance for validation.
            window_shape: shape of the windows that should be sampled.
            nb_max_augment: the max number of random augmentations to apply. 

        """

        logger = logging.getLogger(funcname())
        rng = np.random
        hw, ww = window_shape

        # Define augmentation functions to operate on the frame and mask.
        def rot(a, b):
            deg = rng.randint(1, 360)
            a = transform.rotate(a, deg, mode='reflect', preserve_range=True)
            b = transform.rotate(b, deg, mode='reflect', preserve_range=True)
            return (a, b)

        def stretch(a, b):
            hw_, ww_ = rng.randint(hw, hw * 1.3), rng.randint(ww, ww * 1.3)
            resize = lambda x: transform.resize(x, (hw_, ww_), preserve_range=True)
            a, b = resize(a), resize(b)
            y0 = rng.randint(0, a.shape[0] + 1 - hw)
            x0 = rng.randint(0, a.shape[1] + 1 - ww)
            y1, x1 = y0 + hw, x0 + ww
            return a[y0:y1, x0:x1], b[y0:y1, x0:x1]

        augment_funcs = [
            lambda a, b: (a, b),                      # Identity.
            lambda a, b: (a[:, ::-1], b[:, ::-1]),    # Horizontal flip.
            lambda a, b: (a[::-1, :], b[::-1, :]),    # Vertical flip.
            lambda a, b: rot(a, b),                   # Free rotation.
            lambda a, b: stretch(a, b),               # Make larger and crop.
        ]

        # Pre-compute neuron locations for faster sampling.
        neuron_locs = []
        for ds_idx, m in enumerate(M_summ):
            ymin, ymax = y_coords[ds_idx]
            neuron_locs.append(zip(*np.where(m[ymin:ymax, :] == 1)))

        while True:

            # Shuffle the dataset sampling order for each new batch.
            ds_idxs = np.arange(len(S_summ))
            rng.shuffle(ds_idxs)
            ds_idxs = cycle(ds_idxs)

            # Empty batches to fill.
            s_batch = np.zeros((batch_size, hw, ww), dtype=np.float32)
            m_batch = np.zeros((batch_size, hw, ww), dtype=np.uint8)

            for b_idx in range(batch_size):

                # Pick from next dataset.
                ds_idx = next(ds_idxs)
                s, m = S_summ[ds_idx], M_summ[ds_idx]

                # Dimensions. Height constrained by y range.
                hs, ws = s.shape
                ymin, ymax = y_coords[ds_idx]

                # Pick a random neuron location within this mask to center the window.
                cy, cx = neuron_locs[ds_idx][rng.randint(0, len(neuron_locs[ds_idx]))]

                # Window boundaries with a random offset and extra care to stay in bounds.
                cy = min(max(ymin, cy + rng.randint(-5, 5)), ymax)
                cx = min(max(0, cx + rng.randint(-5, 5)), ws)
                y0 = max(ymin, int(cy - (hw / 2)))
                y1 = min(y0 + hw, ymax)
                x0 = max(0, int(cx - (ww / 2)))
                x1 = min(x0 + ww, ws)

                # Double check.
                assert ymin <= y0 <= y1 <= ymax
                assert 0 <= x0 <= x1 <= ws

                # Slice and store the window.
                m_batch[b_idx, :y1 - y0, :x1 - x0] = m[y0:y1, x0:x1]
                s_batch[b_idx, :y1 - y0, :x1 - x0] = s[y0:y1, x0:x1]

                # Random augmentations.
                nb_augment = rng.randint(0, nb_max_augment + 1)
                for aug in rng.choice(augment_funcs, nb_augment):
                    s_batch[b_idx], m_batch[b_idx] = aug(s_batch[b_idx], m_batch[b_idx])

            yield s_batch, m_batch

    def evaluate(self, datasets, weights_path=None, window_shape=(512, 512), save=False):
        """Evaluates predicted masks vs. true masks for the given sequences."""

        logger = logging.getLogger(funcname())

        model = self.net_builder(window_shape)
        if weights_path is not None:
            model.load_weights(weights_path)
            logger.info('Loaded weights from %s.' % weights_path)

        # Currently only supporting full-sized windows.
        assert window_shape == (512, 512), 'TODO: implement variable window sizes.'

        # Padding helper.
        _, hw, ww = model.input_shape
        pad = lambda x: np.pad(
            x, ((0, hw - x.shape[0]), (0, ww - x.shape[1])), mode='reflect')

        # Evaluate each sequence, mask pair.
        mean_prec, mean_reca, mean_comb = 0., 0., 0.
        for ds in datasets:
            name = ds.attrs['name']
            s = self.series_summary_func(ds)
            m = self.mask_summary_func(ds)
            hs, ws = s.shape

            # Pad and make prediction.
            s_batch = np.zeros((1, ) + window_shape)
            s_batch[0] = pad(s)
            mp = model.predict(s_batch)[0, :hs, :ws].round()

            # Track scores.
            prec, reca, incl, excl, comb = nf_mask_metrics(m, mp)
            logger.info('%s: prec=%.3lf, reca=%.3lf, incl=%.3lf, excl=%.3lf, comb=%.3lf' % (
                name, prec, reca, incl, excl, comb))
            mean_prec += prec / len(datasets)
            mean_reca += reca / len(datasets)
            mean_comb += comb / len(datasets)

            # Save mask and prediction.
            if save:
                imsave('%s/%s_mp.png' % (self.cpdir, name),
                       mask_outlines(s, [m, mp], ['blue', 'red']))

        logger.info('Mean prec=%.3lf, reca=%.3lf, comb=%.3lf' %
                    (mean_prec, mean_reca, mean_comb))

        return mean_comb

    def predict(self, datasets, weights_path=None, window_shape=(512, 512), batch_size=10, save=False):
        """Predicts masks for the given sequences. Optionally saves the masks. Returns the masks as numpy arrays in order corresponding the given sequences."""

        logger = logging.getLogger(funcname())

        model = self.net_builder(window_shape)
        if weights_path is not None:
            model.load_weights(weights_path)
            logger.info('Loaded weights from %s.' % weights_path)

        # Currently only supporting full-sized windows.
        assert window_shape == (512, 512), 'TODO: implement variable window sizes.'

        # Padding helper.
        _, hw, ww = model.input_shape
        pad = lambda x: np.pad(x, ((0, hw - x.shape[0]), (0, ww - x.shape[1])), 'reflect')

        # Store predictions.
        Mp = []

        # Evaluate each sequence, mask pair.
        mean_prec, mean_reca, mean_comb = 0., 0., 0.
        for ds in datasets:
            name = ds.attrs['name']
            s = self.series_summary_func(ds)
            hs, ws = s.shape

            # Pad and make prediction.
            s_batch = np.zeros((1, ) + window_shape)
            s_batch[0] = pad(s)
            mp = model.predict(s_batch)[0, :hs, :ws].round()
            assert mp.shape == s.shape
            Mp.append(mp)

            # Save prediction.
            if save:
                outlined = mask_outlines(s, [mp], ['red'])
                imsave('%s/%s_mp.png' % (self.cpdir, name), outlined)

            logger.info('%s prediction complete.' % name)

        return Mp
