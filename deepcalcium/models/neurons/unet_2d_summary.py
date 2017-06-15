from __future__ import division, print_function
from itertools import cycle
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from math import ceil
from multiprocessing import Pool
from os import path, mkdir, remove
from scipy.misc import imsave
from skimage import transform
from time import time
from tqdm import tqdm
import logging
import numpy as np
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
        self.val_masks = []
        self.names = []
        self.cpdir = cpdir
        self.val_score_max = 0.

        # Store standard, flipped, rotated summary images and their corresponding
        # validation masks.
        for s, m, name, (y0, y1) in zip(S_summ, M_summ, names, y_coords):

            vm = np.zeros(s.shape, dtype=np.uint8)
            vm[y0:y1, :] = 1

            def append(f):
                self.S_summ.append(f(s))
                self.M_summ.append(f(m))
                self.val_masks.append(f(vm))
                self.names.append(name)

            append(lambda x: x)
            append(np.fliplr)
            append(np.flipud)
            append(lambda x: np.rot90(x, 1))
            append(lambda x: np.rot90(x, 2))
            append(lambda x: np.rot90(x, 3))

        return

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

        for s, m, vm, name in zip(self.S_summ, self.M_summ, self.val_masks, self.names):

            # Batch prediction with padding.
            batch = np.zeros((1,) + self.model_val.input_shape[1:])
            batch[0, :s.shape[0], :s.shape[1]] = s
            mp = self.model_val.predict(batch)[0, :s.shape[0], :s.shape[1]]

            # Evaluate metrics masks multiplied by validation mask.
            p, r, i, e, f = nf_mask_metrics(m * vm, mp.round() * vm)
            pp.append(p)
            rr.append(r)
            ff.append(f)

            logger.info('%s p=%.3lf r=%.3lf f=%.3lf' % (name, p, r, f))

        # Compute validation score with added epsilon for early epochs.
        eps = (1e-7 * epoch * int(epoch < 5))
        val_score = np.mean(ff) * np.min(ff) + eps

        logs['val_nf_adj'] = val_score
        logs['val_nf_prec'] = np.mean(pp)
        logs['val_nf_reca'] = np.mean(rr)
        logs['val_nf_f1'] = np.mean(ff)

        logger.info('mean precision  = %.3lf' % logs['val_nf_prec'])
        logger.info('mean recall     = %.3lf' % logs['val_nf_reca'])
        logger.info('mean f1         = %.3lf' % logs['val_nf_f1'])
        logger.info('min  f1         = %.3lf' % np.min(ff))
        logger.info('adjusted score  = %.3lf (%.3lf)' %
                    (val_score, (val_score - self.val_score_max)))
        logger.info('validation time = %.3lf' % (time() - tic))
        self.val_score_max = max(self.val_score_max, val_score)


def _build_compile_unet(window_shape, weights_path):
    '''Builds and compiles the keras UNet model. Can be replaced from outside the class if desired. Returns a compiled keras model.'''

    from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, BatchNormalization, Lambda, Reshape, UpSampling2D
    from keras.models import Model
    from keras.optimizers import Adam
    import keras.backend as K

    logger = logging.getLogger(funcname())

    x = inputs = Input(window_shape)

    x = Reshape(window_shape + (1,))(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.03)(x)

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    dc_0_out = x

    x = Dropout(0.08)(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    dc_1_out = x

    x = Dropout(0.1)(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    dc_2_out = x

    x = Dropout(0.1)(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    dc_3_out = x = Dropout(0.5)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = concatenate([x, dc_3_out], axis=3)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(128, 2, strides=2, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = concatenate([x, dc_2_out], axis=3)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 2, strides=2, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = concatenate([x, dc_1_out], axis=3)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, 2, strides=2, activation='relu')(x)
    x = Dropout(0.08)(x)

    x = concatenate([x, dc_0_out], axis=3)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(2, 1, activation='softmax')(x)
    x = Lambda(lambda x: x[:, :, :, 1], output_shape=window_shape)(x)

    model = Model(inputs=inputs, outputs=x)

    # True positive proportion.
    def ytpos(yt, yp):
        size = K.sum(K.ones_like(yt))
        return K.sum(yt) / (size + K.epsilon())

    # Predicted positive proportion.
    def yppos(yt, yp):
        size = K.sum(K.ones_like(yp))
        return K.sum(K.round(yp)) / (size + K.epsilon())

    def prec(yt, yp):
        yp = K.round(yp)
        tp = K.sum(yt * yp)
        fp = K.sum(K.clip(yp - yt, 0, 1))
        return tp / (tp + fp + K.epsilon())

    def reca(yt, yp):
        yp = K.round(yp)
        tp = K.sum(yt * yp)
        fn = K.sum(K.clip(yt - yp, 0, 1))
        return tp / (tp + fn + K.epsilon())

    def dice_squared(yt, yp):
        nmr = 2 * K.sum(yt * yp)
        dnm = K.sum(yt**2) + K.sum(yp**2) + K.epsilon()
        return (nmr / dnm)

    def dice_squared_loss(yt, yp):
        return (1 - dice_squared(yt, yp))  # + K.abs(ytpos(yt, yp) - yppos(yt, yp))

    model.compile(optimizer=Adam(0.002), loss='binary_crossentropy',
                  metrics=[dice_squared, ytpos, yppos, prec, reca])

    if weights_path is not None:
        model.load_weights(weights_path)
        logger.info('Loaded weights from %s.' % weights_path)

    return model


def _summarize_sequence(ds):
    assert 'series/mean' in ds
    summ = ds.get('series/mean')[...] * 1. / 2**16
    assert np.min(summ) >= 0
    assert np.max(summ) <= 1
    return summ


def _summarize_mask(ds):
    assert 'masks/max' in ds
    return ds.get('masks/max')[...]


class UNet2DSummary(object):

    def __init__(self, cpdir, sequence_summary_func=_summarize_sequence,
                 mask_summary_func=_summarize_mask, model_builder=_build_compile_unet):

        self.cpdir = cpdir
        self.model_builder = model_builder
        self.sequence_summary_func = sequence_summary_func
        self.mask_summary_func = mask_summary_func

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, datasets, weights_path=None, shape_trn=(96, 96), shape_val=(512, 512), batch_size_trn=32,
            batch_size_val=1, nb_steps_trn=200, nb_epochs=20, prop_trn=0.75, prop_val=0.25, keras_callbacks=[]):
        '''Constructs network based on parameters and trains with the given data.'''

        assert len(shape_trn) == 2
        assert len(shape_val) == 2
        assert shape_trn[0] == shape_trn[1]
        assert shape_val[0] == shape_val[1]
        assert 0 < prop_trn < 1
        assert 0 < prop_val < 1

        logger = logging.getLogger(funcname())

        # Define, compile neural net.
        model = self.model_builder(shape_trn, weights_path)
        model.summary()

        # Pre-compute summaries once to avoid problems with accessing HDF5.
        S_summ = [self.sequence_summary_func(ds) for ds in datasets]
        M_summ = [self.mask_summary_func(ds) for ds in datasets]

        # Define generators for training and validation data.
        y_coords_trn = [(0, int(s.shape[0] * prop_trn)) for s in S_summ]
        gen_trn = self.batch_gen_fit(
            S_summ, M_summ, y_coords_trn, batch_size_trn, shape_trn, nb_max_augment=10)

        # Validation setup.
        y_coords_val = [(s.shape[0] - int(s.shape[0] * prop_val), s.shape[0])
                        for s in S_summ]
        names = [ds.attrs['name'] for ds in datasets]
        model_val = self.model_builder(shape_val, weights_path)

        callbacks = [
            ValidationMetricsCB(model_val, S_summ, M_summ,
                                names, y_coords_val, self.cpdir),
            #ValNFMetricsCallback(S_summ, M_summ, names, window_shape, gyr_val, self.cpdir),
            CSVLogger('%s/metrics.csv' % self.cpdir),
            MetricsPlotCallback('%s/metrics.png' % self.cpdir,
                                '%s/metrics.csv' % self.cpdir),
            ModelCheckpoint('%s/weights_val_nf_adj.hdf5' % self.cpdir, mode='max',
                            monitor='val_nf_adj', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_nf_adj', factor=0.5, patience=3,
                              cooldown=1, min_lr=1e-4, verbose=1, mode='max'),
            EarlyStopping(monitor='val_nf_adj', min_delta=1e-3,
                          patience=10, verbose=1, mode='max')

        ] + keras_callbacks

        model.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                            callbacks=callbacks, verbose=1, max_q_size=100)

    def batch_gen_fit(self, S_summ, M_summ, y_coords, batch_size, window_shape, nb_max_augment=0):
        '''Builds and yields random batches used for training.'''

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
            hw_, ww_ = rng.randint(hw, hw * 1.3), np.random.randint(ww, ww * 1.3)
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
            hsmin, hsmax = y_coords[ds_idx]
            neuron_locs.append(zip(*np.where(m[hsmin:hsmax, :] == 1)))

        # Cycle to sequentially sample datasets.
        ds_idxs = cycle(np.arange(len(S_summ)))

        while True:

            # Empty batches to fill.
            s_batch = np.zeros((batch_size, hw, ww), dtype=np.float32)
            m_batch = np.zeros((batch_size, hw, ww), dtype=np.uint8)

            for b_idx in range(batch_size):

                # Pick from next dataset.
                ds_idx = next(ds_idxs)
                s, m = S_summ[ds_idx], M_summ[ds_idx]

                # Dimensions. Height constrained by y range.
                hs, ws = s.shape
                hsmin, hsmax = y_coords[ds_idx]

                # Pick a random neuron location within this mask to center the window.
                cy, cx = neuron_locs[ds_idx][rng.randint(0, len(neuron_locs[ds_idx]))]

                # Window boundaries with a random offset and extra care to stay in bounds.
                cy = min(max(hsmin, cy + rng.randint(-5, 5)), hsmax)
                cx = min(max(0, cx + rng.randint(-5, 5)), ws)
                y0 = max(hsmin, int(cy - (hw / 2)))
                y1 = min(y0 + hw, hsmax)
                x0 = max(0, int(cx - (ww / 2)))
                x1 = min(x0 + ww, ws)

                # Double check.
                assert hsmin <= y0 <= y1 <= hsmax
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
        '''Evaluates predicted masks vs. true masks for the given sequences..'''

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        logger = logging.getLogger(funcname())

        model = self.model_builder(window_shape, weights_path)

        # Currently only supporting full-sized windows.
        assert window_shape == (512, 512), 'TODO: implement variable window sizes.'

        # Evaluate each sequence, mask pair.
        mean_prec, mean_reca, mean_comb = 0., 0., 0.
        for ds in datasets:
            name = ds.attrs['name']
            s = self.sequence_summary_func(ds)
            m = self.mask_summary_func(ds)
            hs, ws = s.shape

            # Pad and make prediction.
            s_batch = np.zeros((1, ) + window_shape)
            s_batch[0, :s.shape[0], :s.shape[1]] = s
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

    def predict(self, datasets, weights_path=None, window_shape=(512, 512), batch_size=10, save=False):
        '''Predicts masks for the given sequences. Optionally saves the masks. Returns the masks as numpy arrays in order corresponding the given sequences.'''

        logger = logging.getLogger(funcname())

        model = self.model_builder(window_shape, weights_path)

        # Currently only supporting full-sized windows.
        assert window_shape == (512, 512), 'TODO: implement variable window sizes.'

        # Store predictions.
        Mp = []

        # Evaluate each sequence, mask pair.
        mean_prec, mean_reca, mean_comb = 0., 0., 0.
        for ds in datasets:
            name = ds.attrs['name']
            s = self.sequence_summary_func(ds)
            hs, ws = s.shape

            # Pad and make prediction.
            s_batch = np.zeros((1, ) + window_shape)
            s_batch[0, :s.shape[0], :s.shape[1]] = s
            mp = model.predict(s_batch)[0, :hs, :ws].round()
            Mp.append(mp)

            # Save prediction.
            if save:
                outlined = mask_outlines(s, [mp], ['red'])
                imsave('%s/%s_mp.png' % (self.cpdir, name), outlined)

            logger.info('%s prediction complete.' % name)

        return Mp
