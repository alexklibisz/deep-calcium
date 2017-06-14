from __future__ import division
from itertools import cycle
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from math import ceil
from multiprocessing import Pool
from os import path, mkdir
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


class ValNFMetricsCallback(Callback):
    '''Compute neurofinder metrics on validation data.'''

    def __init__(self, S_summ, M_summ, names, window_shape, get_y_range, cpdir):
        '''Break S_summ and M_summ into windows stored in batches. Each (s, m) gets
        a single batch. Batch indexing corresponds to names list.'''
        super(Callback, self).__init__()
        self.names = names
        self.cpdir = cpdir
        self.s_batches = []
        self.batch_coords = []
        pad = lambda x: np.pad(x, ((0, window_shape[0] - x.shape[0]),
                                   (0, window_shape[1] - x.shape[1])), mode='reflect')

        # Crop the s and m summaries.
        self.S_summ, self.M_summ = [], []
        for s, m in zip(S_summ, M_summ):
            ymin, ymax = get_y_range(s.shape[0])
            self.S_summ.append(s[ymin:ymax, :])
            self.M_summ.append(m[ymin:ymax, :])

        # Split into batches of windows and lists of their coordinates.
        for s, m in zip(self.S_summ, self.M_summ):
            batch_size = int(ceil(s.shape[0] / window_shape[0])
                             * ceil(s.shape[1] / window_shape[1]))
            s_batch = np.zeros((batch_size,) + window_shape)
            coords = []
            for y0 in xrange(0, s.shape[0], window_shape[0]):
                for x0 in xrange(0, s.shape[1], window_shape[1]):
                    y1, x1 = y0 + window_shape[0], x0 + window_shape[1]
                    s_batch[len(coords)] = pad(s[y0:y1, x0:x1])
                    coords.append((y0, y1, x0, x1))
            self.s_batches.append(s_batch)
            self.batch_coords.append(coords)

    def on_epoch_end(self, epoch, logs={}):
        '''Prediction on s_batches and reconstruction to compute metrics.'''
        logger = logging.getLogger(funcname())
        logger.info('\n')
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        cols = 4
        fig, _ = plt.subplots(len(self.names), cols,
                              figsize=(10, len(self.names) * 1.5))
        logs['val_nf_prec'] = 0.
        logs['val_nf_reca'] = 0.
        logs['val_nf_comb'] = 0.

        # Predict batch, reconstruct window, compute the metrics, make and save plots.
        for idx in xrange(len(self.names)):
            name = self.names[idx]
            s = self.S_summ[idx]
            m = self.M_summ[idx]
            coords = self.batch_coords[idx]
            mp_batch = self.model.predict(self.s_batches[idx])
            mp = np.zeros(m.shape)
            for mp_wdw, (y0, y1, x0, x1) in zip(mp_batch, coords):
                shape = mp[y0:y1, x0:x1].shape
                mp[y0:y1, x0:x1] = mp_wdw[:shape[0], :shape[1]]

            prec, reca, incl, excl, comb = nf_mask_metrics(m, mp.round())
            logs['val_nf_prec'] += prec / len(self.names)
            logs['val_nf_reca'] += reca / len(self.names)
            logs['val_nf_comb'] += comb / len(self.names)
            logger.info('%s: prec=%-6.3lf reca=%-6.3lf incl=%-6.3lf excl=%-6.3lf comb=%-6.3lf' %
                        (name, prec, reca, incl, excl, comb))

            outlined = mask_outlines(s, [m], ['blue'])
            outlined = mask_outlines(outlined, [mp.round()], ['red'])
            fig.axes[cols * idx + 0].imshow(outlined)
            fig.axes[cols * idx + 0].axis('off')
            fig.axes[cols * idx + 1].set_title('%s: p=%.3lf, r=%.3lf, c=%.3lf'
                                               % (name, prec, reca, comb), size=8)
            fig.axes[cols * idx + 1].imshow(m, cmap='gray')
            fig.axes[cols * idx + 1].axis('off')
            fig.axes[cols * idx + 2].imshow(mp.round(), cmap='gray')
            fig.axes[cols * idx + 2].axis('off')

            # Histogram of activations for neuron pixels.
            yy, xx = np.where(m == 1)
            act = mp[yy, xx]
            tp = act[np.where(act >= 0.5)]
            fn = act[np.where(act < 0.5)]
            fig.axes[cols * idx + 3].hist(tp, color='red', label='TP')
            fig.axes[cols * idx + 3].hist(fn, color='black', label='FN')
            fig.axes[cols * idx + 3].set_xlim(0., 1.)
            fig.axes[cols * idx + 3].tick_params(axis='y', labelsize=6)
            fig.axes[cols * idx + 3].tick_params(axis='x', labelsize=6)
            fig.axes[cols * idx + 3].legend()

        logger.info('mean prec=%-6.3lf, mean reca=%-6.3lf, mean comb=%-6.3lf' %
                    (logs['val_nf_prec'], logs['val_nf_reca'], logs['val_nf_comb']))

        plt.savefig('%s/samples_val_%03d.png' % (self.cpdir, epoch), dpi=300)
        plt.savefig('%s/samples_val_latest.png' % self.cpdir, dpi=300)
        plt.close()


def plot_windows(s, m, png_path):
    pass


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
    x = Dropout(0.01)(x)

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    dc_0_out = x
    x = Dropout(0.05)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    dc_1_out = x
    x = Dropout(0.05)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    dc_2_out = x
    x = Dropout(0.05)(x)

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
    x = Dropout(0.05)(x)

    x = concatenate([x, dc_2_out], axis=3)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 2, strides=2, activation='relu')(x)
    x = Dropout(0.05)(x)

    x = concatenate([x, dc_1_out], axis=3)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, 2, strides=2, activation='relu')(x)
    x = Dropout(0.05)(x)

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


def _summarize_sequence(s):
    return s.get('summary_mean')[...]


def _summarize_mask(m):
    return np.max(m.get('m'), axis=0)


class UNet2DSummary(object):

    def __init__(self, cpdir, sequence_summary_func=_summarize_sequence,
                 mask_summary_func=_summarize_mask, model_builder=_build_compile_unet):

        self.cpdir = cpdir
        self.model_builder = model_builder
        self.sequence_summary_func = sequence_summary_func
        self.mask_summary_func = mask_summary_func

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, S, M, weights_path=None, window_shape=(96, 96), batch_size=32, nb_steps_trn=200,
            nb_epochs=20, prop_trn=0.8, prop_val=0.2, keras_callbacks=[]):
        '''Constructs network based on parameters and trains with the given data.'''

        assert len(S) == len(M)
        assert len(window_shape) == 2
        assert window_shape[0] == window_shape[1]
        assert 0 < prop_trn < 1
        assert 0 < prop_val < 1

        logger = logging.getLogger(funcname())

        # Define, compile neural net.
        model = self.model_builder(window_shape, weights_path)
        model.summary()

        # Pre-compute summaries once to avoid problems with accessing HDF5 objects
        # from multiple workers.
        S_summ = [self.sequence_summary_func(s) for s in S]
        M_summ = [self.mask_summary_func(m) for m in M]

        # Define generators for training and validation data.
        # Lambda functions define range of y indexes used for training and validation.
        gyr_trn = lambda hs: (0, int(hs * prop_trn))
        gen_trn = self.batch_gen_fit(S_summ, M_summ, batch_size, window_shape, get_y_range=gyr_trn,
                                     nb_max_augment=10)

        gyr_val = lambda hs: (hs - int(hs * prop_val), hs)
        names = [s.attrs['name'] for s in S]

        callbacks = [
            ValNFMetricsCallback(S_summ, M_summ, names,
                                 window_shape, gyr_val, self.cpdir),
            CSVLogger('%s/metrics.csv' % self.cpdir),
            MetricsPlotCallback('%s/metrics.png' % self.cpdir,
                                '%s/metrics.csv' % self.cpdir),
            ModelCheckpoint('%s/weights_nf_comb_val.hdf5' % self.cpdir, mode='max',
                            monitor='val_nf_comb', save_best_only=True, verbose=1),
            ModelCheckpoint('%s/weights_loss_trn.hdf5' % self.cpdir, mode='min',
                            monitor='loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_nf_comb', factor=0.5, patience=3,
                              cooldown=1, min_lr=1e-4, verbose=1, mode='max'),
            EarlyStopping(monitor='val_nf_comb', min_delta=1e-3,
                          patience=10, verbose=1, mode='max')

        ] + keras_callbacks

        model.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                            callbacks=callbacks, verbose=1, max_q_size=100)

    def batch_gen_fit(self, S_summ, M_summ, batch_size, window_shape, get_y_range, nb_max_augment=0):
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
        for m in M_summ:
            hsmin, hsmax = get_y_range(m.shape[0])
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
                hsmin, hsmax = get_y_range(hs)

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

    def evaluate(self, S, M, weights_path=None, window_shape=(512, 512), save=False):
        '''Evaluates predicted masks vs. true masks for the given sequences..'''

        logger = logging.getLogger(funcname())

        model = self.model_builder(window_shape, weights_path)

        # Pre-compute summaries.
        logger.info('Computing sequence and mask summaries.')
        S_summ = [self.sequence_summary_func(s) for s in tqdm(S)]
        M_summ = [self.mask_summary_func(m) for m in tqdm(M)]

        # Currently only supporting full-sized windows.
        assert window_shape == (512, 512), 'TODO: implement variable window sizes.'

        # Helper to pad up to window shape.
        wh, ww = window_shape
        pad = lambda s: np.pad(s, ((0, wh - s.shape[0]), (0, ww - s.shape[1])), 'reflect')

        # Evaluate each sequence, mask pair.
        mean_prec, mean_reca, mean_comb = 0., 0., 0.
        for i, (s, m) in enumerate(zip(S_summ, M_summ)):
            name = S[i].attrs['name']
            hs, ws = s.shape

            # Pad and make prediction.
            s_batch = np.zeros((1, ) + window_shape)
            s_batch[0] = pad(s)
            mp = model.predict(s_batch)[0, :hs, :ws].round()

            # Track scores.
            prec, reca, incl, excl, comb = nf_mask_metrics(m, mp)
            logger.info('%s: prec=%.3lf, reca=%.3lf, incl=%.3lf, excl=%.3lf, comb=%.3lf' % (
                name, prec, reca, incl, excl, comb))
            mean_prec += prec / len(S)
            mean_reca += reca / len(S)
            mean_comb += comb / len(S)

            # Save mask and prediction.
            if save:
                imsave('%s/%s_m.png' % (self.cpdir, name), m * 255)
                imsave('%s/%s_mp.png' % (self.cpdir, name), mp * 255)

        logger.info('Mean prec=%.3lf, reca=%.3lf, comb=%.3lf' %
                    (mean_prec, mean_reca, mean_comb))

    def predict(self, S, weights_path=None, window_shape=(512, 512), batch_size=10, save=False):
        '''Predicts masks for the given sequences. Optionally saves the masks. Returns the masks as numpy arrays in order corresponding the given sequences.'''

        logger = logging.getLogger(funcname())

        model = self.model_builder(window_shape, weights_path)

        # Pre-compute summaries.
        logger.info('Computing sequence and mask summaries.')
        S_summ = [self.sequence_summary_func(s) for s in tqdm(S)]

        # Currently only supporting full-sized windows.
        assert window_shape == (512, 512), 'TODO: implement variable window sizes.'

        # Helper to pad up to window shape.
        wh, ww = window_shape
        pad = lambda s: np.pad(s, ((0, wh - s.shape[0]), (0, ww - s.shape[1])), 'reflect')

        # Store predictions.
        Mp = []

        # Evaluate each sequence, mask pair.
        mean_prec, mean_reca, mean_comb = 0., 0., 0.
        for i, s in enumerate(S_summ):
            name = S[i].attrs['name']
            hs, ws = s.shape

            # Pad and make prediction.
            s_batch = np.zeros((1, ) + window_shape)
            s_batch[0] = pad(s)
            mp = model.predict(s_batch)[0, :hs, :ws].round()
            Mp.append(mp)

            # Save prediction.
            if save:
                imsave('%s/%s_mp.png' % (self.cpdir, name), mp * 255)

            logger.info('%s prediction complete.' % name)

        return Mp
