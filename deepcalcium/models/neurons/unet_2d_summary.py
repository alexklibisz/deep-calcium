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
from deepcalcium.utils.keras_helpers import HistoryPlotCallback
from deepcalcium.utils.visuals import mask_outlines

class ValNFMetricsCallback(Callback):
    '''Compute neurofinder metrics on validation data.'''

    def __init__(self, S_summ, M_summ, names, window_shape, get_y_range, file_name):
        '''Break S_summ and M_summ into windows stored in batches. Each (s, m) gets
        a single batch. Batch indexing corresponds to names list.'''
        super(Callback, self).__init__()
        self.names = names
        self.S_summ = S_summ
        self.M_summ = M_summ
        self.file_name = file_name
        self.s_batches = []
        self.m_batches = []
        self.batch_coords = []
        self.history = { name: {'prec': [], 'reca': []} for name in self.names }
        pad = lambda x: np.pad(x, ((0, window_shape[0] - x.shape[0]), \
                              (0, window_shape[1] - x.shape[1])), mode='reflect')
        for s, m in zip(S_summ, M_summ):
            ymin, ymax = get_y_range(s.shape[0])
            s, m = s[ymin:ymax,:].copy(), m[ymin:ymax,:].copy()
            batch_size = int(ceil(s.shape[0] / window_shape[0]) * ceil(s.shape[1] / window_shape[1]))
            s_batch, m_batch = np.zeros((batch_size,) + window_shape), np.zeros((batch_size,) + window_shape)
            coords = []
            for y0 in xrange(0, s.shape[0], window_shape[0]):
                for x0 in xrange(0, s.shape[1], window_shape[1]):
                    y1, x1 = y0 + window_shape[0], x0 + window_shape[1]
                    s_batch[len(coords)] = pad(s[y0:y1,x0:x1])
                    m_batch[len(coords)] = pad(m[y0:y1,x0:x1])
                    coords.append((y0,y1,x0,x1))
            self.s_batches.append(s_batch)
            self.m_batches.append(m_batch)
            self.batch_coords.append(coords)
        
    def on_epoch_end(self, epoch, logs={}):
        '''Prediction on s_batches and reconstruction to compute metrics.'''
        logger = logging.getLogger(funcname())
        
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        # Reconstruct windows and compute the metrics.
        for idx in xrange(len(self.names)):
            name = self.names[idx]
            s = self.S_summ[idx]
            m = self.M_summ[idx]
            s_batch = self.s_batches[idx]
            m_batch = self.m_batches[idx]
            coords = self.batch_coords[idx]
            mp_batch = self.model.predict(s_batch)
            mp = np.zeros_like(m)
            for mp_wdw, (y0, y1, x0, x1) in zip(mp_batch, coords):
                shape = mp[y0:y1,x0:x1].shape
                mp[y0:y1,x0:x1] = mp_wdw[:shape[0],:shape[1]]
            prec, reca, incl, excl, comb = nf_mask_metrics(m, mp)
            print(np.sum(m), np.sum(mp))
            logger.info('%s: %-6.3lf %-6.3lf %-6.3lf %-6.3lf %-6.3lf' % \
                        (name, prec, reca, incl, excl, comb))
            self.history[name]['prec'].append(prec)
            self.history[name]['reca'].append(reca)
            
            
        return 
        
        # Plot history.
        linestyles = cycle(['-', '--'])
        colors = cycle(mcolors.cnames)
        plt.figure(figsize=(15, 8))
        for name in self.names:
            color=next(colors)
            plt.plot(self.history[name]['prec'], label='%s prec' % name, color=color, linestyle='-')
            plt.plot(self.history[name]['reca'], label='%s reca' % name, color=color, linestyle='--')
            plt.legend(ncol=2, loc='lower right')
            plt.ylim(0., 1.)
        plt.savefig(self.file_name, dpi=300)
        plt.close()

class ValSamplesCallback(Callback):
    '''Save files with the validation samples and their predicted masks.'''

    def __init__(self, batch_gen, cpdir):
        super(Callback, self).__init__()
        self.batch_gen = batch_gen
        self.cpdir = cpdir
        s_batch, m_batch = next(self.batch_gen)
        self.s_batch = s_batch
        self.m_batch = m_batch

    def on_epoch_end(self, epoch, logs={}):

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        s_batch, m_batch = self.s_batch, self.m_batch
        s_batch, m_batch = s_batch[:25], m_batch[:25]
        mp_batch = self.model.predict(s_batch)
        fig, _ = plt.subplots(len(s_batch), 3, figsize=(3, int(len(s_batch) * 0.4)))

        for ax in fig.axes:
            ax.axis('off')

        for i, (s, m, mp) in enumerate(zip(s_batch, m_batch, mp_batch)):
            outlined = mask_outlines(s, [m], ['blue'])
            outlined = mask_outlines(outlined, [mp.round()], ['red'])
            fig.axes[i * 3 + 0].imshow(outlined, cmap='gray')
            fig.axes[i * 3 + 1].imshow(m, cmap='gray')
            mp[0][0] = 1.
            fig.axes[i * 3 + 2].imshow(mp, cmap='gray')

        plt.suptitle('Epoch %d, val dice squared = %.3lf' %
                     (epoch, logs['val_dice_squared']))
        plt.savefig('%s/samples_val_%03d.png' % (self.cpdir, epoch), dpi=600)
        plt.savefig('%s/samples_val_latest.png' % (self.cpdir), dpi=600)
        plt.close()


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

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    dc_0_out = x = Dropout(0.0)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    dc_1_out = x = Dropout(0.0)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    dc_2_out = x = Dropout(0.0)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    dc_3_out = x = Dropout(0.0)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
    x = Dropout(0.0)(x)

    x = concatenate([x, dc_3_out], axis=3)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(128, 2, strides=2, activation='relu')(x)
    x = Dropout(0.0)(x)

    x = concatenate([x, dc_2_out], axis=3)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, 2, strides=2, activation='relu')(x)
    x = Dropout(0.0)(x)

    x = concatenate([x, dc_1_out], axis=3)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, 2, strides=2, activation='relu')(x)
    x = Dropout(0.0)(x)

    x = concatenate([x, dc_0_out], axis=3)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(2, 1, activation='softmax')(x)
    x = Lambda(lambda x: x[:, :, :, 1], output_shape=window_shape)(x)

    model = Model(inputs=inputs, outputs=x)

    # True positive proportion.
    def tp(yt, yp):
        return K.sum(K.round(yt)) / (K.sum(K.clip(yt, 1, 1)) + K.epsilon())

    # Predicted positive proportion.
    def pp(yt, yp):
        return K.sum(K.round(yp)) / (K.sum(K.clip(yp, 1, 1)) + K.epsilon())

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
        return (1 - dice_squared(yt, yp)) + K.clip(tp(yt, yp) - pp(yt, yp), 0, 1)

    model.compile(optimizer=Adam(0.0008),  # loss='binary_crossentropy',
                  loss=dice_squared_loss,
                  metrics=[dice_squared, tp, pp, prec, reca])

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

    def fit(self, S, M, weights_path=None, window_shape=(96, 96), batch_size=32, nb_steps_trn=150,
            nb_steps_val=100, nb_epochs=20, prop_trn=0.8, prop_val=0.2, keras_callbacks=[]):
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
                                     nb_max_augment=5)

        gyr_val = lambda hs: (hs - int(hs * prop_val), hs)
        gen_val = self.batch_gen_fit(S_summ, M_summ, batch_size, window_shape, get_y_range=gyr_val,
                                     nb_max_augment=1)
                                     
        names = [s.attrs['name'] for s in S]        
                                     
        callbacks = [
            HistoryPlotCallback('%s/history.png' % self.cpdir),
            ValNFMetricsCallback(S_summ, M_summ, names, window_shape, gyr_val, \
                                '%s/history_nf_metrics.png' % self.cpdir),
            ValSamplesCallback(gen_val, self.cpdir),
            CSVLogger('%s/training.csv' % self.cpdir),
            ModelCheckpoint('%s/weights_loss_val.hdf5' % self.cpdir, mode='min',
                            monitor='val_loss', save_best_only=True, verbose=1),
            ModelCheckpoint('%s/weights_loss_trn.hdf5' % self.cpdir, mode='min',
                            monitor='loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='dice_squared', factor=0.5, patience=3,
                              cooldown=2, min_lr=1e-4, verbose=1, mode='max'),
            #ReduceLROnPlateau(monitor='val_dice_squared', factor=0.8, patience=3,
            #                  cooldown=2, min_lr=1e-4, verbose=1, mode='max'),
            #EarlyStopping(monitor='val_loss', min_delta=1e-3,
            #              patience=10, verbose=1, mode='min')

        ] + keras_callbacks

        model.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                            validation_data=gen_val, validation_steps=nb_steps_val,
                            callbacks=callbacks, verbose=1, max_q_size=100)

    def batch_gen_fit(self, S_summ, M_summ, batch_size, window_shape, get_y_range, nb_max_augment=5):
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
            logger.info('%s: prec=%.3lf, reca=%.3lf, inc=%.3lf, excl=%.3lf, comb=%.3lf' % (
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
