from itertools import cycle
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, BatchNormalization, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from math import ceil
from os import path, mkdir
from time import time
from tqdm import tqdm
import keras.backend as K
import logging
import numpy as np
import sys

from deepcalcium.utils.runtime import funcname


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
        fig, _ = plt.subplots(len(s_batch), 4, figsize=(4, int(len(s_batch) * 0.4)))

        for ax in fig.axes:
            ax.axis('off')

        for i, (s, m, mp) in enumerate(zip(s_batch, m_batch, mp_batch)):
            fig.axes[i * 4 + 0].imshow(s, cmap='gray')
            fig.axes[i * 4 + 1].imshow(m, cmap='gray')
            fig.axes[i * 4 + 2].imshow(mp, cmap='gray')
            fig.axes[i * 4 + 3].imshow(mp.round(), cmap='gray')

        plt.suptitle('Epoch %d, val dice squared = %.3lf' %
                     (epoch, logs['val_dice_squared']))
        plt.savefig('%s/samples_val_%03d.png' % (self.cpdir, epoch), dpi=600)
        plt.savefig('%s/samples_val_latest.png' % (self.cpdir), dpi=600)
        plt.close()


def _build_compile_unet(window_shape, weights_path):
    '''Builds and compiles the keras UNet model. Can be replaced from outside the class if desired. Returns a compiled keras model.'''

    x = inputs = Input(window_shape)

    x = Reshape(window_shape + (1,))(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    dc_0_out = x = Dropout(0.1)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    dc_1_out = x = Dropout(0.1)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    dc_2_out = x = Dropout(0.1)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    dc_3_out = x = Dropout(0.1)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2D(512, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
    x = Dropout(0.1)(x)

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
    x = Dropout(0.1)(x)

    x = concatenate([x, dc_0_out], axis=3)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(2, 1, activation='softmax')(x)
    x = Lambda(lambda x: x[:, :, :, 1], output_shape=window_shape)(x)

    model = Model(inputs=inputs, outputs=x)

    def tp(yt, yp):
        return K.sum(K.round(yt)) / (K.sum(K.clip(yt, 1, 1)) + K.epsilon())

    def pp(yt, yp):
        return K.sum(K.round(yp)) / (K.sum(K.clip(yp, 1, 1)) + K.epsilon())

    def dice_squared(yt, yp):
        nmr = 2 * K.sum(yt * yp)
        dnm = K.sum(yt**2) + K.sum(yp**2) + K.epsilon()
        return (nmr / dnm)

    def dice_squared_loss(yt, yp):
        return 1 - dice_squared(yt, yp)

    model.compile(optimizer=Adam(0.0008), loss=dice_squared_loss,
                  metrics=[dice_squared, tp, pp])

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


class UNet2DSummary(object):

    def __init__(self, checkpoint_dir, model_builder=_build_compile_unet,
                 summfunc=lambda s: np.mean(s * 1. / 255., axis=0)):

        self.cpdir = checkpoint_dir
        self.model_builder = model_builder
        self.summfunc = summfunc

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, S, M, weights_path=None, window_shape=(96, 96), nb_epochs=20, batch_size=60,
            val_prop=0.25, keras_callbacks=[]):
        '''Constructs network based on parameters and trains with the given data.'''

        logger = logging.getLogger(funcname())

        # Define, compile neural net.
        model = self.model_builder(window_shape, weights_path)
        model.summary()

        # Pre-compute summaries for generators.
        logger.info('Pre-computing sequence and mask summaries.')
        S_summ = [self.summfunc(s.get('s')[...]) for s in tqdm(S)]
        M_summ = [np.max(m.get('m')[...], axis=0) for m in tqdm(M)]

        # Define generators for training and validation data.
        # Lambda function defines range of y vals used for training and validation.
        gen_trn = self._batch_gen_trn(S_summ, M_summ, batch_size, window_shape,
                                      get_y_range=lambda hs: (0, hs * (1 - val_prop)),
                                      nb_max_augment=2)

        gen_val = self._batch_gen_trn(S_summ, M_summ, batch_size, window_shape,
                                      get_y_range=lambda hs: (hs * (1 - val_prop), hs),
                                      nb_max_augment=2)

        callbacks = [
            ValSamplesCallback(gen_val, self.cpdir),
            CSVLogger('%s/training.csv' % self.cpdir),
            ReduceLROnPlateau(monitor='val_dice_squared', factor=0.8, patience=5,
                              cooldown=2, min_lr=1e-4, verbose=1, mode='max'),
            EarlyStopping('val_dice_squared', min_delta=1e-2,
                          patience=10, mode='max', verbose=1),
            ModelCheckpoint('%s/weights_val_combined.hdf5' % self.cpdir, mode='max',
                            monitor='val_dice_squared', save_best_only=True, verbose=1)

        ] + keras_callbacks

        # nb_steps_trn = ceil(sum([m.get('m').shape[0] for m in M]) * 1. / batch_size)
        nb_steps_trn = 100
        nb_steps_val = min(int(nb_steps_trn * 0.25), 30)

        model.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                            validation_data=gen_val, validation_steps=nb_steps_val,
                            callbacks=callbacks, verbose=1,
                            workers=2, pickle_safe=True, max_q_size=100)

    def evaluate(self, S, M, weights_path, window_shape=(512, 512), batch_size=5, random_mean=False):
        '''Evaluates predicted masks vs. true masks for the given sequences..'''

        print('TODO: evaluate')
        return

    def predict(self, S, weights_path, window_shape=(512, 512), batch_size=10, save_to_checkpoint_dir=False):
        '''Predicts masks for the given sequences. Optionally saves the masks. Returns the masks as numpy arrays in order corresponding the given sequences.'''

        print('TODO: predict')
        return

    def _batch_gen_trn(self, S_summ, M_summ, batch_size, window_shape, get_y_range, nb_max_augment=10):
        '''Builds and yields random batches used for training.'''

        rng = np.random
        hw, ww = window_shape

        augment_funcs = [
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x),
            lambda x: np.rot90(x, 1),
            lambda x: np.rot90(x, 2),
            lambda x: np.rot90(x, 3)
        ]

        while True:

            s_idxs = cycle(rng.choice(np.arange(len(S_summ)), len(S_summ)))

            # Empty batches to fill.
            s_batch = np.zeros((batch_size, hw, ww), dtype=np.float32)
            m_batch = np.zeros((batch_size, hw, ww), dtype=np.uint8)

            for b_idx in range(batch_size):

                # Pick datasets sequentially.
                ds_idx = next(s_idxs)
                s, m = S_summ[ds_idx], M_summ[ds_idx]

                # Dimensions. Height constrained by y range.
                hs, ws = s.shape
                hsmin, hsmax = get_y_range(hs)

                # Pick random window boundaries.
                y0 = rng.randint(hsmin, hsmax - 1 - hw)
                x0 = rng.randint(0, ws - 1 - ww)
                y1, x1 = y0 + hw, x0 + ww

                # Slice the window.
                m_batch[b_idx] = m[y0:y1, x0:x1]
                s_batch[b_idx] = s[y0:y1, x0:x1]

                # Random augmentations.
                nb_augment = rng.randint(0, nb_max_augment + 1)
                for aug in rng.choice(augment_funcs, nb_augment):
                    s_batch[b_idx] = aug(s_batch[b_idx])
                    m_batch[b_idx] = aug(m_batch[b_idx])

            yield s_batch, m_batch
