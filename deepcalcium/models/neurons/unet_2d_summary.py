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
import numpy as np


class ValScoresCallback(Callback):
    '''Callback used to set validation metrics. Makes predictions on larger windows than the ones used in training. Scores them using neurofinder metrics.'''

    def __init__(self, S, M, val_proportion, val_random_mean, random_state):
        super(Callback, self).__init__()
        self.S = S
        self.M = M
        self.val_proportion = val_proportion
        self.val_random_mean = val_random_mean
        self.rng = random_state

    def on_epoch_end(self, epoch, logs={}):

        logs['val_precision'] = 0.0
        logs['val_recall'] = 0.0
        logs['val_combined'] = 0.0

        return


def _build_compile_unet(window_shape, weights_path):
    '''Builds and compiles the keras UNet model. Can be replaced from outside the class if desired. Returns a compiled keras model.'''

    x = inputs = Input(window_shape)

    x = Reshape((*window_shape, 1))(x)
    x = Conv2D(10, 3, padding='same', activation='relu')(x)
    x = Conv2D(10, 3, padding='same', activation='relu')(x)
    dc_0_out = x = Dropout(0.1)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(20, 3, padding='same', activation='relu')(x)
    x = Conv2D(20, 3, padding='same', activation='relu')(x)
    dc_1_out = x = Dropout(0.1)(x)

    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(40, 3, padding='same', activation='relu')(x)
    x = Conv2D(40, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(20, 2, strides=2, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = concatenate([x, dc_1_out], axis=3)
    x = Conv2D(20, 3, padding='same', activation='relu')(x)
    x = Conv2D(20, 3, padding='same', activation='relu')(x)
    x = Conv2DTranspose(10, 2, strides=2, activation='relu')(x)
    x = Dropout(0.1)(x)

    x = concatenate([x, dc_0_out], axis=3)
    x = Conv2D(10, 3, padding='same', activation='relu')(x)
    x = Conv2D(10, 3, padding='same', activation='relu')(x)
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

    model.compile(optimizer=Adam(1e-3), loss=dice_squared_loss,
                  metrics=[dice_squared, tp, pp])

    return model


class UNet2DSummary(object):

    def __init__(self, checkpoint_dir, summary_func, random_state, model_builder=_build_compile_unet):

        self.cpdir = checkpoint_dir
        self.sumfunc = summary_func
        self.rng = random_state
        self.model_builder = model_builder

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, S, M, weights_path=None, window_shape=(96, 96), nb_epochs=20, batch_size=60,
            sample_frames=False, val_proportion=0.25, val_random_mean=False, keras_callbacks=[]):
        '''Constructs network based on parameters and trains with the given data.'''

        model = self.model_builder(window_shape, weights_path)

        gen_trn = self._batch_gen_trn(S, M, self.sumfunc, batch_size, window_shape, 1. - val_proportion, 
                                      sample_frames, self.rng)

        callbacks = [
            ValScoresCallback(S, M, val_proportion, val_random_mean, self.rng),
            CSVLogger('%s/training.csv' % self.cpdir),
            ReduceLROnPlateau(monitor='val_combined', factor=0.8, patience=2,
                              cooldown=2, min_lr=1e-4, verbose=1, mode='max'),
            EarlyStopping('val_combined', min_delta=1e-2,
                          patience=10, mode='max', verbose=1),
            ModelCheckpoint('%s/weights_val_combined.hdf5' % self.cpdir, mode='max',
                            monitor='val_combined', save_best_only=True, verbose=1)

        ] + keras_callbacks

        nb_steps = ceil(sum([m.get('m').shape[0] for m in M]) * 1. / batch_size)
        
        model.fit_generator(gen_trn, steps_per_epoch=nb_steps, epochs=nb_epochs,
                             callbacks=callbacks, verbose=1, workers=2, pickle_safe=True,
                             max_q_size=nb_steps)

        return

    def evaluate(self, S, M, weights_path, window_shape=(512, 512), batch_size=5, random_mean=False):
        '''Evaluates predicted masks vs. true masks for the given sequences..'''

        print('TODO: evaluate')
        return

    def predict(self, S, weights_path, window_shape=(512, 512), batch_size=5, save_to_checkpoint_dir=True):
        '''Predicts masks for the given sequences. Optionally saves the masks. Returns the masks as numpy arrays in order corresponding the given sequences.'''

        print('TODO: predict')
        return

    @staticmethod
    def _batch_gen_trn(S, M, sumfunc, batch_size, window_shape, trn_proportion, sample_frames, rng):
        '''Builds and yields random batches used for training.'''

        hw, ww = window_shape
        s_idxs = cycle(range(len(S)))
                        
        while True:
        
            # Empty batches to fill.
            s_batch = np.zeros((batch_size, hw, ww))
            m_batch = np.zeros((batch_size, hw, ww))
            
            for b_idx in range(batch_size):
            
                # Pick datasets sequentially.
                ds_idx = next(s_idxs)
                s, m = S[ds_idx].get('s'), M[ds_idx].get('m')
            
                # Dimensions.
                fs, hs, ws = s.shape
                hs, ws = int(hs * trn_proportion), int(ws * trn_proportion)
            
                # Pick a random mask (neuron) and compute its boundaries with a random offset.
                m_idx = rng.randint(0, m.shape[0])
                yy, xx = np.where(m[m_idx] == 1)
                hn, wn = np.max(yy) - np.min(yy), np.max(xx) - np.min(xx)  # Height, width
                cy = np.min(yy) + int(hn / 2)
                cx = np.min(xx) + int(wn / 2)
                cy = min(max(0, cy + rng.randint(-hn, hn)), hs)
                cx = min(max(0, cx + rng.randint(-wn, wn)), ws)
                y0 = min(max(0, int(cy - (hw / 2))), hs - 1 - hw)
                x0 = min(max(0, int(cx - (ww / 2))), ws - 1 - ww)
                y1, x1 = y0 + hw, x0 + ww
                
                # Mask window.
                m_wdw = np.max(m[:, y0:y1, x0:x1], axis=0)
                
                # Summarized sequence window.
                if sample_frames:
                    f0 = rng.randint(0, fs - 100)
                    f1 = rng.randint(f0 + 99, fs)
                    s_wdw = sumfunc(s[f0:f1, y0:y1, x0:x1])
                else:
                    s_wdw = sumfunc(s[:, y0:y1, x0:x1])
                    
                m_batch[b_idx] = m_wdw
                s_batch[b_idx] = s_wdw / 255.
                
            yield s_batch, m_batch
