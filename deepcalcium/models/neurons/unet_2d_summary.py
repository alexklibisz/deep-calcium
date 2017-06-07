
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from os import path, mkdir


class TrnValScoresCallback(Callback):
    '''Callback used to set validation metrics. Makes predictions on larger windows than the ones used in training. Scores them using neurofinder metrics.'''

    def __init(self, S, M, val_proportion, val_random_mean, random_state):
        self.S = S
        self.M = M
        self.val_proportion = val_proportion
        self.val_random_mean = val_random_mean
        self.random_state = random_state

    def on_epoch_end(self, epoch, logs):

        logs['val_precision'] = 0.0
        logs['val_recall'] = 0.0
        logs['val_combined'] = 0.0

        return


class UNet2DSummary(object):

    def __init__(self, checkpoint_dir, summary_func, random_state):

        self.cpdir = checkpoint_dir
        self.sumfunc = summary_func
        self.rng = random_state

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, S, M, weights_path=None, window_shape=(96, 96), nb_epochs=20, batch_size=60, keras_callbacks=[],
            sample_frames_min=500, sample_frames_max=2500, val_proportion=0.25, val_random_mean=False):
        '''Constructs network based on parameters and trains with the given data.'''

        print('TODO: fit')
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
    def _batch_gen_trn(S, M, window_shape, trn_proportion, sample_frames_min, sample_frames_max, random_state):
        '''Builds and yields random batches used for training.'''

        return

    @staticmethod
    def _build_compile_unet(window_shape, weights_path):
        '''Builds and compiles the keras UNet model. Can be replaced from outside the class if desired. Returns a compiled keras model.'''

        return
