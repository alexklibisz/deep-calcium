
from os import path, mkdir


class UNet2DSummary(object):

    def __init__(self, checkpoint_dir, summary_func, random_state):

        self.cpdir = checkpoint_dir
        self.sumfunc = summary_func
        self.rng = random_state

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(self, S, M, weights_path=None, window_shape=(96, 96), nb_epochs=20, batch_size=60, keras_callbacks=[],
            sample_frames_min=500, sample_frames_max=2500, val_proportion=0.25, val_random_mean=False):

        print('TODO: fit')
        return

    def evaluate(self, S, M, weights_path, window_shape, batch_size, random_mean=False):

        print('TODO: evaluate')
        return

    def predict(self, S, weights_path, window_shape, batch_size, save_to_checkpoint_dir):

        print('TODO: predict')
        return
