
from os import path, mkdir


class UNet2DSummary(object):

    def __init__(self, checkpoint_dir, summary_func, random_state):

        self.cpdir = checkpoint_dir
        self.sumfunc = summary_func
        self.rng = random_state

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

    def fit(S, M, weights_path, window_shape, nb_epochs, batch_size, keras_callbacks,
            sample_frames_min, sample_frames_max, val_proportion, val_random_average):

        print('TODO: fit')
        pass
        return

    def evaluate(S, M, weights_path, window_shape, batch_size):

        print('TODO: evaluate')
        pass
        return

    def predict(S, M, weights_path, window_shape, batch_size):

        print('TODO: predict')
        pass
        return
