# U-Net 2D Summary module.
# The UNet2DS class is a wrapper around the UNet architecture that simplifies
# training and validation on calcium imaging segmentation tasks like Neurofinder.
# The code for building the UNet network can easily be extracted and re-used
# independent of the specific calcium-imaging features.
from __future__ import division, print_function
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
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
import pickle

from deepcalcium.utils.runtime import funcname
from deepcalcium.utils.config import CHECKPOINTS_DIR
from deepcalcium.datasets.nf import nf_mask_metrics
from deepcalcium.utils.keras_helpers import MetricsPlotCallback, load_model_with_new_input_shape
from deepcalcium.utils.neurons import F1, prec, reca, dice, dicesq, dice_loss, dicesq_loss, posyt, posyp, weighted_binary_crossentropy
from deepcalcium.utils.neurons import mask_outlines
from deepcalcium.utils.neurons import INVERTIBLE_2D_AUGMENTATIONS

MODEL_URL_LATEST = 'https://github.com/alexklibisz/deep-calcium/releases/download/weights-UNet2DS-0.0.1/model_val_F1_0.843_submission_0.569.hdf5'


class _ValidationMetricsCB(Callback):
    """Keras callback that evaluates validation metrics on full-size predictions during training."""

    def __init__(self, model_val, S_summ, M_summ, names, y_coords, scores_path=None):
        self.model_val = model_val
        self.S_summ = []
        self.M_summ = []
        self.val_coords = []
        self.names = []
        self.scores_path = scores_path

        # Standard, flipped, rotated summary images and corresponding validation masks.
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

        # Transfer weights from the training model to the validation model.
        self.model_val.set_weights(self.model.get_weights())

        # Tracking precision, recall, f1 values.
        pp, rr, ff = [], [], []
        name_to_f1 = {n: [] for n in self.names}

        # Padding helper.
        _, hw, ww = self.model_val.input_shape

        def pad(x): return np.pad(
            x, ((0, hw - x.shape[0]), (0, ww - x.shape[1])), 'reflect')

        for s, m, vc, name in zip(self.S_summ, self.M_summ, self.val_coords, self.names):

            # Coordinates for validation.
            y0, y1, x0, x1 = vc

            # Batch prediction with padding.
            [mp] = self.model_val.predict(pad(s)[np.newaxis, :, :])

            # Evaluate metrics masks within validation area.
            p, r, i, e, f = nf_mask_metrics(
                m[y0:y1, x0:x1], mp[y0:y1, x0:x1].round())
            pp.append(p)
            rr.append(r)
            ff.append(f)
            name_to_f1[name].append(f)

            logger.info('%s p=%.3lf r=%.3lf f=%.3lf' % (name, p, r, f))

        if self.scores_path:
            fp = open(self.scores_path, 'wb')
            pickle.dump(name_to_f1, fp)
            fp.close()

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
        logger.info('validation time = %.3lf seconds' % (time() - tic))


def unet(window_shape=(128, 128), nb_filters_base=32, conv_kernel_init='he_normal',
         prop_dropout_base=0.25, upsampling_or_transpose='transpose'):
    """Builds and returns the UNet architecture using Keras.

    # Arguments
        window_shape: tuple of two equivalent integers defining the input/output window shape.
        nb_filters_base: number of convolutional filters used at the first layer. This is doubled
            after every pooling layer, four times until the bottleneck layer, and then it gets
            divided by two four times to the output layer.
        conv_kernel_init: weight initialization for the convolutional kernels. He initialization
            is considered best-practice when using ReLU activations, as is the case in this network.
        prop_dropout_base: proportion of dropout after the first pooling layer. Two-times the
            proportion is used after subsequent pooling layers on the downward pass.
        upsampling_or_transpose: whether to use Upsampling2D or Conv2DTranspose layers on the upward
            pass. The original paper used Conv2DTranspose ("Deconvolution").

    # Returns
        model: Keras model, not compiled.

    """

    from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, BatchNormalization, Lambda, UpSampling2D, Activation
    from keras.models import Model

    drp = prop_dropout_base
    nfb = nb_filters_base
    cki = conv_kernel_init

    # Theano vs. TF setup.
    assert K.backend() == 'tensorflow', 'Theano implementation is incomplete.'

    def up_layer(nb_filters, x):
        if upsampling_or_transpose == 'transpose':
            x = Conv2DTranspose(nb_filters, 2, strides=2,
                                kernel_initializer=cki)(x)
            x = BatchNormalization(momentum=0.5)(x)
            return Activation('relu')(x)
        else:
            return UpSampling2D()(x)

    def conv_layer(nb_filters, x):
        x = Conv2D(nb_filters, (3, 3), strides=(1, 1),
                   padding='same', kernel_initializer=cki)(x)
        x = BatchNormalization(axis=-1)(x)
        return Activation('relu')(x)

    x = inputs = Input(window_shape)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

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
    x = Conv2D(2, 1, activation='softmax')(x)
    x = Lambda(lambda x: x[:, :, :, -1])(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def _summarize_series(dspath):
    """Default summary function for a series. Normalizes the mean summary using
    its mean and standard deviation.

    # Arguments
        dspath: Path to HDF5 dataset where the mask is stored.

    # Returns
        summ: (height x width) normalized mean summary.
    """
    fp = h5py.File(dspath)
    summ = fp.get('series/mean')[...].astype(np.float32)
    summ = (summ - np.mean(summ)) / np.std(summ)
    fp.close()
    return summ


def _summarize_mask(dspath):
    """Default summary function for a mask. Flattens the stack of neuron masks
    into a (height x width) combined mask. Eliminates overlapping and neighboring
    pixels that belong to different neurons to preserve the original number of
    independent neurons.

    # Arguments
        dspath: Path to HDF5 dataset where the mask is stored.

    # Returns
        summ: (height x width) mask summary.
    """

    fp = h5py.File(dspath)
    msks = fp.get('masks/raw')[...]
    fp.close()

    # Coordinates of all 1s in the stack of masks.
    zyx = list(zip(*np.where(msks == 1)))

    # Mapping (y,x) -> z.
    yx_z = {(y, x): [] for z, y, x in zyx}
    for z, y, x in zyx:
        yx_z[(y, x)].append(z)

    # Remove all elements with > 1 z.
    for k in list(yx_z.keys()):
        if len(yx_z[k]) > 1:
            del yx_z[k]
    assert np.max([len(v) for v in yx_z.values()]) == 1.

    # For (y,x), take the union of its z-values with its immediate neighbors' z-values.
    # Delete the (y,x) and its neighbors if |union| > 1.
    for y, x in list(yx_z.keys()):
        nbrs = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1), (y + 1, x + 1),
                (y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1)] + [(y, x)]
        nbrs = [k for k in nbrs if k in yx_z]
        allz = [yx_z[k][0] for k in nbrs]
        if len(np.unique(allz)) > 1:
            for k in nbrs:
                del yx_z[k]

    # The mask consists of the remaining (y,x) keys.
    yy, xx = [y for y, x in yx_z.keys()], [x for y, x in yx_z.keys()]
    summ = np.zeros(msks.shape[1:])
    summ[yy, xx] = 1.

    return summ


def _name_dataset(dspath):
    fp = h5py.File(dspath)
    name = fp.attrs['name']
    fp.close()
    return name


class UNet2DSummary(object):
    """Wrapper class for the UNet2DS model. The constructor arguments are mainly functions that
    make the model more composable. For example, you can easily evaluate a different kind of summary
    by passing in a different series summary function.

    # Arguments
        cpdir: checkpoint directory where training artifacts and predictions will be stored.
        dataset_name_func: function that returns a name for an HDF5 dataset given its path.
        series_summary_func: function that returns a summary image for a series given the path to its HDF5 dataset.
        mask_summary_func: function that returns a summary image for a mask given the path to its HDF5 dataset
        net_builder_func: function that builds and returns the Keras model used for training and predictions.
            This allows swapping out the network architecture without having to re-write or copy all of the
            training and prediction code.
    """

    def __init__(self, cpdir='%s/neurons_unet2ds' % CHECKPOINTS_DIR,
                 dataset_name_func=_name_dataset, series_summary_func=_summarize_series,
                 mask_summary_func=_summarize_mask, net_builder_func=unet):

        self.cpdir = cpdir
        self.dataset_name_func = dataset_name_func
        self.series_summary_func = series_summary_func
        self.mask_summary_func = mask_summary_func
        self.net_builder_func = net_builder_func

        if not path.exists(self.cpdir):
            mkdir(self.cpdir)

        cobj = [F1, prec, reca, dice, dicesq,
                posyt, posyp, dice_loss, dicesq_loss]
        self.custom_objects = {x.__name__: x for x in cobj}

    def fit(self, dataset_paths, model_path=None, proceed=False, shape_trn=(96, 96), shape_val=(512, 512),
            batch_size_trn=32, batch_size_val=1, nb_steps_trn=200, nb_epochs=20, prop_trn=0.75, prop_val=0.25,
            keras_callbacks=[], optimizer=Adam(0.002), loss='binary_crossentropy'):
        """Constructs network based on parameters and trains with the given data.

        # Arguments
            dataset_paths: Paths to HDF5 datasets. Each of these will be passed to self.series_summary_func and
                self.mask_summary_func to compute its series and mask summaries, so those functions should be
                compatible with the HDF5 structure.
            model_path: filesystem path to serialized model that should be loaded into the network.
            proceed: whether to continue training where the model left off or start over. Only relevant when a
                model_path is given because it uses the saved optimizer state.
            shape_trn: (height, width) shape of the windows cropped for training.
            shape_val: (height, width) shape of the windows used for validation.
            batch_size_trn: Batch size used for training.
            batch_size_val: Batch size used for validation.
            nb_steps_trn: Number of updates per training epoch.
            prop_trn: Proportion of each summary image used to train, cropped from the top of the image.
            prop_val: Proportion of each summary image used to validate, cropped from the bottom of the image.
            keras_callbacks: List of callbacks appended to internal callbacks for training.
            optimizer: Instanitated keras optimizer.
            loss: Loss function, one of binary_crossentropy, dice, or dice-squared from https://arxiv.org/abs/1606.04797.

        # Returns
            history: the Keras training history as a dictionary of metrics and their values after each epoch.
            model_path: path to the HDF5 file where the best architecture and weights were serialized.

        """

        # Error check.
        assert len(shape_trn) == 2
        assert len(shape_val) == 2
        assert shape_trn[0] == shape_trn[1]
        assert shape_val[0] == shape_val[1]
        assert 0 < prop_trn < 1
        assert 0 < prop_val < 1
        assert not (proceed and not model_path)

        # Setup loss function.
        losses = {
            'binary_crossentropy': binary_crossentropy,
            'weighted_binary_crossentropy': weighted_binary_crossentropy,
            'dice_loss': dice_loss,
            'dicesq_loss': dicesq_loss
        }
        assert loss in losses.keys() or loss.__name__ in \
            [f.__name__ for f in losses.values()]
        loss = losses[loss] if type(loss) == str else loss

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
            model_val = self.net_builder_func(shape_val)
            model.summary()

        # Recompile network if proceed is false.
        if not proceed:
            model.compile(optimizer=optimizer, loss=loss,
                          metrics=[F1, prec, reca, dice, dicesq, posyt, posyp])

        # Names and summaries.
        names = [self.dataset_name_func(dsp) for dsp in dataset_paths]
        S_summ = [self.series_summary_func(dsp) for dsp in dataset_paths]
        M_summ = [self.mask_summary_func(dsp) for dsp in dataset_paths]

        # Min and max y-coordinates for training and validation sets.
        ycval = [(s.shape[0] - int(s.shape[0] * prop_val), s.shape[0])
                 for s in S_summ]
        yctrn = [(0, int(s.shape[0] * prop_trn)) for s in S_summ]

        # Training generator.
        gen_trn = self._batch_gen(
            S_summ, M_summ, names, yctrn, batch_size_trn, nb_steps_trn, shape_trn, 15)

        # Timestamp to identify checkpoints.
        tic = int(time())

        callbacks = [
            _ValidationMetricsCB(model_val, S_summ, M_summ, names, ycval),
            CSVLogger('%s/%d_metrics.csv' % (self.cpdir, tic)),
            MetricsPlotCallback('%s/%d_metrics.png' % (self.cpdir, tic),
                                '%s/%d_metrics.csv' % (self.cpdir, tic)),
            ModelCheckpoint('%s/%d_model_{epoch:02d}_{val_nf_f1_mean:.3f}.hdf5' % (self.cpdir, tic), mode='max',
                            monitor='val_nf_f1_mean', save_best_only=False, verbose=1),
            ReduceLROnPlateau(monitor='F1', factor=0.5,
                              patience=5, min_lr=1e-4, mode='max'),
        ] + keras_callbacks

        trained = model.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                                      callbacks=callbacks, verbose=1, max_queue_size=1)

        return trained.history, '%s/model_val_nf_f1_mean.hdf5' % self.cpdir

    def _batch_gen(self, S_summ, M_summ, names, y_coords, batch_size, nb_steps, window_shape,
                   nb_max_augment=0, scores_path=None):
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

        rng = np.random
        hw, ww = window_shape
        nb_yields = 0

        # Define augmentation functions to operate on the frame and mask.
        augment_funcs = [
            lambda a, b: (a, b),                            # Identity.
            lambda a, b: (a[:, ::-1], b[:, ::-1]),          # Horizontal flip.
            lambda a, b: (a[::-1, :], b[::-1, :]),          # Vertical flip.
            lambda a, b: (np.rot90(a, 1), np.rot90(b, 1)),  # 90 deg rotations.
            lambda a, b: (np.rot90(a, 2), np.rot90(b, 2)),
            lambda a, b: (np.rot90(a, 3), np.rot90(b, 3)),
        ]

        # Pre-compute neuron locations for faster sampling.
        neuron_locs = []
        for ds_idx, m in enumerate(M_summ):
            ymin, ymax = y_coords[ds_idx]
            neuron_locs.append(list(zip(*np.where(m[ymin:ymax, :] == 1))))

        # Dataset indexes and default probability distribution for sampling
        # them.
        ds_idxs = np.arange(len(S_summ))
        ds_idxp = np.ones((len(ds_idxs))) / len(ds_idxs)

        while True:

            # Update sampling probabilities from scores file.
            if scores_path and os.path.exists(scores_path) and (nb_yields - 1) % nb_steps == 0:
                fp = open(scores_path, 'rb')
                names_to_scores = pickle.load(fp)
                fp.close()
                ds_idxp = np.array(
                    [1 - np.mean(names_to_scores[n]) for n in names])
                ds_idxp /= np.sum(ds_idxp)
                print([(name, '%.4lf' % p) for name, p in zip(names, ds_idxp)])

            # Empty batches to fill.
            s_batch = np.zeros((batch_size, hw, ww), dtype=np.float32)
            m_batch = np.zeros((batch_size, hw, ww), dtype=np.uint8)

            for b_idx in range(batch_size):

                # Sample next dataset.
                ds_idx = rng.choice(np.arange(len(S_summ)), p=ds_idxp)
                s, m = S_summ[ds_idx], M_summ[ds_idx]

                # Dimensions. Height constrained by y range.
                hs, ws = s.shape
                ymin, ymax = y_coords[ds_idx]

                # Pick a random neuron location within this mask to center the
                # window.
                cy, cx = neuron_locs[ds_idx][
                    rng.randint(0, len(neuron_locs[ds_idx]))]

                # Window boundaries with a random offset and extra care to stay
                # in bounds.
                cy = min(max(ymin, cy + rng.randint(-5, 5)), ymax)
                cx = min(max(0, cx + rng.randint(-5, 5)), ws)
                y0 = max(ymin, int(cy - (hw / 2)))
                y1 = min(y0 + hw, ymax)
                x0 = max(0, int(cx - (ww / 2)))
                x1 = min(x0 + ww, ws)

                # Slice and store the window.
                m_batch[b_idx, :y1 - y0, :x1 - x0] = m[y0:y1, x0:x1]
                s_batch[b_idx, :y1 - y0, :x1 - x0] = s[y0:y1, x0:x1]

                # Random augmentations.
                nb_augment = rng.randint(0, nb_max_augment + 1)
                for aug in rng.choice(augment_funcs, nb_augment):
                    s_batch[b_idx], m_batch[b_idx] = aug(
                        s_batch[b_idx], m_batch[b_idx])

            nb_yields += 1
            yield s_batch, m_batch

    def predict(self, dataset_paths, model_path, window_shape=(512, 512), print_scores=False,
                save=False, augmentation=False, threshold=0.5):
        """Make predictions on the given dataset_paths. Currently uses batches of 1.

        Arguments:
            dataset_paths: List of paths to HDF5 datasets. Each of these will be passed to self.series_summary_func and
                self.mask_summary_func to compute its series and mask summaries, so the HDF5 structure
                should be compatible with those functions.
            model_path: Path to the serialized Keras model HDF5 file. This file should include both the
                architecture and the weights.
            window_shape: Tuple window shape used for making predictions. Summary images with windows smaller
                than this are padded up to match this shape.
            print_scores: Flag to print the Neurofinder evaluation metrics. Only works when the datasets include
                ground-truth masks.
            save: Flag to save the predictions as PNGs with outlines around the predicted neurons in red. If
                the ground-truth masks are given, it will also show outlines around the groun-truth neurons.
            augmentation: Flag to perform 8x test-time augmentation. Predictions are made for each of the
                augmentations, the augmentation is inverted to its original orientation, and the average
                of all the augmentations is used as the prediction. In practice, this improved a
                Neurofinder submission from 0.5356 to 0.542.

        Returns:
            Mp: list of the predicted masks stored as Numpy arrays containing raw activation values.
            names: list of the dataset names, useful for making neurofinder submissions.

        """

        logger = logging.getLogger(funcname())
        model = load_model_with_new_input_shape(model_path, window_shape, compile=False,
                                                custom_objects=self.custom_objects)
        logger.info('Loaded model from %s.' % model_path)

        # Currently only supporting full-sized windows.
        assert window_shape == (
            512, 512), 'TODO: implement variable window sizes.'

        # Padding helper.
        def pad(x):
            _, hw, ww = model.input_shape
            return np.pad(x, ((0, hw - x.shape[0]), (0, ww - x.shape[1])), mode='reflect')

        # Store predicted masks and scores.
        Mp, names = [], []
        mean_prec, mean_reca, mean_comb = 0., 0., 0.

        # Evaluate each sequence, mask pair.
        for dsp in dataset_paths:
            name = self.dataset_name_func(dsp)
            s = self.series_summary_func(dsp)
            hs, ws = s.shape

            # Pad and make prediction(s).
            s_batch = pad(s)[np.newaxis, :, :]
            if augmentation:
                mp = np.zeros(s.shape)
                for _, aug, inv in INVERTIBLE_2D_AUGMENTATIONS:
                    mpaug = model.predict(aug(s_batch))
                    mp += inv(mpaug)[0, :hs, :ws] / \
                        len(INVERTIBLE_2D_AUGMENTATIONS)
            else:
                mp = model.predict(s_batch)[0, :hs, :ws]

            # Round about the given threshold and store prediction, name
            mp = (mp > threshold).astype(np.uint8)
            Mp.append(mp)
            names.append(name)

            # Track scores.
            if print_scores:
                m = self.mask_summary_func(dsp)
                prec, reca, incl, excl, comb = nf_mask_metrics(m, mp.round())
                logger.info('%s: prec=%.3lf, reca=%.3lf, incl=%.3lf, excl=%.3lf, comb=%.3lf' % (
                    name, prec, reca, incl, excl, comb))
                mean_prec += prec / len(dataset_paths)
                mean_reca += reca / len(dataset_paths)
                mean_comb += comb / len(dataset_paths)

            # Save mask and prediction.
            if save:
                if 'masks' in h5py.File(dsp):
                    m = self.mask_summary_func(dsp)
                    outlined = mask_outlines(
                        s, [m, mp.round()], ['blue', 'red'])
                else:
                    outlined = mask_outlines(s, [mp.round()], ['red'])
                save_path = '%s/%s_mp.png' % (self.cpdir, name)
                imsave(save_path, outlined)
                logger.info('Saved %s' % save_path)

        if print_scores:
            logger.info('Mean prec=%.3lf, reca=%.3lf, comb=%.3lf' %
                        (mean_prec, mean_reca, mean_comb))

        return Mp, names
