# Pre-training the UNet2DS model on the ISBI 2012 EM Segmentation challenge.
# Findings from some quick experimentation:
# 1. Train on ISBI data up to validation F1=0.95, evaluate neurofinder with the
# resulting weights: all precision, recall, F1 = 0.0.
# 2. Train on ISBI data up to validation F1=0.95, initialize with the weights and
# re-train full network for Neurofinder: reaches validation F1 = 0.85 within six
# epochs and scores ~0.52 on the test data (this is faster than training from scratch).
# 3. Train on ISBI data up to validation F1=0.95, initialize with the weights,
# freeze all but the last conv and softmax layers: reaches precision ~0.15, recall < 0.1.
# There's a noteworthy difference in the scale and distribution of the Neurofidner
# and ISBI datasets. ISBI has values in [0,255] with a mean ~125. Neurofinder has
# [0,65535] with 1000 < mean < 10000 depending on the dataset. So the Neurofinder
# data in general is much darker. Maybe there is some pre-processing that could
# account for this and improve the results.
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import logging
import numpy as np
import tensorflow as tf
import keras.backend as K
import tifffile as tif

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import UNet2DSummary
from deepcalcium.datasets.nf import nf_load_hdf5

np.random.seed(865)
tf.set_random_seed(7535)
logging.basicConfig(level=logging.INFO)

CPDIR = 'checkpoints/unet2ds_isbi_nf'
ISBI_WEIGHTS_PATH = '%s/unet_isbi_val_F1.hdf5' % CPDIR


def train_isbi():
    """Use the UNet2DS architecture to train a model from scratch on ISBI data. 
    Use the stored weights and the unet2ds_nf script to evaluate directly on the
    Neurofinder data."""

    nb_epochs = 20
    nb_steps_trn = 150
    nb_steps_val = 12
    batch_size = 16
    window_shape = (256, 256)

    def batch_gen(imgs, msks, augment=False):
        """Generator for randomly sampling training and validation data
        with minimal augmentations."""

        rng = np.random
        batch_shape = (batch_size,) + window_shape
        h, w = window_shape

        # Identity, flipping, rotation.
        augment_funcs = [lambda x: x] + \
            [lambda x: np.fliplr(x), lambda x: np.flipud(x)] + \
            [lambda x: np.rot90(x, n) for n in range(4)]

        while True:
            ib, mb = np.zeros(batch_shape), np.zeros(batch_shape)
            for bidx in range(batch_size):
                # Random window for image and mask.
                iidx = rng.randint(0, imgs.shape[0])
                y0 = rng.randint(0, imgs.shape[1] - h)
                x0 = rng.randint(0, imgs.shape[2] - w)
                y1, x1 = y0 + h, x0 + w
                ib[bidx] = imgs[iidx, y0:y1, x0:x1]
                mb[bidx] = msks[iidx, y0:y1, x0:x1]

                # Apply random selection of augmentations to both image and mask.
                for f in rng.choice(augment_funcs, rng.randint(0, 5)):
                    ib[bidx], mb[bidx] = f(ib[bidx]), f(mb[bidx])

                # Sanity checks.
                assert np.min(ib) >= 0 and np.max(ib) <= 1
                assert np.min(mb) == 0 and np.max(mb) == 1

            yield ib, mb

    # Model, data, generators. Use the first 25 slices for training and the
    # last five for validation.
    model = UNet2DSummary(cpdir=CPDIR)
    net = model.net_builder_func(window_shape)
    imgs_trn = tif.imread('%s/isbi-train-volume.tif' % CPDIR)[:25, :, :] / 255.
    msks_trn = tif.imread('%s/isbi-train-labels.tif' % CPDIR)[:25, :, :] / 255.
    imgs_val = tif.imread('%s/isbi-train-volume.tif' % CPDIR)[25:, :, :] / 255.
    msks_val = tif.imread('%s/isbi-train-labels.tif' % CPDIR)[25:, :, :] / 255.
    gen_trn = batch_gen(imgs_trn, msks_trn, augment=True)
    gen_val = batch_gen(imgs_val, msks_val, augment=False)

    def prec(yt, yp):
        yp = K.round(yp)
        return K.sum(yp * yt) / (K.sum(yp) + K.epsilon())

    def reca(yt, yp):
        yp = K.round(yp)
        tp = K.sum(yp * yt)
        fn = K.sum(K.clip(yt - yp, 0, 1))
        return K.sum(yp * yt) / (tp + fn + K.epsilon())

    def F1(yt, yp):
        p = prec(yt, yp)
        r = reca(yt, yp)
        return (2 * p * r) / (p + r + K.epsilon())

    cb = [ModelCheckpoint(ISBI_WEIGHTS_PATH, monitor='val_F1',
                          mode='max', save_best_only=True, verbose=1)]

    net.compile(optimizer=Adam(0.0007), loss='binary_crossentropy',
                metrics=[F1, prec, reca])
    net.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                      callbacks=cb, verbose=1, validation_data=gen_val,
                      validation_steps=nb_steps_val)


def finetune_neurofinder(method='full'):
    """Starting with weights trained on ISBI data, fine tune the network for Neurofinder data.
    Can fine-tune the full network or only the softmax layer. Once the weights have been saved,
    you can evaluate on the Neurofinder data using the unet2ds_nf.py script."""

    assert method in {'softmax', 'full'}

    # Load all sequences and masks as hdf5 File objects.
    ds_trn = nf_load_hdf5('all_train')

    # Modify the network builder function to freeze all but the last four layers
    # and lower the learning rate.
    if method == 'softmax':
        from deepcalcium.models.neurons.unet_2d_summary import _build_compile_unet

        def net_builder(window_shape):
            net = _build_compile_unet(window_shape)
            for layer in net.layers[:len(net.layers) - 3]:
                layer.trainable = False
            return net

        model = UNet2DSummary(cpdir=CPDIR, net_builder_func=net_builder)

    # Keep the default network.
    else:
        model = UNet2DSummary(cpdir=CPDIR)

    # Training.
    model.fit(
        ds_trn,
        model_path=ISBI_WEIGHTS_PATH,
        shape_trn=(128, 128),
        shape_val=(512, 512),
        batch_size_trn=20,
        nb_steps_trn=250,
        nb_epochs=10,
        keras_callbacks=[],
        prop_trn=0.70,
        prop_val=0.25
    )

    return


def cotrain():
    """TODO: Train UNet2DS model on Neurofinder and ISBI data at the same time."""
    pass

if __name__ == "__main__":

    # Un-comment the function you want to run.
    # train_isbi()
    finetune_neurofinder('full')
    # finetune_neurofinder('softmax')
