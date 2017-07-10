# Example of using UNet for frame-by-frame segmentation on the 3DEM
# dataset: http://cvlab.epfl.ch/data/em.
# The following metrics are reached after training for 10 epochs with
# keras' built-in binary_crossentropy loss.
# Training loss  0.01529115
# Training F1    0.94134005
# Training prec  0.94447301
# Training reca  0.93936959
# Training jacc  0.83308420
# Testing  loss  0.03041491
# Testing  F1    0.90911399
# Testing  prec  0.90598474
# Testing  reca  0.91567677
# Testing  jacc  0.78539054
from keras.models import load_model, Model
from keras.layers import Input, Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from time import time
import argparse
import logging
import numpy as np
import tensorflow as tf
import keras.backend as K
import tifffile as tif

import sys
sys.path.append('.')

from deepcalcium.models.neurons.unet_2d_summary import unet_builder
from deepcalcium.utils.keras_helpers import F1, prec, reca, jacc, load_model_with_new_input_shape

np.random.seed(865)
tf.set_random_seed(7535)


def batch_gen(imgs, msks, window_shape, batch_size, augment=False):
    """Generator for randomly sampling training and validation data
    with minimal augmentations."""

    rng = np.random
    batch_shape = (batch_size,) + window_shape
    h, w = window_shape

    # Identity, flipping, rotation.
    augment_funcs = [lambda x: x, lambda x: np.fliplr(x), lambda x: np.flipud(x)] + \
        [lambda x: np.rot90(x, n) for n in range(4)]

    # Compute the locations of cells in masks.
    cell_locs = [zip(*np.where(m == 1)) for m in msks]

    while True:
        ib, mb = np.zeros(batch_shape), np.zeros(batch_shape)
        for bidx in range(batch_size):

            # Window boundaries with a random offset and extra care to stay in bounds.
            iidx = rng.randint(0, imgs.shape[0])
            ymin, ymax = 0, imgs[iidx].shape[0]
            xmin, xmax = 0, imgs[iidx].shape[1]
            cidx = rng.randint(0, len(cell_locs[iidx]))
            cy, cx = cell_locs[iidx][cidx]
            cy = max((h / 2), min(cy, ymax - (h / 2)))
            cx = max((w / 2), min(cx, xmax - (w / 2)))
            y0, x0 = cy - (h / 2), cx - (w / 2)
            y1, x1 = y0 + h, x0 + w

            # Grab windows.
            ib[bidx] = imgs[iidx, y0:y1, x0:x1]
            mb[bidx] = msks[iidx, y0:y1, x0:x1]

            # Apply random selection of augmentations to both image and mask.
            for f in rng.choice(augment_funcs, rng.randint(0, 5)):
                ib[bidx], mb[bidx] = f(ib[bidx]), f(mb[bidx])

            # Sanity check.
            assert np.min(ib) >= 0 and np.max(ib) <= 1
            assert np.min(mb) == 0 and np.max(mb) == 1

        yield ib, mb


def train_3DEM(model_path, weights_path):
    """Use the UNet architecture to train a model from scratch on 3DEM data.
    Use the stored weights and the unet2ds_nf script to evaluate directly on the
    Neurofinder data."""

    cpdir = 'checkpoints/unet2d_3dem'
    model_save_path = '%s/model_tst_jacc.hdf5' % cpdir
    path_imgs_trn = '%s/training.tif' % cpdir
    path_msks_trn = '%s/training_groundtruth.tif' % cpdir
    path_imgs_tst = '%s/testing.tif' % cpdir
    path_msks_tst = '%s/testing_groundtruth.tif' % cpdir
    nb_epochs = 10
    nb_steps_trn = 150
    nb_steps_tst = 12
    batch_size = 16
    window_shape = (256, 256)

    # Import data and define generators.
    imgs_trn = tif.imread(path_imgs_trn) / 255.
    msks_trn = tif.imread(path_msks_trn) / 255.
    imgs_tst = tif.imread(path_imgs_tst) / 255.
    msks_tst = tif.imread(path_msks_tst) / 255.
    gen_trn = batch_gen(imgs_trn, msks_trn, window_shape, batch_size, augment=True)
    gen_tst = batch_gen(imgs_tst, msks_tst, window_shape, batch_size, augment=False)

    # Define UNet Keras model.
    if model_path:
        cobj = {'F1': F1, 'prec': prec, 'reca': reca, 'jacc': jacc, 'tf': tf}
        net = load_model_with_new_input_shape(model_path, window_shape,
                                              custom_objects=cobj)
    else:
        net = unet_builder(window_shape)

    if weights_path:
        net.load_weights(weights_path)

    # Training callbacks.
    cb = [
        ModelCheckpoint(model_save_path, monitor='val_jacc', mode='max',
                        save_best_only=True, verbose=1),
        CSVLogger('%s/history.csv' % cpdir)
    ]

    # Network compilation with loss and metrics.
    net.compile(optimizer=Adam(0.0007), loss='binary_crossentropy',
                metrics=[F1, prec, reca, jacc])

    # Train using generators for training and validation data.
    net.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=nb_epochs,
                      callbacks=cb, verbose=1, validation_data=gen_tst,
                      validation_steps=nb_steps_tst)


def predict_3DEM(model_path):

    cpdir = 'checkpoints/unet2d_3dem'
    model_save_path = '%s/model_tst_jacc.hdf5' % cpdir
    path_imgs_trn = '%s/training.tif' % cpdir
    path_msks_trn = '%s/training_groundtruth.tif' % cpdir
    path_imgs_tst = '%s/testing.tif' % cpdir
    path_msks_tst = '%s/testing_groundtruth.tif' % cpdir
    window_shape = (1024, 1024)

    # Load tiffs.
    print('Loading tiffs..')
    imgs_trn = tif.imread(path_imgs_trn) / 255.
    msks_trn = tif.imread(path_msks_trn) / 255.
    imgs_tst = tif.imread(path_imgs_tst) / 255.
    msks_tst = tif.imread(path_msks_tst) / 255.

    # Pad to fit window shape.
    print('Padding tiffs..')
    h, w = window_shape
    pad_shape = lambda x: ((0, 0), (0, h - x.shape[1]), (0, w - x.shape[2]))
    imgs_trn = np.pad(imgs_trn, pad_shape(imgs_trn), mode='reflect')
    msks_trn = np.pad(msks_trn, pad_shape(msks_trn), mode='reflect')
    imgs_tst = np.pad(imgs_tst, pad_shape(imgs_tst), mode='reflect')
    msks_tst = np.pad(msks_tst, pad_shape(msks_tst), mode='reflect')

    # Load net and set it to use window_shape input size.
    cobj = {'F1': F1, 'prec': prec, 'reca': reca, 'jacc': jacc, 'tf': tf}
    net = load_model_with_new_input_shape(model_path, window_shape, custom_objects=cobj)

    # Training evaluation.
    print('Training evaluation...')
    metrics_tsts = net.evaluate(imgs_trn, msks_trn, batch_size=1, verbose=0)
    for name, val in zip(net.metrics_names, metrics_tsts):
        print('Training %-5s %.8lf' % (name, val))

    # Testing evaluation.
    print('Testing  evaluation...')
    metrics_tsts = net.evaluate(imgs_tst, msks_tst, batch_size=1, verbose=0)
    for name, val in zip(net.metrics_names, metrics_tsts):
        print('Testing  %-5s %.8lf' % (name, val))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model runner.')
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    p = sub.add_parser('train', help='training')
    p.set_defaults(which='train')
    p.add_argument('-m', '--model', help='path to model hdf5 file.')
    p.add_argument('-w', '--weights', help='path to load weights from hdf5 file.')

    p = sub.add_parser('predict', help='training')
    p.set_defaults(which='predict')
    p.add_argument('-m', '--model', help='path to model hdf5 file.', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in {'train', 'predict'}

    if args['which'] == 'train':
        train_3DEM(args['model'], args['weights'])

    if args['which'] == 'predict':
        predict_3DEM(args['model'])
